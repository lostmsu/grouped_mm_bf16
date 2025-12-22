from __future__ import annotations

from typing import Optional

import os
import sys

import torch

import triton
import triton.language as tl


# PyTorch enforces < 1024 groups; 2^10 = 1024.
MAX_GROUPS_LOG2 = tl.constexpr(10)


def _patch_triton_hip_windows() -> None:
    if sys.platform != "win32":
        return
    if getattr(torch.version, "hip", None) is None:
        return

    try:
        from triton.backends.amd import driver as amd_driver
    except Exception:
        return

    # Ensure HIP headers are discoverable for Triton's tiny C helpers on Windows.
    try:
        import _rocm_sdk_core  # type: ignore[import-not-found]

        rocm_root = os.path.dirname(_rocm_sdk_core.__file__)
        rocm_include = os.path.join(rocm_root, "include")
        if os.path.isdir(rocm_include) and rocm_include not in amd_driver.include_dirs:  # type: ignore[attr-defined]
            amd_driver.include_dirs.append(rocm_include)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Provide a Windows dlfcn shim for Triton's Linux-oriented helper C sources.
    compat_include = os.path.join(os.path.dirname(__file__), "compat")
    if os.path.isdir(compat_include):
        try:
            if compat_include not in amd_driver.include_dirs:  # type: ignore[attr-defined]
                amd_driver.include_dirs.append(compat_include)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Fix Triton AMD backend temp file handling on Windows:
    # it uses NamedTemporaryFile() and then re-opens the path, which fails on Windows.
    try:
        from triton.backends.amd import compiler as amd_compiler
        import tempfile

        def _make_hsaco_windows(src, metadata, options):
            target_features = ""
            if amd_compiler.knobs.compilation.enable_asan:  # type: ignore[attr-defined]
                target_features = "+xnack"
            hsaco = amd_compiler.amd.assemble_amdgcn(src, options.arch, target_features)  # type: ignore[attr-defined]

            tmp_out = tempfile.NamedTemporaryFile(delete=False)
            tmp_in = tempfile.NamedTemporaryFile(delete=False)
            tmp_out.close()
            tmp_in.close()
            try:
                with open(tmp_in.name, "wb") as fd_in:
                    fd_in.write(hsaco)
                amd_compiler.amd.link_hsaco(tmp_in.name, tmp_out.name)  # type: ignore[attr-defined]
                with open(tmp_out.name, "rb") as fd_out:
                    return fd_out.read()
            finally:
                try:
                    os.remove(tmp_in.name)
                except OSError:
                    pass
                try:
                    os.remove(tmp_out.name)
                except OSError:
                    pass

        amd_compiler.HIPBackend.make_hsaco = staticmethod(_make_hsaco_windows)  # type: ignore[attr-defined]
    except Exception:
        pass

    def _get_path_to_hip_runtime_dylib_windows() -> str:
        def _c_path(path: str) -> str:
            # This path is injected into a C string literal in Triton's `driver.c`;
            # backslashes would be parsed as escapes (e.g. \U...), so normalize.
            return path.replace("\\", "/")

        env_path = os.getenv("TRITON_LIBHIP_PATH") or os.getenv("TRITON_LIBHIP_DLL_PATH")
        if env_path and os.path.exists(env_path):
            return _c_path(env_path)

        attempted: list[str] = []

        def _try_dir(directory: str) -> Optional[str]:
            if not directory or not os.path.isdir(directory):
                return None
            candidates = [
                "amdhip64.dll",
                "amdhip64_7.dll",
                "amdhip64_6.dll",
                "amdhip64_5.dll",
            ]
            for name in candidates:
                path = os.path.join(directory, name)
                attempted.append(path)
                if os.path.exists(path):
                    return path
            try:
                for name in os.listdir(directory):
                    name_l = name.lower()
                    if name_l.startswith("amdhip64_") and name_l.endswith(".dll"):
                        path = os.path.join(directory, name)
                        attempted.append(path)
                        if os.path.exists(path):
                            return path
            except Exception:
                pass
            return None

        # ROCm SDK wheels on Windows (PyTorch ROCm dependency).
        try:
            import _rocm_sdk_core  # type: ignore[import-not-found]

            root = os.path.dirname(_rocm_sdk_core.__file__)
            hit = _try_dir(os.path.join(root, "bin"))
            if hit:
                return _c_path(hit)
        except Exception:
            pass

        # Common env vars.
        for base in (os.getenv("HIP_PATH"), os.getenv("ROCM_PATH"), os.getenv("ROCM_HOME")):
            if not base:
                continue
            for sub in ("bin", "lib", "lib64"):
                hit = _try_dir(os.path.join(base, sub))
                if hit:
                    return _c_path(hit)

        # Search PATH.
        for d in os.getenv("PATH", "").split(";"):
            hit = _try_dir(d)
            if hit:
                return _c_path(hit)

        raise RuntimeError(
            "Triton HIP on Windows could not locate an amdhip64 DLL. "
            "Set TRITON_LIBHIP_PATH to a full path to amdhip64_*.dll. "
            f"Attempted paths: {attempted[:20]}{' ...' if len(attempted) > 20 else ''}"
        )

    amd_driver._get_path_to_hip_runtime_dylib = _get_path_to_hip_runtime_dylib_windows  # type: ignore[attr-defined]


_patch_triton_hip_windows()


@triton.jit
def _upper_bound_offs_i32(offs_ptr, group_count: tl.constexpr, idx):
    low = tl.full(idx.shape, 0, dtype=tl.int32)
    high = tl.full(idx.shape, group_count, dtype=tl.int32)
    for _ in tl.static_range(MAX_GROUPS_LOG2):
        mid = (low + high) // 2
        mid_val = tl.load(offs_ptr + mid, mask=mid < group_count, other=0x7FFFFFFF).to(tl.int32)
        go_left = idx < mid_val
        high = tl.where(go_left, mid, high)
        low = tl.where(go_left, low, mid + 1)
    return low


def _as_tl_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise RuntimeError(f"Unsupported dtype: {dtype}")


@triton.jit
def _grouped_mm_3d3d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_ag: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_g = tl.program_id(axis=2)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_g = A_ptr + pid_g * stride_ag
    b_g = B_ptr + pid_g * stride_bg
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_g + m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_g + k[:, None] * stride_bk + n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)

    c = C_ptr + pid_g * stride_cg + m[:, None] * stride_cm + n[None, :] * stride_cn
    tl.store(c, tl.cast(acc, OUT_DTYPE), mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _grouped_mm_2d2d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    offs_ptr,
    G: tl.constexpr,
    M,
    N,
    K_TOTAL: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_g = tl.program_id(axis=2)

    # K segment for this group.
    k_end = tl.load(offs_ptr + pid_g).to(tl.int32)
    k_start = tl.load(offs_ptr + pid_g - 1, mask=pid_g > 0, other=0).to(tl.int32)
    k_len = k_end - k_start

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_TOTAL, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_in = k < k_len
        k_glob = k_start + k
        a = tl.load(
            A_ptr + m[:, None] * stride_am + k_glob[None, :] * stride_ak,
            mask=(m[:, None] < M) & k_in[None, :] & (k_glob[None, :] < K_TOTAL),
            other=0.0,
        )
        b = tl.load(
            B_ptr + k_glob[:, None] * stride_bk + n[None, :] * stride_bn,
            mask=k_in[:, None] & (k_glob[:, None] < K_TOTAL) & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)

    c = C_ptr + pid_g * stride_cg + m[:, None] * stride_cm + n[None, :] * stride_cn
    tl.store(c, tl.cast(acc, OUT_DTYPE), mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _grouped_mm_2d3d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    offs_ptr,
    G: tl.constexpr,
    M_TOTAL,
    N,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # row index
    pid_n = tl.program_id(axis=1)  # col-tile

    m = pid_m
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    g = _upper_bound_offs_i32(offs_ptr, G, m.to(tl.int32))
    b_g = B_ptr + g * stride_bg

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + m * stride_am + k * stride_ak,
            mask=(m < M_TOTAL) & (k < K),
            other=0.0,
        )
        b = tl.load(
            b_g + k[:, None] * stride_bk + n[None, :] * stride_bn,
            mask=(k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc += tl.sum(b.to(tl.float32) * a[:, None].to(tl.float32), axis=0)

    c = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c, tl.cast(acc, OUT_DTYPE), mask=(m < M_TOTAL) & (n < N))


@triton.jit
def _grouped_mm_3d2d_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    offs_ptr,
    G: tl.constexpr,
    M,
    N_TOTAL,
    K: tl.constexpr,
    stride_ag: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)  # column index (potentially large)
    pid_m = tl.program_id(axis=1)  # m-tile

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n

    g = _upper_bound_offs_i32(offs_ptr, G, n.to(tl.int32))
    a_g = A_ptr + g * stride_ag

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_g + m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            B_ptr + k * stride_bk + n * stride_bn,
            mask=k < K,
            other=0.0,
        )
        acc += tl.sum(a.to(tl.float32) * b[None, :].to(tl.float32), axis=1)

    c = C_ptr + m * stride_cm + n * stride_cn
    tl.store(c, tl.cast(acc, OUT_DTYPE), mask=(m < M) & (n < N_TOTAL))


def grouped_mm_3d3d(mat_a: torch.Tensor, mat_b: torch.Tensor, out: torch.Tensor) -> None:
    G, M, K = mat_a.shape
    _, _, N = mat_b.shape
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64), G)
    _grouped_mm_3d3d_kernel[grid](
        mat_a,
        mat_b,
        out,
        M=M,
        N=N,
        K=K,
        stride_ag=mat_a.stride(0),
        stride_am=mat_a.stride(1),
        stride_ak=mat_a.stride(2),
        stride_bg=mat_b.stride(0),
        stride_bk=mat_b.stride(1),
        stride_bn=mat_b.stride(2),
        stride_cg=out.stride(0),
        stride_cm=out.stride(1),
        stride_cn=out.stride(2),
        OUT_DTYPE=_as_tl_dtype(out.dtype),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
    )


def grouped_mm_2d2d(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor, out: torch.Tensor) -> None:
    G = offs.numel()
    M, K_TOTAL = mat_a.shape
    _, N = mat_b.shape
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64), G)
    _grouped_mm_2d2d_kernel[grid](
        mat_a,
        mat_b,
        out,
        offs,
        G=G,
        M=M,
        N=N,
        K_TOTAL=K_TOTAL,
        stride_am=mat_a.stride(0),
        stride_ak=mat_a.stride(1),
        stride_bk=mat_b.stride(0),
        stride_bn=mat_b.stride(1),
        stride_cg=out.stride(0),
        stride_cm=out.stride(1),
        stride_cn=out.stride(2),
        OUT_DTYPE=_as_tl_dtype(out.dtype),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        num_warps=4,
    )


def grouped_mm_2d3d(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor, out: torch.Tensor) -> None:
    G = offs.numel()
    M_TOTAL, K = mat_a.shape
    _, _, N = mat_b.shape
    grid = (M_TOTAL, triton.cdiv(N, 128))
    _grouped_mm_2d3d_kernel[grid](
        mat_a,
        mat_b,
        out,
        offs,
        G=G,
        M_TOTAL=M_TOTAL,
        N=N,
        K=K,
        stride_am=mat_a.stride(0),
        stride_ak=mat_a.stride(1),
        stride_bg=mat_b.stride(0),
        stride_bk=mat_b.stride(1),
        stride_bn=mat_b.stride(2),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        OUT_DTYPE=_as_tl_dtype(out.dtype),
        BLOCK_N=128,
        BLOCK_K=32,
        num_warps=4,
    )


def grouped_mm_3d2d(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor, out: torch.Tensor) -> None:
    G = offs.numel()
    G_a, M, K = mat_a.shape
    _, N_TOTAL = mat_b.shape
    if G_a != G:
        raise RuntimeError("matrix batch sizes have to match")
    # Put the potentially-large N dimension on axis 0 to avoid CUDA grid-y limits.
    grid = (N_TOTAL, triton.cdiv(M, 128))
    _grouped_mm_3d2d_kernel[grid](
        mat_a,
        mat_b,
        out,
        offs,
        G=G,
        M=M,
        N_TOTAL=N_TOTAL,
        K=K,
        stride_ag=mat_a.stride(0),
        stride_am=mat_a.stride(1),
        stride_ak=mat_a.stride(2),
        stride_bk=mat_b.stride(0),
        stride_bn=mat_b.stride(1),
        stride_cm=out.stride(0),
        stride_cn=out.stride(1),
        OUT_DTYPE=_as_tl_dtype(out.dtype),
        BLOCK_M=128,
        BLOCK_K=32,
        num_warps=4,
    )
