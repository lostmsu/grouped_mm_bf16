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
def _grouped_mm_2d3d_prologue_main_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    m = m0 + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Expert for the tile start row; mask rows beyond this expert's end.
    g = _upper_bound_offs_i32(offs_ptr, G, m0.to(tl.int32)).to(tl.int32)
    valid_g = g < G
    g_safe = tl.where(valid_g, g, 0).to(tl.int32)
    row_end = tl.load(offs_ptr + g_safe, mask=valid_g, other=0).to(tl.int32)
    row_end = tl.minimum(row_end, M_TOTAL).to(tl.int32)
    valid_row = (m.to(tl.int32) < row_end) & (m < M_TOTAL)

    b_g = B_ptr + g_safe * stride_bg
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=valid_g & valid_row[:, None] & (k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_g + k[:, None] * stride_bk + n[None, :] * stride_bn,
            mask=valid_g & (k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)

    c = C_ptr + m[:, None] * stride_cm + n[None, :] * stride_cn
    # Store all rows; rows outside `valid_row` are zero because `a` is masked to 0.
    tl.store(c, tl.cast(acc, OUT_DTYPE), mask=valid_g & (m[:, None] < M_TOTAL) & (n[None, :] < N))


@triton.jit
def _grouped_mm_2d3d_prologue_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_g = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    g = pid_g.to(tl.int32)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    end = tl.load(offs_ptr + g, mask=g < G, other=0).to(tl.int32)
    start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0).to(tl.int32)
    start_mod = start % BLOCK_M
    has_prefix = (start_mod != 0) & (start < end)
    prefix_end = tl.where(has_prefix, tl.minimum(end, start + (BLOCK_M - start_mod)), start).to(tl.int32)

    if has_prefix:
        m = start + tl.arange(0, BLOCK_M)
        valid_row = (m.to(tl.int32) < prefix_end) & (m < M_TOTAL)

        b_g = B_ptr + g * stride_bg
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            a = tl.load(
                A_ptr + m[:, None] * stride_am + k[None, :] * stride_ak,
                mask=valid_row[:, None] & (k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_g + k[:, None] * stride_bk + n[None, :] * stride_bn,
                mask=(k[:, None] < K) & (n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc=acc)

        c = C_ptr + m[:, None] * stride_cm + n[None, :] * stride_cn
        tl.store(c, tl.cast(acc, OUT_DTYPE), mask=valid_row[:, None] & (n[None, :] < N))


@triton.jit
def _grouped_mm_2d3d_rowwise_kernel(
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
    # Baseline v0.1.1 behavior: one program per row, B is reloaded for each row.
    pid_m = tl.program_id(axis=0)  # row index
    pid_n = tl.program_id(axis=1)  # col-tile

    m = pid_m
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    g = _upper_bound_offs_i32(offs_ptr, G, m.to(tl.int32)).to(tl.int32)
    valid_g = g < G
    g_safe = tl.where(valid_g, g, 0).to(tl.int32)
    b_g = B_ptr + g_safe * stride_bg

    acc = tl.zeros((1, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + m * stride_am + k * stride_ak,
            mask=(m < M_TOTAL) & (k < K),
            other=0.0,
        )
        b = tl.load(
            b_g + k[:, None] * stride_bk + n[None, :] * stride_bn,
            mask=valid_g & (k[:, None] < K) & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a[None, :], b, acc=acc)

    c = C_ptr + m * stride_cm + n[None, :] * stride_cn
    tl.store(c, tl.cast(acc, OUT_DTYPE), mask=valid_g & (m < M_TOTAL) & (n[None, :] < N))


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
    if M_TOTAL == 0 or N == 0:
        return

    impl = os.getenv("GROUPED_MM_2D3D_IMPL", "prologue").strip().lower()
    if impl in ("hybrid", "tiled"):
        impl = "prologue"
    if impl == "rowwise":
        block_n = int(os.getenv("GROUPED_MM_2D3D_ROWWISE_BLOCK_N", "128"))
        block_k = int(os.getenv("GROUPED_MM_2D3D_ROWWISE_BLOCK_K", "32"))
        num_warps = int(os.getenv("GROUPED_MM_2D3D_ROWWISE_NUM_WARPS", "4"))
        grid = (M_TOTAL, triton.cdiv(N, block_n))
        _grouped_mm_2d3d_rowwise_kernel[grid](
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
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    elif impl == "prologue":
        block_m_env = os.getenv("GROUPED_MM_2D3D_BLOCK_M")
        block_k_env = os.getenv("GROUPED_MM_2D3D_BLOCK_K")
        block_n_env = os.getenv("GROUPED_MM_2D3D_BLOCK_N")
        num_warps_env = os.getenv("GROUPED_MM_2D3D_NUM_WARPS")

        default_block_n = 128 if N >= 128 else 64
        default_block_k = 64 if K >= 128 else 32
        default_block_m = 32 if K >= 128 else 64
        default_num_warps = 8 if default_block_n >= 128 else 4

        block_m = int(block_m_env) if block_m_env is not None else default_block_m
        block_n = int(block_n_env) if block_n_env is not None else default_block_n
        block_k = int(block_k_env) if block_k_env is not None else default_block_k
        num_warps = int(num_warps_env) if num_warps_env is not None else default_num_warps

        grid = (triton.cdiv(M_TOTAL, block_m), triton.cdiv(N, block_n))
        _grouped_mm_2d3d_prologue_main_kernel[grid](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )

        grid_p = (G, triton.cdiv(N, block_n))
        _grouped_mm_2d3d_prologue_kernel[grid_p](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    else:
        raise RuntimeError(f"Unknown GROUPED_MM_2D3D_IMPL={impl!r}")


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


@triton.jit
def _grouped_mm_2d3d_dA_prologue_main_kernel(
    dC_ptr,
    B_ptr,
    dA_ptr,
    offs_ptr,
    G: tl.constexpr,
    M_TOTAL,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_dam: tl.constexpr,
    stride_dak: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    m = m0 + tl.arange(0, BLOCK_M)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    g = _upper_bound_offs_i32(offs_ptr, G, m0.to(tl.int32)).to(tl.int32)
    valid_g = g < G
    g_safe = tl.where(valid_g, g, 0).to(tl.int32)
    row_end = tl.load(offs_ptr + g_safe, mask=valid_g, other=0).to(tl.int32)
    row_end = tl.minimum(row_end, M_TOTAL).to(tl.int32)
    valid_row = (m.to(tl.int32) < row_end) & (m < M_TOTAL)

    b_g = B_ptr + g_safe * stride_bg
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in tl.static_range(0, N, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        dc = tl.load(
            dC_ptr + m[:, None] * stride_dcm + n[None, :] * stride_dcn,
            mask=valid_g & valid_row[:, None] & (n[None, :] < N),
            other=0.0,
        )
        bT = tl.load(
            b_g + k[None, :] * stride_bk + n[:, None] * stride_bn,
            mask=valid_g & (k[None, :] < K) & (n[:, None] < N),
            other=0.0,
        )
        acc = tl.dot(dc, bT, acc=acc)

    out_ptr = dA_ptr + m[:, None] * stride_dam + k[None, :] * stride_dak
    tl.store(out_ptr, tl.cast(acc, OUT_DTYPE), mask=valid_g & (m[:, None] < M_TOTAL) & (k[None, :] < K))


@triton.jit
def _grouped_mm_2d3d_dA_prologue_kernel(
    dC_ptr,
    B_ptr,
    dA_ptr,
    offs_ptr,
    G: tl.constexpr,
    M_TOTAL,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_dam: tl.constexpr,
    stride_dak: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    g = pid_g.to(tl.int32)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    end = tl.load(offs_ptr + g, mask=g < G, other=0).to(tl.int32)
    start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0).to(tl.int32)
    start_mod = start % BLOCK_M
    has_prefix = (start_mod != 0) & (start < end)
    prefix_end = tl.where(has_prefix, tl.minimum(end, start + (BLOCK_M - start_mod)), start).to(tl.int32)

    if has_prefix:
        m = start + tl.arange(0, BLOCK_M)
        valid_row = (m.to(tl.int32) < prefix_end) & (m < M_TOTAL)

        b_g = B_ptr + g * stride_bg
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n0 in tl.static_range(0, N, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            dc = tl.load(
                dC_ptr + m[:, None] * stride_dcm + n[None, :] * stride_dcn,
                mask=valid_row[:, None] & (n[None, :] < N),
                other=0.0,
            )
            bT = tl.load(
                b_g + k[None, :] * stride_bk + n[:, None] * stride_bn,
                mask=(k[None, :] < K) & (n[:, None] < N),
                other=0.0,
            )
            acc = tl.dot(dc, bT, acc=acc)

        out_ptr = dA_ptr + m[:, None] * stride_dam + k[None, :] * stride_dak
        tl.store(out_ptr, tl.cast(acc, OUT_DTYPE), mask=valid_row[:, None] & (k[None, :] < K))


@triton.jit
def _grouped_mm_2d3d_dB_reduce_kernel(
    A_ptr,
    dC_ptr,
    dB_ptr,
    offs_ptr,
    G: tl.constexpr,
    M_TOTAL,
    N,
    K,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_dbg: tl.constexpr,
    stride_dbk: tl.constexpr,
    stride_dbn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_g = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    g = pid_g.to(tl.int32)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    end = tl.load(offs_ptr + g, mask=g < G, other=0).to(tl.int32)
    start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0).to(tl.int32)

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    m0 = start
    while m0 < end:
        m = m0 + tl.arange(0, BLOCK_M)
        mask_m = m < end
        aT = tl.load(
            A_ptr + m[None, :] * stride_am + k[:, None] * stride_ak,
            mask=mask_m[None, :] & (k[:, None] < K),
            other=0.0,
        )
        dc = tl.load(
            dC_ptr + m[:, None] * stride_dcm + n[None, :] * stride_dcn,
            mask=mask_m[:, None] & (n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(aT, dc, acc=acc)
        m0 += BLOCK_M

    out_ptr = dB_ptr + g * stride_dbg + k[:, None] * stride_dbk + n[None, :] * stride_dbn
    tl.store(out_ptr, acc, mask=(g < G) & (k[:, None] < K) & (n[None, :] < N))


@triton.jit
def _grouped_mm_3d2d_dB_kernel(
    A_ptr,
    dC_ptr,
    dB_ptr,
    offs_ptr,
    G: tl.constexpr,
    M: tl.constexpr,
    N_TOTAL,
    K: tl.constexpr,
    stride_ag: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_dbk: tl.constexpr,
    stride_dbn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    n = pid_n
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    g = _upper_bound_offs_i32(offs_ptr, G, n.to(tl.int32))
    a_g = A_ptr + g * stride_ag

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for m0 in tl.static_range(0, M, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)
        a = tl.load(
            a_g + m[:, None] * stride_am + k[None, :] * stride_ak,
            mask=(m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        dc = tl.load(
            dC_ptr + m * stride_dcm + n * stride_dcn,
            mask=(m < M) & (n < N_TOTAL),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a * dc[:, None], axis=0)

    tl.store(
        dB_ptr + k * stride_dbk + n * stride_dbn,
        tl.cast(acc, OUT_DTYPE),
        mask=(k < K) & (n < N_TOTAL),
    )


@triton.jit
def _grouped_mm_3d2d_dA_atomic_kernel(
    dC_ptr,
    B_ptr,
    dA_ptr,
    offs_ptr,
    G: tl.constexpr,
    M,
    N_TOTAL,
    K: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_dag: tl.constexpr,
    stride_dam: tl.constexpr,
    stride_dak: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    n = pid_n
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    g = _upper_bound_offs_i32(offs_ptr, G, n.to(tl.int32))

    dc = tl.load(
        dC_ptr + m * stride_dcm + n * stride_dcn,
        mask=(m < M) & (n < N_TOTAL),
        other=0.0,
    ).to(tl.float32)
    b = tl.load(
        B_ptr + k * stride_bk + n * stride_bn,
        mask=(k < K) & (n < N_TOTAL),
        other=0.0,
    ).to(tl.float32)

    contrib = dc[:, None] * b[None, :]
    out_ptr = dA_ptr + g * stride_dag + m[:, None] * stride_dam + k[None, :] * stride_dak
    tl.atomic_add(out_ptr, contrib, mask=(m[:, None] < M) & (k[None, :] < K) & (n < N_TOTAL))


@triton.jit
def _grouped_mm_2d2d_dA_kernel(
    dC_ptr,
    B_ptr,
    dA_ptr,
    offs_ptr,
    G: tl.constexpr,
    M,
    N: tl.constexpr,
    K_TOTAL,
    stride_dcg: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_dam: tl.constexpr,
    stride_dak: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    k = pid_k
    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    g = _upper_bound_offs_i32(offs_ptr, G, k.to(tl.int32))
    dc_g = dC_ptr + g * stride_dcg

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in tl.static_range(0, N, BLOCK_N):
        n = n0 + tl.arange(0, BLOCK_N)
        dc = tl.load(
            dc_g + m[:, None] * stride_dcm + n[None, :] * stride_dcn,
            mask=(m[:, None] < M) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_ptr + k * stride_bk + n * stride_bn,
            mask=(k < K_TOTAL) & (n < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(dc * b[None, :], axis=1)

    tl.store(
        dA_ptr + m * stride_dam + k * stride_dak,
        tl.cast(acc, OUT_DTYPE),
        mask=(m < M) & (k < K_TOTAL),
    )


@triton.jit
def _grouped_mm_2d2d_dB_kernel(
    A_ptr,
    dC_ptr,
    dB_ptr,
    offs_ptr,
    G: tl.constexpr,
    M: tl.constexpr,
    N,
    K_TOTAL,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_dcg: tl.constexpr,
    stride_dcm: tl.constexpr,
    stride_dcn: tl.constexpr,
    stride_dbk: tl.constexpr,
    stride_dbn: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    k = pid_k
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    g = _upper_bound_offs_i32(offs_ptr, G, k.to(tl.int32))
    dc_g = dC_ptr + g * stride_dcg

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for m0 in tl.static_range(0, M, BLOCK_M):
        m = m0 + tl.arange(0, BLOCK_M)
        a = tl.load(
            A_ptr + m * stride_am + k * stride_ak,
            mask=(m < M) & (k < K_TOTAL),
            other=0.0,
        ).to(tl.float32)
        dc = tl.load(
            dc_g + m[:, None] * stride_dcm + n[None, :] * stride_dcn,
            mask=(m[:, None] < M) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(a[:, None] * dc, axis=0)

    tl.store(
        dB_ptr + k * stride_dbk + n * stride_dbn,
        tl.cast(acc, OUT_DTYPE),
        mask=(k < K_TOTAL) & (n < N),
    )


def grouped_mm_3d3d_backward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    grad_out: torch.Tensor,
    grad_a: Optional[torch.Tensor],
    grad_b: Optional[torch.Tensor],
) -> None:
    G, M, K = mat_a.shape
    _, _, N = mat_b.shape
    if grad_a is not None:
        grid = (triton.cdiv(M, 64), triton.cdiv(K, 64), G)
        _grouped_mm_3d3d_kernel[grid](
            grad_out,
            mat_b,
            grad_a,
            M=M,
            N=K,
            K=N,
            stride_ag=grad_out.stride(0),
            stride_am=grad_out.stride(1),
            stride_ak=grad_out.stride(2),
            stride_bg=mat_b.stride(0),
            stride_bk=mat_b.stride(2),  # treat B as [N, K]
            stride_bn=mat_b.stride(1),
            stride_cg=grad_a.stride(0),
            stride_cm=grad_a.stride(1),
            stride_cn=grad_a.stride(2),
            OUT_DTYPE=_as_tl_dtype(grad_a.dtype),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )

    if grad_b is not None:
        grid = (triton.cdiv(K, 64), triton.cdiv(N, 64), G)
        _grouped_mm_3d3d_kernel[grid](
            mat_a,
            grad_out,
            grad_b,
            M=K,
            N=N,
            K=M,
            stride_ag=mat_a.stride(0),
            stride_am=mat_a.stride(2),  # treat A as [K, M]
            stride_ak=mat_a.stride(1),
            stride_bg=grad_out.stride(0),
            stride_bk=grad_out.stride(1),
            stride_bn=grad_out.stride(2),
            stride_cg=grad_b.stride(0),
            stride_cm=grad_b.stride(1),
            stride_cn=grad_b.stride(2),
            OUT_DTYPE=_as_tl_dtype(grad_b.dtype),
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=32,
            num_warps=4,
        )


def grouped_mm_2d3d_backward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    grad_out: torch.Tensor,
    grad_a: Optional[torch.Tensor],
    grad_b: Optional[torch.Tensor],
) -> None:
    G = offs.numel()
    M_TOTAL, K = mat_a.shape
    _, _, N = mat_b.shape
    if grad_a is not None:
        impl = os.getenv("GROUPED_MM_2D3D_IMPL", "prologue").strip().lower()
        if impl in ("hybrid", "tiled"):
            impl = "prologue"

        block_m = 64
        block_k = 64
        block_n = 32
        num_warps = 4
        grid = (triton.cdiv(M_TOTAL, block_m), triton.cdiv(K, block_k))

        if impl == "prologue":
            _grouped_mm_2d3d_dA_prologue_main_kernel[grid](
                grad_out,
                mat_b,
                grad_a,
                offs,
                G=G,
                M_TOTAL=M_TOTAL,
                N=N,
                K=K,
                stride_dcm=grad_out.stride(0),
                stride_dcn=grad_out.stride(1),
                stride_bg=mat_b.stride(0),
                stride_bk=mat_b.stride(1),
                stride_bn=mat_b.stride(2),
                stride_dam=grad_a.stride(0),
                stride_dak=grad_a.stride(1),
                OUT_DTYPE=_as_tl_dtype(grad_a.dtype),
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
            grid_p = (G, triton.cdiv(K, block_k))
            _grouped_mm_2d3d_dA_prologue_kernel[grid_p](
                grad_out,
                mat_b,
                grad_a,
                offs,
                G=G,
                M_TOTAL=M_TOTAL,
                N=N,
                K=K,
                stride_dcm=grad_out.stride(0),
                stride_dcn=grad_out.stride(1),
                stride_bg=mat_b.stride(0),
                stride_bk=mat_b.stride(1),
                stride_bn=mat_b.stride(2),
                stride_dam=grad_a.stride(0),
                stride_dak=grad_a.stride(1),
                OUT_DTYPE=_as_tl_dtype(grad_a.dtype),
                BLOCK_M=block_m,
                BLOCK_K=block_k,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        else:
            raise RuntimeError(f"Unknown GROUPED_MM_2D3D_IMPL={impl!r}")

    if grad_b is not None:
        dB_fp32 = torch.zeros((G, K, N), device=mat_a.device, dtype=torch.float32)

        # dB[g] = A_g^T @ dC_g (per-expert reduction, avoids ~O(M_total) atomics).
        block_m = 64
        block_k = 32
        block_n = 64
        grid = (G, triton.cdiv(K, block_k), triton.cdiv(N, block_n))
        _grouped_mm_2d3d_dB_reduce_kernel[grid](
            mat_a,
            grad_out,
            dB_fp32,
            offs,
            G=G,
            M_TOTAL=M_TOTAL,
            N=N,
            K=K,
            stride_am=mat_a.stride(0),
            stride_ak=mat_a.stride(1),
            stride_dcm=grad_out.stride(0),
            stride_dcn=grad_out.stride(1),
            stride_dbg=dB_fp32.stride(0),
            stride_dbk=dB_fp32.stride(1),
            stride_dbn=dB_fp32.stride(2),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_N=block_n,
            num_warps=4,
        )
        grad_b.copy_(dB_fp32.to(dtype=grad_b.dtype))


def grouped_mm_3d2d_backward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    grad_out: torch.Tensor,
    grad_a: Optional[torch.Tensor],
    grad_b: Optional[torch.Tensor],
) -> None:
    G = offs.numel()
    G_a, M, K = mat_a.shape
    _, N_TOTAL = mat_b.shape
    if G_a != G:
        raise RuntimeError("matrix batch sizes have to match")

    if grad_b is not None:
        grid = (N_TOTAL, triton.cdiv(K, 128))
        _grouped_mm_3d2d_dB_kernel[grid](
            mat_a,
            grad_out,
            grad_b,
            offs,
            G=G,
            M=M,
            N_TOTAL=N_TOTAL,
            K=K,
            stride_ag=mat_a.stride(0),
            stride_am=mat_a.stride(1),
            stride_ak=mat_a.stride(2),
            stride_dcm=grad_out.stride(0),
            stride_dcn=grad_out.stride(1),
            stride_dbk=grad_b.stride(0),
            stride_dbn=grad_b.stride(1),
            OUT_DTYPE=_as_tl_dtype(grad_b.dtype),
            BLOCK_M=32,
            BLOCK_K=128,
            num_warps=4,
        )

    if grad_a is not None:
        dA_fp32 = torch.zeros((G, M, K), device=mat_a.device, dtype=torch.float32)
        grid = (
            N_TOTAL,
            triton.cdiv(M, 16),
            triton.cdiv(K, 32),
        )
        _grouped_mm_3d2d_dA_atomic_kernel[grid](
            grad_out,
            mat_b,
            dA_fp32,
            offs,
            G=G,
            M=M,
            N_TOTAL=N_TOTAL,
            K=K,
            stride_dcm=grad_out.stride(0),
            stride_dcn=grad_out.stride(1),
            stride_bk=mat_b.stride(0),
            stride_bn=mat_b.stride(1),
            stride_dag=dA_fp32.stride(0),
            stride_dam=dA_fp32.stride(1),
            stride_dak=dA_fp32.stride(2),
            BLOCK_M=16,
            BLOCK_K=32,
            num_warps=4,
        )
        grad_a.copy_(dA_fp32.to(dtype=grad_a.dtype))


def grouped_mm_2d2d_backward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    grad_out: torch.Tensor,
    grad_a: Optional[torch.Tensor],
    grad_b: Optional[torch.Tensor],
) -> None:
    G = offs.numel()
    M, K_TOTAL = mat_a.shape
    _, N = mat_b.shape

    if grad_a is not None:
        grid = (K_TOTAL, triton.cdiv(M, 128))
        _grouped_mm_2d2d_dA_kernel[grid](
            grad_out,
            mat_b,
            grad_a,
            offs,
            G=G,
            M=M,
            N=N,
            K_TOTAL=K_TOTAL,
            stride_dcg=grad_out.stride(0),
            stride_dcm=grad_out.stride(1),
            stride_dcn=grad_out.stride(2),
            stride_bk=mat_b.stride(0),
            stride_bn=mat_b.stride(1),
            stride_dam=grad_a.stride(0),
            stride_dak=grad_a.stride(1),
            OUT_DTYPE=_as_tl_dtype(grad_a.dtype),
            BLOCK_M=128,
            BLOCK_N=32,
            num_warps=4,
        )

    if grad_b is not None:
        grid = (K_TOTAL, triton.cdiv(N, 128))
        _grouped_mm_2d2d_dB_kernel[grid](
            mat_a,
            grad_out,
            grad_b,
            offs,
            G=G,
            M=M,
            N=N,
            K_TOTAL=K_TOTAL,
            stride_am=mat_a.stride(0),
            stride_ak=mat_a.stride(1),
            stride_dcg=grad_out.stride(0),
            stride_dcm=grad_out.stride(1),
            stride_dcn=grad_out.stride(2),
            stride_dbk=grad_b.stride(0),
            stride_dbn=grad_b.stride(1),
            OUT_DTYPE=_as_tl_dtype(grad_b.dtype),
            BLOCK_M=32,
            BLOCK_N=128,
            num_warps=4,
        )
