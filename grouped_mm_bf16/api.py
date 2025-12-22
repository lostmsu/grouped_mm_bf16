from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .kernels import (
    grouped_mm_2d2d,
    grouped_mm_2d2d_backward,
    grouped_mm_2d3d,
    grouped_mm_2d3d_backward,
    grouped_mm_3d2d,
    grouped_mm_3d2d_backward,
    grouped_mm_3d3d,
    grouped_mm_3d3d_backward,
)


@dataclass(frozen=True)
class _Validated:
    a_is_2d: bool
    b_is_2d: bool
    out_dtype: torch.dtype


def _is_hip() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _check_valid_strides_and_return_transposed(mat: torch.Tensor) -> bool:
    if mat.dim() not in (2, 3):
        raise RuntimeError(f"mat must be 2D or 3D, got dim={mat.dim()}")

    end_dim = mat.dim() - 1
    stride_m = mat.stride(end_dim - 1)
    stride_n = mat.stride(end_dim)
    size_m = mat.size(end_dim - 1)
    size_n = mat.size(end_dim)

    # Row-major: last dim contiguous
    if stride_n == 1 and stride_m >= max(1, size_n):
        return True

    # Column-major: second-to-last dim contiguous
    if stride_m == 1 and stride_n >= max(1, size_m):
        return False

    raise RuntimeError(
        f"Invalid strides/sizes, got {tuple(mat.stride())} for strides and {tuple(mat.size())} for sizes"
    )


def _resolve_out_dtype(mat_a: torch.Tensor, out_dtype: Optional[torch.dtype]) -> torch.dtype:
    resolved = out_dtype if out_dtype is not None else mat_a.dtype
    if resolved != mat_a.dtype:
        raise RuntimeError("Grouped gemm output dtype must match `mat_a` dtype")
    return resolved


def _validate_inputs(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
) -> _Validated:
    if mat_a.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError(
            f"Expected mat_a to be Float32, BFloat16 or Float16 matrix, got {mat_a.dtype}"
        )
    if mat_b.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError(
            f"Expected mat_b to be Float32, BFloat16 or Float16 matrix, got {mat_b.dtype}"
        )
    if mat_a.dim() not in (2, 3):
        raise RuntimeError("mat_a has to be 2 or 3d")
    if mat_b.dim() not in (2, 3):
        raise RuntimeError("mat_b has to be 2 or 3d")

    if mat_a.dtype != mat_b.dtype:
        raise RuntimeError("mat_a and mat_b must have the same dtype")

    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    if (not a_is_2d) or (not b_is_2d):
        if mat_a.size(-1) != mat_b.size(-2):
            raise RuntimeError("contraction dimension of mat_a and mat_b must match")
    else:
        if mat_a.size(1) != mat_b.size(0):
            raise RuntimeError("contraction dimension of mat_a and mat_b must match")

    if not mat_a.is_cuda or not mat_b.is_cuda:
        raise RuntimeError("grouped_mm_bf16 is GPU-only (CUDA/HIP); CPU tensors are not supported")
    if mat_a.device != mat_b.device:
        raise RuntimeError("mat_a and mat_b must be on the same device")

    _check_valid_strides_and_return_transposed(mat_a)
    _check_valid_strides_and_return_transposed(mat_b)

    if (offs is None) != (not (a_is_2d or b_is_2d)):
        raise RuntimeError(
            "Have to provide offsets if there is a 2d matrix, or no offset if both matrices are 3d"
        )

    if offs is not None:
        if not offs.is_cuda:
            raise RuntimeError("offs must be a CUDA/HIP tensor (CPU offs are not supported)")
        if offs.device != mat_a.device:
            raise RuntimeError("offs must be on the same device as inputs")
        if offs.dim() != 1:
            raise RuntimeError("offs has to be 1D")
        if offs.dtype != torch.int32:
            raise RuntimeError("Offsets have to be int32")
        if offs.numel() >= 1024:
            raise RuntimeError("Can't process more than 1024 groups")

    if bias is not None:
        raise RuntimeError("Bias not supported yet")

    resolved_out_dtype = _resolve_out_dtype(mat_a, out_dtype)
    return _Validated(a_is_2d=a_is_2d, b_is_2d=b_is_2d, out_dtype=resolved_out_dtype)


def _create_output(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2

    if a_is_2d:
        if b_is_2d:
            out_size = (offs.size(0), mat_a.size(0), mat_b.size(1))
        else:
            if offs.size(0) != mat_b.size(0):
                raise RuntimeError("matrix batch sizes have to match")
            out_size = (mat_a.size(0), mat_b.size(-1))
    else:
        if b_is_2d:
            if offs.size(0) != mat_a.size(0):
                raise RuntimeError("matrix batch sizes have to match")
            out_size = (mat_a.size(1), mat_b.size(1))
        else:
            if mat_a.size(0) != mat_b.size(0):
                raise RuntimeError("batched dimension has to match")
            out_size = (mat_a.size(0), mat_a.size(1), mat_b.size(-1))

    # Match PyTorch CUDA behavior: pad the last-dim stride for alignment.
    # On ROCm, PyTorch uses a regular contiguous allocation.
    if _is_hip():
        return torch.empty(out_size, device=mat_a.device, dtype=out_dtype)

    alignment = 16 // torch.empty((), device=mat_a.device, dtype=out_dtype).element_size()
    last = out_size[-1]
    size_padded = ((last + alignment - 1) // alignment) * alignment
    if a_is_2d != b_is_2d:
        strides = (size_padded, 1)
    else:
        strides = (out_size[1] * size_padded, size_padded, 1)
    return torch.empty_strided(out_size, strides, device=mat_a.device, dtype=out_dtype)


class _GroupedMMFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        offs: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        out_dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        v = _validate_inputs(mat_a, mat_b, offs, bias, out_dtype)
        out = _create_output(mat_a, mat_b, offs, v.out_dtype)

        ctx.a_is_2d = v.a_is_2d
        ctx.b_is_2d = v.b_is_2d
        ctx.has_offs = offs is not None
        ctx.save_for_backward(
            mat_a,
            mat_b,
            offs if offs is not None else torch.empty((), device=mat_a.device, dtype=torch.int32),
        )

        if out.numel() == 0:
            return out

        if (not v.a_is_2d) and (not v.b_is_2d):
            if mat_a.size(2) == 0:
                out.zero_()
                return out
            grouped_mm_3d3d(mat_a, mat_b, out)
            return out
        if v.a_is_2d and v.b_is_2d:
            if mat_a.size(1) == 0:
                out.zero_()
                return out
            grouped_mm_2d2d(mat_a, mat_b, offs, out)
            return out
        if v.a_is_2d and (not v.b_is_2d):
            if mat_a.size(1) == 0:
                out.zero_()
                return out
            grouped_mm_2d3d(mat_a, mat_b, offs, out)
            return out
        # 3d x 2d
        if mat_a.size(2) == 0:
            out.zero_()
            return out
        grouped_mm_3d2d(mat_a, mat_b, offs, out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        mat_a, mat_b, offs_or_empty = ctx.saved_tensors
        offs = offs_or_empty if ctx.has_offs else None

        if grad_out.numel() == 0:
            grad_a = torch.zeros_like(mat_a) if ctx.needs_input_grad[0] else None
            grad_b = torch.zeros_like(mat_b) if ctx.needs_input_grad[1] else None
            return grad_a, grad_b, None, None, None

        try:
            _check_valid_strides_and_return_transposed(grad_out)
        except Exception:
            grad_out = grad_out.contiguous()

        grad_a = torch.empty_like(mat_a) if ctx.needs_input_grad[0] else None
        grad_b = torch.empty_like(mat_b) if ctx.needs_input_grad[1] else None

        if (not ctx.a_is_2d) and (not ctx.b_is_2d):
            if mat_a.size(2) == 0:
                if grad_a is not None:
                    grad_a.zero_()
                if grad_b is not None:
                    grad_b.zero_()
                return grad_a, grad_b, None, None, None
            grouped_mm_3d3d_backward(mat_a, mat_b, grad_out, grad_a, grad_b)
            return grad_a, grad_b, None, None, None

        if ctx.a_is_2d and ctx.b_is_2d:
            if mat_a.size(1) == 0:
                if grad_a is not None:
                    grad_a.zero_()
                if grad_b is not None:
                    grad_b.zero_()
                return grad_a, grad_b, None, None, None
            grouped_mm_2d2d_backward(mat_a, mat_b, offs, grad_out, grad_a, grad_b)
            return grad_a, grad_b, None, None, None

        if ctx.a_is_2d and (not ctx.b_is_2d):
            if mat_a.size(1) == 0:
                if grad_a is not None:
                    grad_a.zero_()
                if grad_b is not None:
                    grad_b.zero_()
                return grad_a, grad_b, None, None, None
            grouped_mm_2d3d_backward(mat_a, mat_b, offs, grad_out, grad_a, grad_b)
            return grad_a, grad_b, None, None, None

        # 3d x 2d
        if mat_a.size(2) == 0:
            if grad_a is not None:
                grad_a.zero_()
            if grad_b is not None:
                grad_b.zero_()
            return grad_a, grad_b, None, None, None
        grouped_mm_3d2d_backward(mat_a, mat_b, offs, grad_out, grad_a, grad_b)
        return grad_a, grad_b, None, None, None


def grouped_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Triton GPU-only implementation of `torch.nn.functional.grouped_mm` semantics
    (matches the private operator signature: `torch._grouped_mm`), including backprop.
    """

    return _GroupedMMFn.apply(mat_a, mat_b, offs, bias, out_dtype)
