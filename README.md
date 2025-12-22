# grouped_mm_bf16 (Triton)

## Install

1) Install PyTorch for your system (CUDA / ROCm / CPU): https://pytorch.org/get-started/locally/

2) Install this package (and Triton):

```bash
pip install grouped-mm-bf16[triton]
```

## Usage

```py
import torch
from grouped_mm_bf16 import grouped_mm

dev = "cuda"

# 3D x 3D (batched matmul): [G,M,K] @ [G,K,N] -> [G,M,N]
a = torch.randn((4, 32, 64), device=dev, dtype=torch.bfloat16)
b = torch.randn((4, 64, 48), device=dev, dtype=torch.bfloat16)
out = grouped_mm(a, b)

# 2D x 3D with offsets: [M_total,K] @ [G,K,N] -> [M_total,N]
sizes = torch.tensor([0, 7, 1, 13, 0, 11], device=dev, dtype=torch.int32)
offs = sizes.cumsum(0)  # group end indices along M_total
a = torch.randn((int(offs[-1].item()), 64), device=dev, dtype=torch.bfloat16)
b = torch.randn((sizes.numel(), 64, 48), device=dev, dtype=torch.bfloat16)
out = grouped_mm(a, b, offs=offs)
```

## About

Universal (NVIDIA CUDA + AMD HIP) Triton implementation of `torch.nn.functional.grouped_mm` semantics for BF16.

This repo provides `grouped_mm_bf16.grouped_mm(...)`, mirroring the private operator signature:

```py
grouped_mm(mat_a, mat_b, offs=None, bias=None, out_dtype=None) -> Tensor
```

Notes:
- GPU-only: raises if any inputs are on CPU; never falls back to CPU.
- Matches PyTorch grouped_mm shape semantics (2D/3D + `offs`) and dtype rules.
- `bias` is rejected, matching current PyTorch behavior (`Bias not supported yet`).
- Like PyTorch’s current CUDA implementation, inputs must have “valid” (row/column-major) strides and 16-byte alignment; contiguous matrices with odd inner dimensions may need padding.
