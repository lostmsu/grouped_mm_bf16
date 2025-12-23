import pytest
import torch

from grouped_mm_bf16 import grouped_mm


def _device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP required")
    return torch.device("cuda")


def _bf16_supported() -> bool:
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    return True if fn is None else bool(fn())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_3d3d_bf16_matches_bmm():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, m, k, n = 4, 32, 64, 48
    a = torch.randn((g, m, k), device=dev, dtype=torch.bfloat16)
    b = torch.randn((g, k, n), device=dev, dtype=torch.bfloat16)
    out = grouped_mm(a, b, offs=None)
    ref = torch.bmm(a, b)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_2d3d_bf16_matches_rowwise_reference():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, k, n = 6, 32, 40
    sizes_list = [0, 7, 1, 13, 0, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    m_total = sum(sizes_list)
    assert m_total > 0

    a = torch.randn((m_total, k), device=dev, dtype=torch.bfloat16)
    b = torch.randn((g, k, n), device=dev, dtype=torch.bfloat16)

    out = grouped_mm(a, b, offs=offs)

    # GPU-only reference using searchsorted + gather + bmm.
    rows = torch.arange(m_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, rows, right=True).to(torch.int64)
    b_g = b.index_select(0, gid)  # [m_total, k, n]
    ref = torch.bmm(a.unsqueeze(1), b_g).squeeze(1)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    # CPU-orchestrated reference (mm runs on GPU), mirroring PyTorch's fallback structure.
    offs_cpu = offs.cpu()
    starts_cpu = torch.cat([torch.zeros((1,), dtype=offs_cpu.dtype), offs_cpu[:-1]])
    assert ((offs_cpu - starts_cpu) == 0).any()
    ref_loop = torch.empty_like(out)
    start = 0
    for gi in range(int(offs_cpu.numel())):
        end = int(offs_cpu[gi].item())
        if end > start:
            ref_loop[start:end] = torch.mm(a[start:end], b[gi])
        start = end
    torch.testing.assert_close(out, ref_loop, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_2d2d_bf16_matches_masked_reference():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, m, n = 5, 16, 24
    sizes_list = [7, 0, 1, 13, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    k_total = sum(sizes_list)
    assert k_total > 0
    a = torch.randn((m, k_total), device=dev, dtype=torch.bfloat16)
    b = torch.randn((k_total, n), device=dev, dtype=torch.bfloat16)

    out = grouped_mm(a, b, offs=offs)

    # Reference: build group masks over K on GPU, then broadcast matmul.
    k = torch.arange(k_total, device=dev, dtype=torch.int32)
    start = torch.cat([torch.zeros((1,), device=dev, dtype=torch.int32), offs[:-1]])
    mask = (k[None, :] >= start[:, None]) & (k[None, :] < offs[:, None])
    a_masked = a.unsqueeze(0) * mask.to(a.dtype).unsqueeze(1)
    ref = torch.matmul(a_masked, b)  # [g, m, n]
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    offs_cpu = offs.cpu()
    starts_cpu = torch.cat([torch.zeros((1,), dtype=offs_cpu.dtype), offs_cpu[:-1]])
    assert ((offs_cpu - starts_cpu) == 0).any()
    ref_loop = torch.empty_like(out)
    start = 0
    for gi in range(int(offs_cpu.numel())):
        end = int(offs_cpu[gi].item())
        if end > start:
            ref_loop[gi] = torch.mm(a[:, start:end], b[start:end, :])
        else:
            ref_loop[gi].zero_()
        start = end
    torch.testing.assert_close(out, ref_loop, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_3d2d_bf16_matches_colwise_reference():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, m, k = 5, 32, 48
    sizes_list = [7, 0, 1, 13, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    n_total = sum(sizes_list)
    assert n_total > 0

    a = torch.randn((g, m, k), device=dev, dtype=torch.bfloat16)
    b = torch.randn((k, n_total), device=dev, dtype=torch.bfloat16)

    out = grouped_mm(a, b, offs=offs)

    cols = torch.arange(n_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, cols, right=True).to(torch.int64)
    a_g = a.index_select(0, gid)  # [n_total, m, k]
    b_cols = b.t().unsqueeze(-1)  # [n_total, k, 1]
    ref_cols = torch.bmm(a_g, b_cols).squeeze(-1)  # [n_total, m]
    ref = ref_cols.t().contiguous()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    offs_cpu = offs.cpu()
    starts_cpu = torch.cat([torch.zeros((1,), dtype=offs_cpu.dtype), offs_cpu[:-1]])
    assert ((offs_cpu - starts_cpu) == 0).any()
    ref_loop = torch.empty_like(out)
    start = 0
    for gi in range(int(offs_cpu.numel())):
        end = int(offs_cpu[gi].item())
        if end > start:
            ref_loop[:, start:end] = torch.mm(a[gi], b[:, start:end])
        start = end
    torch.testing.assert_close(out, ref_loop, atol=2e-2, rtol=2e-2)


def _grads(loss_out: torch.Tensor, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    grads = torch.autograd.grad(loss_out, inputs, allow_unused=False, retain_graph=False)
    return [g.detach() for g in grads]


class _Env:
    def __init__(self, **items: str):
        self._items = items
        self._prev: dict[str, str | None] = {}

    def __enter__(self):
        import os

        for k, v in self._items.items():
            self._prev[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, exc_type, exc, tb):
        import os

        for k, prev in self._prev.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
        return False


def _torch_grouped_mm():
    try:
        import torch.nn.functional as F

        fn = getattr(F, "grouped_mm", None)
        return fn
    except Exception:
        return None


def _maybe_compare_to_torch_grouped_mm(
    out: torch.Tensor, mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor
) -> None:
    fn = _torch_grouped_mm()
    if fn is None:
        return
    try:
        out_torch = fn(mat_a=mat_a, mat_b=mat_b, offs=offs, bias=None, out_dtype=mat_a.dtype)
    except TypeError:
        return
    except RuntimeError as e:
        msg = str(e).lower()
        if "not implemented" in msg or "not supported" in msg:
            return
        raise
    if not out_torch.is_cuda:
        return
    torch.testing.assert_close(out, out_torch, atol=2e-2, rtol=2e-2)


def _maybe_compare_grads_to_torch_grouped_mm(
    da: torch.Tensor,
    db: torch.Tensor,
    a0: torch.Tensor,
    b0: torch.Tensor,
    offs: torch.Tensor,
    w: torch.Tensor,
) -> None:
    fn = _torch_grouped_mm()
    if fn is None:
        return
    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    try:
        out2 = fn(mat_a=a2, mat_b=b2, offs=offs, bias=None, out_dtype=a2.dtype)
    except TypeError:
        return
    except RuntimeError as e:
        msg = str(e).lower()
        if "not implemented" in msg or "not supported" in msg:
            return
        raise
    if not out2.is_cuda:
        return
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])
    torch.testing.assert_close(da, da2, atol=6e-4, rtol=6e-4)
    torch.testing.assert_close(db, db2, atol=8e-4, rtol=8e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_3d3d_float32_matches_bmm():
    dev = _device()
    g, m, k, n = 3, 17, 19, 23
    a0 = torch.randn((g, m, k), device=dev, dtype=torch.float32)
    b0 = torch.randn((g, k, n), device=dev, dtype=torch.float32)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=None)
    w = torch.randn_like(out1)
    grads1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    out2 = torch.bmm(a2, b2)
    grads2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(grads1[0], grads2[0], atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(grads1[1], grads2[1], atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_3d3d_bf16_matches_bmm():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, m, k, n = 3, 17, 19, 23
    a0 = torch.randn((g, m, k), device=dev, dtype=torch.bfloat16)
    b0 = torch.randn((g, k, n), device=dev, dtype=torch.bfloat16)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=None)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    out2 = torch.bmm(a2, b2)
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(db1, db2, atol=4e-2, rtol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_2d3d_float32_matches_reference_with_zero_groups():
    dev = _device()
    g, k, n = 6, 29, 31
    sizes_list = [0, 7, 1, 13, 0, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    m_total = sum(sizes_list)
    assert m_total > 0

    a0 = torch.randn((m_total, k), device=dev, dtype=torch.float32)
    b0 = torch.randn((g, k, n), device=dev, dtype=torch.float32)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=offs)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    rows = torch.arange(m_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, rows, right=True).to(torch.int64)
    b_g = b2.index_select(0, gid)  # [m_total, k, n]
    out2 = torch.bmm(a2.unsqueeze(1), b_g).squeeze(1)
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(db1, db2, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_2d3d_bf16_matches_reference_with_zero_groups():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, k, n = 6, 29, 31
    sizes_list = [0, 7, 1, 13, 0, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    m_total = sum(sizes_list)
    assert m_total > 0

    a0 = torch.randn((m_total, k), device=dev, dtype=torch.bfloat16)
    b0 = torch.randn((g, k, n), device=dev, dtype=torch.bfloat16)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=offs)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    rows = torch.arange(m_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, rows, right=True).to(torch.int64)
    b_g = b2.index_select(0, gid)  # [m_total, k, n]
    out2 = torch.bmm(a2.unsqueeze(1), b_g).squeeze(1)
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(db1, db2, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
@pytest.mark.parametrize(
    "sizes_list,block_m",
    [
        # All experts aligned to block boundary -> prologue kernel should do nothing.
        ([32, 32, 32], 32),
        # Multi-boundary in one tile: 3 experts inside first 32-row tile.
        ([10, 10, 10, 2], 32),
        # Single-token expert ending exactly on a tile boundary.
        ([31, 1, 31], 32),
        # Total tokens not divisible by BLOCK_M, and the final partial tile contains multiple experts.
        # Tile [64..95) includes boundaries at 70 and 80.
        ([40, 30, 10, 7], 32),
        # Empty-only tail experts.
        ([32, 16, 0, 0, 0], 32),
        # Tiny experts (including many within a tile), with empties interspersed.
        ([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], 32),
        # Starts and ends mid-block for the same expert; also includes empty experts.
        ([5, 0, 7, 0, 3, 9], 32),
    ],
)
def test_grouped_mm_2d3d_prologue_forward_matches_reference(sizes_list, block_m):
    dev = _device()
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    g = len(sizes_list)
    k, n = 33, 29
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    m_total = int(offs[-1].item())
    if m_total == 0:
        pytest.skip("degenerate all-empty case")

    a = torch.randn((m_total, k), device=dev, dtype=torch.bfloat16)
    b = torch.randn((g, k, n), device=dev, dtype=torch.bfloat16)

    with _Env(
        GROUPED_MM_2D3D_IMPL="prologue",
        GROUPED_MM_2D3D_BLOCK_M=str(block_m),
        GROUPED_MM_2D3D_BLOCK_N="64",
        GROUPED_MM_2D3D_BLOCK_K="32",
        GROUPED_MM_2D3D_NUM_WARPS="4",
    ):
        out = grouped_mm(a, b, offs=offs)

    rows = torch.arange(m_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, rows, right=True).to(torch.int64)
    ref = torch.bmm(a.unsqueeze(1), b.index_select(0, gid)).squeeze(1)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    _maybe_compare_to_torch_grouped_mm(out, a, b, offs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
@pytest.mark.parametrize(
    "sizes_list,block_m",
    [
        # Total tokens not divisible by BLOCK_M, and the final partial tile contains multiple experts.
        ([32, 1, 30], 32),
        # Single-token expert ending exactly on a tile boundary.
        ([31, 1, 31], 32),
        # Cross-boundary expert: starts mid-block and spans into next block(s).
        ([11, 90, 7], 32),
        # Swiss-cheese offsets: consecutive empty experts around small experts.
        ([0, 3, 0, 0, 5, 0, 1, 0, 9], 32),
    ],
)
def test_grouped_mm_2d3d_prologue_backward_da_matches_reference(sizes_list, block_m):
    dev = _device()
    g = len(sizes_list)
    k, n = 41, 37
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    m_total = int(offs[-1].item())
    if m_total == 0:
        pytest.skip("degenerate all-empty case")

    for seed in (0, 1):
        torch.manual_seed(seed)
        a0 = torch.randn((m_total, k), device=dev, dtype=torch.float32)
        b0 = torch.randn((g, k, n), device=dev, dtype=torch.float32)

        with _Env(
            GROUPED_MM_2D3D_IMPL="prologue",
            GROUPED_MM_2D3D_BLOCK_M=str(block_m),
            GROUPED_MM_2D3D_BLOCK_N="64",
            GROUPED_MM_2D3D_BLOCK_K="32",
            GROUPED_MM_2D3D_NUM_WARPS="4",
        ):
            a1 = a0.clone().requires_grad_(True)
            b1 = b0.clone().requires_grad_(True)
            out1 = grouped_mm(a1, b1, offs=offs)
            w = torch.randn_like(out1)
            da1, db1 = _grads((out1 * w).sum(), [a1, b1])

        a2 = a0.clone().requires_grad_(True)
        b2 = b0.clone().requires_grad_(True)
        rows = torch.arange(m_total, device=dev, dtype=torch.int32)
        gid = torch.searchsorted(offs, rows, right=True).to(torch.int64)
        out2 = torch.bmm(a2.unsqueeze(1), b2.index_select(0, gid)).squeeze(1)
        da2, db2 = _grads((out2 * w).sum(), [a2, b2])

        torch.testing.assert_close(da1, da2, atol=6e-4, rtol=6e-4)
        torch.testing.assert_close(db1, db2, atol=8e-4, rtol=8e-4)
        _maybe_compare_grads_to_torch_grouped_mm(da1, db1, a0, b0, offs, w)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_3d2d_float32_matches_reference_with_zero_groups():
    dev = _device()
    g, m, k = 5, 23, 17
    sizes_list = [7, 0, 1, 13, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    n_total = sum(sizes_list)
    assert n_total > 0

    a0 = torch.randn((g, m, k), device=dev, dtype=torch.float32)
    b0 = torch.randn((k, n_total), device=dev, dtype=torch.float32)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=offs)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    cols = torch.arange(n_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, cols, right=True).to(torch.int64)
    a_g = a2.index_select(0, gid)  # [n_total, m, k]
    b_cols = b2.t().unsqueeze(-1)  # [n_total, k, 1]
    ref_cols = torch.bmm(a_g, b_cols).squeeze(-1)  # [n_total, m]
    out2 = ref_cols.t().contiguous()
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=6e-4, rtol=6e-4)
    torch.testing.assert_close(db1, db2, atol=6e-4, rtol=6e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_2d2d_float32_matches_masked_reference_with_weird_sizes():
    dev = _device()
    g, m, n = 5, 19, 21
    sizes_list = [7, 0, 1, 13, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    k_total = sum(sizes_list)
    assert k_total > 0

    a0 = torch.randn((m, k_total), device=dev, dtype=torch.float32)
    b0 = torch.randn((k_total, n), device=dev, dtype=torch.float32)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=offs)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    k_idx = torch.arange(k_total, device=dev, dtype=torch.int32)
    start = torch.cat([torch.zeros((1,), device=dev, dtype=torch.int32), offs[:-1]])
    mask = (k_idx[None, :] >= start[:, None]) & (k_idx[None, :] < offs[:, None])
    a_masked = a2.unsqueeze(0) * mask.to(a2.dtype).unsqueeze(1)
    out2 = torch.matmul(a_masked, b2)
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(db1, db2, atol=5e-4, rtol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_3d2d_bf16_matches_reference_with_zero_groups():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, m, k = 5, 23, 17
    sizes_list = [7, 0, 1, 13, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    n_total = sum(sizes_list)
    assert n_total > 0

    a0 = torch.randn((g, m, k), device=dev, dtype=torch.bfloat16)
    b0 = torch.randn((k, n_total), device=dev, dtype=torch.bfloat16)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=offs)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    cols = torch.arange(n_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, cols, right=True).to(torch.int64)
    a_g = a2.index_select(0, gid)  # [n_total, m, k]
    b_cols = b2.t().unsqueeze(-1)  # [n_total, k, 1]
    ref_cols = torch.bmm(a_g, b_cols).squeeze(-1)  # [n_total, m]
    out2 = ref_cols.t().contiguous()
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(db1, db2, atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_backward_2d2d_bf16_matches_masked_reference_with_weird_sizes():
    if not _bf16_supported():
        pytest.skip("bf16 not supported on this device")
    dev = _device()
    g, m, n = 5, 19, 21
    sizes_list = [7, 0, 1, 13, 11]
    sizes = torch.tensor(sizes_list, device=dev, dtype=torch.int32)
    offs = sizes.cumsum(0).to(torch.int32)
    k_total = sum(sizes_list)
    assert k_total > 0

    a0 = torch.randn((m, k_total), device=dev, dtype=torch.bfloat16)
    b0 = torch.randn((k_total, n), device=dev, dtype=torch.bfloat16)

    a1 = a0.clone().requires_grad_(True)
    b1 = b0.clone().requires_grad_(True)
    out1 = grouped_mm(a1, b1, offs=offs)
    w = torch.randn_like(out1)
    da1, db1 = _grads((out1 * w).sum(), [a1, b1])

    a2 = a0.clone().requires_grad_(True)
    b2 = b0.clone().requires_grad_(True)
    k_idx = torch.arange(k_total, device=dev, dtype=torch.int32)
    start = torch.cat([torch.zeros((1,), device=dev, dtype=torch.int32), offs[:-1]])
    mask = (k_idx[None, :] >= start[:, None]) & (k_idx[None, :] < offs[:, None])
    a_masked = a2.unsqueeze(0) * mask.to(a2.dtype).unsqueeze(1)
    out2 = torch.matmul(a_masked, b2)
    da2, db2 = _grads((out2 * w).sum(), [a2, b2])

    torch.testing.assert_close(da1, da2, atol=6e-2, rtol=6e-2)
    torch.testing.assert_close(db1, db2, atol=6e-2, rtol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_2d3d_float32_large_group_count_matches_reference():
    dev = _device()
    g, k, n = 1023, 8, 8
    sizes = torch.zeros((g,), device=dev, dtype=torch.int32)
    sizes[::2] = 1  # 0,1 alternating
    sizes[-1] = 0  # trailing zero-sized group at the boundary
    offs = sizes.cumsum(0).to(torch.int32)
    m_total = int(offs[-1].item())
    assert m_total > 0

    a = torch.randn((m_total, k), device=dev, dtype=torch.float32, requires_grad=True)
    b = torch.randn((g, k, n), device=dev, dtype=torch.float32, requires_grad=True)

    out = grouped_mm(a, b, offs=offs)

    rows = torch.arange(m_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, rows, right=True).to(torch.int64)
    b_g = b.index_select(0, gid)  # [m_total, k, n]
    ref = torch.bmm(a.unsqueeze(1), b_g).squeeze(1)
    torch.testing.assert_close(out, ref, atol=3e-4, rtol=3e-4)

    w = torch.randn_like(out)
    da, db = _grads((out * w).sum(), [a, b])
    da_ref, db_ref = _grads((ref * w).sum(), [a, b])
    torch.testing.assert_close(da, da_ref, atol=6e-4, rtol=6e-4)
    torch.testing.assert_close(db, db_ref, atol=8e-4, rtol=8e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_3d2d_float32_large_group_count_matches_reference():
    dev = _device()
    g, m, k = 1023, 4, 8
    sizes = torch.zeros((g,), device=dev, dtype=torch.int32)
    sizes[::2] = 1
    sizes[-1] = 0
    offs = sizes.cumsum(0).to(torch.int32)
    n_total = int(offs[-1].item())
    assert n_total > 0

    a = torch.randn((g, m, k), device=dev, dtype=torch.float32, requires_grad=True)
    b = torch.randn((k, n_total), device=dev, dtype=torch.float32, requires_grad=True)

    out = grouped_mm(a, b, offs=offs)

    cols = torch.arange(n_total, device=dev, dtype=torch.int32)
    gid = torch.searchsorted(offs, cols, right=True).to(torch.int64)
    a_g = a.index_select(0, gid)  # [n_total, m, k]
    b_cols = b.t().unsqueeze(-1)  # [n_total, k, 1]
    ref_cols = torch.bmm(a_g, b_cols).squeeze(-1)  # [n_total, m]
    ref = ref_cols.t().contiguous()
    torch.testing.assert_close(out, ref, atol=3e-4, rtol=3e-4)

    w = torch.randn_like(out)
    da, db = _grads((out * w).sum(), [a, b])
    da_ref, db_ref = _grads((ref * w).sum(), [a, b])
    torch.testing.assert_close(da, da_ref, atol=8e-4, rtol=8e-4)
    torch.testing.assert_close(db, db_ref, atol=8e-4, rtol=8e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP required")
def test_grouped_mm_2d2d_float32_large_group_count_matches_reference():
    dev = _device()
    g, m, n = 1023, 4, 8
    sizes = torch.zeros((g,), device=dev, dtype=torch.int32)
    sizes[::2] = 1
    sizes[-1] = 0
    offs = sizes.cumsum(0).to(torch.int32)
    k_total = int(offs[-1].item())
    assert k_total > 0

    a = torch.randn((m, k_total), device=dev, dtype=torch.float32, requires_grad=True)
    b = torch.randn((k_total, n), device=dev, dtype=torch.float32, requires_grad=True)

    out = grouped_mm(a, b, offs=offs)

    k_idx = torch.arange(k_total, device=dev, dtype=torch.int32)
    start = torch.cat([torch.zeros((1,), device=dev, dtype=torch.int32), offs[:-1]])
    mask = (k_idx[None, :] >= start[:, None]) & (k_idx[None, :] < offs[:, None])
    a_masked = a.unsqueeze(0) * mask.to(a.dtype).unsqueeze(1)
    ref = torch.matmul(a_masked, b)
    torch.testing.assert_close(out, ref, atol=3e-4, rtol=3e-4)

    w = torch.randn_like(out)
    da, db = _grads((out * w).sum(), [a, b])
    da_ref, db_ref = _grads((ref * w).sum(), [a, b])
    torch.testing.assert_close(da, da_ref, atol=8e-4, rtol=8e-4)
    torch.testing.assert_close(db, db_ref, atol=8e-4, rtol=8e-4)
