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
