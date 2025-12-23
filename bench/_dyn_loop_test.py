import torch
import triton
import triton.language as tl

# Ensure HIP Windows patch is applied.
import grouped_mm_bf16.kernels  # noqa: F401

@triton.jit
def dyn_loop_kernel(X_ptr, Y_ptr, offs_ptr, G: tl.constexpr, BLOCK: tl.constexpr):
    g = tl.program_id(0)
    end = tl.load(offs_ptr + g).to(tl.int32)
    start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0).to(tl.int32)
    acc = tl.zeros((BLOCK,), dtype=tl.int32)
    m0 = start
    while m0 < end:
        offs = m0 + tl.arange(0, BLOCK)
        x = tl.load(X_ptr + offs, mask=offs < end, other=0).to(tl.int32)
        acc += x
        m0 += BLOCK
    tl.store(Y_ptr + g * BLOCK + tl.arange(0, BLOCK), acc)

def main():
    dev=torch.device('cuda')
    G=4
    sizes=torch.tensor([0,7,1,13], device=dev, dtype=torch.int32)
    offs=sizes.cumsum(0).to(torch.int32)
    M=int(offs[-1].item())
    x=torch.arange(M, device=dev, dtype=torch.int32)
    y=torch.empty((G*32,), device=dev, dtype=torch.int32)
    dyn_loop_kernel[(G,)](x,y,offs,G=G,BLOCK=32,num_warps=4)
    torch.cuda.synchronize()
    print('ok', y[:8].tolist())

if __name__=='__main__':
    main()
