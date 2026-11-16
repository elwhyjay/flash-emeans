"""FP32 squared norms of FP16 rows in one pass (no FP32 copy of the rows)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _row_norms_kernel(x_ptr, out_ptr, n, D: tl.constexpr,
                      BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    mask = rows < n
    acc = tl.zeros((BLOCK_R,), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        v = tl.load(x_ptr + rows[:, None].to(tl.int64) * D + d[None, :],
                    mask=mask[:, None] & (d[None, :] < D), other=0.0)
        v = v.to(tl.float32)
        acc += tl.sum(v * v, axis=1)
    tl.store(out_ptr + rows, acc, mask=mask)


def row_norms(x16):
    n, D = x16.shape
    assert x16.is_contiguous()
    out = torch.empty(n, dtype=torch.float32, device=x16.device)
    _row_norms_kernel[(triton.cdiv(n, 32),)](x16, out, n, D=D,
                                             BLOCK_R=32, BLOCK_D=128)
    return out
