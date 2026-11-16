"""IVF routing: FP16 GEMM fused with a running top-nprobe.

The (B, nlist) score matrix is never written. Operands match
IndexIVFFlat.route: FP16 rows and centroids, FP32 accumulation and FP32
norms (+inf for empty lists). Probe order within a row is unspecified.
"""
import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BLOCK_B": bb, "BLOCK_K": bk, "BLOCK_D": bd},
                  num_warps=w, num_stages=s)
    for bb, bk, bd, w, s in [
        (128, 128, 64, 8, 3), (128, 128, 64, 8, 4), (64, 128, 64, 4, 4),
        (128, 64, 64, 4, 4), (128, 128, 128, 8, 3), (256, 128, 64, 8, 3),
    ]
]


@triton.autotune(configs=_CONFIGS, key=["K", "D", "NPROBE"])
@triton.jit
def _route_kernel(x_ptr, c_ptr, norm_ptr, out_ptr, B, K,
                  D: tl.constexpr, NPROBE: tl.constexpr,
                  BLOCK_B: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_D: tl.constexpr):
    rows = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
    rmask = rows < B
    slot = tl.arange(0, NPROBE)
    top_s = tl.full((BLOCK_B, NPROBE), float("inf"), tl.float32)
    # Distinct initial ids keep rows with < NPROBE routable lists valid.
    top_i = tl.zeros((BLOCK_B, NPROBE), tl.int32) + slot[None, :]
    lane = tl.arange(0, BLOCK_K)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + lane
        cmask = cols < K
        acc = tl.zeros((BLOCK_B, BLOCK_K), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            x = tl.load(x_ptr + rows[:, None].to(tl.int64) * D + d[None, :],
                        mask=rmask[:, None], other=0.0)
            c = tl.load(c_ptr + cols[:, None] * D + d[None, :],
                        mask=cmask[:, None], other=0.0)
            acc = tl.dot(x, tl.trans(c), acc)
        norm = tl.load(norm_ptr + cols, mask=cmask, other=float("inf"))
        score = norm[None, :] - 2.0 * acc
        for _ in tl.static_range(NPROBE):
            m, j = tl.min(score, axis=1, return_indices=True,
                          return_indices_tie_break_left=True)
            worst, w = tl.max(top_s, axis=1, return_indices=True,
                              return_indices_tie_break_left=True)
            take = (m < worst)[:, None] & (slot[None, :] == w[:, None])
            top_s = tl.where(take, m[:, None], top_s)
            top_i = tl.where(take, (j + k0)[:, None], top_i)
            score = tl.where(lane[None, :] == j[:, None], float("inf"), score)
    out = out_ptr + rows[:, None].to(tl.int64) * NPROBE + slot[None, :]
    tl.store(out, top_i.to(tl.int64), mask=rmask[:, None])


def route_topk(x16, c16, norms, nprobe):
    """Ids (B, nprobe) of the nprobe smallest ``norms - 2 x16 @ c16.T``."""
    B, D = x16.shape
    assert x16.dtype == c16.dtype == torch.float16
    assert norms.dtype == torch.float32
    assert x16.is_contiguous() and c16.is_contiguous() and D % 128 == 0
    assert nprobe & (nprobe - 1) == 0, "nprobe must be a power of two"
    out = torch.empty((B, nprobe), dtype=torch.int64, device=x16.device)
    grid = lambda meta: (triton.cdiv(B, meta["BLOCK_B"]),)  # noqa: E731
    _route_kernel[grid](x16, c16, norms, out, B, c16.shape[0], D,
                        NPROBE=nprobe)
    return out
