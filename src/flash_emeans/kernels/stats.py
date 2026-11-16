"""Owner statistics in one pass per row: atomic sums and counts, plus the
FP64 objective of FP32 row-to-centroid squared errors."""
import triton
import triton.language as tl


@triton.jit
def _owned_stats_kernel(rows_ptr, winner_ptr, cent_ptr, sums_ptr,
                        counts_ptr, obj_ptr, n, begin, k, stats_stride,
                        D: tl.constexpr, BLOCK_R: tl.constexpr,
                        BLOCK_D: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    w = tl.load(winner_ptr + r, mask=r < n, other=-1).to(tl.int64) - begin
    owned = (r < n) & (w >= 0) & (w < k)
    sq = tl.zeros((BLOCK_R,), tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        m = owned[:, None] & (d[None, :] < D)
        v = tl.load(rows_ptr + r[:, None].to(tl.int64) * D + d[None, :],
                    mask=m, other=0.0)
        c = tl.load(cent_ptr + w[:, None] * D + d[None, :], mask=m, other=0.0)
        tl.atomic_add(sums_ptr + w[:, None] * stats_stride + d[None, :], v,
                      mask=m, sem="relaxed")
        diff = v - c
        sq += tl.sum(diff * diff, axis=1)
    ones = tl.full((BLOCK_R,), 1.0, tl.float32)
    tl.atomic_add(counts_ptr + w * stats_stride, ones, mask=owned,
                  sem="relaxed")
    tl.atomic_add(obj_ptr, tl.sum(sq.to(tl.float64), axis=0), sem="relaxed")


def add_owned(rows, winner, centroids, sums, counts, objective, begin, k):
    """Add rows whose winner falls in [begin, begin + k) to the statistics.

    ``sums``/``counts`` are column views of one (k, d + 1) FP32 buffer.
    """
    n, D = rows.shape
    _owned_stats_kernel[(triton.cdiv(n, 16),)](
        rows, winner, centroids, sums, counts, objective, n, begin, k,
        sums.stride(0), D=D, BLOCK_R=16, BLOCK_D=128)
