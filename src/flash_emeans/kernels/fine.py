"""IVF fine search: ragged per-list GEMM fused with argmin.

A work item is (list, BLOCK_Q routed queries of that list). Programs find
their item by binary search over a device-side tile prefix sum, so there is
no padding, no host-side planning and no sync. Each (row, probe) slot is
written by exactly one item; the final min over probes keeps the lowest id
on ties, as IndexIVFFlat.search_1nn_batched does.
"""
import torch
import triton
import triton.language as tl

BLOCK_Q = 64

_CONFIGS = [
    triton.Config({"BLOCK_C": bc, "BLOCK_D": bd}, num_warps=w, num_stages=s)
    for bc, bd, w, s in [
        (64, 64, 4, 3), (64, 128, 4, 3), (64, 64, 8, 4), (32, 128, 4, 3),
        (128, 64, 8, 3), (64, 128, 8, 3),
    ]
]


@triton.autotune(configs=_CONFIGS, key=["D", "NPROBE"])
@triton.jit
def _fine_kernel(x_ptr, c_ptr, cnorm_ptr, cid_ptr, xb_lims_ptr, qids_ptr,
                 q_lims_ptr, tile_off_ptr, out_s_ptr, out_i_ptr, NLIST,
                 D: tl.constexpr, NPROBE: tl.constexpr,
                 LOG_NLIST: tl.constexpr, BLOCK_Q: tl.constexpr,
                 BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr):
    t = tl.program_id(0)
    if t >= tl.load(tile_off_ptr + NLIST):
        return
    # The owning list is the largest l with tile_off[l] <= t.
    lo = 0
    hi = NLIST
    for _ in tl.static_range(LOG_NLIST + 1):
        mid = (lo + hi) // 2
        go = tl.load(tile_off_ptr + mid) <= t
        lo = tl.where(go, mid, lo)
        hi = tl.where(go, hi, mid)
    tile = t - tl.load(tile_off_ptr + lo)
    qpos = tl.load(q_lims_ptr + lo) + tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    qmask = qpos < tl.load(q_lims_ptr + lo + 1)
    qid = tl.load(qids_ptr + qpos, mask=qmask, other=0)
    row = (qid // NPROBE).to(tl.int64)
    c_end = tl.load(xb_lims_ptr + lo + 1)
    best_s = tl.full((BLOCK_Q,), float("inf"), tl.float32)
    best_i = tl.full((BLOCK_Q,), -1, tl.int32)
    for c0 in range(tl.load(xb_lims_ptr + lo), c_end, BLOCK_C):
        cpos = c0 + tl.arange(0, BLOCK_C)
        cmask = cpos < c_end
        acc = tl.zeros((BLOCK_Q, BLOCK_C), tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            xq = tl.load(x_ptr + row[:, None] * D + d[None, :],
                         mask=qmask[:, None], other=0.0)
            c = tl.load(c_ptr + cpos[:, None].to(tl.int64) * D + d[None, :],
                        mask=cmask[:, None], other=0.0)
            acc = tl.dot(xq, tl.trans(c), acc)
        norm = tl.load(cnorm_ptr + cpos, mask=cmask, other=float("inf"))
        s = norm[None, :] - 2.0 * acc
        # Ids ascend within a list (stable bucket sort): leftmost is lowest.
        m, j = tl.min(s, axis=1, return_indices=True,
                      return_indices_tie_break_left=True)
        better = m < best_s
        best_s = tl.where(better, m, best_s)
        cid = tl.load(cid_ptr + c0 + j, mask=better, other=-1)
        best_i = tl.where(better, cid, best_i)
    tl.store(out_s_ptr + qid, best_s, mask=qmask)
    tl.store(out_i_ptr + qid, best_i, mask=qmask)


def fine_search(x16, vectors, norms, ids, xb_lims, qids, q_lims, nprobe):
    """(score, id) of each row's nearest candidate in its routed lists.

    ``vectors``/``norms``/``ids`` are list-sorted candidates delimited by
    ``xb_lims``; ``qids``/``q_lims`` are the routed (row * nprobe + probe)
    slots bucket-sorted by list. Rows without candidates get (inf, -1).
    """
    nq, D = x16.shape
    nlist = len(q_lims) - 1
    assert x16.dtype == vectors.dtype == torch.float16
    assert x16.is_contiguous() and D % 128 == 0
    tiles = (q_lims[1:] - q_lims[:-1] + BLOCK_Q - 1) // BLOCK_Q
    tile_off = torch.zeros(nlist + 1, dtype=torch.int32, device=x16.device)
    torch.cumsum(tiles, 0, dtype=torch.int32, out=tile_off[1:])
    slots = nq * nprobe
    scores = torch.full((slots,), float("inf"), device=x16.device)
    labels = torch.full((slots,), -1, dtype=torch.int32, device=x16.device)
    # Each list adds at most one partial tile.
    grid = (triton.cdiv(slots, BLOCK_Q) + nlist,)
    _fine_kernel[grid](x16, vectors, norms, ids, xb_lims, qids, q_lims,
                       tile_off, scores, labels, nlist, D=D, NPROBE=nprobe,
                       LOG_NLIST=max(1, (nlist - 1).bit_length()),
                       BLOCK_Q=BLOCK_Q)
    scores, labels = scores.view(nq, nprobe), labels.view(nq, nprobe)
    best = scores.min(1).values
    tied = torch.where(scores == best[:, None], labels,
                       torch.iinfo(torch.int32).max)
    return best, tied.min(1).values
