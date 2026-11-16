"""Install the kernels into faiss.contrib.e_means.

Tested against the faiss commit pinned in third_party/faiss. Every patch
falls back to the original method for inputs the kernels do not cover and
returns that original.
"""
import numpy as np
import torch

from .kernels.fine import fine_search
from .kernels.norms import row_norms
from .kernels.route import route_topk
from .kernels.stats import add_owned

FEATURES = ("route", "fine", "rebuild", "stats")


def enable(*features):
    """Apply ``features`` (default: all); return {feature: original(s)}."""
    return {f: globals()[f"patch_{f}"]() for f in features or FEATURES}


def _fp16(index, stage):
    return index._mm_dtype(stage) is torch.float16


def _coarse16(index):
    if getattr(index, "_c16_src", None) is not index.centroids:
        index._c16 = index.centroids.half().contiguous()
        index._c16_src = index.centroids
    return index._c16


def patch_route():
    """IndexIVFFlat.route: GEMM + top-nprobe in one kernel."""
    from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

    original = IndexIVFFlat.route

    def route(self, x):
        if (not x.is_cuda or not _fp16(self, "coarse") or x.shape[1] % 128
                or self.nprobe & (self.nprobe - 1)):
            return original(self, x)
        self.coarse_ndis += int(x.shape[0]) * self.nlist
        return route_topk(x.half().contiguous(), _coarse16(self),
                          self._current_routing_norms(), self.nprobe)

    IndexIVFFlat.route = route
    return original


def patch_fine():
    """IndexIVFFlat.search_1nn_batched: ragged GEMM + argmin, no planning."""
    from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

    original = IndexIVFFlat.search_1nn_batched

    def search(self, x, grouped, coarse_idx=None):
        if (not x.is_cuda or not _fp16(self, "fine") or self.ids is None
                or self.vectors_lowp is None or x.shape[1] % 128):
            return original(self, x, grouped, coarse_idx)
        self._check_fine_storage()
        qids, q_lims = self.coarse_quantize(x, coarse_idx)
        return fine_search(x.half().contiguous(), self.vectors_lowp,
                           self.vectors_norms, self.ids, self.xb_lims, qids,
                           q_lims, self.nprobe)

    IndexIVFFlat.search_1nn_batched = search
    return original


def patch_rebuild():
    """IndexIVFFlat._add: one FP16 cast, no chunk loop, no host sync."""
    from faiss.contrib.e_means import ivf_flat
    from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

    original = IndexIVFFlat._add
    # Only the original padded search reads the host copy of the limits.
    IndexIVFFlat.xb_lims_cpu = property(
        lambda self: self.xb_lims.cpu().numpy().astype(np.int64),
        lambda self, value: None)

    def _add(self, x):
        if (not x.is_cuda or self.implem < 4 or not _fp16(self, "fine")
                or not _fp16(self, "coarse") or x.shape[1] % 128):
            return original(self, x)
        ivf_flat._check_int32_count(len(x), "candidate rows")
        self.coarse_ndis += len(x) * self.nlist
        x16 = x.half().contiguous()
        # nprobe=1 routing equals the Triton argmin, lowest list on ties.
        lists = route_topk(x16, _coarse16(self), self.centroids_norms.float(),
                           1).view(-1)
        order, lims = ivf_flat._bucket_sort(lists, self.nlist)
        self.list_sizes = lims[1:] - lims[:-1]
        self.set_routing_nonempty(self.list_sizes > 0)
        self.xb_lims = lims.long()
        self.ids = order.int()
        self.vectors = None
        self.vectors_lowp = x16[order]
        self.vectors_norms = row_norms(x16)[order]
        self._fine_storage_key = self._policy_key("fine")

    IndexIVFFlat._add = _add
    return original


def patch_stats():
    """EMA and owner statistics of the sharded classes.

    The IVF class resolves the "native" backend because it assigns rows
    itself, which also drops its EMA from Triton ema_update to ~10 Torch
    passes over the centroid table.
    """
    from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
    from faiss.contrib.e_means.torch_backend import TorchBackend
    from faiss.contrib.e_means.triton_assignment import ema_update

    original_update = TorchBackend.update
    original_add = EmeansShardedCentroids._add_owned

    def update(self, centroids, usage, ids, sums, counts, alpha, decay,
               replace_usage):
        if ids is None and centroids.is_cuda and ema_update(
                sums, counts, centroids, usage, alpha, decay, replace_usage):
            return
        original_update(self, centroids, usage, ids, sums, counts, alpha,
                        decay, replace_usage)

    def _add_owned(self, rows, winner, sums, counts, objective, score=None):
        if (self.spherical or not rows.is_cuda
                or torch.are_deterministic_algorithms_enabled()
                or rows.dtype != torch.float32 or not rows.is_contiguous()
                or sums.dtype != torch.float32 or sums.stride(1) != 1
                or counts.stride(0) != sums.stride(0)
                or objective.dtype != torch.float64
                or not self._centroids.is_contiguous()):
            return original_add(self, rows, winner, sums, counts, objective,
                                score)
        add_owned(rows, winner, self._centroids, sums, counts, objective,
                  self.centroid_begin, self.k)

    TorchBackend.update = update
    EmeansShardedCentroids._add_owned = _add_owned
    return original_update, original_add
