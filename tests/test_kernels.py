"""Each patch against the faiss.contrib.e_means code it replaces (CUDA).

    python tests/test_kernels.py      # or pytest
"""
from types import SimpleNamespace

import torch
from faiss.contrib.e_means.ivf_flat import IndexIVFFlat
from faiss.contrib.e_means.sharded_centroids import EmeansShardedCentroids
from faiss.contrib.e_means.torch_backend import TorchBackend

import flash_emeans

ORIG = flash_emeans.enable()


def data(n, d=256, seed=0):
    """Clustered unit-variance rows, so IVF lists are unbalanced."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    centers = 2 * torch.randn(64, d, device="cuda", generator=g)
    x = centers[torch.randint(0, 64, (n,), device="cuda", generator=g)]
    x += torch.randn(n, d, device="cuda", generator=g)
    return (x / x.std()).contiguous()


def reference_index(nprobe, d=256, nlist=128, k=8192):
    x = data(nlist + k, d)
    index = IndexIVFFlat(x[:nlist], nprobe=nprobe)
    ORIG["rebuild"](index, x[nlist:])
    return index, x[nlist:]


def test_route():
    for d, nprobe in ((256, 4), (256, 16), (96, 4)):  # d=96: fallback
        index, _ = reference_index(nprobe, d)
        q = data(4096, d, seed=1)
        ref, got = ORIG["route"](index, q), index.route(q)
        same = (ref.sort(1).values == got.sort(1).values).all(1)
        assert same.float().mean() > 0.999, (d, nprobe)


def test_fine():
    for nprobe in (4, 16):
        index, _ = reference_index(nprobe)
        q = data(4096, seed=1)
        routing = index.route(q)
        ref_s, ref_i = ORIG["fine"](index, q, False, routing)
        got_s, got_i = index.search_1nn_batched(q, False, routing)
        assert (ref_i == got_i).float().mean() > 0.999 and (got_i >= 0).all()
        assert torch.allclose(ref_s, got_s, rtol=1e-4, atol=1e-2)


def test_rebuild():
    ref, cands = reference_index(4)
    got = IndexIVFFlat(ref.centroids, nprobe=4)
    got.add(cands)
    assert torch.equal(ref.list_sizes, got.list_sizes)
    assert (ref.ids == got.ids).float().mean() > 0.999
    assert torch.allclose(ref.vectors_norms, got.vectors_norms, rtol=1e-5)
    assert (got.xb_lims_cpu == ref.xb_lims_cpu).all()  # on-demand host copy


def test_stats():
    k, d = 4096, 256
    rows, centroids = data(16384, d), data(k, d, seed=2)
    winner = torch.randint(-3, k + 3, (len(rows),), device="cuda")
    owner = SimpleNamespace(spherical=False, centroid_begin=0, k=k,
                            _centroids=centroids)
    original_update, original_add = ORIG["stats"]

    def accumulate(add):
        acc = torch.zeros(k, d + 1, device="cuda")
        obj = torch.zeros((), dtype=torch.float64, device="cuda")
        for b in range(0, len(rows), 4096):
            add(owner, rows[b:b + 4096], winner[b:b + 4096], acc[:, :-1],
                acc[:, -1], obj)
        return acc, obj

    (ref, ref_obj), (got, got_obj) = (
        accumulate(original_add),
        accumulate(EmeansShardedCentroids._add_owned))
    assert torch.equal(ref[:, -1], got[:, -1])
    assert torch.allclose(ref, got, rtol=1e-5, atol=1e-3)
    assert ((ref_obj - got_obj) / ref_obj).abs() < 1e-6

    backend, usage = TorchBackend(4096), torch.rand(k, device="cuda") * 100
    states = [(centroids.clone(), usage.clone()) for _ in range(2)]
    for update, (c, u) in zip((original_update, TorchBackend.update), states):
        update(backend, c, u, None, got[:, :-1], got[:, -1], 0.05, 0.95, False)
    assert torch.equal(states[0][0], states[1][0])
    assert torch.equal(states[0][1], states[1][1])


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
