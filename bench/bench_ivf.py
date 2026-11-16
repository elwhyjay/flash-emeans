"""DINO e-means timing (exact vs IVF): baseline and cumulative patches in
one process, with the paper's protocol (untimed identical warm-up, then
timed runs).

Mirrors faiss benchs/e_means/demo_sharded_centroids.py on one GPU, minus
per-epoch test evaluation. Compare variants only within one run: the
unpatched IVF path is partly host-bound, so its time depends on the CPU.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

import flash_emeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "third_party/faiss/benchs/e_means"))
from dino_data import DinoBvecs, DinoQueries, query_mse  # noqa: E402
from faiss.contrib.e_means import EmeansShardedCentroids, EmeansShardedCentroidsIVF  # noqa: E402
from faiss.contrib.e_means.base import _epoch_seed  # noqa: E402
from faiss.contrib.e_means.e_means_torch import _torch_generator  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/DINO10B")
p.add_argument("--n", type=int, default=1 << 24)
p.add_argument("--k", type=int, default=1 << 16)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--seeds", default="0", help="comma list; one timed run per seed (plus --repeat on the first)")
p.add_argument("--ivf", action="store_true")
p.add_argument("--nlist", type=int, default=1024)
p.add_argument("--nprobe", type=int, default=4)
p.add_argument("--repeat", type=int, default=3)
p.add_argument("--profile", action="store_true", help="torch.profiler over one extra run")
p.add_argument("--stages", action="store_true", help="synchronized per-stage wall time, one extra run")
p.add_argument("--patches", default="",
               help="flash_emeans features applied cumulatively after the baseline, e.g. route,fine; "
                    "'+' joins features into one step, e.g. route+fine+rebuild+stats")
p.add_argument("--ncu", action="store_true",
               help="capture one mid-epoch step per variant; run under ncu --profile-from-start off")
args = p.parse_args()
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
dist.init_process_group("nccl", rank=0, world_size=1)
device = torch.device("cuda")

x = DinoBvecs(args.data, args.n).resident(0, args.n, device)
mean = x.mean(dim=0)
x.sub_(mean)
scale = 1.0
if args.ivf:  # FP16 IVF needs unit scale; chunked so no second n*d temporary
    sq = sum(x[i:i + (1 << 20)].square().sum(dtype=torch.float64) for i in range(0, len(x), 1 << 20))
    scale = (sq / x.numel()).sqrt().item()
    x.div_(scale)


seeds = [int(s) for s in args.seeds.split(",")]


def train(seed=seeds[0], capture=False):
    cls = EmeansShardedCentroidsIVF if args.ivf else EmeansShardedCentroids
    extra = {"nlist": args.nlist, "nprobe": args.nprobe} if args.ivf else {}
    model = cls(x.shape[1], args.k, nepochs=args.epochs, batch_law="quality", seed=seed,
                group=dist.group.WORLD, verbose=False, **extra)
    init = torch.randperm(len(x), device=device, generator=_torch_generator(device, seed))[:args.k]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.begin(args.n, x[init].float())
    batch = model.recipe["batch_size"]
    for epoch in range(args.epochs):
        order = torch.randperm(len(x), device=device,
                               generator=_torch_generator(device, _epoch_seed(seed, epoch)))
        for b in range(0, len(x), batch):
            mid = capture and epoch == 0 and b == len(x) // batch // 2 * batch
            if mid:
                torch.cuda.profiler.start()
            model.update(model.accumulate(x[order[b:b + batch]]))
            if mid:
                torch.cuda.profiler.stop()
        model.end_epoch()
    model.finalize()
    torch.cuda.synchronize()
    return time.perf_counter() - t0, model


tag = f"{'ivf%d.np%d' % (args.nlist, args.nprobe) if args.ivf else 'exact'} n={args.n} k={args.k} T={args.epochs}"


def measure(label):
    warm, model = train()
    times = []
    for _ in range(args.repeat):
        model = None
        t, model = train()
        times.append(t)
    times.sort()
    print(f"[{tag} {label}] warm-up {warm:.2f}s timed {['%.2f' % t for t in times]} "
          f"median={times[len(times) // 2]:.2f}s", flush=True)
    queries = DinoQueries(args.data, x.shape[1])
    for seed in seeds:
        if seed != seeds[0]:
            model = None
            _, model = train(seed)
        test = query_mse(model, queries.batches(mean, scale, 1024)) * scale**2
        print(f"[{tag} {label}] seed={seed} train_mse={model.obj[-1].item() / args.n * scale**2:.1f} "
              f"test_mse={test:.1f}", flush=True)
    if args.ncu:
        model = None
        train(capture=True)


measure("baseline")
for name in filter(None, args.patches.split(",")):
    flash_emeans.enable(*name.split("+"))
    measure(f"+{name}")

if args.profile:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        train()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30, max_name_column_width=90))

if args.stages:
    from faiss.contrib.e_means.base import _EmeansBase
    from faiss.contrib.e_means.ivf_flat import IndexIVFFlat

    spent, calls = {}, {}

    def wrap(cls, name, label):
        fn = getattr(cls, name)

        def timed(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                torch.cuda.synchronize()
                spent[label] = spent.get(label, 0.0) + time.perf_counter() - t
                calls[label] = calls.get(label, 0) + 1
        setattr(cls, name, timed)

    cls = EmeansShardedCentroidsIVF if args.ivf else EmeansShardedCentroids
    wrap(cls, "_prepare_assignment", "1 prepare centroids / rebuild IVF")
    wrap(cls, "_gather_assignment_rows", "2 batch rows + route")
    wrap(IndexIVFFlat, "route", "2a   route (coarse top-nprobe)")
    wrap(cls, "_local_assignment_batch", "3 assignment")
    wrap(IndexIVFFlat, "coarse_quantize", "3a   bucket sort by list")
    wrap(IndexIVFFlat, "list_chunks", "3b   chunk planning (host)")
    wrap(cls, "_add_owned", "4 accumulate sums/counts")
    wrap(_EmeansBase, "update", "5 EMA update + revival")
    wrap(cls, "_apply_statistics", "5a   EMA (apply statistics)")
    wrap(cls, "_revive_unused", "5b   revival")
    total, model = train()
    print(f"[{tag}] stages (synchronized run, total {total:.2f}s, n_splits={model.n_splits}):")
    for label in sorted(spent):
        print(f"  {label:38s} {spent[label]:7.2f}s {100 * spent[label] / total:5.1f}%  calls={calls[label]}")
    print(f"  {'(sum of top-level stages)':38s} {sum(v for k, v in spent.items() if not k[1].isalpha()):7.2f}s")
dist.destroy_process_group()
