# flash-emeans

Fused Triton kernels for the IVF assignment path of
[e-means](https://arxiv.org/abs/2611.09501) (`faiss.contrib.e_means`).
Unofficial; not affiliated with the e-means authors or Meta.

## Result

One H100 SXM, first 2^24 rows of DINO10B (d=1024), k=2^16, IVF1024,
nprobe=4, 2 epochs (the paper's App. M.3 setting). All variants run in the
same process; test MSE is over 5 seeds.

| | time | test MSE |
|---|---|---|
| exact assignment | 8.25 s | 187519 ± 41 |
| IVF, faiss | 10.18 s | 187157 ± 27 |
| IVF, flash-emeans | **1.97 s** | 187159 ± 34 |

| feature | replaces | change |
|---|---|---|
| `route` | coarse GEMM + `torch.topk` | GEMM fused with running top-nprobe |
| `fine` | padded BMM + host chunk planning | ragged per-list GEMM fused with argmin |
| `rebuild` | per-chunk gather/cast/norm + host sync | one FP16 cast, reuses `route` |
| `stats` | Torch EMA, masked index_add | Triton EMA, one-pass owner statistics |

Each kernel returns the same winners as the faiss code it replaces, up to
FP32 accumulation-order ties (`tests/test_kernels.py`).

### Breakdown

Features enabled cumulatively, same process as above:

| | time | test MSE |
|---|---|---|
| faiss | 10.18 s | 187157 ± 27 |
| + `route` | 8.00 s | 187184 ± 39 |
| + `fine` | 4.48 s | 187178 ± 54 |
| + `rebuild` | 4.13 s | 187155 ± 29 |
| + `stats` | 1.97 s | 187159 ± 34 |

Per stage (`--stages`, one synchronized run each). Both columns come from one
container on a slower host than above; the unpatched path is partly
host-bound.

| stage | faiss | flash-emeans |
|---|---|---|
| rebuild IVF | 1.24 s | 0.62 s |
| batch rows + route | 4.38 s | 0.45 s |
| &nbsp;&nbsp;route | 4.21 s | 0.29 s |
| assignment | 4.59 s | 0.51 s |
| &nbsp;&nbsp;bucket sort | 0.19 s | 0.19 s |
| &nbsp;&nbsp;chunk planning (host) | 0.94 s | |
| accumulate sums/counts | 1.77 s | 0.32 s |
| EMA + revival | 1.41 s | 0.46 s |
| &nbsp;&nbsp;EMA | 1.15 s | 0.20 s |
| &nbsp;&nbsp;revival | 0.24 s | 0.23 s |
| total | 13.84 s | 2.78 s |

## Usage

```python
import flash_emeans
flash_emeans.enable()  # or enable("route", "fine")
# then train EmeansShardedCentroidsIVF as usual
```

Unsupported inputs (CPU, d % 128 != 0, non power-of-two nprobe,
deterministic mode for `stats`) fall back to the original faiss code.

## Setup

Needs a CUDA GPU with torch and triton. `faiss.contrib.e_means` is not in a
faiss release yet, so link the copy pinned in `third_party/faiss`:

```bash
git clone --recurse-submodules https://github.com/elwhyjay/flash-emeans.git
cd flash-emeans
pip install -e . faiss-cpu==1.15.1
ln -s "$PWD/third_party/faiss/contrib/e_means" \
    "$(python -c 'import faiss, os; print(os.path.dirname(faiss.__file__))')/contrib/e_means"
python tests/test_kernels.py
```

## Benchmark

First 2^24 DINO10B rows (17 GB) and the held-out queries:

```bash
U=https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B
mkdir -p data/DINO10B/chunked_base_10B
curl -L -o data/DINO10B/queries_clean.bvecs $U/queries_clean.bvecs
curl -L --range 0-17246978047 -o data/DINO10B/chunked_base_10B/chunk_0000.bvecs \
    $U/chunked_base_10B/chunk_0000.bvecs
python bench/bench_ivf.py --ivf --patches route,fine,rebuild,stats
```

Rows stay on the GPU in FP32 (64 GiB at 2^24); use `--n` on smaller GPUs.
The exact baseline (no `--ivf`) on H100 also needs the CuTe packages listed
in `third_party/faiss/benchs/e_means/README.md`.

## Layout

- `src/flash_emeans/kernels/`: faiss-independent kernels (upstream candidates)
- `src/flash_emeans/patch.py`: installs them into `faiss.contrib.e_means`
- `tests/`: each patch against the original, synthetic data
- `bench/`: DINO benchmark
