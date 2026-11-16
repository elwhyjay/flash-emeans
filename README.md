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

### Larger k

Same data and IVF1024/nprobe 4, k raised toward the paper's IVF regime
(Table 3 uses k=2^18 at 100M). The quality law gives batch 2^16 for all three.
One container per k, so compare within a row; faiss IVF time depends on the
host (13.08 s here vs 10.18 s above at k=2^16).

| k | exact | IVF, faiss | IVF, flash-emeans | vs exact | vs faiss |
|---|---|---|---|---|---|
| 2^16 | 8.48 s | 13.08 s | **2.20 s** | 3.9× | 5.9× |
| 2^17 | 15.09 s | 17.19 s | **2.72 s** | 5.5× | 6.3× |
| 2^18 | 28.09 s | 18.19 s | **3.72 s** | 7.6× | 4.9× |

Test MSE, 5 seeds:

| k | exact | IVF, faiss | IVF, flash-emeans |
|---|---|---|---|
| 2^16 | 187519 ± 41 | 187106 ± 28 | 187150 ± 25 |
| 2^17 | 184182 ± 68 | 183189 ± 34 | 183227 ± 55 |
| 2^18 | 182764 ± 159 | 181249 ± 48 | 181235 ± 61 |

With flash-emeans, route stays at 0.28 s while rebuild (0.61 → 1.38 s) and
EMA (0.20 → 0.65 s) grow with k: rebuild routes all k centroids every step,
and EMA reads and writes k×d every step.

### Kernel efficiency

`ncu --set full --clock-control none` on one mid-epoch step with all features
enabled (`--ncu`). SM clocks are not locked (1.77–1.99 GHz), and ncu flushes
caches before each kernel. The tensor rate counts executed FP16 WGMMA,
including tile padding; the percentage is of ncu's dense FP16 peak at the
measured clock.

| kernel | k | time | FP16 tensor | DRAM | L2 |
|---|---|---|---|---|---|
| `route`, batch rows (nprobe 4) | 2^16 | 407 µs | 338 TFLOP/s (35%) | 0.55 TB/s (17%) | 74% |
| | 2^18 | 393 µs | 349 TFLOP/s (36%) | 0.48 TB/s (14%) | 43% |
| `route`, rebuild (nprobe 1) | 2^16 | 400 µs | 344 TFLOP/s (36%) | 2.70 TB/s (81%) | 79% |
| | 2^18 | 1249 µs | 440 TFLOP/s (46%) | 0.65 TB/s (19%) | 61% |
| `fine` | 2^16 | 514 µs | 115 TFLOP/s (12%) | 2.58 TB/s (77%) | 87% |
| | 2^18 | 814 µs | 295 TFLOP/s (31%) | 2.07 TB/s (62%) | 81% |

- `route` is bound by neither compute nor memory. It runs 8 warps per SM
  (188–244 registers, up to 193 KB shared memory) and stalls mostly on
  `wait` and `barrier`. nprobe 4 costs 27% more per row than nprobe 1,
  because the top-k epilogue does not overlap the MMA. In the k=2^16 rebuild,
  the autotuner picked 256-row tiles whose operands do not stay in L2: it
  reads 1 GiB from DRAM for 128 MiB of rows.
- `fine` is bandwidth-bound (L2 at 81–87%). Each query tile re-reads the
  centroids of its list, and query rows are gathered once per probe.
- The memory-bound kernels: `row_norms` reaches 2.8–3.0 TB/s (82–90% of
  DRAM peak) and the `stats` accumulation 2.4 TB/s (72%). The faiss Triton
  EMA uses 2 rows per CTA, so it launches k/2 CTAs, and it falls from
  2.73 TB/s at k=2^16 to 1.09 TB/s at k=2^18.

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
`--ncu` under `ncu --profile-from-start off` captures one mid-epoch step per
variant (add `--clock-control none` where clocks cannot be locked).
The exact baseline (no `--ivf`) on H100 also needs the CuTe packages listed
in `third_party/faiss/benchs/e_means/README.md`.

## Layout

- `src/flash_emeans/kernels/`: faiss-independent kernels (upstream candidates)
- `src/flash_emeans/patch.py`: installs them into `faiss.contrib.e_means`
- `tests/`: each patch against the original, synthetic data
- `bench/`: DINO benchmark
