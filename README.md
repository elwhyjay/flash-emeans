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
| exact assignment | 8.25 s | |
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

## Usage

```python
import flash_emeans
flash_emeans.enable()  # or enable("route", "fine")
# then train EmeansShardedCentroidsIVF as usual
```

Requires a CUDA GPU, torch, triton, and `faiss.contrib.e_means` from the
faiss commit pinned in `third_party/faiss` (not in a faiss release yet).
Unsupported inputs (CPU, d % 128 != 0, non power-of-two nprobe,
deterministic mode for `stats`) fall back to the original faiss code.

## Layout

- `src/flash_emeans/kernels/`: faiss-independent kernels (upstream candidates)
- `src/flash_emeans/patch.py`: installs them into `faiss.contrib.e_means`
- `tests/`: each patch against the original, synthetic data
- `bench/`: DINO benchmark and the Modal runner

```bash
uv sync
uv run modal run bench/modal_app.py::download
uv run modal run bench/modal_app.py::run --cmd "python tests/test_kernels.py"
uv run modal run bench/modal_app.py::run \
    --cmd "python bench/bench_ivf.py --ivf --patches route,fine,rebuild,stats"
```
