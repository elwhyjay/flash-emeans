"""Run tests and benchmarks on one Modal H100.

    uv run modal run bench/modal_app.py::download
    uv run modal run bench/modal_app.py::run --cmd "python tests/test_kernels.py"
    uv run modal run bench/modal_app.py::run \
        --cmd "python bench/bench_ivf.py --ivf --patches route,fine,rebuild,stats"
    uv run modal volume get emeans-data out/ ./out/
"""
from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent
FAISS = ROOT / "third_party/faiss"
DINO = "https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B"
DEEP = ("https://huggingface.co/datasets/suchun/deep1b-mirror/resolve/main/"
        "base.1B.fbin")

# (url, path in volume, bytes): first 2**24 DINO rows (paper App. M.3) and
# the DEEP* 10M prefix used by faiss benchs/e_means/demo_deep10m.py.
FILES = [
    (f"{DINO}/queries_clean.bvecs", "DINO10B/queries_clean.bvecs",
     102_800_000),
    (f"{DINO}/chunked_base_10B/chunk_0000.bvecs",
     "DINO10B/chunked_base_10B/chunk_0000.bvecs", (1 << 24) * 1028),
    (DEEP, "deep/base.1B.fbin.part", 3_843_840_008),
]

data = modal.Volume.from_name("emeans-data", create_if_missing=True)
cache = modal.Volume.from_name("emeans-cache", create_if_missing=True)
volumes = {"/data": data, "/cache": cache}
skip = ["**/__pycache__"]

image = (
    # Host driver is CUDA 13.3; the devel image ships ncu, nsys is added.
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu24.04",
                              add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "cuda-nsight-systems-13-0")
    # Versions from faiss benchs/e_means/README.md (tested on H100).
    .pip_install("torch==2.15.0", "faiss-cpu==1.15.1", "quack-kernels==0.6.5",
                 "nvidia-cutlass-dsl==4.7.1", "cuda-python>=13,<14", "numpy",
                 "cuda-cccl[cu13]")
    .env({"TRITON_CACHE_DIR": "/cache/triton", "XDG_CACHE_HOME": "/cache/xdg",
          "PYTHONPATH": "/root/src"})
    # Mounted at container start: local edits apply without a rebuild.
    .add_local_dir(FAISS / "contrib/e_means", "/root/e_means",
                   ignore=skip + ["figure"])
    .add_local_dir(FAISS / "benchs/e_means", "/root/faiss_bench", ignore=skip)
    .add_local_dir(ROOT / "src", "/root/src", ignore=skip)
    .add_local_dir(ROOT / "tests", "/root/tests", ignore=skip)
    .add_local_dir(ROOT / "bench", "/root/bench", ignore=skip)
)
app = modal.App("flash-emeans", image=image)


@app.function(volumes=volumes, timeout=4 * 3600, cpu=4)
def download():
    import os
    import subprocess

    for url, rel, size in FILES:
        path = f"/data/{rel}"
        if os.path.exists(path) and os.path.getsize(path) == size:
            print("have", rel)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["curl", "--fail", "--location", "--retry", "5",
                        "--range", f"0-{size - 1}", "-o", path, url],
                       check=True)
        assert os.path.getsize(path) == size, rel
    data.commit()


@app.function(gpu="H100!", volumes=volumes, timeout=4 * 3600, cpu=8,
              memory=32 * 1024)
def run(cmd: str):
    """Run ``cmd`` from /root (src/, tests/, bench/ mirror the repo)."""
    import os
    import subprocess

    import faiss

    # faiss wheels lack contrib/e_means; link the pinned checkout's copy.
    target = os.path.join(os.path.dirname(faiss.__file__), "contrib",
                          "e_means")
    if not os.path.exists(target):
        os.symlink("/root/e_means", target)
    if not os.path.exists("/tmp/base.1B.fbin.part"):  # demo_deep10m.py path
        os.symlink("/data/deep/base.1B.fbin.part", "/tmp/base.1B.fbin.part")
    os.makedirs("/data/out", exist_ok=True)
    subprocess.run("nvidia-smi --query-gpu=name,power.limit,driver_version "
                   "--format=csv", shell=True)
    try:
        subprocess.run(cmd, shell=True, check=True, cwd="/root")
    finally:
        data.commit()
        cache.commit()
