"""End-to-end smoke test on data/example.h5ad."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "smoke"


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    data_path = str(ROOT / "data" / "example.h5ad")
    stage1_dir = str(OUT / "stage1")
    bank_dir = str(OUT / "bank")
    stage2_dir = str(OUT / "stage2")
    infer_dir = str(OUT / "inference")

    run([
        "python", "-u", "pipeline/01_train_clip.py",
        "--data_path", data_path,
        "--output_dir", stage1_dir,
        "--epochs", "8",
        "--batch_size", "256",
        "--n_top_genes", "1000",
        "--device", "cpu",
    ])
    run([
        "python", "-u", "pipeline/02_build_bank.py",
        "--data_path", data_path,
        "--stage1_path", str(Path(stage1_dir) / "stage1_best.pt"),
        "--output_dir", bank_dir,
        "--n_top_genes", "1000",
        "--device", "cpu",
    ])
    run([
        "python", "-u", "pipeline/03_train_residual_sklearn.py",
        "--data_path", data_path,
        "--stage1_path", str(Path(stage1_dir) / "stage1_best.pt"),
        "--bank_path", str(Path(bank_dir) / "memory_bank.npz"),
        "--output_dir", stage2_dir,
        "--n_top_genes", "1000",
        "--k", "5",
        "--hidden_dims", "256", "128",
        "--max_iter", "300",
        "--device", "cpu",
    ])
    run([
        "python", "-u", "pipeline/04_inference_residual_sklearn.py",
        "--data_path", data_path,
        "--stage1_path", str(Path(stage1_dir) / "stage1_best.pt"),
        "--bank_path", str(Path(bank_dir) / "memory_bank.npz"),
        "--residual_path", str(Path(stage2_dir) / "residual_mlp.pkl"),
        "--output_dir", infer_dir,
        "--n_top_genes", "1000",
        "--device", "cpu",
    ])


if __name__ == "__main__":
    main()
