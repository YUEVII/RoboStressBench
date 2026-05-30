# Environments

RoboStressBench intentionally provides two requirement files.

## Default Environment

Use `requirements.txt` for API evaluation and mainstream local checkpoints such as Qwen and InternVL. This environment keeps `transformers` unpinned so users can install a version compatible with newer checkpoints.

```bash
conda create -n robostressbench python=3.10
conda activate robostressbench
pip install -r requirements.txt
```

If your CUDA setup requires a specific PyTorch wheel, install PyTorch following the official PyTorch selector before installing the rest of the requirements.

## Molmo Environment

Use `requirements-molmo.txt` only for Molmo evaluation. Molmo is pinned to `transformers==4.57.1`, which may be incompatible with newer local checkpoints.

```bash
conda create -n robostressbench-molmo python=3.10
conda activate robostressbench-molmo
pip install -r requirements-molmo.txt
```

Keep Molmo runs in this separate environment and use `configs/molmo.yaml`.
