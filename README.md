<h1 align="center">
RoboStressBench: Benchmarking VLM Robustness to
Physical Visual Stress in Embodied Scenes
</h1>

<p align="center">
Leyi Wu<sup>1,3,*</sup>, Yifan Zhao<sup>1,*</sup>, Jinjie
Zhang<sup>1,*</sup>, Suzeyu Chen<sup>1,3,*</sup>, Wosong Chen<sup>1,3</sup>, Zhifei Chen<sup>1</sup>, Tianshuo Xu<sup>1</sup>, Qingchun He<sup>1</sup>, Hongxin Hu<sup>1</sup>, Haojian Huang<sup>1,3</sup>, Yangkai Wei<sup>3</sup>, Wenqian Li<sup>3</sup>, Yinchuan Li<sup>3</sup>, Ying-Cong
Chen<sup>1,2,†</sup>
</p>

<p align="center">
<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>Knowin
</p>

<p align="center">
Paper: Coming soon &nbsp;|&nbsp; Project Page:
<a href="https://yuevii.github.io/robostressbench-page/">RoboStressBench Page</a>
</p>

## Repository Contents

```text
run.py                  # CLI entry point
eval/                   # dataset loading, prompting, adapters, parsing, metrics, results
configs/                # release-ready YAML templates
requirements.txt        # default environment
requirements-molmo.txt  # Molmo-only environment, use separately
docs/                   # dataset and evaluation notes
```

Generated outputs are written under `outputs/` by default and are ignored by git.

## Dataset

The dataset is intended to be released separately on Hugging Face Datasets, with Zenodo recommended for DOI archival. Download or symlink the dataset so this path exists:

```bash
data/RoboStressBench-Dataset/manifest.jsonl
```

Example after downloading from Hugging Face:

```bash
mkdir -p data
huggingface-cli download RoboStressBench/RoboStressBench-Dataset --repo-type dataset --local-dir data/RoboStressBench-Dataset
```

Dataset page: [RoboStressBench/RoboStressBench-Dataset](https://huggingface.co/datasets/RoboStressBench/RoboStressBench-Dataset)

The evaluation code reads `manifest.jsonl` and per-sample `records/*.json`. The optional `metadata.jsonl` file is provided for easier dataset browsing and external scripts.

## Installation

Default environment for API, Qwen-style, and InternVL-style evaluation:

```bash
conda create -n robostressbench python=3.10
conda activate robostressbench
pip install -r requirements.txt
```

Molmo requires a separate environment because its supported `transformers` version can conflict with newer checkpoints:

```bash
conda create -n robostressbench-molmo python=3.10
conda activate robostressbench-molmo
pip install -r requirements-molmo.txt
```

Do not install both requirement files into the same environment unless you have verified the model-specific `transformers` constraints yourself.

## Configuration

Edit `configs/model_registry.yaml` before local evaluation. Replace `checkpoints/...` paths with your actual model checkpoint directories.

API evaluation uses environment variables for credentials:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
```

## Running

API example:

```bash
python run.py --config configs/api.yaml
```

Local checkpoint example:

```bash
python run.py --config configs/local.yaml --gpu-ids 0
```

The release defaults are conservative (`batch_size: 1`, `max_workers: 1`) so the first run is less likely to hit memory limits. After a smoke test passes, increase `batch_size`, `max_workers`, or `--gpu-ids` to match your hardware.

Molmo example, from the Molmo-specific environment:

```bash
python run.py --config configs/molmo.yaml --gpu-ids 0,1,2,3
```

See `docs/evaluation.md` for smoke-test commands and output details.

## Notes

- Bounding-box ground truth uses `xyxy` coordinates in permille space `[0, 1000]`.
- Placement grounding expects the model to output one point `(x, y)` in permille space; the prediction is correct if it lands on the ground-truth mask.
- Multiple-choice prompts are expanded with lettered options at runtime.

## Citation

BibTeX entry coming soon. We will update this section once the citation is finalized.

```bibtex
@misc{robostressbench2026,
  title  = {TBD},
  author = {TBD},
  year   = {2026},
  note   = {BibTeX placeholder -- to be updated}
}
```

## License

The evaluation framework code in this repository is released under the Apache License 2.0. The RoboStressBench dataset is released separately and remains subject to its dataset-specific terms and the licenses/terms of its source data.
