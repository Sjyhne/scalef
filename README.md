# ScaleF

ScaleF is the large-scale continuation of SuperF for test-time optimization based multi-image super-resolution.

This repository contains the core code migrated from SuperF, focused on what is most important to keep momentum:
- Optimization pipelines: `optimize.py`, `optimize_against_folder.py`
- Core data loaders and utilities: `data.py`, `utils.py`, `losses.py`, `viz_utils.py`
- Model and projection modules: `models/`, `input_projections/`
- Data preparation helpers: `create_data_from_single_image.py`, `extract_force_datacube.py`

## Quick Start

```bash
pip install -e .

# Main optimization entrypoint
python optimize.py --dataset satburst_synth --sample_id sample_1 --df 4 --iters 1000

# Or via console script
scalef-train --dataset satburst_synth --sample_id sample_1 --df 4 --iters 1000

# Evaluate in mixed precision (FP16/BF16) to benchmark PSNR vs float32
python optimize.py --dataset satburst_synth --sample_id sample_1 --df 4 --iters 1000 --eval_mixed_precision auto
```

## Benchmarking inference speed

Compare inference time with and without dynamic quantization (CPU):

```bash
python benchmark_speed.py
```

Options: `--warmup`, `--repeat`, `--height`, `--width`, `--device cpu|cuda`, `--checkpoint path/to.pt`. Quantization is run on CPU only (PyTorch dynamic int8).

## Notes

- Large datasets, experiment outputs, and docs/media are intentionally not migrated to keep this repo lean.
- The code remains compatible with the original SuperF workflow while giving you a cleaner base for scaling and performance work.
