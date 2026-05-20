# ScaleF

ScaleF is the large-scale continuation of SuperF for test-time optimization based multi-image super-resolution.

This repository contains the core code migrated from SuperF, focused on what is most important to keep momentum:
- Optimization pipelines: `optimize.py`
- Core data loaders and utilities: `data.py`, `utils.py`, `losses.py`, `viz_utils.py`
- Model and projection modules: `models/`, `input_projections/`
- Data preparation helpers: `create_data_from_single_image.py`

## Attention experiment suite (exam ablation)

Fair comparison of Fourier vs HashGrid baselines and decoder-side attention (methods A–D). See [`docs/ATTENTION_EXPERIMENTS.md`](docs/ATTENTION_EXPERIMENTS.md) for hyperparameters, reproduction commands, and saved metrics under `single_samples/`.

```bash
SUITE=attention_full DEVICE=cuda:0 ./scripts/run_attention_experiment_suite.sh
```

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

## HashGrid (tiny-cuda-nn) training

HashGrid uses **CUDA** (`tinycudann`). Use `--device cuda:0` (or an index) or rely on CPU fallback only for non-hash runs.

**Projection aliases:** `hashgrid`, `hash`, `ngp_hash`, `hashgrid_tcnn`, `hash_tcnn`, `ngp_hash_tcnn` all resolve to the same tcnn encoder.

**Encoder hyperparameters (all optional; shown with typical values):**

| Flag | Meaning |
|------|---------|
| `--hash_n_levels` | Multires levels `L` |
| `--hash_n_features_per_level` | Features per level `F` (decoder input is `L*F`) |
| `--hash_log2_hashmap_size` | Log2 hash table size |
| `--hash_base_resolution` | Coarsest grid resolution |
| `--hash_max_resolution` | Finest target resolution (sets per-level scale with `n_levels`) |
| `--hash_encoding_dtype` | `fp16` or `fp32` (tcnn encoding output) |
| `--hash_encoding_preset` | `hashgrid` (Linear) or `smoothstep_grid` (Grid+Hash+Smoothstep) |
| `--hash_grid_type` | With `smoothstep_grid`: `Hash`, `Dense`, or `Tiled` |

**Decoder / training (shared with Fourier):** `--model mlp` or `mlp_tcnn`, `--network_depth`, `--network_hidden_dim`, `--tcnn_mlp_dtype`, `--optimizer adamw` (required for hash; not `muon`), `--learning_rate`, `--weight_decay`, `--num_samples`, `--aug`, `--lr_shift`, etc.

### Default-style HashGrid (repo defaults)

```bash
python optimize.py \
  --dataset satburst_synth \
  --sample_id sample_1 \
  --df 4 \
  --lr_shift 1.0 \
  --aug none \
  --num_samples 16 \
  --iters 2000 \
  --device cuda:0 \
  --input_projection hashgrid \
  --hash_n_levels 16 \
  --hash_n_features_per_level 2 \
  --hash_log2_hashmap_size 21 \
  --hash_base_resolution 16 \
  --hash_max_resolution 2048 \
  --hash_encoding_dtype fp32 \
  --hash_encoding_preset hashgrid \
  --model mlp \
  --network_depth 4 \
  --network_hidden_dim 256 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.05
```

### Recommended settings for ~64×64 LR → 256×256 HR (`df` 4)

Aligned with the internal HashGrid experiment doc (moderate resolution, lighter table):

```bash
python optimize.py \
  --dataset satburst_synth \
  --sample_id sample_1 \
  --df 4 \
  --lr_shift 1.0 \
  --aug none \
  --num_samples 16 \
  --iters 2000 \
  --device cuda:0 \
  --input_projection hashgrid \
  --hash_n_levels 16 \
  --hash_n_features_per_level 2 \
  --hash_log2_hashmap_size 21 \
  --hash_base_resolution 8 \
  --hash_max_resolution 256 \
  --hash_encoding_dtype fp32 \
  --hash_encoding_preset smoothstep_grid \
  --hash_grid_type Hash \
  --model mlp_tcnn \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --tcnn_mlp_dtype fp16 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0
```

Console script equivalent: prefix with `scalef-train` instead of `python optimize.py`.

## Notes

- Large datasets, experiment outputs, and docs/media are intentionally not migrated to keep this repo lean.
- The code remains compatible with the original SuperF workflow while giving you a cleaner base for scaling and performance work.
