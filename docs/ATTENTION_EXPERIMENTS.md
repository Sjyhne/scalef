# Attention experiment suite (exam ablation)

This document describes the **reproducible configuration** for the A/B/C/D comparison and where to find **saved metrics** for the project report.

## Specification sources

| Source | Role |
|--------|------|
| [`updated_attention_experiment_agent_brief.md`](../updated_attention_experiment_agent_brief.md) | Research goals, constraints, method matrix, report framing |
| [`scripts/run_experiment_suite.py`](../scripts/run_experiment_suite.py) | Canonical hyperparameters (`build_attention_experiment_suite`, `attention_full`) |
| [`scripts/run_attention_experiment_suite.sh`](../scripts/run_attention_experiment_suite.sh) | Convenience wrapper (`SUITE`, `DATASET`, `DEVICE`, …) |
| Per-run `run_command.txt` / `metrics.json` → `"command"` | Exact CLI used on the machine that produced each run |

## How to reproduce (synthetic report set)

```bash
cd /path/to/scalef
SUITE=attention_full DEVICE=cuda:0 ./scripts/run_attention_experiment_suite.sh
```

Equivalent:

```bash
python scripts/run_experiment_suite.py \
  --suite attention_full \
  --dataset satburst_synth \
  --satburst_data_root data \
  --device cuda:0 \
  --weight_decay 0 \
  --seed 6 \
  --iters 2000
```

## Shared settings (all methods A–D)

| Setting | Value |
|---------|--------|
| Dataset | `satburst_synth` → scenes under `data/` |
| `df` | 4 |
| `lr_shift` | 1.0 |
| `aug` | none |
| `num_samples` | 16 |
| `iters` | 2000 |
| `batch_size` | 1 |
| `optimizer` | adamw |
| `learning_rate` | 2e-3 |
| `weight_decay` | 0 |
| `seed` | 6 |
| Decoder | depth 3, hidden 64 |
| Eval crop request | `--eval_crop_lr_size 64 --eval_crop_anchor topleft` |
| Plain MLP decoder | `mlp_tcnn` + `--tcnn_mlp_dtype fp16` |

### Method-specific knobs

| ID | `run_name` | Projection | Model | Extra flags |
|----|------------|------------|-------|-------------|
| A | `fourier_mlp_baseline` | Fourier | `mlp_tcnn` | `--projection_dim 256 --fourier_scale 10` |
| B | `hashgrid_mlp_baseline` | HashGrid | `mlp_tcnn` | 12 levels, `smoothstep_grid`, base 8, max 256, fp32 encoding |
| C | `hashgrid_level_attention` | HashGrid | `hash_attn` | `--hash_attn_token_dim 32 --log_attention` |
| D | `fourier_band_attention` | Fourier | `fourier_band_attn` | `--attn_token_dim 32 --fourier_num_bands 8 --log_attention` |

## Saved artifacts

Under `single_samples/<dataset>/<scene_id>/<run_name>/`:

- `metrics.json` / `metrics.txt` — PSNR, SSIM, LPIPS, timing, eval crop info, full command
- `run_command.txt` — CLI copy
- `attention_log.json` — methods C and D only (level/band weights over training)
- `experiment_results.csv` — per-scene aggregate
- `experiment_results_all_samples.csv` — full suite table (report-ready)

PNG visualizations are generated locally but **not** committed (see `.gitignore`).

## Report results (`satburst_synth`, 20 scenes, `attention_full`)

Mean over 20 scenes (from `single_samples/satburst_synth/experiment_results_all_samples.csv`):

| Method | Mean PSNR | Mean SSIM | Mean time/iter |
|--------|-----------|-----------|----------------|
| A Fourier + mlp_tcnn | 33.47 dB | 0.871 | 3.7 ms |
| B HashGrid + mlp_tcnn | **37.57 dB** | **0.953** | 3.8 ms |
| C HashGrid + level attn | 36.00 dB | 0.938 | 7.4 ms |
| D Fourier + band attn | 27.43 dB | 0.618 | 4.8 ms |

**Head-to-head A vs B:** HashGrid wins PSNR on **20/20** scenes; SSIM also higher on all scenes.

**Attention:** C is usually below B but above A; D is consistently weak on synth (often below bilinear PSNR).

## Known limitations / flaws to mention in the report

1. **Single seed (6)** — no variance estimate across random seeds.
2. **Eval crop often not applied on synth** — LR patches are already 64×64, so `--eval_crop_lr_size 64` is a no-op (`eval_crop.applied: false` in metrics). Metrics are on the **full** HR patch; this is **fair across methods** but differs from real-data scenes where crop is active.
3. **Fourier-band attention (D)** — poor synth performance; treat as negative result or debug (band count, attention capacity, optimization difficulty).
4. **HashGrid capacity vs Fourier** — B uses a high-resolution multiscale hash (max 256); A uses fixed Fourier features. The comparison is “exam-fair” per brief, not parameter-matched for representation size.
5. **Metrics** — PSNR / SSIM / LPIPS only; no MSE/MAE in `metrics.json`.
6. **Timing** — `time_per_iteration_seconds` includes periodic eval every 100 steps during training.
7. **Paths in `command`** — may show `/workspaces/scalef/...` from the machine that ran jobs; reproducibility is via flags, not path.
8. **Brief vs synth outcome** — the brief was written when Fourier looked strongest on some setups; on **this** synth suite, **HashGrid MLP is the clear baseline winner**.

## Real-data runs (optional)

`single_samples/satburst_real/` may exist from the same suite with `DATASET=satburst_real` and `MAX_LR_SIDE=256`. Eval crop behavior differs by scene; use synth for the primary controlled comparison unless the report explicitly contrasts datasets.
