# Single-sample experiment outputs

Training outputs from `scripts/run_experiment_suite.py` / `optimize.py` (`--run_name`).

## Layout

```text
single_samples/<dataset>/<scene_id>/<run_name>/
  metrics.json          # primary numbers for the report
  metrics.txt
  run_command.txt
  attention_log.json    # hash / fourier attention runs only
  experiment_results.csv   # per-scene (parent folder)
```

Dataset-level aggregate:

```text
single_samples/<dataset>/experiment_results_all_samples.csv
```

## Exam ablation (report)

- **Synthetic (primary):** `satburst_synth/` — 20 scenes × 4 methods (`attention_full`)
- Configuration: [`docs/ATTENTION_EXPERIMENTS.md`](../docs/ATTENTION_EXPERIMENTS.md)

PNGs (`comparison.png`, etc.) are produced on disk but gitignored to keep the repository small.
