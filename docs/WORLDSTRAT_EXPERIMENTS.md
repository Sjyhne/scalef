# WorldStrat experiment pipeline

Run the same attention / SuperF suites as satburst on **WorldStrat sweet** or **bitter** splits.

## Data location

WorldStrat data lives **inside this repo** (not under `superf/`):

- `worldstrat_test_data/` — full corpus (394 areas, ~648 MB)
- `worldstrat_datasets/worldstrat_sweet` and `worldstrat_bitter` — 25 areas each (symlinks into `worldstrat_test_data/`)

Both paths are gitignored.

## Expected layout

Each area is a folder under the dataset root with native HR and LR stacks:

```text
worldstrat_datasets/worldstrat_sweet/
  UNHCR-TURs005204/
    hr/*.png
    lr/*.png
  ...
```

Defaults (override with `--worldstrat_data_root` or `WORLDSTRAT_DATA_ROOT`):

| `--dataset` | Default root |
|-------------|----------------|
| `worldstrat_sweet` | `worldstrat_datasets/worldstrat_sweet` |
| `worldstrat_bitter` | `worldstrat_datasets/worldstrat_bitter` |
| `worldstrat_test` | `worldstrat_test_data` |

Area lists are in [`worldstrat_datasets.txt`](../worldstrat_datasets.txt).

## Run attention suite (sweet, 25 areas × 8 runs)

```bash
SUITE=attention_full DEVICE=cuda:0 DATASET=worldstrat_sweet ./scripts/run_attention_experiment_suite.sh
```

Bitter:

```bash
SUITE=attention_full DEVICE=cuda:0 DATASET=worldstrat_bitter ./scripts/run_attention_experiment_suite.sh
```

Smoke test (one area, short training):

```bash
DATASET=worldstrat_sweet LIMIT_SAMPLES=1 ITERS=100 SUITE=attention ./scripts/run_attention_experiment_suite.sh
```

## Outputs

Same layout as satburst:

`single_samples/<dataset>/<area_name>/<run_name>/metrics.json`

Aggregate: `single_samples/worldstrat_sweet/experiment_results_all_samples.csv`

## Notes

- Loader: `WorldStratTestDataset` — center-cropped LR (max 64×64), HR resized to 4× LR crop, global LR standardization.
- `--num_samples` caps how many LR frames per area are used (default 16, same as satburst suite).
- `--eval_crop_lr_size auto` resolves to **64** for WorldStrat (fixed crop size).
- HashGrid adaptive resolution uses the area’s LR crop side (≤ 64) when CLI `hash_*_resolution` is 0.
- `s2_psf` uses RGB band order B04, B03, B02 on channels 0–2.

Single-area debug:

```bash
python optimize.py --dataset worldstrat_sweet --sample_id "UNHCR-TURs005204" \
  --worldstrat_data_root worldstrat_datasets/worldstrat_sweet \
  --num_samples 16 --iters 2000 --input_projection hashgrid --model mlp_tcnn
```
