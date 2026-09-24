# Paper

| Path | Role |
|--|--|
| [`OUTLINE.md`](OUTLINE.md) | Claims, section plan, figure/table wishlist |
| [`DRAFT.md`](DRAFT.md) | Working manuscript draft (prose + frozen numbers) |
| [`results/`](results/) | **Only** canonical JSON for tables / figures |
| [`results/MANIFEST.json`](results/MANIFEST.json) | Provenance: source path, mtime, SHA-256 |

`results/` is **not** a working sweep directory. Exploratory runs, smokes, production mosaics, and one-off benches stay in `single_samples/sweep_results/`.

## Layout (results)

## What is in `results/`

| File | Role |
|--|--|
| `paper_ablations.json` | B0–B4 + `S_aoi*` (7 cities, 52 runs) — quality **and** speed |
| `paper_fourier_fair.json` | Fair Fourier `F_aoi*_s{2,5,10}` + hash `H_aoi*` (Asker, 24) |
| `paper_fourier_fair_s2.json` | Related fair / s=2 cells |
| `bench_complete_patches_size_ladder_nested.json` | Nested LR64→512 same-footprint summary |
| `bench_complete_patches_nest_lr{64,128,256,512}.json` | Per-tile nest ladder rows |
| `spatial_alignment.json` | Frozen HR↔LR shifts for eval AOIs |

## Updating

1. Re-run the driver (`scripts/run_paper_ablations.py`, size ladder, …) into `single_samples/sweep_results/` as usual.
2. After accepting the summaries, run `python scripts/build_paper_results.py`. Use
   `--dry-run` to validate without writing; incomplete inputs require the explicit
   `--allow-partial` opt-in.
3. The builder copies canonical JSON here, refreshes SHA-256 provenance in
   `MANIFEST.json`, and writes generated exports only below
   `ScaleF_Overleaf/generated/`. Existing `tables/` and `figures/` remain handwritten.
4. Keep large per-run dirs (`single_samples/*/sample/...`) out of this folder.

Drivers still default to `sweep_results/`; this folder is the curated freeze for writing.
