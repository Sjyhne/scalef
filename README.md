# ScaleF

ScaleF is the large-scale continuation of SuperF for test-time optimization based multi-image super-resolution.

This repository contains the core code migrated from SuperF, focused on what is most important to keep momentum:
- Optimization pipeline: `optimize.py`
- Utilities: `data.py`, `utils.py`, `losses.py`
- Model and projection modules: `models/`, `input_projections/`
- Sentinel-2 revisit download: `scripts/fetch_s2_revisits.py`

## Quick Start

```bash
pip install -e .

# Download Sentinel-2 revisits (full MGRS 10 m tile; --size-km is the study/crop window)
python scripts/fetch_s2_revisits.py \
  --date 2019-07-15 --lon -76.53 --lat 37.41 --size-km 1.28 \
  --num-samples 8 --cloud-method omnicloudmask --out data/s2_revisits/demo
```

## Method

The current training method, rejected alternatives, and open questions are in [docs/METHOD.md](docs/METHOD.md).

## External evaluation

The optional [MuS2 integration](docs/MUS2.md) provides a manifest-driven,
dry-run-by-default adapter and the benchmark's aligned single-band evaluation.
It does not download the approximately 3.1 GB dataset automatically.

## Notes

- Large datasets and experiment outputs are gitignored.
- Download Sentinel-2 revisits with `scripts/fetch_s2_revisits.py`. Training loaders for WorldStrat / satburst were removed.
