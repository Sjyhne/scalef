# MuS2 external evaluation

This integration adapts the public MuS2 benchmark to ScaleF without bundling,
automatically downloading, or modifying MuS2 data.

## Authoritative sources and provenance

- Dataset: Kowaleczko et al., *Data for: MuS2: A Benchmark for Sentinel-2
  Multi-Image Super-Resolution*, Harvard Dataverse, version 2.0,
  [doi:10.7910/DVN/1JMRAT](https://doi.org/10.7910/DVN/1JMRAT).
- Dataset paper: Kowaleczko et al., *A Real-World Benchmark for Sentinel-2
  Multi-Image Super-Resolution*, Scientific Data 10, 644 (2023),
  [doi:10.1038/s41597-023-02538-9](https://doi.org/10.1038/s41597-023-02538-9).
- Official evaluation capsule:
  [Code Ocean capsule 8131193, v2](https://codeocean.com/capsule/8131193/tree/v2).
- Public author repository used to inspect the evaluator:
  [pk94/WVS2Benchmark](https://github.com/pk94/WVS2Benchmark).

DataCite metadata for dataset version 2.0 declares open access and
[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode). It
lists three archives of 1,804,384,808, 1,152,975,259, and 130,289,087 bytes
(about 3.09 GB total). The paper itself is CC BY 4.0. The underlying imagery
comes from Sentinel-2 Level-2A and the WorldView-2 European Cities dataset;
users should retain the dataset citation and review any upstream imagery terms
for their intended redistribution.

The public `pk94/WVS2Benchmark` repository does not contain a license file and
GitHub reports no detected license. This integration therefore copies no code;
it independently implements the protocol described by the paper and observable
behavior of the public evaluator.

No manual-access restriction is declared by the DOI metadata. Harvard
Dataverse controls the current download links, so download the three archives
from the dataset DOI and extract them locally. If the Dataverse page presents
an account, terms-acceptance, or access-request prompt, complete that step
manually; the scripts intentionally do not bypass it.

## Published layout and supported bands

MuS2 contains 91 scenes. Each scene has 14 or 15 Sentinel-2 revisits under
band directories such as `b2/lrs`, plus `hr_resized` WorldView-2 references.
The published evaluation pairs are:

- Sentinel-2 B02 (`b2`) with WorldView-2 blue (`mul_band_1`)
- Sentinel-2 B03 (`b3`) with WorldView-2 green (`mul_band_2`)
- Sentinel-2 B04 (`b4`) with WorldView-2 red (`mul_band_4`)
- Sentinel-2 B08 (`b8`) with WorldView-2 NIR1 (`mul_band_6`)

The adapter prepares one ScaleF dataset per scene and band. Because ScaleF is
an RGB model, the single band is replicated into three channels. This preserves
the single-band MuS2 task without changing `optimize.py`; the evaluator reads
the first output channel. Synthetic, mutually consistent georeferencing is
assigned because MuS2's distributed crops are already pixel-aligned.

## Prepare (dry-run first)

```bash
# Prints discovery and intended outputs; writes nothing.
python scripts/prepare_mus2.py \
  --source /path/to/extracted/MuS2 \
  --output data/mus2_scalef

# Materialize the adapters and manifest.
python scripts/prepare_mus2.py \
  --source /path/to/extracted/MuS2 \
  --output data/mus2_scalef \
  --execute
```

Mask discovery defaults to MuS2 final masks. Use `--mask-root` if masks were
extracted separately, or `--mask-mode none` for an unmasked diagnostic run.
In the official masks, non-zero/white pixels are excluded.

## Batch plan, run, and evaluate

```bash
# Side-effect-free command plan (default).
python scripts/run_mus2.py --manifest data/mus2_scalef/manifest.json

# Explicitly launch ScaleF, then evaluate.
python scripts/run_mus2.py \
  --manifest data/mus2_scalef/manifest.json \
  --evaluate --execute

# Evaluate existing single-band predictions at DIR/SCENE/BAND.tif.
python scripts/run_mus2.py \
  --manifest data/mus2_scalef/manifest.json \
  --predictions-root /path/to/predictions \
  --evaluate-only --execute
```

`--lpips` opts into LPIPS-Alex and may download model weights. It is disabled
by default. cPSNR and cSSIM run on CPU and need no model weights.

The evaluator follows the public implementation where practical:

1. histogram-match each prediction to its WorldView-2 reference;
2. use the final/change/relevance mask with non-zero pixels excluded;
3. crop a 3-pixel border and search the official integer-shift range;
4. select the shift with bias-corrected PSNR and reuse it for corrected SSIM;
5. compare against the mean of bicubically enlarged revisits; and
6. report MuS2's balanced score (lower than 1 is better than bicubic).

The public evaluator's asymmetric six-by-six search for `max_shift=3`
(`[-3, 2]` on each axis) is retained for compatibility. Perceptual LPIPS masks
excluded prediction pixels by replacing them with the reference and uses the
full co-registered image (not the cPSNR-selected shift), as in the public code.
