# Repository Agent Brief: Minimal HashGrid-Level Attention for ScaleF/SuperF

## Goal

Implement a minimal decoder-side attention module for the `course-base-2026-05-08` branch of `Sjyhne/scalef`.

The exam experiment should compare:

1. `HashGrid + plain MLP decoder` baseline
2. `HashGrid + lightweight level-attention decoder` proposed method

The attention mechanism must operate on the multiresolution HashGrid feature levels. Do **not** change the SuperF observation model, alignment path, LR masking, LR reconstruction loss, or fixed downsampling/alignment code.

---

## Non-negotiable scope constraints

Keep this current pipeline intact:

```text
HR coordinates
-> per-frame affine warp
-> HashGrid encoder
-> decoder
-> predicted HR image
-> existing LR alignment/downsampling path
-> LR reconstruction loss
```

Only replace this part:

```text
HashGrid features -> plain MLP -> RGB
```

with:

```text
HashGrid features + warped coordinates -> level-attention decoder -> RGB
```

Do not add:

- learned downsampler
- PSF model
- pixel/window Transformer attention
- attention inside average pooling / LR observation model
- new dataset logic
- new loss beyond optional attention diagnostics

---

## Target files

Implement/edit only these files unless absolutely necessary:

```text
models/hash_attention_decoder.py      # new
models/inr.py                         # support coordinate-aware decoder
models/utils.py                       # register hash_attn decoder
optimize.py                           # add CLI args, model validation, metrics logging
```

Optional helper if needed:

```text
scripts/run_hash_attn_ablation.sh      # optional convenience script
```

---

## Architecture to implement

HashGrid output has shape:

```text
[N, L * F]
```

where:

```text
L = hash_n_levels
F = hash_n_features_per_level
```

Reshape it into per-level tokens:

```text
[N, L, F]
```

Use the warped coordinate `q = [x, y]` as a query and compute a softmax over HashGrid levels:

```text
level features -> token projection -> [N, L, D]
coords -> query projection -> [N, D]
attention logits = dot(token_l, query) / sqrt(D)
attention weights = softmax(logits over L)
fused token = weighted sum over levels
RGB = MLP([fused token, coords])
```

This is coordinate-conditioned scale selection over HashGrid levels.

---

## New module: `models/hash_attention_decoder.py`

Create a small PyTorch module. Keep it simple and debuggable.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HashLevelAttentionDecoderLite(nn.Module):
    """Lightweight coordinate-conditioned attention over HashGrid levels.

    Expected call signature:
        rgb, aux = decoder(coords, hash_features)

    coords:        [N, 2]
    hash_features: [N, L * F]
    rgb:           [N, out_dim]
    aux: diagnostics with detached attention stats
    """

    requires_coords = True

    def __init__(
        self,
        n_levels: int,
        n_features_per_level: int,
        token_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 3,
        coord_dim: int = 2,
    ):
        super().__init__()
        self.n_levels = int(n_levels)
        self.n_features_per_level = int(n_features_per_level)
        self.token_dim = int(token_dim)
        self.coord_dim = int(coord_dim)

        self.level_proj = nn.Linear(self.n_features_per_level, self.token_dim)
        self.level_embed = nn.Parameter(torch.zeros(1, self.n_levels, self.token_dim))

        self.coord_query = nn.Sequential(
            nn.Linear(self.coord_dim, self.token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
        )

        self.rgb_mlp = nn.Sequential(
            nn.Linear(self.token_dim + self.coord_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords: torch.Tensor, hash_features: torch.Tensor):
        if coords.ndim != 2 or coords.shape[-1] != self.coord_dim:
            raise ValueError(f"coords must be [N,{self.coord_dim}], got {tuple(coords.shape)}")
        if hash_features.ndim != 2:
            raise ValueError(f"hash_features must be [N,L*F], got {tuple(hash_features.shape)}")

        expected = self.n_levels * self.n_features_per_level
        if hash_features.shape[-1] != expected:
            raise ValueError(
                f"hash_features last dim must be {expected}, got {hash_features.shape[-1]}"
            )

        n = hash_features.shape[0]
        h = hash_features.reshape(n, self.n_levels, self.n_features_per_level)

        tokens = self.level_proj(h) + self.level_embed              # [N, L, D]
        query = self.coord_query(coords).unsqueeze(1)               # [N, 1, D]

        logits = (tokens * query).sum(dim=-1) / math.sqrt(self.token_dim)  # [N, L]
        attn = torch.softmax(logits, dim=-1)                        # [N, L]
        fused = (attn.unsqueeze(-1) * tokens).sum(dim=1)            # [N, D]

        rgb = self.rgb_mlp(torch.cat([fused, coords], dim=-1))

        with torch.no_grad():
            entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)
            fine_k = min(3, self.n_levels)
            aux = {
                "attn": attn.detach(),
                "attn_mean": attn.detach().mean(dim=0),
                "attn_entropy_mean": entropy.detach().mean(),
                "attn_fine_mass_last3": attn[:, -fine_k:].detach().sum(dim=-1).mean(),
            }

        return rgb, aux
```

---

## Edit `models/inr.py`

Modify `INRBase._encode_and_decode` so existing decoders still work, while `hash_attn` can receive both warped coordinates and HashGrid features.

Current behavior is roughly:

```python
projected = self.input_projection(warped_q, progress=progress)
raw_flat = projected.reshape(n_pix, -1)
output_flat = self.decoder(raw_flat)
```

Change to this pattern:

```python
projected = self.input_projection(warped_q, progress=progress) if self.input_projection is not None else warped_q
n_pix = B * H * W
raw_flat = projected.reshape(n_pix, -1)

if getattr(self.decoder, "requires_coords", False):
    q_flat = warped_q.reshape(n_pix, -1)
    output_flat, aux = self.decoder(q_flat, raw_flat)
    self.last_decoder_aux = aux
else:
    output_flat = self.decoder(raw_flat)
    self.last_decoder_aux = {}

output = output_flat.reshape(B, H, W, -1)
```

Keep return signature unchanged:

```python
return output, shifts, warped_coords
```

Important: pass **warped** coordinates, not base coordinates. The decoder should attend based on the actual queried coordinate after the per-frame affine warp.

---

## Edit `models/utils.py`

Extend `get_decoder(...)` with optional HashGrid-attention arguments:

```python
hash_n_levels=None,
hash_n_features_per_level=None,
hash_attn_token_dim=32,
```

Add a branch:

```python
elif network_name == "hash_attn":
    from models.hash_attention_decoder import HashLevelAttentionDecoderLite
    if hash_n_levels is None or hash_n_features_per_level is None:
        raise ValueError("hash_attn requires hash_n_levels and hash_n_features_per_level")
    return HashLevelAttentionDecoderLite(
        n_levels=hash_n_levels,
        n_features_per_level=hash_n_features_per_level,
        token_dim=hash_attn_token_dim,
        hidden_dim=network_hidden_dim,
        out_dim=output_dim,
    )
```

Update the error message to include `hash_attn`.

---

## Edit `optimize.py`

### 1. Allow the new model

In `build_input_projection_decoder_bundle`, change:

```python
if args.model not in {"mlp", "mlp_tcnn", "nir"}:
```

to:

```python
if args.model not in {"mlp", "mlp_tcnn", "nir", "hash_attn"}:
```

After projection normalization, enforce:

```python
if args.model == "hash_attn" and canon != "hashgrid_tcnn":
    raise ValueError("--model hash_attn requires --input_projection hashgrid/hashgrid_tcnn")
```

### 2. Pass HashGrid attention params into `get_decoder`

Change the `get_decoder(...)` call so it includes:

```python
hash_n_levels=args.hash_n_levels,
hash_n_features_per_level=args.hash_n_features_per_level,
hash_attn_token_dim=args.hash_attn_token_dim,
```

### 3. Add CLI arg

In the argument parser, add:

```python
parser.add_argument("--hash_attn_token_dim", type=int, default=32)
```

Update `--model` help text to include `hash_attn`.

### 4. Log attention diagnostics

When writing `metrics.json`, add this if available:

```python
aux = getattr(model, "last_decoder_aux", {}) or {}
if "attn_mean" in aux:
    metrics_dict["attention"] = {
        "mean_by_level": aux["attn_mean"].detach().cpu().tolist(),
        "entropy_mean": float(aux["attn_entropy_mean"].detach().cpu()),
        "fine_mass_last3": float(aux["attn_fine_mass_last3"].detach().cpu()),
    }
```

Optional but useful: save a small bar plot called:

```text
attention_mean_by_level.png
```

inside the same sample output directory.

---

## Smoke tests

Run these before any long experiment.

### Baseline smoke test

```bash
python optimize.py \
  --dataset satburst_synth \
  --sample_id sample_1 \
  --df 4 \
  --lr_shift 1.0 \
  --aug none \
  --num_samples 16 \
  --iters 20 \
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
  --model mlp \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --run_name smoke_hashgrid_mlp
```

### Attention smoke test

```bash
python optimize.py \
  --dataset satburst_synth \
  --sample_id sample_1 \
  --df 4 \
  --lr_shift 1.0 \
  --aug none \
  --num_samples 16 \
  --iters 20 \
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
  --model hash_attn \
  --hash_attn_token_dim 32 \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --run_name smoke_hashgrid_level_attention
```

The attention smoke test passes if:

- training starts without shape errors
- loss decreases or remains finite
- final evaluation runs
- `metrics.json` exists
- `metrics.json` includes an `attention` block

---

## Main experiments for the exam

Keep the experiment matrix small.

| Run | Model | Purpose |
|---|---|---|
| `hashgrid_mlp` | `--model mlp` | Main baseline |
| `hashgrid_level_attention` | `--model hash_attn` | Proposed method |

Use identical settings except for `--model` and attention-specific args.

### Baseline: HashGrid + plain MLP

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
  --model mlp \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --run_name hashgrid_mlp
```

### Proposed: HashGrid + level attention

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
  --model hash_attn \
  --hash_attn_token_dim 32 \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --run_name hashgrid_level_attention
```

Optional if time permits:

```bash
# Same two runs with --iters 1000 for a faster preliminary table.
# Same two runs with another sample_id if sample_1 gives unstable/ambiguous results.
```

Do not expand beyond this unless the two core runs are complete.

---

## Metrics to extract

From each run collect:

- PSNR
- SSIM
- LPIPS
- final reconstruction loss
- runtime / time per iteration if available
- comparison image
- training curve

For the attention run additionally collect:

- mean attention per HashGrid level
- mean attention entropy
- mass on the finest 3 levels
- `attention_mean_by_level.png` if implemented

Suggested result table columns:

```text
method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | final recon loss ↓ | time/iter ↓
```

Suggested attention table/figure:

```text
level index | mean attention weight
```

---

## Expected output artifacts

Each run should produce enough for the report:

```text
metrics.json
metrics.txt
comparison.png
training_metrics.png
prediction/model output image
bilinear baseline image
attention_mean_by_level.png      # attention run only, optional but recommended
```

The report only needs:

1. one quantitative table
2. one qualitative comparison figure
3. one attention-level bar plot

---

## Acceptance criteria for the implementation

The patch is acceptable when:

1. Existing `--model mlp`, `--model mlp_tcnn`, and `--model nir` still work.
2. `--model hash_attn` works only with `--input_projection hashgrid` / `hashgrid_tcnn`.
3. The SuperF LR observation/alignment/loss path is unchanged.
4. Attention receives warped coordinates and HashGrid features.
5. Attention weights have shape `[N, hash_n_levels]`.
6. `metrics.json` for the attention run contains attention diagnostics.
7. The two 20-iteration smoke tests run without crashing.
8. The two 2000-iteration exam runs can be compared directly.

---

## Minimal report claim enabled by this implementation

> I add a lightweight coordinate-conditioned attention decoder that adaptively fuses multiresolution HashGrid levels inside a SuperF-style test-time optimized neural field, while preserving the original LR image-formation model. The ablation tests whether representation-side scale selection improves reconstruction compared with plain HashGrid feature concatenation.
