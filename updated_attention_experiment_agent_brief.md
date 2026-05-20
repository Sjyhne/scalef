# Updated Agent Brief: Attention Experiments After Fourier Baseline Wins

## Purpose

Implement and run the **minimal** code and experiment changes needed for the exam project after the current finding that **standard Fourier features are the strongest baseline so far**.

The project should now be framed as a controlled ablation:

> Does attention-based feature fusion help SuperF-style test-time neural super-resolution, or are fixed Fourier features better conditioned than learned HashGrid features under the available optimization budget?

This is not a failure case. If Fourier remains best, the report should present that as the main empirical insight.

---

## Current empirical status

- Standard Fourier + MLP is currently the best-performing method.
- HashGrid + plain MLP is still useful as a learned multiscale representation baseline.
- HashGrid + level attention is still useful as the main attention method.
- Optional: Fourier-band attention can be added if it is quick, because Fourier is currently the strongest representation.

Do **not** aggressively tune new architectures. Prioritize a fair, reproducible comparison.

---

## Hard constraints

Do not change the SuperF observation model.

Keep this pipeline unchanged:

```text
HR coordinate grid
-> per-frame affine warp
-> input projection
-> decoder
-> predicted HR image
-> fixed avg_pool2d(pred_hr, kernel=df, stride=df)
-> strict valid LR mask / LR alignment path
-> LR reconstruction loss
```

Do not add:

```text
learned downsampler
PSF model
stochastic footprint integration
pixel-wise Transformer attention
new dataset
new loss function
large architecture rewrite
```

The attention mechanism should only change the **decoder-side feature fusion**.

---

## Required experiment matrix

Run these three methods first:

| ID | Method | Input projection | Decoder | Priority |
|---|---|---|---|---|
| A | Fourier baseline | Fourier | plain MLP | required |
| B | HashGrid baseline | HashGrid | plain MLP / mlp_tcnn | required |
| C | HashGrid attention | HashGrid | level-attention decoder | required |
| D | Fourier-band attention | Fourier | band-attention decoder | optional only if quick |

All runs should use the same:

```text
dataset
sample_id
df
lr_shift
aug
num_samples
iters
seed if available
optimizer
learning_rate
weight_decay
eval_crop_lr_size if used
```

Save the exact command for every run.

---

## Recommended final framing

Use this research question:

```text
Does attention-based feature fusion improve test-time optimized neural representations
for multi-image super-resolution, and how does its behavior differ between fixed Fourier
features and learned HashGrid features?
```

Likely conclusion if Fourier stays best:

```text
Fixed Fourier features provide a stable spectral basis that is easier to optimize in
the limited test-time setting. HashGrid attention adds adaptive multiscale fusion, but
the additional trainable representation and attention parameters may make optimization
harder unless more iterations, regularization, or better schedules are used.
```

---

## Implementation tasks

### 1. Keep or add coordinate-aware decoder interface

In `models/inr.py`, modify `INRBase._encode_and_decode` so decoders can optionally receive both coordinates and encoded features.

Current behavior is roughly:

```python
projected = self.input_projection(warped_q, progress=progress)
raw_flat = projected.reshape(n_pix, -1)
output_flat = self.decoder(raw_flat)
```

Use:

```python
projected = self.input_projection(warped_q, progress=progress)
feat_flat = projected.reshape(n_pix, -1)
coord_flat = warped_q.reshape(n_pix, -1)

if getattr(self.decoder, "requires_coords", False):
    decoder_out = self.decoder(coord_flat, feat_flat)

    if isinstance(decoder_out, tuple):
        output_flat, aux = decoder_out
        self.last_aux = aux
    else:
        output_flat = decoder_out
        self.last_aux = {}
else:
    output_flat = self.decoder(feat_flat)
    self.last_aux = {}
```

This allows both HashGrid level attention and Fourier-band attention.

---

### 2. HashGrid level-attention decoder

Create or update:

```text
models/attention_decoders.py
```

Add:

```python
class HashLevelAttentionDecoderLite(nn.Module):
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
        ...
```

Expected input/output:

```text
coord_flat: [N, 2]
hash_flat:  [N, L * F]
output:     [N, 3]
aux:
  attention_mean_per_level: [L]
  attention_entropy: scalar
  fine_level_mass: scalar
```

Architecture:

```text
hash_flat
-> reshape [N, L, F]
-> per-level linear projection to [N, L, token_dim]
-> add learnable level embedding [1, L, token_dim]
-> coordinate query MLP(coord_flat) -> [N, token_dim]
-> dot-product logits over levels
-> softmax over L
-> weighted sum of level tokens -> [N, token_dim]
-> MLP([fused_token, coord_flat]) -> RGB
```

Minimal code structure:

```python
class HashLevelAttentionDecoderLite(nn.Module):
    requires_coords = True

    def __init__(self, n_levels, n_features_per_level, token_dim=32,
                 hidden_dim=64, out_dim=3, coord_dim=2):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.token_dim = token_dim

        self.level_proj = nn.Linear(n_features_per_level, token_dim)
        self.level_embed = nn.Parameter(torch.zeros(1, n_levels, token_dim))

        self.coord_query = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, token_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(token_dim + coord_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, coords, features):
        N = features.shape[0]
        x = features.view(N, self.n_levels, self.n_features_per_level)

        tokens = self.level_proj(x) + self.level_embed
        query = self.coord_query(coords)

        logits = (tokens * query.unsqueeze(1)).sum(dim=-1) / (self.token_dim ** 0.5)
        attn = torch.softmax(logits, dim=1)

        fused = (attn.unsqueeze(-1) * tokens).sum(dim=1)
        out = self.decoder(torch.cat([fused, coords], dim=-1))

        eps = 1e-8
        entropy = -(attn * (attn + eps).log()).sum(dim=1).mean()
        fine_mass = attn[:, -min(3, self.n_levels):].sum(dim=1).mean()

        aux = {
            "attention_mean_per_level": attn.detach().mean(dim=0),
            "attention_entropy": entropy.detach(),
            "fine_level_mass": fine_mass.detach(),
        }
        return out, aux
```

Keep this simple. Do not add multi-head attention unless all required experiments are already finished.

---

### 3. Optional Fourier-band attention decoder

Only implement this if it can be done quickly.

Purpose: since Fourier is currently best, test whether attention helps **the winning representation** rather than only HashGrid.

Add:

```python
class FourierBandAttentionDecoderLite(nn.Module):
    requires_coords = True
```

Expected behavior:

```text
coord_flat:    [N, 2]
fourier_flat:  [N, D]
output:        [N, 3]
```

Group Fourier features into frequency-band tokens. The exact grouping depends on the Fourier projection implementation. The agent should inspect `input_projections/` and infer the safest grouping.

Suggested robust grouping:

```python
# If D is divisible by num_bands, reshape directly.
# Otherwise choose num_bands so D % num_bands == 0.
# Start with num_bands = 8, then fallback to 4, 2, 1.
```

Architecture:

```text
fourier_flat
-> reshape [N, K, F_band]
-> band projection to [N, K, token_dim]
-> add band embedding
-> coordinate query
-> softmax over K bands
-> weighted fused token
-> MLP([fused_token, coord_flat]) -> RGB
```

CLI model name:

```text
--model fourier_band_attn
```

If this takes too long or causes shape confusion, skip it. The report can still focus on Fourier baseline vs HashGrid attention.

---

### 4. Register decoders

In `models/utils.py`, update `get_decoder`.

It currently accepts:

```python
get_decoder(
    network_name,
    network_depth,
    input_dim,
    network_hidden_dim,
    output_dim=3,
    tcnn_mlp_dtype="fp16",
    mlp_init="kaiming",
    device=None,
)
```

Add optional arguments:

```python
hash_n_levels=None
hash_n_features_per_level=None
attn_token_dim=32
fourier_num_bands=8
```

Add branches:

```python
elif network_name == "hash_attn":
    from models.attention_decoders import HashLevelAttentionDecoderLite
    if hash_n_levels is None or hash_n_features_per_level is None:
        raise ValueError("hash_attn requires hash_n_levels and hash_n_features_per_level")
    return HashLevelAttentionDecoderLite(
        n_levels=hash_n_levels,
        n_features_per_level=hash_n_features_per_level,
        token_dim=attn_token_dim,
        hidden_dim=network_hidden_dim,
        out_dim=output_dim,
    )

elif network_name == "fourier_band_attn":
    from models.attention_decoders import FourierBandAttentionDecoderLite
    return FourierBandAttentionDecoderLite(
        input_dim=input_dim,
        num_bands=fourier_num_bands,
        token_dim=attn_token_dim,
        hidden_dim=network_hidden_dim,
        out_dim=output_dim,
    )
```

Update the error message to include:

```text
mlp, mlp_tcnn, nir, hash_attn, fourier_band_attn
```

---

### 5. Update `optimize.py`

In `build_input_projection_decoder_bundle`, allow new models:

```python
if args.model not in {"mlp", "mlp_tcnn", "nir", "hash_attn", "fourier_band_attn"}:
    raise ValueError(...)
```

When calling `get_decoder`, pass:

```python
hash_n_levels=args.hash_n_levels,
hash_n_features_per_level=args.hash_n_features_per_level,
attn_token_dim=args.attn_token_dim,
fourier_num_bands=args.fourier_num_bands,
```

Add CLI args:

```python
parser.add_argument("--attn_token_dim", type=int, default=32)
parser.add_argument("--fourier_num_bands", type=int, default=8)
parser.add_argument("--log_attention", action="store_true")
```

---

### 6. Save attention diagnostics

If `model.last_aux` exists, periodically save attention diagnostics.

Minimal approach inside the existing training loop, at the same interval as PSNR logging:

```python
if getattr(args, "log_attention", False):
    aux = getattr(model, "last_aux", {})
    if aux:
        # Convert tensors to Python lists/floats.
        # Append to attention_log list.
```

At the end of each sample, save:

```text
sample_xxx/attention_log.json
```

Each record should contain:

```json
{
  "iteration": 100,
  "attention_mean_per_level": [...],
  "attention_entropy": 0.0,
  "fine_level_mass": 0.0
}
```

This is mainly for the final report figure. Do not over-engineer.

---

## Commands to run

Adjust paths/seed/output names as needed.

### A. Fourier baseline

Use the exact currently best Fourier command if already known. If not, use this template:

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
  --input_projection fourier \
  --model mlp \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --run_name fourier_mlp_baseline
```

If the current best Fourier run uses a different `projection_dim`, `fourier_scale`, depth, hidden dimension, LR, or iterations, keep the best known version and document it.

---

### B. HashGrid baseline

Use the repo-recommended moderate HashGrid setting:

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
  --hash_n_levels 12 \
  --hash_n_features_per_level 2 \
  --hash_log2_hashmap_size 16 \
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
  --weight_decay 0.0 \
  --run_name hashgrid_mlp_baseline
```

If `mlp_tcnn` causes issues with the new interface or fair comparison, use `--model mlp`.

---

### C. HashGrid level attention

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
  --hash_n_levels 12 \
  --hash_n_features_per_level 2 \
  --hash_log2_hashmap_size 16 \
  --hash_base_resolution 8 \
  --hash_max_resolution 256 \
  --hash_encoding_dtype fp32 \
  --hash_encoding_preset smoothstep_grid \
  --hash_grid_type Hash \
  --model hash_attn \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --attn_token_dim 32 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --log_attention \
  --run_name hashgrid_level_attention
```

---

### D. Optional Fourier-band attention

Only run if implemented quickly.

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
  --input_projection fourier \
  --model fourier_band_attn \
  --network_depth 3 \
  --network_hidden_dim 64 \
  --attn_token_dim 32 \
  --fourier_num_bands 8 \
  --optimizer adamw \
  --learning_rate 2e-3 \
  --weight_decay 0.0 \
  --log_attention \
  --run_name fourier_band_attention
```

---

## Metrics to collect

For each method, collect:

```text
model_psnr
model_ssim
model_lpips
model_mse
model_mae
final_recon_loss
training_time_seconds
time_per_iteration_seconds
```

Also save:

```text
comparison.png
prediction_aligned.png
ground_truth.png
training_metrics.png
results JSON/CSV
exact command
```

For attention methods, additionally collect:

```text
attention_mean_per_level or per_band
attention_entropy
fine_level_mass for HashGrid
```

---

## Final table template

```text
Method                  Projection   Attention   PSNR   SSIM   LPIPS   LR loss   Time/iter
Fourier + MLP           Fourier      No          ...    ...    ...     ...       ...
HashGrid + MLP          HashGrid     No          ...    ...    ...     ...       ...
HashGrid + level attn   HashGrid     Yes         ...    ...    ...     ...       ...
Fourier-band attn       Fourier      Yes         ...    ...    ...     ...       ...  optional
```

---

## Result interpretation guide

### If Fourier wins

Use:

```text
Fixed Fourier features were better conditioned for the limited-budget test-time optimization.
HashGrid had higher adaptive capacity but was harder to optimize.
Attention did not automatically improve reconstruction quality; it introduced useful scale-selection diagnostics but also extra optimization complexity.
```

### If HashGrid attention beats HashGrid MLP but not Fourier

Use:

```text
Attention improved learned multiscale HashGrid fusion, but the fixed Fourier baseline remained stronger overall.
This suggests the main bottleneck is not only feature fusion, but optimization stability and representation conditioning.
```

### If HashGrid attention beats all methods

Use:

```text
Coordinate-conditioned scale selection over HashGrid levels improved the representation-side inductive bias while preserving the SuperF observation model.
```

### If attention is uniform

Use:

```text
The attention module did not learn meaningful scale selection under this optimization budget.
```

### If fine levels dominate immediately

Use:

```text
The attention module likely amplified high-frequency bias or overfitting. A progressive schedule or regularization may be needed.
```

---

## Stop rules

### Today / tonight

Must have:

```text
A. Fourier baseline result
B. HashGrid baseline result
C. HashGrid attention result
one qualitative comparison figure
```

### Tuesday midday

If no attention method beats Fourier, stop tuning attention. Start writing the report around the negative/mixed result.

### Tuesday evening

Only rerun:

```text
best Fourier
best attention method
optional one extra seed
```

No new architecture ideas after Tuesday evening.

### Wednesday

No new experiments unless a run is broken. Focus on:

```text
final table
final figures
discussion
slides
rehearsal
```

---

## Acceptance criteria

The agent implementation is acceptable when:

```text
1. Existing Fourier + MLP and HashGrid + MLP runs still work.
2. --model hash_attn runs end-to-end with --input_projection hashgrid.
3. The SuperF LR degradation/observation model is unchanged.
4. Results are saved with PSNR, SSIM, LPIPS, runtime, and final loss.
5. Attention diagnostics are saved for attention models if --log_attention is enabled.
6. At least one comparison figure and one training curve are produced.
```

---

## Do not spend time on

```text
full Transformer over pixels
cross-image attention
learned downsampling
PSF estimation
new datasets
large hyperparameter sweeps
making HashGrid win at all costs
```

The final report can be strong even if Fourier wins.
