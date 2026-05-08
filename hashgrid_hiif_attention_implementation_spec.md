# HashGrid-SuperF + HIIF-Inspired Attention Decoder

## Purpose

This spec describes how to add a **HIIF-inspired attention decoder** to the current HashGrid-SuperF implementation **without changing the SuperF observation model**.

The current baseline should remain:

```text
HR coordinate grid
-> per-frame affine warp
-> HashGrid encoding
-> decoder
-> pred_hr
-> avg_pool2d(pred_hr, kernel=df, stride=df)
-> strict valid LR mask
-> LR reconstruction loss
```

No PSF, no stochastic footprint integration, no learned downsampler, and no attention inside the downsampling operator are introduced in this step.

The only intended change is:

```text
HashGrid features -> attention-based decoder -> RGB
```

instead of:

```text
HashGrid features -> plain MLP -> RGB
```

The goal is to improve the **representation-side inductive bias** of the HashGrid field by borrowing HIIF's hierarchical/attention idea: use the multiresolution HashGrid levels as tokens and learn how to combine coarse, middle, and fine levels for each queried coordinate.

---

## Current Baseline to Preserve

For a frame `i`, the current forward model is:

```text
q_base_grid: [H_hr, W_hr, 2], coordinates in [0,1]^2
    -> affine warp T_i
    -> q_warped
    -> HashGrid(q_warped)
    -> decoder
    -> pred_hr_i
    -> avg_pool2d(kernel=df, stride=df)
    -> pred_lr_i
    -> compare to LR target where strict footprint mask is valid
```

Mathematically:

\[
\hat{x}^{HR}_i(q) = f_\theta(T_i q)
\]

\[
\hat{y}^{LR}_i = D_{\text{box}, df}(\hat{x}^{HR}_i)
\]

where `D_box,df` is the existing fixed average-pooling downsampler.

The new attention decoder should not change this image-formation model.

---

## Coordinate Convention

Use the current simplified overlap-only convention:

```text
HashGrid domain [0,1]^2 == base frame.
Affine-warped samples outside [0,1]^2 are ignored in the LR loss.
```

Important implementation details:

1. The strict valid mask must be computed from **unclamped** warped coordinates.
2. Coordinates may be clamped only as a numerical guard before tiny-cuda-nn encoding.
3. The base frame should remain fixed to identity.
4. The affine warp should preferably be center-based:

\[
q' = c + A_i(q - c) + t_i,
\quad c=(0.5,0.5)
\]

rather than a raw affine around the origin.

---

## Recommended Architecture

### High-level design

Treat the HashGrid's multiresolution levels as a sequence of tokens:

```text
HashGrid(q) -> [e_0(q), e_1(q), ..., e_{L-1}(q)]
```

where:

```text
L = n_levels
F = n_features_per_level
HashGrid output dimension = L * F
```

Instead of concatenating all levels and passing them directly to an MLP, project each level into a token, then use a coordinate-conditioned attention/gating mechanism:

```text
hash_features: [N, L*F]
    -> reshape [N, L, F]
    -> per-level token projection [N, L, D]
    -> add level embeddings
    -> coordinate query from q: [N, D]
    -> attention/gating over levels
    -> fused token [N, D]
    -> MLP([fused_token, q])
    -> RGB or residual RGB
```

This is the most practical HIIF-inspired adaptation because the HashGrid already provides a natural hierarchy of spatial scales.

---

## Recommended First Implementation: Lightweight Level Attention

Use a lightweight coordinate-conditioned level attention module instead of full `nn.MultiheadAttention` first.

Why:

- It is cheaper than full attention.
- It is easy to debug.
- It gives explicit attention weights over HashGrid levels.
- The number of tokens is small, usually 10-12 levels.
- It is less likely to dominate runtime than a Transformer-style block.

### Module: `HashLevelAttentionDecoderLite`

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HashLevelAttentionDecoderLite(nn.Module):
    """
    HIIF-inspired level-attention decoder for HashGrid features.

    Input:
        q:             [N, 2]
        hash_features: [N, n_levels * n_features_per_level]

    Output:
        rgb_or_residual: [N, out_dim]
        aux: dictionary with attention weights and diagnostics
    """

    def __init__(
        self,
        n_levels: int,
        n_features_per_level: int,
        token_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 3,
        use_coord_concat: bool = True,
        use_lowfreq_coord_pe: bool = False,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.token_dim = token_dim
        self.use_coord_concat = use_coord_concat
        self.use_lowfreq_coord_pe = use_lowfreq_coord_pe

        self.level_proj = nn.Linear(n_features_per_level, token_dim)
        self.level_embed = nn.Parameter(torch.randn(n_levels, token_dim) * 0.02)

        coord_dim = 2
        if use_lowfreq_coord_pe:
            # [x, y, sin(pi*x), cos(pi*x), sin(pi*y), cos(pi*y)]
            coord_dim = 6

        self.coord_query = nn.Sequential(
            nn.Linear(coord_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )

        mlp_in_dim = token_dim
        if use_coord_concat:
            mlp_in_dim += coord_dim

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def coord_features(self, q: torch.Tensor) -> torch.Tensor:
        if not self.use_lowfreq_coord_pe:
            return q

        x = q[..., 0:1]
        y = q[..., 1:2]
        return torch.cat(
            [
                x,
                y,
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(math.pi * y),
                torch.cos(math.pi * y),
            ],
            dim=-1,
        )

    def forward(self, q: torch.Tensor, hash_features: torch.Tensor):
        N = q.shape[0]
        assert hash_features.shape[-1] == self.n_levels * self.n_features_per_level

        h = hash_features.view(N, self.n_levels, self.n_features_per_level)
        tokens = self.level_proj(h) + self.level_embed[None, :, :]
        # tokens: [N, L, D]

        q_feat = self.coord_features(q)
        query = self.coord_query(q_feat)
        # query: [N, D]

        # Dot-product attention over levels.
        scores = (tokens * query[:, None, :]).sum(dim=-1) / math.sqrt(self.token_dim)
        attn = torch.softmax(scores, dim=-1)
        # attn: [N, L]

        fused = (attn[..., None] * tokens).sum(dim=1)
        # fused: [N, D]

        if self.use_coord_concat:
            decoder_in = torch.cat([fused, q_feat], dim=-1)
        else:
            decoder_in = fused

        out = self.mlp(decoder_in)

        aux = {
            "level_attention": attn,
            "level_attention_mean": attn.mean(dim=0).detach(),
            "level_attention_entropy": (-(attn * (attn + 1e-8).log()).sum(dim=-1)).mean().detach(),
        }
        return out, aux
```

---

## Optional Full Multi-Head Cross Attention Decoder

After the lightweight version is working, a more literal HIIF-style cross-attention decoder can be tested.

This is more expensive and less CUDA-fused, but it is useful as an ablation.

```python
class HashLevelCrossAttentionDecoder(nn.Module):
    def __init__(
        self,
        n_levels: int,
        n_features_per_level: int,
        token_dim: int = 32,
        n_heads: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 3,
        use_coord_concat: bool = True,
    ):
        super().__init__()

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.use_coord_concat = use_coord_concat

        self.level_proj = nn.Linear(n_features_per_level, token_dim)
        self.level_embed = nn.Parameter(torch.randn(n_levels, token_dim) * 0.02)

        self.coord_query = nn.Sequential(
            nn.Linear(2, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        mlp_in = token_dim + (2 if use_coord_concat else 0)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, q: torch.Tensor, hash_features: torch.Tensor):
        N = q.shape[0]
        h = hash_features.view(N, self.n_levels, self.n_features_per_level)

        tokens = self.level_proj(h) + self.level_embed[None, :, :]
        query = self.coord_query(q).unsqueeze(1)

        z, attn_weights = self.attn(
            query=query,
            key=tokens,
            value=tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        z = z.squeeze(1)
        if self.use_coord_concat:
            z = torch.cat([z, q], dim=-1)

        out = self.mlp(z)

        aux = {
            "level_attention": attn_weights.squeeze(1),
            "level_attention_mean": attn_weights.squeeze(1).mean(dim=0).detach(),
        }
        return out, aux
```

---

## Strongly Recommended: Residual-over-Bilinear Skip

HIIF uses a skip connection from a bilinearly upsampled image to the implicit decoder output. For this HashGrid-SuperF setting, this is highly recommended.

Instead of predicting the whole HR image:

\[
f_\theta(q) = D_\theta(q)
\]

predict a residual over a stable baseline:

\[
f_\theta(q) = B(q) + \alpha r_\theta(q)
\]

where:

- `B(q)` is a bilinear or bicubic upsampled base LR frame, or a robust fused baseline if available.
- `r_theta(q)` is the HashGrid-attention decoder output.
- `alpha` is a small residual scale.

Recommended first values:

```text
residual_scale = 0.05 or 0.1
```

This is likely to reduce HashGrid artifacts because the model is no longer required to learn the whole image from LR averages.

### Baseline sampling helper

```python
def sample_bilinear_base(base_lr: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Args:
        base_lr: [C, H_lr, W_lr]
        q:       [N, 2], normalized coordinates in [0,1]^2, q[...,0]=x, q[...,1]=y

    Returns:
        sampled: [N, C]
    """
    C, H, W = base_lr.shape

    grid = q.reshape(1, -1, 1, 2)
    grid = grid * 2.0 - 1.0

    sampled = F.grid_sample(
        base_lr.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    sampled = sampled.reshape(C, -1).T.contiguous()
    return sampled
```

### Field wrapper

```python
class HashAttentionField(nn.Module):
    def __init__(
        self,
        hashgrid,
        decoder,
        residual_mode: bool = False,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.hashgrid = hashgrid
        self.decoder = decoder
        self.residual_mode = residual_mode
        self.residual_scale = residual_scale

    def forward(self, q, base_lr=None):
        """
        q: [N,2]
        base_lr: optional [C,H_lr,W_lr]
        """
        h = self.hashgrid(q)
        residual_or_rgb, aux = self.decoder(q, h)

        if not self.residual_mode:
            return residual_or_rgb, aux

        if base_lr is None:
            raise ValueError("base_lr must be provided when residual_mode=True")

        base = sample_bilinear_base(base_lr, q)
        out = base + self.residual_scale * residual_or_rgb
        return out, aux
```

If integrating inside the existing `INRBase`, the same idea can be implemented without a separate wrapper: after `input_projection`, call the new decoder with both `hash_features` and the raw warped coordinates, and optionally add the sampled base LR skip.

---

## Integration Points in Current Project

Based on the current project layout, these are the likely modifications.

### `input_projections/hashgrid_tcnn.py`

No major changes required.

Make sure `HashGridTcnn` exposes or stores:

```python
self.n_levels
self.n_features_per_level
self.output_dim = n_levels * n_features_per_level
```

The output should remain:

```python
hash_features: [..., n_levels * n_features_per_level]
```

Do not reshape into levels inside the projection module. Leave that to the decoder.

---

### `models/inr.py`

The current path probably looks approximately like:

```python
coords = warped_coords
coords = self.input_projection(coords, progress=progress)
output = self.decoder(coords_flat)
```

For attention decoder support, preserve the raw warped coordinates:

```python
warped_coords = self.apply_affine(coords, A)
raw_q = warped_coords
hash_features = self.input_projection(raw_q, progress=progress)
```

Then flatten both consistently:

```python
orig_shape = hash_features.shape[:-1]
hash_flat = hash_features.reshape(-1, hash_features.shape[-1])
q_flat = raw_q.reshape(-1, 2)
```

Call decoder depending on type:

```python
if getattr(self.decoder, "requires_coords", False):
    out_flat, aux = self.decoder(q_flat, hash_flat)
else:
    out_flat = self.decoder(hash_flat)
    aux = {}
```

Then reshape:

```python
output = out_flat.reshape(*orig_shape, channels)
```

If using residual-over-bilinear, the decoder/field needs access to a base LR image. There are two implementation options:

1. Pass `base_lr` into `forward` and sample it inside the decoder/field.
2. Precompute a bilinear HR baseline and add it after rendering `pred_hr`.

Option 2 is easier for the current fixed-HR-grid path:

```python
pred_hr = residual_scale * residual_hr + baseline_hr
```

where `baseline_hr` is the base LR upsampled to `[H_hr, W_hr]` using bilinear/bicubic.

---

### New decoder file suggestion

Create one of:

```text
models/hash_attention_decoder.py
models/decoders/hash_attention.py
```

and register it in the model factory / CLI selection as something like:

```text
--model hash_attn
```

Suggested CLI flags:

```text
--hash_attn_token_dim 32
--hash_attn_hidden_dim 64
--hash_attn_type lite           # lite | mha
--hash_attn_heads 4             # only for mha
--hash_attn_use_coord_concat true
--hash_attn_use_lowfreq_coord_pe false
--hash_attn_residual_mode false # turn on later
--hash_attn_residual_scale 0.1
--hash_attn_log_attention true
```

---

## Forward Pass With Attention Decoder

This is the target forward pass for the no-PSF baseline:

```text
input_canonical: [B, H_hr, W_hr, 2]
    -> get affine A_i
    -> warped_coords: [B, H_hr, W_hr, 2]
    -> strict validity computed from warped_coords
    -> hash_features = HashGridTcnn(warped_coords)
    -> flatten:
        q_flat:    [B*H_hr*W_hr, 2]
        hash_flat: [B*H_hr*W_hr, L*F]
    -> attention decoder:
        rgb_flat, aux = decoder(q_flat, hash_flat)
    -> pred_hr: [B, H_hr, W_hr, C]
    -> pred_hr_nchw = pred_hr.permute(0,3,1,2)
    -> pred_lr = avg_pool2d(pred_hr_nchw, kernel=df, stride=df)
    -> compare to LR target using strict valid LR mask
```

The downsampling remains exactly:

```python
pred_lr = F.avg_pool2d(pred_hr_nchw, kernel_size=df, stride=df)
```

---

## Loss Function Remains SuperF-Style

Do not add attention to the LR pooling operator.

Current strict mask remains:

```python
x = warped_grid[..., 0]
y = warped_grid[..., 1]
valid_hr = ((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)).float().unsqueeze(1)
coverage_lr = F.avg_pool2d(valid_hr, kernel_size=df, stride=df)
valid_lr = coverage_lr >= (1.0 - 1e-6)
```

Recommended masked loss implementation:

```python
err = (pred_lr - lr_target) ** 2
valid = valid_lr.expand_as(err).float()
recon_loss = (err * valid).sum() / valid.sum().clamp_min(1.0)
```

Do not compute the valid mask from clamped coordinates.

---

## Optional Attention Regularization

If the decoder immediately puts all attention on the finest levels, the artifacts may remain.

Add an optional weak regularizer during early training that discourages finest-level dominance.

### Simple fine-level penalty

```python
def fine_attention_penalty(attn: torch.Tensor, fine_start: int) -> torch.Tensor:
    """
    attn: [N, L]
    fine_start: first level index considered fine.
    """
    return attn[:, fine_start:].mean()
```

Example schedule:

```text
iterations 0-300:
    lambda_fine_attn = 1e-3
iterations 300+:
    lambda_fine_attn = 0
```

Loss:

```python
loss = recon_loss + lambda_fine_attn * fine_attention_penalty(attn, fine_start=8)
```

For `n_levels=12`, a reasonable `fine_start` is 8 or 9.

For `n_levels=10`, a reasonable `fine_start` is 7 or 8.

Use this only if logs show the finest levels dominate too early.

---

## Optional HR Priors Still Make Sense

Even though this spec does not introduce PSF or stochastic footprint integration, the HR output may still be underconstrained by LR average pooling.

These priors are optional but useful:

### TV loss

```python
def tv_loss(pred_hr_nchw):
    dx = (pred_hr_nchw[..., :, 1:] - pred_hr_nchw[..., :, :-1]).abs().mean()
    dy = (pred_hr_nchw[..., 1:, :] - pred_hr_nchw[..., :-1, :]).abs().mean()
    return dx + dy
```

Suggested weight:

```text
lambda_tv = 1e-5 to 1e-4
```

### Intra-cell variance warmup

```python
def intra_cell_variance_loss(pred_hr_nchw, df):
    B, C, H_hr, W_hr = pred_hr_nchw.shape
    H_lr = H_hr // df
    W_lr = W_hr // df

    blocks = pred_hr_nchw.reshape(B, C, H_lr, df, W_lr, df)
    block_mean = blocks.mean(dim=(3, 5), keepdim=True)
    return ((blocks - block_mean) ** 2).mean()
```

Suggested schedule:

```text
lambda_cell = 1e-2 for first 300-500 iterations
lambda_cell = 1e-3 afterward
```

This is not part of HIIF, but it directly targets the known failure mode where bad HR structure averages away under `avg_pool2d`.

---

## Recommended Hyperparameters for Current 64x64 -> 256x256 Case

Assuming:

```text
LR: 64 x 64
scale / df: 4
HR: 256 x 256
HashGrid domain: [0,1]^2, no halo
```

Recommended HashGrid:

```text
hash_grid_type: Hash
hash_encoding_preset: smoothstep_grid
hash_n_levels: 10 or 12
hash_n_features_per_level: 2
hash_base_resolution: 8
hash_max_resolution: 256
hash_log2_hashmap_size: 16
hash_encoding_dtype: fp32
interpolation: Smoothstep
```

Recommended attention decoder:

```text
attention type: lite
attention token_dim: 32
attention hidden_dim: 64
coord concat: true
lowfreq coord PE: false initially
residual mode: false initially, then true as ablation
residual scale: 0.05 or 0.1 when enabled
```

Recommended optimizer:

```text
learning_rate: 1e-3 to 2e-3 for first tests
weight_decay: 0.0 initially
```

If there are separate parameter groups:

```text
HashGrid + decoder LR: 1e-3 to 2e-3
Affine LR: 3e-4 to 1e-3
```

---

## Suggested Experiment Ladder

Run these in order. Keep dataset, seed, iterations, and device fixed.

### Experiment A: Existing baseline

```text
HashGrid + plain MLP
fixed avg_pool2d
```

Purpose: reference point.

---

### Experiment B: Add raw coordinate concatenation only

```text
MLP([hash_features, x, y])
```

Purpose: determine whether global coordinate awareness already helps.

---

### Experiment C: Residual-over-bilinear with plain MLP

```text
pred_hr = bilinear_base_hr + 0.1 * HashGridMLP(q)
```

Purpose: test the HIIF-style skip connection before adding attention.

---

### Experiment D: Hash level-attention decoder

```text
HashGrid(q)
-> reshape by level
-> lite attention over levels
-> MLP([fused_token, q])
-> RGB
```

Purpose: test the core HIIF-inspired decoder.

---

### Experiment E: Hash level-attention + residual-over-bilinear

```text
pred_hr = bilinear_base_hr + 0.1 * HashAttentionResidual(q)
```

Purpose: likely best no-PSF variant.

---

### Experiment F: Attention regularization

Only if attention collapses to fine levels:

```text
fine-level attention penalty for first 300 iterations
```

Purpose: reduce early high-frequency artifacts.

---

## What Not to Implement in This Step

Do not change the observation model in this attention-only ablation.

Avoid:

```text
- PSF blur
- stochastic footprint integration
- learned downsampler
- attention-weighted downsampling over the 4x4 block
- unconstrained attention pooling for LR prediction
- frame-ID conditioning inside the canonical field
```

Especially avoid:

```text
pred_hr 4x4 block -> attention pooling -> LR
```

That changes the SuperF downsampling model and may simply hide HR artifacts.

The correct attention insertion point is:

```text
HashGrid features -> attention decoder -> pred_hr
```

not:

```text
pred_hr -> attention downsampler
```

---

## Logging Requirements

Add logs for:

```text
LR reconstruction loss / PSNR
valid fraction per frame
translation magnitude in LR pixels
affine deviation from identity
mean attention per level
attention entropy
attention to finest 2-3 levels
intra-cell variance of pred_hr
TV of pred_hr
residual magnitude if residual mode is enabled
```

For attention logs:

```python
attn_mean = aux["level_attention_mean"]
```

Save/print something like:

```text
level_attn: [0.08, 0.10, 0.12, ..., 0.03]
fine_attn_sum: attn_mean[-2:].sum()
attn_entropy: aux["level_attention_entropy"]
```

Interpretation:

```text
If finest levels dominate from the start:
    expect possible HR artifacts.
    try residual mode, fine-level attention penalty, or lower max_resolution.

If attention is almost uniform:
    decoder may behave similarly to concat+MLP.

If coarse/mid levels dominate early and fine levels increase later:
    this is the desired behavior.
```

---

## Evaluation and Visualization

Evaluation should remain the same as the current baseline:

```text
render full pred_hr at 256x256 HR pixel centers
avg_pool2d(pred_hr, df)
compare to LR target
visualize pred_hr, avg_pool(pred_hr), LR target nearest-up
```

For diagnosing whether attention helps, always inspect:

```text
1. pred_hr
2. avg_pool(pred_hr) nearest-up
3. LR target nearest-up
4. attention per level
```

Desired result:

```text
pred_hr has fewer lattice/grid artifacts
avg_pool(pred_hr) still matches LR target
attention does not collapse to only the finest levels
```

---

## Expected Failure Modes

### Failure mode 1: LR reprojection good, HR still bad

Interpretation:

```text
Attention helped little; the LR average-pooling null space remains dominant.
```

Next steps:

```text
- enable residual-over-bilinear
- add TV or intra-cell variance warmup
- add attention prior against finest levels
- later consider stochastic footprint integration / PSF in a separate ablation
```

---

### Failure mode 2: Attention collapses to fine levels

Interpretation:

```text
Decoder is using high-frequency HashGrid capacity too aggressively.
```

Next steps:

```text
- add fine-level attention penalty for first 300 iterations
- lower hash_max_resolution to 128 as a sanity check
- enable residual-over-bilinear
```

---

### Failure mode 3: Runtime too slow

Interpretation:

```text
PyTorch attention decoder is now the bottleneck.
```

Next steps:

```text
- use the lite attention decoder rather than nn.MultiheadAttention
- reduce token_dim from 32 to 16
- use n_levels=10 instead of 12
- avoid local-offset attention
```

---

### Failure mode 4: Output is overly smooth

Interpretation:

```text
Decoder is underusing fine levels or residual scale is too low.
```

Next steps:

```text
- increase residual_scale from 0.05 to 0.1 or 0.25
- remove fine-level attention penalty
- increase token_dim to 64
- try n_features_per_level=4
```

---

## Optional Later Extension: Local-Offset Attention

Do not implement this first.

A more HIIF-like local context variant is to query HashGrid at nearby offsets:

```text
q
q + [1/W_hr, 0]
q - [1/W_hr, 0]
q + [0, 1/H_hr]
q - [0, 1/H_hr]
```

Then attend over these local-offset tokens.

This may reduce speckle by giving the decoder local context, but it multiplies HashGrid queries by 5 or 9 and should be considered a later ablation.

---

## Summary Implementation Target

Implement this first:

```text
HashGrid-SuperF baseline preserved:
    HR grid -> affine -> HashGrid -> decoder -> avg_pool2d -> LR loss

New decoder:
    HashGrid output reshaped into [levels, features_per_level]
    coordinate-conditioned attention over levels
    MLP receives fused token + raw coordinate
    output RGB or residual RGB

Recommended first model:
    HashLevelAttentionDecoderLite
    token_dim=32
    hidden_dim=64
    coord concat=true
    n_levels=10 or 12
    n_features_per_level=2

Most promising ablation:
    residual-over-bilinear + level attention
```

The goal is to test whether a HIIF-inspired hierarchical attention decoder gives the HashGrid field a better inductive bias while keeping the SuperF loss and downsampling path unchanged.
