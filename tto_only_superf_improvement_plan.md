# TTO-Only SuperF Improvement Plan

## Purpose

This document describes a complete experiment and implementation plan for improving **SuperF-style test-time optimization (TTO)** for Sentinel-2 multi-image super-resolution **without adding a pretrained backbone, external HR training data, or supervised feature encoder**.

The goal is to improve SuperF by making the inverse problem:

1. better represented,
2. better conditioned,
3. better optimized,
4. more sensor-aware,
5. more robust to real Sentinel-2 inconsistencies.

The main research question is:

> Can SuperF be improved purely through test-time changes to the coordinate representation, optimization curriculum, and observation model?

---

## 1. Scope and non-goals

### 1.1 In scope

The following are allowed:

- Test-time optimized neural fields.
- HashGrid coordinate encoders initialized from scratch.
- Progressive/coarse-to-fine optimization schedules.
- Test-time learned alignment parameters.
- Test-time learned radiometric correction.
- Test-time learned or constrained degradation model.
- Robust losses and uncertainty/reliability weighting.
- Hash-level scale gates initialized from scratch.
- Held-out LR consistency evaluation.
- Synthetic Sentinel-2-like burst evaluation.
- Real Sentinel-2 qualitative and LR-consistency evaluation.

### 1.2 Out of scope

The following should not be used for the main method:

- Pretrained CNN or Transformer backbones.
- Supervised HR/LR training pairs.
- Dataset-level training before test time.
- External learned priors.
- Foundation models.
- Learned image encoders trained outside the current scene.
- Full image Transformers unless implemented from scratch and optimized only per scene, which is not recommended for this project.

The method should remain:

> scene-specific, test-time optimized, and training-free with respect to external HR data.

---

## 2. Core idea

SuperF already shows that multi-image super-resolution can be formulated as a test-time optimization problem: a shared implicit neural representation is optimized from several low-resolution observations, together with frame-specific alignment parameters.

This project improves the TTO formulation itself rather than adding a learned prior.

The main improvement categories are:

| Category | Purpose |
|---|---|
| HashGrid representation | Improve coordinate-field capacity and convergence. |
| Progressive coarse-to-fine TTO | Improve alignment and reduce early high-frequency overfitting. |
| Sensor-aware observation model | Better match Sentinel-2 image formation. |
| Radiometric correction | Handle per-frame brightness/atmospheric differences. |
| Robust losses/reliability | Downweight clouds, noise, temporal changes, and outliers. |
| Hash-level scale gates | Optional explicit multiscale selection. |
| Held-out LR consistency | Evaluate real scenes without HR ground truth. |

---

## 3. Baseline: SuperF-style TTO formulation

Given \(N\) low-resolution observations:

\[
\{y_i\}_{i=1}^{N},
\]

where:

\[
y_i \in \mathbb{R}^{H \times W \times C},
\]

optimize a continuous high-resolution field:

\[
f_\theta : [0,1]^2 \rightarrow \mathbb{R}^{C}.
\]

Each frame has a trainable alignment transform:

\[
A_i,
\]

typically parameterized by translation and small rotation.

The predicted LR observation is:

\[
\hat{y}_i =
D_s(f_\theta(A_i x)),
\]

where:

- \(D_s\) is the downsampling/degradation operator,
- \(s\) is the scale factor,
- \(x\) is a coordinate grid.

The basic optimization objective is:

\[
\min_{\theta,\{A_i\}}
\frac{1}{N}
\sum_{i=1}^{N}
\mathcal{L}(\hat{y}_i, y_i).
\]

The baseline loss is usually MSE:

\[
\mathcal{L}_{\text{MSE}}
=
\lVert \hat{y}_i - y_i \rVert_2^2.
\]

---

## 4. Proposed method overview

The proposed TTO-only improvement stack is:

\[
\text{SuperF}
\rightarrow
\text{HashGrid-SuperF}
\rightarrow
\text{Progressive HashGrid-SuperF}
\rightarrow
\text{Sensor/Radiometric/Robust HashGrid-SuperF}.
\]

The optional attention/gating extension is:

\[
\text{Progressive HashGrid-SuperF}
\rightarrow
\text{Progressive HashGrid-SuperF + Scale Gates}.
\]

The recommended final method name for experiments is:

> **Progressive Robust HashGrid-SuperF**

Optional gated version:

> **Scale-Gated Progressive Robust HashGrid-SuperF**

---

# Part A — Core implementation modules

---

## 5. Module 1: HashGrid coordinate representation

### 5.1 Motivation

Fourier-feature coordinate MLPs can be sensitive to frequency-band selection and may optimize slowly. A multiresolution HashGrid encoder gives the neural field trainable features at multiple spatial resolutions and can improve high-frequency fitting and convergence.

### 5.2 Formulation

Replace the baseline coordinate field:

\[
f_\theta(x)
\]

with:

\[
f_{\theta,\phi}(x)
=
d_\theta(E_\phi(x)),
\]

where:

- \(E_\phi(x)\) is a multiresolution HashGrid encoder,
- \(d_\theta\) is a compact MLP decoder,
- both \(\phi\) and \(\theta\) are optimized from scratch at test time.

For \(L\) HashGrid levels and \(F\) features per level:

\[
E_\phi(x)
=
[h_1(x), h_2(x), \ldots, h_L(x)],
\]

where:

\[
h_\ell(x) \in \mathbb{R}^{F}.
\]

The default HashGrid baseline concatenates all levels:

\[
z(x)
=
[h_1(x), h_2(x), \ldots, h_L(x)].
\]

Then:

\[
f_{\theta,\phi}(x)=d_\theta(z(x)).
\]

### 5.3 Recommended config

```yaml
hashgrid:
  n_levels: 12
  features_per_level: 2
  base_resolution: 8
  max_resolution: 256
  log2_hashmap_size: 16
  interpolation: linear

decoder:
  type: mlp
  input_dim: 24
  hidden_dim: 64
  n_layers: 3
  activation: relu
  output_dim: 3
```

If using `features_per_level: 4`:

```yaml
decoder:
  input_dim: 48
```

### 5.4 Implementation requirements

The HashGrid implementation must support:

```text
coords: [B, 2]
features_flat: [B, L * F]
features_by_level: [B, L, F]
```

If the encoder returns only the flattened representation:

```python
features_by_level = features_flat.view(B, n_levels, features_per_level)
```

Verify that the feature ordering is level-major before relying on this reshape.

---

## 6. Module 2: Progressive coarse-to-fine TTO

### 6.1 Motivation

Jointly optimizing the field and frame alignment is difficult. If high-frequency representation capacity is available too early, the field may explain away misalignment instead of correcting it.

A coarse-to-fine schedule improves conditioning by first fitting low-frequency structure and alignment, then gradually introducing finer spatial detail.

### 6.2 Progressive HashGrid levels

Let \(L\) be the number of HashGrid levels. Define a time-dependent mask:

\[
m_\ell(t) \in [0,1],
\]

where \(t\) is the optimization iteration.

The encoded feature becomes:

\[
z_t(x)
=
[m_1(t)h_1(x), m_2(t)h_2(x), \ldots, m_L(t)h_L(x)].
\]

At the start:

\[
m_\ell(t) = 0
\quad
\text{for fine levels}.
\]

Later:

\[
m_\ell(t) \rightarrow 1.
\]

### 6.3 Simple hard schedule

For 2000 iterations and 12 levels:

| Iteration range | Active levels |
|---:|---|
| 0–300 | 1–4 |
| 300–700 | 1–6 |
| 700–1200 | 1–9 |
| 1200–2000 | 1–12 |

Implementation:

```python
def get_active_level_mask(iteration: int, n_levels: int):
    if iteration < 300:
        active = 4
    elif iteration < 700:
        active = 6
    elif iteration < 1200:
        active = 9
    else:
        active = n_levels

    mask = torch.zeros(n_levels)
    mask[:active] = 1.0
    return mask
```

### 6.4 Smooth schedule

A smoother alternative is to use a sigmoid ramp per level:

\[
m_\ell(t)
=
\sigma
\left(
\frac{t - t_\ell}{\tau}
\right).
\]

Where:

- \(t_\ell\) is the activation iteration for level \(\ell\),
- \(\tau\) controls smoothness.

Recommended:

```yaml
progressive_levels:
  enabled: true
  mode: smooth
  start_iteration: 0
  end_iteration: 1200
  min_active_levels: 4
  tau: 100
```

### 6.5 Progressive TTO schedule

Recommended high-level training schedule:

| Phase | Iterations | Trainable components | Purpose |
|---|---:|---|---|
| Phase 1 | 0–300 | coarse levels + decoder + alignment | stabilize alignment |
| Phase 2 | 300–700 | coarse/mid levels + decoder + alignment | refine structure |
| Phase 3 | 700–1200 | most levels + decoder + alignment | add detail |
| Phase 4 | 1200–2000 | all levels + decoder + alignment, lower LR | final refinement |

### 6.6 Metrics to monitor

Log:

```text
progressive/active_levels
progressive/mean_level_mask
alignment/error
eval/psnr
eval/lpips
train/loss
```

---

## 7. Module 3: Per-frame radiometric correction

### 7.1 Motivation

Real Sentinel-2 time series are not perfectly radiometrically consistent. Frames may differ due to:

- illumination,
- atmosphere,
- haze,
- BRDF effects,
- seasonal differences,
- preprocessing variation.

If these differences are ignored, the neural field may waste capacity fitting frame-specific brightness changes.

### 7.2 Formulation

Use per-frame, per-band gain and offset:

\[
\hat{y}_{i,b}
=
a_{i,b}
D_s(f_{\theta,b}(A_i x))
+
c_{i,b}.
\]

Where:

- \(a_{i,b}\) is a multiplicative gain,
- \(c_{i,b}\) is an additive offset,
- \(i\) indexes frame,
- \(b\) indexes spectral band.

Initialize:

\[
a_{i,b}=1,
\quad
c_{i,b}=0.
\]

Regularize:

\[
\mathcal{L}_{\text{radio}}
=
\lambda_a
\sum_{i,b}(a_{i,b}-1)^2
+
\lambda_c
\sum_{i,b}c_{i,b}^2.
\]

Total loss:

\[
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\mathcal{L}_{\text{radio}}.
\]

### 7.3 Recommended config

```yaml
radiometric:
  enabled: true
  model: gain_offset
  per_frame: true
  per_band: true
  init_gain: 1.0
  init_offset: 0.0
  lambda_gain: 1.0e-3
  lambda_offset: 1.0e-3
  lr_gain: 1.0e-3
  lr_offset: 1.0e-3
```

### 7.4 Implementation notes

Use constrained gain to avoid negative values:

```python
gain = 1.0 + 0.1 * torch.tanh(raw_gain)
offset = 0.05 * torch.tanh(raw_offset)
```

This keeps corrections modest.

Alternative:

```python
gain = torch.exp(raw_log_gain)
```

but this can be less bounded unless regularized strongly.

### 7.5 What to log

```text
radiometric/gain_mean
radiometric/gain_std
radiometric/gain_min
radiometric/gain_max
radiometric/offset_mean
radiometric/offset_std
radiometric/offset_min
radiometric/offset_max
radiometric/loss
```

### 7.6 Main ablation

| Method | Radiometric model |
|---|---|
| HashGrid-SuperF | none |
| HashGrid-SuperF + gain | per-frame gain |
| HashGrid-SuperF + gain/offset | per-frame gain and offset |
| HashGrid-SuperF + gain/offset + robust loss | full robust version |

---

## 8. Module 4: Sensor-aware degradation model

### 8.1 Motivation

The low-resolution observation is not just an average-pooled version of a high-resolution image. Sentinel-2 observations are affected by sensor blur, band-specific point-spread functions, modulation transfer function, interpolation, and resampling.

A TTO-only improvement is to optimize a small constrained degradation model jointly with the field.

### 8.2 Fixed degradation baseline

Baseline:

\[
\hat{y}_i =
D_s(f_\theta(A_i x)).
\]

Where \(D_s\) is fixed average pooling or fixed blur + downsampling.

### 8.3 Learned Gaussian PSF

Replace fixed downsampling with:

\[
\hat{y}_i =
D_{s,k_b}(f_\theta(A_i x)),
\]

where \(k_b\) is a per-band Gaussian blur kernel.

Use:

\[
k_b = \text{Gaussian}(\sigma_b).
\]

Optimize \(\sigma_b\) at test time.

### 8.4 Constrained parameterization

```python
sigma = sigma_min + (sigma_max - sigma_min) * torch.sigmoid(raw_sigma)
```

Recommended:

```yaml
psf:
  enabled: true
  type: gaussian
  per_band: true
  sigma_min: 0.2
  sigma_max: 2.5
  init_sigma: 1.0
  kernel_size: 7
  lambda_sigma: 1.0e-3
  lr_sigma: 1.0e-3
```

Regularize toward the initial value:

\[
\mathcal{L}_{\text{psf}}
=
\lambda_\sigma
\sum_b
(\sigma_b-\sigma_{0})^2.
\]

### 8.5 Optional anisotropic PSF

Higher-risk extension:

\[
k_b = \text{Gaussian}(\sigma_{x,b},\sigma_{y,b},\rho_b).
\]

Not recommended for the first report unless the isotropic version works.

### 8.6 Main ablation

| Method | Degradation model |
|---|---|
| SuperF baseline | fixed average/downsample |
| HashGrid-SuperF | fixed average/downsample |
| HashGrid + fixed Gaussian blur | fixed PSF |
| HashGrid + learned Gaussian PSF | learned \(\sigma\) |
| HashGrid + learned per-band PSF | learned \(\sigma_b\) |

### 8.7 Metrics

Log:

```text
psf/sigma_band_0
psf/sigma_band_1
psf/sigma_band_2
psf/sigma_mean
psf/loss
```

Evaluate:

- PSNR on synthetic data,
- robustness when the synthetic blur differs from the assumed blur,
- held-out LR consistency on real Sentinel-2.

---

## 9. Module 5: Robust reconstruction loss

### 9.1 Motivation

Real Sentinel-2 time series contain outliers:

- clouds,
- haze,
- cloud shadows,
- seasonal changes,
- snow,
- crop changes,
- registration errors,
- moving objects,
- atmospheric artifacts.

MSE can overfit these inconsistencies.

### 9.2 Option A: Huber loss

\[
\mathcal{L}_{\text{Huber}}(r)
=
\begin{cases}
\frac{1}{2}r^2, & |r|\leq \delta \\
\delta(|r|-\frac{1}{2}\delta), & |r|>\delta
\end{cases}
\]

Recommended config:

```yaml
loss:
  type: huber
  delta: 0.03
```

### 9.3 Option B: Charbonnier loss

\[
\mathcal{L}_{\text{charb}}
=
\sqrt{r^2 + \epsilon^2}.
\]

Recommended:

```yaml
loss:
  type: charbonnier
  epsilon: 1.0e-3
```

### 9.4 Option C: GNLL uncertainty loss

Predict or optimize a log variance:

\[
s_i(x)
\]

and use:

\[
\mathcal{L}_{\text{GNLL}}
=
\frac{1}{2}
\exp(-s_i(x))
\lVert \hat{y}_i(x)-y_i(x)\rVert^2
+
\frac{1}{2}s_i(x).
\]

This downweights inconsistent observations while penalizing excessive uncertainty.

Recommended config:

```yaml
uncertainty:
  enabled: true
  type: gnll
  predict_logvar: true
  logvar_min: -4.0
  logvar_max: 2.0
  lambda_smooth: 1.0e-4
```

### 9.5 Option D: Frame reliability weights

Optimize one reliability scalar per frame:

\[
r_i \in [0,1].
\]

Use:

\[
\mathcal{L}
=
\sum_i r_i
\lVert \hat{y}_i-y_i\rVert^2
+
\lambda_r
\sum_i (r_i-1)^2.
\]

Constrain:

```python
reliability = torch.sigmoid(raw_reliability)
```

Initialize near 1:

```python
raw_reliability = inverse_sigmoid(0.95)
```

Recommended config:

```yaml
frame_reliability:
  enabled: false
  init: 0.95
  lambda_reliability: 1.0e-3
  lr_reliability: 1.0e-3
```

### 9.6 Main robust-loss ablation

| Method | Robustness model |
|---|---|
| HashGrid-SuperF | MSE |
| HashGrid-SuperF + Huber | robust residual |
| HashGrid-SuperF + Charbonnier | robust residual |
| HashGrid-SuperF + GNLL | learned uncertainty |
| HashGrid-SuperF + frame reliability | learned frame weights |
| HashGrid-SuperF + cloud mask | quality-mask-weighted loss |

### 9.7 Metrics

Synthetic:

- PSNR,
- SSIM,
- LPIPS,
- performance under corrupted frames,
- performance under cloud-like masks.

Real:

- held-out LR consistency,
- uncertainty/reliability maps,
- visual artifacts,
- whether cloudy frames receive lower reliability.

---

## 10. Module 6: Optional Hash-level scale gates

### 10.1 Motivation

A multiresolution HashGrid exposes features at several spatial resolutions, but standard concatenation treats all levels equally at every coordinate. Scale gates make the level mixing explicit and interpretable.

This is optional. It should not be the only contribution.

### 10.2 Softmax scale gate

Given:

\[
H(x)= [h_1(x),\ldots,h_L(x)],
\]

predict:

\[
\alpha(x)
=
\text{softmax}(g_\psi(\text{flatten}(H(x))) / \tau).
\]

Use:

\[
w_\ell(x)=L\alpha_\ell(x).
\]

Then:

\[
z(x)
=
\text{concat}(w_1h_1,\ldots,w_Lh_L).
\]

### 10.3 Sigmoid scale gate

\[
w_\ell(x)
=
2\sigma(g_\psi(H(x))_\ell).
\]

This allows independent level recalibration.

### 10.4 Coord-conditioned gates

Optional upgrade:

\[
w(x)
=
g_\psi([h_1(x),...,h_L(x),\gamma(x)]),
\]

where:

\[
\gamma(x)
=
[x,y,\sin(2\pi x),\cos(2\pi x),
\sin(2\pi y),\cos(2\pi y)].
\]

### 10.5 Gate warmup

Because gates can fight the HashGrid early, use:

```yaml
scale_gates:
  enabled: true
  gate_type: softmax
  hidden_dim: 64
  temperature: 1.0
  warmup_iterations: 300
  lr_gate: 5.0e-4
```

During warmup:

\[
w_\ell = 1.
\]

After warmup, enable learned gates.

### 10.6 Gate logging

Log:

```text
gate/entropy
gate/gate_mean_level_*
gate/mass_mean_level_*
gate/coarse_mass
gate/mid_mass
gate/fine_mass
gate/effective_contribution_level_*
gate/effective_coarse_contribution
gate/effective_mid_contribution
gate/effective_fine_contribution
```

Effective contribution:

\[
c_\ell(x)=\lVert w_\ell(x)h_\ell(x)\rVert_2.
\]

This accounts for both gate value and feature magnitude.

---

## 11. Module 7: Held-out LR consistency

### 11.1 Motivation

Real Sentinel-2 scenes often have no HR reference. Therefore, evaluate whether the optimized HR field predicts unseen LR observations.

### 11.2 Protocol

Split frames into:

\[
S_{\text{train}}
\quad \text{and} \quad
S_{\text{val}}.
\]

Optimize using only \(S_{\text{train}}\).

Evaluate:

\[
\mathcal{E}_{\text{heldout}}
=
\frac{1}{|S_{\text{val}}|}
\sum_{i\in S_{\text{val}}}
\lVert
D_s(f(A_i x)) - y_i
\rVert.
\]

Use cloud masks if available:

\[
\mathcal{E}_{\text{masked}}
=
\frac{
\sum_p M_i(p)
\lVert \hat{y}_i(p)-y_i(p)\rVert
}{
\sum_p M_i(p)
}.
\]

### 11.3 Recommended split

For \(N=16\):

```yaml
heldout:
  enabled: true
  train_frames: 12
  val_frames: 4
  split: stratified_by_time_or_random
```

For \(N=8\):

```yaml
heldout:
  train_frames: 6
  val_frames: 2
```

### 11.4 Metrics

```text
real/train_lr_consistency
real/heldout_lr_consistency
real/masked_heldout_lr_consistency
real/valid_pixel_fraction
```

This is one of the most important metrics for real Sentinel-2.

---

# Part B — Recommended final method variants

---

## 12. Minimal method family

Implement variants in this order.

### M0 — SuperF-Fourier baseline

Purpose:

> Reproduce or approximate the original SuperF baseline.

Components:

- Fourier coordinate encoding,
- MLP decoder,
- joint alignment,
- fixed downsampling,
- MSE loss.

---

### M1 — HashGrid-SuperF

Purpose:

> Test whether HashGrid improves the coordinate representation.

Components:

- HashGrid encoder,
- MLP decoder,
- joint alignment,
- fixed downsampling,
- MSE loss.

---

### M2 — Progressive HashGrid-SuperF

Purpose:

> Test whether coarse-to-fine activation improves alignment and convergence.

Components:

- HashGrid encoder,
- progressive level schedule,
- MLP decoder,
- joint alignment,
- fixed downsampling,
- MSE loss.

---

### M3 — Progressive HashGrid-SuperF + Radiometric

Purpose:

> Test whether per-frame gain/offset improves real and synthetic robustness.

Components:

- M2,
- per-frame per-band gain/offset,
- gain/offset regularization.

---

### M4 — Progressive Robust HashGrid-SuperF

Purpose:

> Test whether robust loss improves noisy/cloudy/inconsistent cases.

Components:

- M3,
- Huber or Charbonnier loss,
- optional GNLL uncertainty.

Recommended as main final method:

> **Progressive Robust HashGrid-SuperF**

---

### M5 — Progressive Robust HashGrid-SuperF + Learned PSF

Purpose:

> Test whether a constrained learned degradation model improves sensor mismatch robustness.

Components:

- M4,
- learned Gaussian PSF,
- PSF regularization.

This is scientifically strong but should be added after M4 works.

---

### M6 — Scale-Gated Progressive Robust HashGrid-SuperF

Purpose:

> Test whether explicit level gating improves scale selection, convergence, or interpretability.

Components:

- M4 or M5,
- softmax or sigmoid hash-level gates,
- gate warmup,
- effective contribution logging.

This is optional but useful if attention/gating remains part of the report.

---

## 13. Recommended experiment matrix

### 13.1 Core ablation

| Method | Encoding | Schedule | Observation model | Loss |
|---|---|---|---|---|
| SuperF-Fourier | Fourier | joint | fixed downsample | MSE |
| HashGrid-SuperF | HashGrid | joint | fixed downsample | MSE |
| Progressive HashGrid | HashGrid | coarse-to-fine | fixed downsample | MSE |
| Progressive + Radiometric | HashGrid | coarse-to-fine | gain/offset | MSE |
| Progressive + Robust | HashGrid | coarse-to-fine | gain/offset | Huber/Charbonnier |
| Progressive + PSF | HashGrid | coarse-to-fine | gain/offset + learned PSF | Huber/Charbonnier |

### 13.2 Optional gating ablation

| Method | Gate |
|---|---|
| Progressive HashGrid | none |
| Progressive HashGrid + SoftmaxGate | competitive scale selection |
| Progressive HashGrid + SigmoidGate | independent level recalibration |
| Progressive HashGrid + CoordSoftmaxGate | coordinate-conditioned gate |

### 13.3 Capacity control

| Method | Purpose |
|---|---|
| HashGrid-Concat | baseline |
| HashGrid-Concat-BiggerMLP | extra parameter control |
| HashGrid-ScaleGate | scale-gating test |

---

# Part C — Datasets and evaluation

---

## 14. Dataset A: Synthetic Sentinel-2-like bursts

### 14.1 Purpose

Use synthetic data for quantitative evaluation because HR ground truth and true alignment are known.

### 14.2 Generation pipeline

\[
\text{HR image}
\rightarrow
\text{random affine transforms}
\rightarrow
\text{blur/PSF}
\rightarrow
\text{downsample}
\rightarrow
\text{radiometric variation}
\rightarrow
\text{noise/outliers}
\rightarrow
\text{LR burst}.
\]

### 14.3 Recommended settings

```yaml
synthetic_dataset:
  hr_crop_size: 256
  scale_factor: 4
  lr_crop_size: 64
  n_frames: 16
  channels: 3
  shift_range_lr_pixels: 1.0
  rotation_range_degrees: 1.0
  gaussian_noise_std: 0.01
  spectral_variation: true
  radiometric_gain_range: [0.9, 1.1]
  radiometric_offset_range: [-0.03, 0.03]
  blur_model: gaussian_or_mtf_like
```

### 14.4 Corrupted synthetic setting

For robustness experiments:

```yaml
corruptions:
  cloudy_frames_fraction: 0.25
  cloud_mask_type: random_blobs
  cloud_opacity_range: [0.3, 0.8]
  haze_strength_range: [0.05, 0.2]
  outlier_frame_fraction: 0.125
```

### 14.5 Metrics

Use:

```text
PSNR
SSIM
LPIPS
alignment_error
runtime
peak_gpu_memory
iterations_to_95_percent_final_psnr
```

If using multispectral bands:

```text
SAM
ERGAS
bandwise_PSNR
```

---

## 15. Dataset B: Real Sentinel-2 time series

### 15.1 Purpose

Use real data to show that the method behaves sensibly without HR ground truth.

### 15.2 Recommended setup

```yaml
real_sentinel2:
  product: L2A
  bands: [B2, B3, B4]
  optional_bands: [B8]
  n_frames: 8_to_16
  time_window: 1_to_3_months
  crop_size_lr: 64_or_128
  cloud_filtering: true
  use_scl_mask: true
```

### 15.3 Scene types

Use two or three real crops:

| Scene type | Purpose |
|---|---|
| Urban/roads | high-frequency structure |
| Agriculture/fields | smooth + boundaries |
| Coast/water | sharp edges + smooth regions |
| Noisy/cloudy time series | robustness |

### 15.4 Real-data metrics

Use:

```text
train_lr_consistency
heldout_lr_consistency
masked_heldout_lr_consistency
valid_pixel_fraction
runtime
visual_quality
uncertainty_or_reliability_maps
```

Do not claim true HR accuracy on real Sentinel-2 unless there is a valid HR reference.

---

# Part D — Experiments

---

## 16. Experiment 1: Main synthetic quantitative comparison

### 16.1 Purpose

Test whether each TTO-only improvement helps under controlled conditions.

### 16.2 Setup

```yaml
dataset: synthetic_sentinel2_burst
scale_factor: 4
n_frames: 16
hr_crop_size: 256
channels: RGB
iterations: 2000
seeds: [0, 1, 2]
```

### 16.3 Methods

| Method | Include |
|---|---|
| Bicubic | required |
| SuperF-Fourier | required |
| HashGrid-SuperF | required |
| Progressive HashGrid | required |
| Progressive + Radiometric | required if implemented |
| Progressive + Robust | required if implemented |
| Progressive + PSF | optional |

### 16.4 Report table

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Align. err ↓ | Runtime ↓ | Memory ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Bicubic |  |  |  | — |  |  |
| SuperF-Fourier |  |  |  |  |  |  |
| HashGrid-SuperF |  |  |  |  |  |  |
| Progressive HashGrid |  |  |  |  |  |  |
| Progressive + Radiometric |  |  |  |  |  |  |
| Progressive + Robust |  |  |  |  |  |  |
| Progressive + PSF |  |  |  |  |  |  |

---

## 17. Experiment 2: Convergence and optimization dynamics

### 17.1 Purpose

Test whether the TTO changes improve optimization efficiency.

### 17.2 Track every 100 iterations

```text
eval/psnr
eval/ssim
eval/lpips
train/loss
alignment/error
runtime/elapsed
progressive/active_levels
```

### 17.3 Plot

```text
PSNR vs optimization iteration
alignment error vs optimization iteration
training loss vs optimization iteration
```

### 17.4 Report table

| Method | Final PSNR ↑ | Iter. to 95% final PSNR ↓ | Final align. err ↓ | Runtime ↓ |
|---|---:|---:|---:|---:|
| SuperF-Fourier |  |  |  |  |
| HashGrid-SuperF |  |  |  |  |
| Progressive HashGrid |  |  |  |  |
| Progressive + Robust |  |  |  |  |

---

## 18. Experiment 3: Alignment robustness

### 18.1 Purpose

Test whether coarse-to-fine TTO improves alignment stability.

### 18.2 Setup

Vary initial alignment perturbation:

```yaml
alignment_stress:
  shift_init_error_lr_pixels: [0.0, 0.5, 1.0, 2.0]
  rotation_init_error_degrees: [0.0, 0.5, 1.0, 2.0]
```

### 18.3 Methods

| Method | Purpose |
|---|---|
| SuperF-Fourier | baseline |
| HashGrid-SuperF | representation change |
| Progressive HashGrid | alignment curriculum |
| Progressive HashGrid + gates | optional |

### 18.4 Metrics

```text
alignment_error
PSNR
failure_rate
```

Define failure:

```text
failure if final alignment_error > threshold or PSNR < baseline_threshold
```

### 18.5 Report table

| Init perturbation | SuperF PSNR | HashGrid PSNR | Progressive PSNR | Progressive align. err |
|---|---:|---:|---:|---:|
| small |  |  |  |  |
| medium |  |  |  |  |
| large |  |  |  |  |

---

## 19. Experiment 4: Robustness to radiometric variation

### 19.1 Purpose

Test whether per-frame gain/offset helps when LR frames have brightness/spectral inconsistencies.

### 19.2 Setup

Use synthetic bursts with controlled gain/offset perturbations.

```yaml
radiometric_stress:
  gain_ranges:
    - [1.0, 1.0]
    - [0.95, 1.05]
    - [0.9, 1.1]
    - [0.8, 1.2]
  offset_ranges:
    - [0.0, 0.0]
    - [-0.01, 0.01]
    - [-0.03, 0.03]
    - [-0.05, 0.05]
```

### 19.3 Methods

| Method | Purpose |
|---|---|
| Progressive HashGrid | no correction |
| Progressive + gain | simple radiometric correction |
| Progressive + gain/offset | full correction |
| Progressive + gain/offset + robust loss | robust correction |

### 19.4 Metrics

```text
PSNR
SSIM
LPIPS
learned_gain_error
learned_offset_error
heldout_lr_consistency
```

### 19.5 Report table

| Radiometric stress | No correction PSNR | Gain PSNR | Gain/offset PSNR | Robust PSNR |
|---|---:|---:|---:|---:|
| none |  |  |  |  |
| mild |  |  |  |  |
| medium |  |  |  |  |
| strong |  |  |  |  |

---

## 20. Experiment 5: Robustness to clouds/outliers/noise

### 20.1 Purpose

Test whether robust losses and reliability models improve corrupted frame settings.

### 20.2 Setup

```yaml
corruption_stress:
  gaussian_noise_std: [0.0, 0.01, 0.03]
  cloudy_frame_fraction: [0.0, 0.125, 0.25, 0.5]
  cloud_opacity: [0.3, 0.6, 0.9]
```

### 20.3 Methods

| Method | Loss |
|---|---|
| Progressive + Radiometric | MSE |
| Progressive + Radiometric | Huber |
| Progressive + Radiometric | Charbonnier |
| Progressive + Radiometric | GNLL |
| Progressive + Radiometric | frame reliability |

### 20.4 Metrics

```text
PSNR
SSIM
LPIPS
heldout_lr_consistency
uncertainty_map_quality
frame_reliability_values
```

### 20.5 Report table

| Corruption | MSE PSNR | Huber PSNR | Charbonnier PSNR | GNLL PSNR | Reliability PSNR |
|---|---:|---:|---:|---:|---:|
| clean |  |  |  |  |  |
| noisy |  |  |  |  |  |
| cloudy 25% |  |  |  |  |  |
| cloudy 50% |  |  |  |  |  |

---

## 21. Experiment 6: Sensor/degradation mismatch

### 21.1 Purpose

Test whether learned PSF helps when the assumed downsampling model is wrong.

### 21.2 Setup

Generate LR bursts with different blur kernels:

```yaml
degradation_stress:
  true_sigma_values: [0.5, 1.0, 1.5, 2.0]
  model_assumed_sigma: 1.0
```

### 21.3 Methods

| Method | Degradation model |
|---|---|
| Progressive HashGrid | fixed average |
| Progressive HashGrid | fixed Gaussian sigma=1.0 |
| Progressive HashGrid | learned Gaussian sigma |
| Progressive HashGrid | learned per-band sigma |

### 21.4 Metrics

```text
PSNR
SSIM
LPIPS
learned_sigma_error
heldout_lr_consistency
```

### 21.5 Report table

| True blur sigma | Fixed avg PSNR | Fixed sigma PSNR | Learned sigma PSNR | Learned sigma |
|---|---:|---:|---:|---:|
| 0.5 |  |  |  |  |
| 1.0 |  |  |  |  |
| 1.5 |  |  |  |  |
| 2.0 |  |  |  |  |

---

## 22. Experiment 7: Number of LR frames

### 22.1 Purpose

Test data efficiency.

### 22.2 Setup

\[
N \in \{4, 8, 16\}.
\]

### 22.3 Methods

| Method | Purpose |
|---|---|
| SuperF-Fourier | baseline |
| HashGrid-SuperF | representation |
| Progressive HashGrid | schedule |
| Progressive + Robust | full method |
| Progressive + Robust + Gate | optional |

### 22.4 Report table

| Method | 4 frames PSNR ↑ | 8 frames PSNR ↑ | 16 frames PSNR ↑ |
|---|---:|---:|---:|
| SuperF-Fourier |  |  |  |
| HashGrid-SuperF |  |  |  |
| Progressive HashGrid |  |  |  |
| Progressive + Robust |  |  |  |
| Progressive + Robust + Gate |  |  |  |

---

## 23. Experiment 8: Real Sentinel-2 evaluation

### 23.1 Purpose

Show real-scene behavior without HR ground truth.

### 23.2 Setup

Use two to three real Sentinel-2 L2A time series.

```yaml
real_eval:
  n_frames: 8_to_16
  crop_size_lr: 64_or_128
  train_val_split: true
  cloud_mask: true
```

### 23.3 Methods

| Method |
|---|
| Bicubic |
| SuperF-Fourier |
| HashGrid-SuperF |
| Progressive HashGrid |
| Progressive + Radiometric |
| Progressive + Robust |
| Progressive + Robust + PSF |

### 23.4 Metrics

```text
train_lr_consistency
heldout_lr_consistency
masked_heldout_lr_consistency
runtime
visual_quality
uncertainty/reliability maps
```

### 23.5 Qualitative figure

Suggested columns:

| LR input | Bicubic | SuperF | HashGrid | Progressive | Progressive + Robust | Uncertainty/reliability |
|---|---|---|---|---|---|---|

### 23.6 Report caveat

Use this wording:

> Since no higher-resolution ground truth is available for real Sentinel-2 scenes, the real-data experiment evaluates LR consistency, held-out LR consistency, robustness, and visual plausibility rather than absolute HR accuracy.

---

# Part E — Recommended configs

---

## 24. Base config: SuperF-Fourier

```yaml
experiment:
  name: superf_fourier_x4
  seed: 0

dataset:
  type: synthetic_sentinel2_burst
  scale_factor: 4
  hr_crop_size: 256
  lr_crop_size: 64
  n_frames: 16
  channels: 3

model:
  type: superf_fourier

fourier:
  n_frequencies: 10
  scale: 10.0

decoder:
  type: mlp
  hidden_dim: 128
  n_layers: 4
  activation: relu
  output_dim: 3

alignment:
  enabled: true
  model: affine_small
  optimize_translation: true
  optimize_rotation: true

degradation:
  type: fixed_average_pool

loss:
  type: mse

optimization:
  optimizer: adamw
  iterations: 2000
  lr_decoder: 0.001
  lr_alignment: 0.001
  weight_decay: 0.01
  scheduler: cosine
```

---

## 25. Base config: HashGrid-SuperF

```yaml
experiment:
  name: hashgrid_superf_x4
  seed: 0

dataset:
  type: synthetic_sentinel2_burst
  scale_factor: 4
  hr_crop_size: 256
  lr_crop_size: 64
  n_frames: 16
  channels: 3

model:
  type: hashgrid_superf

hashgrid:
  n_levels: 12
  features_per_level: 2
  base_resolution: 8
  max_resolution: 256
  log2_hashmap_size: 16
  interpolation: linear

decoder:
  type: mlp
  input_dim: 24
  hidden_dim: 64
  n_layers: 3
  activation: relu
  output_dim: 3

alignment:
  enabled: true
  model: affine_small
  optimize_translation: true
  optimize_rotation: true

degradation:
  type: fixed_average_pool

loss:
  type: mse

optimization:
  optimizer: adamw
  iterations: 2000
  lr_hashgrid: 0.01
  lr_decoder: 0.001
  lr_alignment: 0.001
  weight_decay: 0.01
  scheduler: cosine
```

---

## 26. Config: Progressive HashGrid-SuperF

```yaml
experiment:
  name: progressive_hashgrid_superf_x4
  seed: 0

model:
  type: hashgrid_superf

hashgrid:
  n_levels: 12
  features_per_level: 2
  base_resolution: 8
  max_resolution: 256
  log2_hashmap_size: 16
  interpolation: linear

progressive_levels:
  enabled: true
  mode: hard
  schedule:
    - {start: 0, active_levels: 4}
    - {start: 300, active_levels: 6}
    - {start: 700, active_levels: 9}
    - {start: 1200, active_levels: 12}

decoder:
  type: mlp
  input_dim: 24
  hidden_dim: 64
  n_layers: 3
  activation: relu
  output_dim: 3

optimization:
  optimizer: adamw
  iterations: 2000
  lr_hashgrid: 0.01
  lr_decoder: 0.001
  lr_alignment: 0.001
  weight_decay: 0.01
  scheduler: cosine

loss:
  type: mse
```

---

## 27. Config: Progressive Robust HashGrid-SuperF

```yaml
experiment:
  name: progressive_robust_hashgrid_superf_x4
  seed: 0

model:
  type: hashgrid_superf

hashgrid:
  n_levels: 12
  features_per_level: 2
  base_resolution: 8
  max_resolution: 256
  log2_hashmap_size: 16
  interpolation: linear

progressive_levels:
  enabled: true
  mode: hard
  schedule:
    - {start: 0, active_levels: 4}
    - {start: 300, active_levels: 6}
    - {start: 700, active_levels: 9}
    - {start: 1200, active_levels: 12}

radiometric:
  enabled: true
  model: gain_offset
  per_frame: true
  per_band: true
  lambda_gain: 1.0e-3
  lambda_offset: 1.0e-3

degradation:
  type: fixed_average_pool

loss:
  type: charbonnier
  epsilon: 1.0e-3

optimization:
  optimizer: adamw
  iterations: 2000
  lr_hashgrid: 0.01
  lr_decoder: 0.001
  lr_alignment: 0.001
  lr_gain: 0.001
  lr_offset: 0.001
  weight_decay: 0.01
  scheduler: cosine
```

---

## 28. Config: Progressive Robust HashGrid-SuperF + learned PSF

```yaml
experiment:
  name: progressive_robust_hashgrid_superf_psf_x4
  seed: 0

degradation:
  type: learned_gaussian_psf
  per_band: true
  sigma_min: 0.2
  sigma_max: 2.5
  init_sigma: 1.0
  kernel_size: 7
  lambda_sigma: 1.0e-3

optimization:
  lr_sigma: 0.001
```

This config extends the Progressive Robust HashGrid-SuperF config.

---

## 29. Config: Scale-Gated Progressive Robust HashGrid-SuperF

```yaml
experiment:
  name: scalegated_progressive_robust_hashgrid_superf_x4
  seed: 0

scale_gates:
  enabled: true
  gate_type: softmax
  hidden_dim: 64
  temperature: 1.0
  use_layernorm: true
  coord_conditioned: false
  warmup_iterations: 300
  lr_gate: 5.0e-4

logging:
  log_gate_maps: true
  log_effective_contribution: true
```

Sigmoid variant:

```yaml
scale_gates:
  enabled: true
  gate_type: sigmoid
  hidden_dim: 64
  use_layernorm: true
  coord_conditioned: false
  warmup_iterations: 300
  lr_gate: 5.0e-4
```

Coord-conditioned variant:

```yaml
scale_gates:
  enabled: true
  gate_type: softmax
  coord_conditioned: true
  coord_n_bands: 1
  hidden_dim: 64
  temperature: 1.0
  warmup_iterations: 300
  lr_gate: 5.0e-4
```

---

# Part F — Logging schema

---

## 30. General metrics

Log these for all methods:

```text
train/loss
train/reconstruction_loss
eval/psnr
eval/ssim
eval/lpips
eval/alignment_error
runtime/seconds
runtime/iterations_per_second
memory/peak_gpu_memory
```

---

## 31. Alignment metrics

```text
alignment/translation_error_mean
alignment/translation_error_median
alignment/rotation_error_mean
alignment/rotation_error_median
alignment/failure_rate
```

For real data, where true alignment is unknown:

```text
alignment/estimated_translation_mean
alignment/estimated_rotation_mean
alignment/estimated_transform_norm
```

---

## 32. Progressive-level metrics

```text
progressive/active_levels
progressive/mask_level_00
progressive/mask_level_01
...
progressive/mask_level_11
```

---

## 33. Radiometric metrics

```text
radiometric/gain_mean
radiometric/gain_std
radiometric/gain_min
radiometric/gain_max
radiometric/offset_mean
radiometric/offset_std
radiometric/offset_min
radiometric/offset_max
radiometric/loss
```

---

## 34. PSF metrics

```text
psf/sigma_band_00
psf/sigma_band_01
psf/sigma_band_02
psf/sigma_mean
psf/sigma_std
psf/loss
```

---

## 35. Robustness/uncertainty metrics

```text
uncertainty/logvar_mean
uncertainty/logvar_std
uncertainty/logvar_min
uncertainty/logvar_max
uncertainty/map_entropy_or_variance
frame_reliability/mean
frame_reliability/min
frame_reliability/max
frame_reliability/frame_00
...
frame_reliability/frame_15
```

---

## 36. Gate metrics

```text
gate/entropy
gate/gate_mean_level_00
gate/gate_mean_level_01
...
gate/gate_mean_level_11
gate/mass_mean_level_00
...
gate/coarse_mass
gate/mid_mass
gate/fine_mass
gate/effective_contribution_level_00
...
gate/effective_coarse_contribution
gate/effective_mid_contribution
gate/effective_fine_contribution
```

---

## 37. Real-data metrics

```text
real/train_lr_consistency
real/heldout_lr_consistency
real/masked_train_lr_consistency
real/masked_heldout_lr_consistency
real/valid_pixel_fraction
```

---

# Part G — Command templates

---

## 38. Main synthetic comparison

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/superf_fourier.yaml \
  --seed 0
```

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/hashgrid_superf.yaml \
  --seed 0
```

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/progressive_hashgrid_superf.yaml \
  --seed 0
```

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/progressive_robust_hashgrid_superf.yaml \
  --seed 0
```

---

## 39. Multiple seeds

```bash
for seed in 0 1 2; do
  python train_superf.py --config configs/synthetic_x4_16frames/superf_fourier.yaml --seed $seed
  python train_superf.py --config configs/synthetic_x4_16frames/hashgrid_superf.yaml --seed $seed
  python train_superf.py --config configs/synthetic_x4_16frames/progressive_hashgrid_superf.yaml --seed $seed
  python train_superf.py --config configs/synthetic_x4_16frames/progressive_robust_hashgrid_superf.yaml --seed $seed
done
```

---

## 40. Frame-count ablation

```bash
for n_frames in 4 8 16; do
  python train_superf.py --config configs/synthetic_x4_${n_frames}frames/hashgrid_superf.yaml --seed 0
  python train_superf.py --config configs/synthetic_x4_${n_frames}frames/progressive_hashgrid_superf.yaml --seed 0
  python train_superf.py --config configs/synthetic_x4_${n_frames}frames/progressive_robust_hashgrid_superf.yaml --seed 0
done
```

---

## 41. Robustness experiments

```bash
for corruption in clean noisy cloudy25 cloudy50; do
  python train_superf.py \
    --config configs/robustness/${corruption}/progressive_hashgrid_mse.yaml \
    --seed 0

  python train_superf.py \
    --config configs/robustness/${corruption}/progressive_hashgrid_charbonnier.yaml \
    --seed 0

  python train_superf.py \
    --config configs/robustness/${corruption}/progressive_hashgrid_gnll.yaml \
    --seed 0
done
```

---

## 42. PSF/degradation mismatch experiments

```bash
for sigma in 0.5 1.0 1.5 2.0; do
  python train_superf.py \
    --config configs/psf_mismatch/true_sigma_${sigma}/fixed_psf.yaml \
    --seed 0

  python train_superf.py \
    --config configs/psf_mismatch/true_sigma_${sigma}/learned_psf.yaml \
    --seed 0
done
```

---

## 43. Scale-gating ablation

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/progressive_robust_hashgrid_superf.yaml \
  --override scale_gates.enabled=false \
  --seed 0
```

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/scalegated_progressive_robust_hashgrid_superf.yaml \
  --override scale_gates.gate_type=softmax \
  --seed 0
```

```bash
python train_superf.py \
  --config configs/synthetic_x4_16frames/scalegated_progressive_robust_hashgrid_superf.yaml \
  --override scale_gates.gate_type=sigmoid \
  --seed 0
```

---

# Part H — Implementation roadmap

---

## 44. Recommended development order

### Step 1: Reproduce baseline

- [ ] Run existing SuperF/Fourier version.
- [ ] Verify LR reconstruction loss decreases.
- [ ] Verify synthetic PSNR/SSIM/LPIPS evaluation.
- [ ] Verify alignment error logging.

### Step 2: Implement HashGrid encoder

- [ ] Add HashGrid module.
- [ ] Match decoder input to \(L \times F\).
- [ ] Verify gradients flow through hash table.
- [ ] Compare against Fourier baseline.

### Step 3: Add progressive level schedule

- [ ] Implement level masks.
- [ ] Add hard schedule.
- [ ] Optional: add smooth schedule.
- [ ] Log active levels.
- [ ] Evaluate alignment stability.

### Step 4: Add radiometric gain/offset

- [ ] Add per-frame gain.
- [ ] Add per-frame offset.
- [ ] Add regularization.
- [ ] Log learned parameters.
- [ ] Test on synthetic radiometric stress.

### Step 5: Add robust loss

- [ ] Implement Huber.
- [ ] Implement Charbonnier.
- [ ] Optional: implement GNLL.
- [ ] Test on corrupted synthetic bursts.

### Step 6: Add held-out LR evaluation

- [ ] Split frames into train/held-out.
- [ ] Evaluate held-out LR consistency.
- [ ] Add masked consistency for cloud masks.

### Step 7: Add learned PSF

- [ ] Implement Gaussian kernel from trainable sigma.
- [ ] Add per-band sigma.
- [ ] Add sigma regularization.
- [ ] Test on synthetic PSF mismatch.

### Step 8: Add optional scale gates

- [ ] Implement softmax gates.
- [ ] Implement sigmoid gates.
- [ ] Add warmup.
- [ ] Log effective contribution.
- [ ] Save gate maps.

---

# Part I — Report framing

---

## 45. Suggested title

> **Improving Test-Time Neural Field Super-Resolution with HashGrid Encoding, Progressive Alignment, and Robust Observation Modeling**

Optional if scale gates are included:

> **Scale-Gated HashGrid Neural Fields for Test-Time Sentinel-2 Multi-Image Super-Resolution**

---

## 46. Suggested abstract skeleton

> Test-time multi-image super-resolution avoids the need for high-resolution training data by optimizing a scene-specific representation directly from low-resolution observations. This report studies whether the SuperF test-time optimization pipeline can be improved without adding pretrained backbones or external learned priors. We replace Fourier coordinate features with a multiresolution HashGrid encoder, introduce a progressive coarse-to-fine level schedule to stabilize joint alignment and reconstruction, and add robust observation modeling through per-frame radiometric correction and robust losses. Experiments on synthetic Sentinel-2-like bursts and real Sentinel-2 time series evaluate reconstruction quality, convergence speed, alignment stability, and held-out LR consistency. The results show whether improvements to the representation, optimization curriculum, and observation model can make test-time neural-field super-resolution more accurate and robust.

---

## 47. Suggested contribution statement

Use this version if not including gates:

> This report makes three contributions. First, it replaces SuperF’s fixed coordinate encoding with a trainable multiresolution HashGrid representation optimized from scratch at test time. Second, it introduces a progressive coarse-to-fine HashGrid schedule designed to stabilize joint field and alignment optimization. Third, it evaluates robust observation modeling through per-frame radiometric correction, robust reconstruction losses, and optionally a constrained learned PSF.

Use this version if including gates:

> In addition, we evaluate lightweight hash-level scale gates that explicitly modulate the contribution of each HashGrid resolution level, providing an interpretable mechanism for spatially varying scale selection.

---

## 48. Suggested related-work framing

Keep related work short:

1. **SuperF and test-time MISR**
   - SuperF as the direct baseline.
   - DIP/ZSSR as broader test-time/image-specific reconstruction context.

2. **HashGrid neural fields**
   - Instant-NGP as the basis for multiresolution hash encoding.
   - Spatially adaptive hash encodings as motivation for level selection or level masking.

3. **Coarse-to-fine optimization**
   - BARF-style logic: high-frequency encodings can hurt alignment; coarse-to-fine schedules can stabilize registration.

4. **Robust remote-sensing SR**
   - Sentinel-2 time series contain radiometric differences, clouds, seasonal variation, and sensor mismatch.
   - QA-Net/PIUnet/RAMS can be mentioned as examples of remote-sensing MISR models handling quality, uncertainty, or attention, while your method remains TTO-only.

---

## 49. Suggested method-section structure

### 49.1 Baseline

Describe SuperF objective:

\[
\min_{\theta,\{A_i\}}
\sum_i
\mathcal{L}(D_s(f_\theta(A_i x)), y_i).
\]

### 49.2 HashGrid representation

Describe:

\[
f_{\theta,\phi}(x)=d_\theta(E_\phi(x)).
\]

### 49.3 Progressive optimization

Describe:

\[
z_t(x)=
[m_1(t)h_1(x),\ldots,m_L(t)h_L(x)].
\]

### 49.4 Robust observation model

Describe:

\[
\hat{y}_{i,b}
=
a_{i,b}
D_{s,k_b}(f_{\theta,b}(A_i x))
+
c_{i,b}.
\]

### 49.5 Robust loss

Describe Huber/Charbonnier/GNLL.

### 49.6 Optional gates

Describe:

\[
z(x)=
\text{concat}(w_1(x)h_1(x),\ldots,w_L(x)h_L(x)).
\]

---

## 50. Suggested experiment-section structure

Use four experiments in the report:

### Experiment 1: Main synthetic ablation

Compares:

- SuperF-Fourier,
- HashGrid-SuperF,
- Progressive HashGrid,
- Progressive + Robust,
- optional Progressive + PSF/Gates.

### Experiment 2: Convergence and alignment

Shows:

- PSNR vs iteration,
- alignment error vs iteration.

### Experiment 3: Robustness stress tests

Shows:

- radiometric perturbation,
- clouds/noise/outliers,
- degradation mismatch.

### Experiment 4: Real Sentinel-2

Shows:

- held-out LR consistency,
- qualitative crops,
- uncertainty/reliability maps if available.

---

## 51. Suggested conclusion wording

> The experiments test whether SuperF can be improved without external training by modifying the test-time inverse problem itself. HashGrid encoding targets representational efficiency, progressive level activation targets optimization and alignment stability, and robust observation modeling targets real Sentinel-2 inconsistencies. This framing keeps the method scene-specific and training-free while addressing practical weaknesses of direct test-time neural-field fitting.

---

# Part J — References to start from

Use these as starting references for the report.

1. **SuperF** — main baseline and TTO MISR formulation.  
   https://arxiv.org/abs/2512.09115

2. **Instant-NGP** — multiresolution HashGrid encoding.  
   Müller et al., 2022.  
   https://arxiv.org/abs/2201.05989

3. **BARF** — coarse-to-fine positional encoding for joint alignment and neural field optimization.  
   Lin et al., 2021.  
   https://arxiv.org/abs/2104.06405

4. **Deep Image Prior** — image-specific test-time inverse problem prior.  
   Ulyanov et al., 2018.  
   https://arxiv.org/abs/1711.10925

5. **ZSSR** — zero-shot single-image super-resolution.  
   Shocher et al., 2018.  
   https://arxiv.org/abs/1712.06087

6. **KernelGAN** — blind image-specific downsampling kernel estimation.  
   Bell-Kligler et al., 2019.  
   https://arxiv.org/abs/1909.06581

7. **Spatially-Adaptive Hash Encodings** — spatially adaptive resolution selection in HashGrid-like neural fields.  
   https://arxiv.org/abs/2412.05179

8. **Mip-NeRF** — anti-aliased/integrated positional encoding concept.  
   Barron et al., 2021.  
   https://arxiv.org/abs/2103.13415

9. **QA-Net** — quality-map associated attention for satellite MISR.  
   https://arxiv.org/abs/2202.13124

10. **PIUnet** — permutation invariance and uncertainty in multitemporal SR.  
    https://arxiv.org/abs/2105.12409

11. **RAMS** — residual attention for remote-sensing multi-image super-resolution.  
    https://arxiv.org/abs/2007.03107

12. **Selective Kernel Networks** — adaptive scale selection by softmax branch weighting.  
    https://arxiv.org/abs/1903.06586

13. **Squeeze-and-Excitation Networks** — lightweight feature/channel recalibration.  
    https://arxiv.org/abs/1709.01507

14. **Coordinate Attention** — attention with positional information.  
    https://arxiv.org/abs/2103.02907

---

# Part K — Priority summary

## 52. Highest-value low-risk experiments

Run these first:

1. SuperF-Fourier baseline.
2. HashGrid-SuperF.
3. Progressive HashGrid-SuperF.
4. Progressive HashGrid + radiometric gain/offset.
5. Progressive HashGrid + Charbonnier/Huber.
6. Held-out LR consistency on real Sentinel-2.

## 53. Medium-risk high-value experiments

Run these after the core stack works:

7. Learned Gaussian PSF.
8. GNLL uncertainty.
9. Hash-level softmax/sigmoid gates.
10. Alignment stress test.

## 54. Optional/future-work experiments

Use these only if time allows:

11. Coord-conditioned gates.
12. Frame reliability weights.
13. Jittered area sampling / anti-aliasing.
14. Multi-start TTO.
15. Local coordinate attention.

---

## 55. Final instruction for local agent

Implement the TTO-only SuperF improvement stack in this order:

1. Add a HashGrid coordinate encoder and compare against Fourier-SuperF.
2. Add progressive coarse-to-fine HashGrid level activation.
3. Add per-frame, per-band radiometric gain/offset correction.
4. Add robust reconstruction losses, starting with Charbonnier or Huber.
5. Add held-out LR consistency evaluation.
6. Add a constrained learned Gaussian PSF if the previous modules work.
7. Optionally add hash-level softmax/sigmoid scale gates with warmup and effective contribution logging.

Run the core synthetic experiment at \(4\times\) scale with 16 LR frames and 2000 TTO iterations. Report PSNR, SSIM, LPIPS, alignment error, runtime, memory, and convergence speed. Then run frame-count, radiometric, corruption, degradation-mismatch, and real Sentinel-2 held-out consistency experiments.

The preferred final comparison is:

| Method | Encoding | Schedule | Observation model | Loss |
|---|---|---|---|---|
| SuperF-Fourier | Fourier | joint | fixed | MSE |
| HashGrid-SuperF | HashGrid | joint | fixed | MSE |
| Progressive HashGrid | HashGrid | coarse-to-fine | fixed | MSE |
| Progressive + Radiometric | HashGrid | coarse-to-fine | gain/offset | MSE |
| Progressive + Robust | HashGrid | coarse-to-fine | gain/offset | Charbonnier/Huber |
| Progressive + PSF | HashGrid | coarse-to-fine | gain/offset + learned PSF | Charbonnier/Huber |

The main scientific claim should be:

> SuperF can be improved without external training by modifying the test-time inverse problem: HashGrid encoding improves representation and convergence, progressive level activation improves alignment stability, and robust observation modeling improves real-scene reliability.
