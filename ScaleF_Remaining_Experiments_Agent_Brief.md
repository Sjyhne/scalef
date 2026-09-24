# ScaleF: remaining experiments and evaluation fixes

**Prepared:** 9 September 2026  
**Audience:** Implementation and experiment agent working on ScaleF  
**Implementation repository:** https://github.com/Sjyhne/scalef  
**Paper repository:** https://github.com/Sjyhne/ScaleF_Overleaf  
**Reviewed paper snapshot:** `04af6b76a9fe9e0ce867b91c964ce81c1e6efdd4`

## 1. Objective and scope

Implement the evaluation fixes and complete the experiments needed to support the ScaleF paper. Audit the current implementation and experiment artifacts before scheduling work. **The user explicitly requires a fresh full benchmark rerun on all 17 locations after implementing per-LR512 spatial correction. Rescoring historical outputs is not a substitute for that rerun.** Other supporting experiments may be reusable after corrected evaluation. The snapshot above identifies the reviewed draft; inspect the current repositories for subsequent changes.

The project goals are:

1. Produce Sentinel-2 RGB maps of Norway and Denmark on a 2.5 m GSD grid.
2. Evaluate against real high-resolution orthophotos using a reproducible harmonization protocol.
3. Establish a stable, scalable SuperF extension using an Instant-NGP-style multiresolution representation and fused computation.
4. Optionally demonstrate downstream value. Downstream evaluation is not a prerequisite for the core reconstruction experiments.

This brief specifies experiments and deliverables, not expected outcomes. Preserve negative results, failed jobs, and uncertainty. Distinguish measured quantities from estimates. Treat 2.5 m as output GSD unless additional evidence supports an effective-resolution claim.

## 2. Mandatory requirement: spatial correction for EVERY LR512 tile

**Estimate the orthophoto-to-Sentinel-2 spatial correction independently for EACH LR512 tile. Do not estimate one correction at a site center, in one representative tile, or for a whole NIB project/MGRS granule and extrapolate or copy it to the remaining LR512 tiles.**

An LR512 tile covers approximately 5.12 km × 5.12 km at the 10 m Sentinel-2 input GSD. Every such evaluation tile must have its own locally estimated correction and quality assessment.

### Required behavior

1. Assign every LR512 evaluation tile a stable identifier tied to its geographic footprint and input package.
2. For each tile, use its own local LR base-frame content and corresponding local orthophoto content to estimate reference alignment. Match spatial bandwidth appropriately for cross-sensor registration; do not depend on invented HR detail in an SR output.
3. Select and document the registration model on development data. Audit whether translation is sufficient or a constrained affine is justified. Freeze the model, parameter bounds, estimator, and QC rules before confirmatory evaluation. Do not silently introduce flexible warping to improve reference scores.
4. Freeze one accepted correction for that tile and reference/base-frame pair. Apply it consistently to all methods, seeds, checkpoints, and ablations evaluated in that tile.
5. Do not optimize reference alignment separately for ScaleF, Fourier, SuperF, bilinear, or another method. A method must not receive a more favorable reference transform because of its output.
6. A tile that cannot be registered reliably must have an explicit failed/insufficient-confidence status. Report the reason and its effect on the evaluation denominator. Do not silently fall back to another tile's correction.
7. Cache corrections by tile identity, input/reference identities, registration configuration, and code version. A change in base frame, reference, footprint, or relevant preprocessing invalidates the cache.

### Nested-field and larger-field evaluation

- **Each LR512 parent is registered separately.** When comparing LR64/128/256 children with an LR512 fit on that same parent footprint, assemble the child outputs and evaluate all variants using the parent's fixed correction, reference, masks, and evaluation windows.
- Do not independently realign each LR64/128/256 child for the primary field-size comparison; that would change the evaluator with training-field size.
- If a reconstruction is larger than LR512, evaluate its LR512 subregions using their individually estimated reference corrections. Document this protocol and retain separate absolute-geolocation diagnostics.
- These are reference-evaluation corrections. They do not constitute a method for georeferencing or joining the delivered SR map.
- Keep reference alignment distinct from the per-revisit geometric transforms learned from LR observations during reconstruction. Both need explicit coordinate conventions and provenance.

### Correction artifacts and QC

For every attempted LR512 tile, store at least:

| Field | Required information |
|---|---|
| Identity | Tile ID, footprint, CRS, geotransform, base-frame ID, reference ID |
| Transform | Model, numerical parameters, direction, coordinate units, pixel-center convention |
| Estimation | Code/configuration version, local estimation support, masks, preprocessing |
| Quality | Available overlap, estimator confidence, residual diagnostics appropriate to the estimator |
| Outcome | Accepted/rejected status and reason; no inherited-transform fallback |
| Reproducibility | Input checksums or stable identifiers and registration-artifact checksum |

Generate before/after overlays for a predeclared representative subset and all rejected/suspicious cases. Inspect structures across the tile, not only the center. Report registration acceptance and residual distributions across the full cohort.

**Acceptance criteria:** Every reference-scored LR512 tile has its own accepted, auditable local estimate; every method on that tile uses the same estimate; excluded tiles are counted and explained. Identical numerical estimates on different tiles are allowed if independently estimated—the requirement concerns provenance and local estimation, not forced numerical differences.

## 3. Priorities and dependencies

| Priority / ID | Work | Depends on | Main deliverable |
|---|---|---|---|
| P0 / E0 | Inventory artifacts and version the evaluation protocol | Repository audit | Run inventory and reuse/rescore/rerun decisions |
| P0 / E1 | Per-LR512 spatial correction and harmonization | E0 | Tile correction registry and evaluator |
| P0 / E2 | Common-footprint rescoring | E1 | Corrected supporting comparisons and evaluator diagnostics |
| P0 / E3 | Encoding audit and essential baselines | E1; audit can start earlier | Actual grid allocation and matched-footprint comparisons |
| P1 / E4 | Controlled encoding and update studies | E2–E3 | Quality–time curves and mechanism evidence |
| P1 / E5 | Validate LR-only checkpoint selection | E1; reuse E4 traces | Selection regret, runtime, and failure analysis |
| Required / B17 | Fresh full benchmark on all 17 locations | E1–E5; freeze the final benchmark configurations first | New complete run matrix and replacement paper results |
| P1 / E6 | Freeze configuration and evaluate new geography | E2–E5 | Untouched geographic evaluation |
| P1 / E7 | Forward-model sensitivity and fidelity diagnostics | E1–E2 | Targeted robustness results |
| P1 / E8 | Contiguous regional production pilot | E2–E5 and B17 | Seams, total memory, failures, and end-to-end cost |
| P2 / E9 | Norway and Denmark national delivery | E8 | Two documented national map products |
| P2 / E10 | Dataset packaging and external comparability | E1 onward | Reusable evaluation package; MuS2 comparison |
| Optional | Downstream task, 1 m queries, expanded sweeps | Core evidence | Only experiments answering a concrete remaining question |

P2 indicates execution order, not optional status: both national maps are required to claim the complete two-country contribution. Dataset documentation should progress alongside the experiments.

## 4. E0 — Audit existing work before spending compute

Inventory code versions, configurations, scene manifests, saved fields/outputs, reference transforms, masks, checkpoints, trajectories, raw metrics, timings, and logs.

The reviewed draft already reports:

- A seven-site, three-seed fixed-LR128 update study with K = 1, 2, 4, 8, and full coverage.
- Encoding comparisons at LR64/128/256/512.
- Selected-configuration results on 17 named Norwegian sites.
- Results on 85 successful complete-reference tiles out of 86 materializable candidates, from 91 complete-reference candidates overall.
- A nested field-size study with crop-dependent evaluation.
- A 5 m output-GSD experiment.
- Norwegian observation-availability analyses, distinct from completed national reconstruction.

Verify these statuses against actual artifacts. Classify each experiment as `reusable`, `rescore`, `rerender`, `rerun`, `missing`, or `blocked`, with a reason. **Every configuration included in the final 17-location benchmark must be classified for a fresh rerun under B17, even if its historical outputs remain available.**

Reference-only alignment does not ordinarily change the LR training objective, because orthophotos must not enter optimization or deployment checkpoint selection. Nevertheless, the user requires a fresh end-to-end 17-location benchmark to establish one coherent experimental freeze. Follow that requirement. Rescoring old outputs can validate the new evaluator and support historical diagnostics. If reference data previously affected training or checkpoint selection, additionally document and remove that dependency.

Reconcile older paper notes explicitly. They contain inconsistent baseline requirements, crop-scoring instructions, and priorities for the 1 m study. This brief requests an actual tiled-SuperF comparison, common-footprint evaluation, and per-LR512 correction. Preserve older results as historical evidence rather than silently relabeling them.

**Deliverable:** A concise experiment inventory and a dependency-aware execution manifest. Do not launch the historical full matrix unchanged.

## 5. E1 — Freeze a reproducible harmonization and scoring protocol

Implement the mandatory tile-specific correction from Section 2 before generating new headline reference metrics.

Specify:

- Orthophoto acquisition date/year and precision; exact LR acquisition IDs/dates; reference-to-LR temporal offsets.
- The temporal window used for reference experiments, cloud/shadow/snow handling, and valid support.
- Reference resampling kernel, antialiasing, nodata propagation, and output grid.
- Spatial estimator and QC rules, separately from learned LR-revisit alignment.
- Color handling: model, fit domain, input pair, bounds, and application direction. Prefer a common calibration derived from the reference and designated LR anchor at matched bandwidth. Freeze it per tile and share it across methods. Any output-fitted calibration must be separately labeled as a secondary metric protocol, not silently used for the primary comparison.
- PSNR data range; SSIM parameters; LPIPS network/version, normalization, window geometry, and aggregation.
- How masks and borders interact with metric receptive fields. Do not zero-fill invalid regions and treat the resulting score as if those regions were excluded.

Report clearly separated views where applicable:

1. Structural/appearance agreement after the shared spatial and color harmonization.
2. Cross-sensor reference agreement before color normalization, with its radiometric limitations stated.
3. Radiometric/LR observation consistency in the Sentinel-2 domain.
4. Absolute geolocation diagnostics before reference alignment, distinct from aligned structural quality.

Add focused correctness checks for known synthetic shifts, transform direction and units, per-tile cache isolation, shared transforms across methods, masked boundaries, and metric reproducibility. These checks protect the experiment's scientific validity.

**Acceptance criteria:** One versioned evaluator can reproduce each reported metric from saved outputs and the tile manifest; reference-only transformations cannot enter the training or deployment stopping paths.

## 6. E2 — Rescore on identical geographic support

The reviewed nested table reports ScaleF LPIPS 0.359 at LR64 and 0.394 at LR512, while bilinear changes from 0.417 to 0.445. These historical values are not an isolated measure of training-field-size effects.

For each common LR512 parent:

1. Assemble all LR64, LR128, and LR256 children onto the parent's output grid, and include the direct LR512 reconstruction.
2. Use the same parent-specific reference correction, color protocol, valid mask, and canonical baseline image.
3. Score all outputs on the same predeclared evaluation windows. The scoring-window size must not depend on training-field size.
4. Report any gaps and the number of complete parent comparisons. Use explicit common-support rules and report lost support; never change support silently by method.
5. Aggregate within parents, then by independent project/region. Preserve per-window and per-parent results.
6. Report total processing cost for producing each parent, summing child jobs. Distinguish serial GPU work from elapsed parallel execution time.

**Required invariant:** Given identical baseline pixels, reference, transform, windows, and masks, the baseline score is identical across training-field-size variants within numerical tolerance. A failure of this invariant blocks publication of that comparison.

Rescore available named-site, complete-tile, K, and encoding outputs under the corrected protocol where useful for debugging and development. **Historical rescoring does not complete B17.** The final 17-location benchmark and its comparisons must come from the new runs specified below. Mark legacy scores as superseded and do not mix evaluator versions or experimental freezes in one comparison.

**Deliverables:** Corrected raw metrics and tables; a regenerated nested quality–cost plot; geographically identical qualitative crops. Avoid one misleading horizontal baseline line when the underlying evaluations differ.

## 7. E3 — Audit the representation and establish essential baselines

### E3a. Actual encoding behavior

Log per-level resolution, dense-grid entry requirement, allocated entries, active indexing mode, features per entry, parameter counts, and memory for each field size.

The paper's approximate 10 m finest cells over a 5.12 km field imply roughly 512² grid locations. This is below the stated 2²¹ table cap. The hypothesis to check is that the default configuration uses dense indexing throughout, despite selecting the HashGrid backend. This is an inference from the documented configuration, not an established implementation finding.

Inspect the pinned tiny-cuda-nn version. If all levels fit densely, describe the measured benefit accordingly. If a hashing/compression claim is retained, run a focused capacity sweep around the observed dense-to-hashed transition. A dense-grid control is useful where supported. Do not build a large sweep around arbitrary capacities before examining actual allocations.

### E3b. Required practical comparisons

| Method | Implementation requirement | Purpose |
|---|---|---|
| Bilinear base frame | Canonical shared base image and output grid | Single-frame reference |
| Registered multi-frame fusion | Transparent registration and aggregation, using the same observations and anchor | Contribution beyond alignment and aggregation |
| Original SuperF, tiled | Original implementation or a documented faithful reproduction on feasible smaller tiles, assembled into LR512 parents | Direct comparison with the predecessor |
| ScaleF | Selected configuration; full coverage where relevant | Proposed practical system |

The Fourier path within ScaleF remains useful as an encoding control, but must not be labeled as an independent original-SuperF reproduction.

Inspect original SuperF before selecting tile size. Tune manageable LR64/128 or other feasible sizes on development data, document any sensor/data adaptations, and freeze the chosen baseline configuration. Use the same LR observations, parent anchor, geographic output, reference corrections, and evaluator. Preserve methods' legitimate internal differences and report them.

Start with the existing seven-site cohort and seeds 6/7/8 for stochastic fits. Deterministic baselines need one run. Extend the selected baselines to the confirmatory cohort in E6.

Report parent-level LPIPS, PSNR, SSIM, LR consistency, processing time, total device memory, and tile-boundary behavior. Show both a practical deployment comparison and matched-time quality when trajectories permit it.

**Acceptance criteria:** A reader can determine whether ScaleF improves the quality–cost trade-off over tiled SuperF on identical geographic outputs. If a baseline is blocked, record the exact reason and narrow claims accordingly.

## 8. E4 — Resolve the encoding and partial-update questions

### E4a. Encoding versus field extent

- Compare LR64/128/256/512 using matched geographic centers and LR inputs.
- Express Fourier bandwidth in cycles per meter, documenting the coordinate convention. With normalized coordinates, adjust bandwidth with field extent to keep physical frequency coverage comparable.
- Use development data to establish an adequate bandwidth range; do not treat a best setting at the search boundary as a completed tuning study.
- Use full coverage for both encodings in the controlled encoding comparison.
- Match decoder architecture where feasible. Document feature count, parameter count, precision, optimizer, implementation/backend differences, and any remaining confounds.
- Preserve checkpoints to examine both iteration and wall-time convergence. A fixed iteration budget alone does not establish convergence or practical superiority.
- Evaluate corresponding outputs on identical physical scoring windows within each matched comparison. Do not interpret scores across changing geographic crops as a pure size effect.

Begin with a small representative development subset to resolve configuration and convergence issues, then run the frozen seven-site, three-seed comparison. Expand budgets only to address an observed unresolved convergence question.

### E4b. Partial coverage

Reuse the completed K study if it has the required inputs and logs. Rescore first. Rerun only missing or invalid comparisons.

- Hold the field at LR512 and the update tile at LR128; compare K = 1, 2, 4, 8, and full.
- Specify sampling with/without replacement, permitted overlap, revisit sampling, and loss normalization. Ensure the estimator targets the intended full loss or state its weighting explicitly.
- Ensure sampled tiles include correct blur support. Artificial reflection boundaries at every sampled tile must not silently change the forward model relative to full-field evaluation.
- Check sampled-versus-full degradation and, where relevant, gradients on a controlled small example.
- Record actual observation pixels and coordinate queries, validation cost, time per step, stop iteration, memory, and final quality.
- Derive matched-time and matched-work comparisons from saved traces where possible.

Treat K = 4 as an operating choice, not a universal optimum. Report when K = 2 or full coverage is preferable under a different time/quality objective.

### E4c. Field size and same-area cost

After E2 rescoring, determine whether a controlled field-size rerun is still needed. If so, predeclare a development subset spanning projects and land cover—approximately 10–20 LR512 parents is a starting scope, not a statistical guarantee.

Keep two views separate:

1. Controlled field-size comparison with a shared update policy and stated budget convention.
2. Operational comparison using the chosen policy at each size.

For both, assemble and score on common parent support. Do not describe the operational view as isolating field size alone.

## 9. E5 — Validate LR-only stopping

Use existing or newly saved trajectories on representative easy and difficult development scenes. Compare:

- The current LR hold-out EMA-selected checkpoint.
- Predeclared fixed-budget checkpoints.
- An HR-oracle checkpoint used only for retrospective analysis.

Report HR selection regret: the selected checkpoint's metric relative to the best recorded HR checkpoint on that trajectory, respecting whether higher or lower is better. State checkpoint sampling frequency, since the recorded oracle is not a continuous-time optimum. Pair regret with compute saved, variability across sites/seeds, and LR/HR metric trajectories.

Document validation interval, frame subset, block construction, EMA definition, improvement threshold, patience, minimum/maximum iterations, and restoration of field, alignment, and radiometric parameters.

Verify that withheld observations do not enter the training loss through implementation errors. Explain that independent masks across revisits can supervise the same ground location; this is withheld-observation prediction, not geographic generalization.

The HR oracle must never select deployment outputs, tune on the final test set, or be mixed with LR-selected results in a primary table.

**Deliverables:** Selection-regret/runtime table and representative trajectories. Change the stopping policy only on development data, then freeze it before E6.

## 10. B17 — REQUIRED fresh full benchmark on all 17 locations

**User requirement: once each LR512 tile has its own spatial correction, run the full benchmark on the 17 locations again. Do not stop after estimating corrections, updating metric files, rerunning one example, or rescoring existing predictions.**

### Freeze and enumerate the full matrix

1. Recover the authoritative 17-location inventory and explicitly enumerate all LR512 tiles included in each benchmark cohort. Keep named-site fields and complete-reference fields as distinct labeled cohorts when both are reported.
2. Estimate and QC an independent correction for every LR512 tile in that inventory; freeze the correction registry and common evaluator before the benchmark.
3. Freeze the reconstruction implementation, stopping policy, input packages, seeds, and complete benchmark configuration list after the development fixes and baseline preparation.
4. Include ScaleF's selected configuration, the essential baselines, and every method/configuration retained in the final 17-location benchmark. Run the frozen listed configurations across all 17 locations; do not substitute the earlier seven-site study for this coverage. Explicitly enumerate retained headline ablations in the matrix so that “full benchmark” cannot silently become “B0 only.” Additional exploratory sweeps remain separately scoped development work.
5. Use seeds 6/7/8 for stochastic benchmark configurations unless an explicitly documented final protocol changes them. Run deterministic baselines once per input package. For one named-site LR512 field at each location, the selected ScaleF configuration alone requires **17 × 3 = 51 fresh runs**; this is only one component of the full matrix.

### Execute and replace the evidence

- Run fresh reconstruction/optimization and rendering for the stochastic reconstruction configurations. Do not import old fitted fields as completed B17 runs.
- Regenerate baseline outputs from the frozen inputs and baseline implementations.
- Evaluate every output against its own tile's independently corrected reference, using the shared masks, color protocol, scoring windows, and metric settings.
- Use the same per-tile reference correction across all methods, seeds, and checkpoints. Re-estimate it only if its input pair or protocol changes, in which case invalidate and repeat the affected comparisons.
- Record the full attempted run matrix, input/registration failures, optimization failures, retries, output availability, timings, memory, and quality metrics. Do not silently drop a location or report successful runs as if the matrix were complete.
- Regenerate all affected 17-location tables, pooled summaries, figures, and manuscript statements from the new records. Reconcile seven-site subset summaries against this same new freeze where their configurations match.
- Compare legacy and corrected results only in a clearly labeled diagnostic report. They must not be mixed into a single headline mean or used selectively to preserve an earlier conclusion.

**Acceptance criteria:** The complete frozen benchmark has been attempted on every one of the 17 locations with all specified configurations and seeds; every scored LR512 tile uses its own accepted correction; failures are explicitly accounted for; and the affected paper results have been replaced using the new run records. Any missing jobs remain a visible incomplete/blocked status, not a completed benchmark claim.

The 17 locations remain a development/evaluation cohort despite this rerun. B17 does not turn them into an untouched geographic test set; E6 addresses that separately.

## 11. E6 — Frozen geographic evaluation, uncertainty, and failures

The existing 17 named sites have informed development. Retain that description even after freezing their configuration.

Before inspecting new test outcomes:

1. Select independent projects or regions with a documented overlap check and selection rationale.
2. Freeze reconstruction, stopping, baseline, registration, color, and scoring rules.
3. Predeclare geographic/land-cover coverage, exclusions, and aggregation. Select cohort size using available coverage and development variability; explain limitations rather than inventing a universal minimum.
4. Run the frozen methods and tile-local reference-registration procedure on the new cohort.

Denmark is a valuable transfer setting. Keep shared reconstruction settings frozen from Norway, document country-specific acquisition/reference processing, and disclose any later adaptation. Per-tile estimation using the frozen reference-registration algorithm is evaluation preprocessing, not a license to tune the method on Danish reference scores.

Report paired method differences by tile and independent project/region. Use clustered/hierarchical uncertainty estimates consistent with the sampling design, separating seed variability from geographic variability. Do not count neighboring children or seeds as independent geographic samples.

Account for all attempted fields, including registration rejection, input preparation failures, optimization failures, and reconstruction losses to baseline. Resolve the historical 91 → 86 → 85 counts and explain the two successful complete tiles where ScaleF lost to bilinear.

Select failure examples systematically: worst paired gains, all failed jobs, and predeclared cloud/shadow, vegetation, water/snow, temporal-change, and alignment cases where present. Report insufficient examples if a category is absent.

## 12. E7 — Targeted forward-model and fidelity checks

Use a representative development subset to compare area-only degradation, the nominal band-wise Gaussian MTF approximation, and sigma scaled by 0.8 and 1.2. These perturbations are a sensitivity experiment, not a calibrated sensor uncertainty interval.

Keep inputs, output GSD, optimization policy, tile correction, and scoring fixed. Reuse the existing area-versus-MTF result where compatible. Report whether the conclusion is robust and whether radiometric/LR fidelity and appearance metrics disagree.

Bring PSNR and SSIM into the main reconstruction comparison. Include appropriate Sentinel-2-domain consistency measurements and distinguish them from color-normalized cross-sensor appearance scores.

If the paper claims improved effective resolution, add independent spatial-detail evidence, such as controlled frequency recovery and verified edge/structure comparisons. Synthetic evidence alone does not establish nationwide real-image resolution.

The completed 5 m result needs corrected evaluation if affected by E1. A new 1 m query/train sweep and an expanded loss sweep are optional; they should not delay the evaluator, essential baselines, stopping validation, or map pilot. Never rank different output GSDs by unqualified raw LPIPS comparisons.

## 13. E8 — Contiguous regional pilot

Select and document contiguous regions that exercise neighboring fields, different base frames, difficult observations, and at least one granule boundary where feasible. Run the frozen pipeline from acquisition discovery through final export and mosaicking.

Measure:

| Property | Required reporting |
|---|---|
| Runtime | Catalog, download, filtering, preparation, optimization, validation, rendering, export, mosaicking, retries |
| Throughput | GPU work and elapsed time separately; unique land km² as denominator |
| Memory | Total process/device memory including tiny-cuda-nn; account for other GPU processes when using device-wide counters |
| Reliability | Attempted/completed/failed/retried jobs, reasons, runtime distribution including tail behavior |
| Coverage | Unique geographic union, partial fields, overlaps, gaps, masked/no-output land |
| Provenance | Input dates/IDs, observation count, base frame, configuration and code versions |
| Seams | Geometry and color differences before and after the selected assembly policy |

Evaluate adjacent-field differences in shared overlap where available. Compare boundary error with nearby interior behavior and independent reference where available; natural scene edges must not automatically count as seams. Include a boundary crossing a real structure in qualitative inspection.

Independent fits may inherit different reference-frame geometry and radiometry. Choose the mosaic policy using measured evidence. Feathering can be a delivery operation, but do not treat reduced visible contrast as proof that geometric disagreement is fixed. Report raw and delivered behavior.

Maintain valid geospatial metadata through learned alignment, rendering, and export. Orthophoto-based evaluation correction must not be silently used to improve production geolocation in places without orthophotos.

**Acceptance criteria:** Reproducible regional delivery, explicit coverage/failure accounting, and quantified boundary behavior before national scaling.

## 14. E9 — Complete Norway and Denmark products

After the pilot, freeze each country's acquisition schedule, geographic grid, valid-land definition, overlap ownership, edge policy, reconstruction settings, output format, and provenance fields.

- Produce and assess both countries before claiming two completed national maps.
- Count unique delivered land area; MGRS job counts and observation-availability percentages are not equivalent to this quantity.
- Document gaps and low-observation regions. Keep the current policy of retaining thin stacks unless development evidence motivates a documented change; six observations is not an established quality threshold.
- Report dates actually contributing to each field and describe seasonal reconstructions accordingly.
- Provide country-level coverage and quality summaries, processing/failure statistics, and a way to locate individual fields and their metadata.
- Keep reference-based results limited to areas with suitable reference. Do not extrapolate their measured accuracy to every output pixel.

A regional pilot supports a regional demonstration claim. It does not complete the national-map contribution.

## 15. E10 — Dataset package and external evaluation

Package the Norwegian collection with stable scene/tile manifests, acquisition metadata and temporal offsets, footprints/CRS, masks, independently estimated LR512 corrections and QC, split definitions, preprocessing/configurations, evaluator, baseline outputs where distributable, and versioned metric records.

State which imagery can be redistributed and which requires an external authorized access route. Do not describe unavailable orthophotos as a public download. Document settlement-biased site selection and the broader land cover in complete-reference fields. Danish reference expansion remains optional even though the Danish map product is a project goal.

Add an external MuS2 comparison if accessible. Inspect existing adapters, sensor scale/band conventions, masks, radiometric treatment, and evaluator requirements before execution. Keep official-comparable results separate from adaptations and NIB scores. Do not copy a fourfold RGB protocol into another benchmark without validating its task definition.

Explain the dataset's relationship to SEN2NAIP and the evaluation protocol's relationship to OpenSR-test. Their existence does not remove the value of Norwegian real-reference MISR data; state the concrete differences and added coverage.

## 16. Run records, reporting, and completion

Use a consistent machine-readable record with, at minimum:

`experiment_id`, `run_id`, `status`, `failure_reason`, `code_commit`, `dependency_versions`, `config_hash`, `scene_id`, `project_id`, `country`, `parent_lr512_id`, `training_field_size`, `seed`, `input_manifest_hash`, `base_frame_id`, `reference_id`, `alignment_artifact_hash`, `evaluator_version`, `mask_hash`, `checkpoint_selection`, `stop_iteration`, `metrics`, `timing_stages`, `peak_total_gpu_memory`, and output/trajectory artifact locations.

Adapt field names to the repository's existing schema rather than introducing a competing system unnecessarily. Mark unavailable measurements explicitly; do not replace them with zero or infer them from unrelated runs.

Generate tables and figures from versioned raw records. Include cohort sizes, exclusions, evaluator/configuration versions, metric direction, uncertainty, and measured-versus-extrapolated cost labels. Record both complete-case comparisons and failures so successful-run averages cannot hide operational failure.

If compute, credentials, or data are unavailable, finish the relevant implementation and local validation, export the exact remaining run commands and prerequisites, and label experiments as not executed. Do not invent script paths that have not been implemented or verified, and do not present an execution plan as completed evidence.

### Completion checklist

- [ ] Every scored LR512 tile has an independently estimated, accepted local reference correction.
- [ ] No site/granule/project correction is extrapolated to other LR512 tiles.
- [ ] Methods and nested variants share the same fixed evaluator within each LR512 parent.
- [ ] Common-footprint baseline-invariance check passes.
- [ ] The full benchmark has been freshly rerun on all 17 locations, including the frozen method/configuration matrix and all specified seeds.
- [ ] All affected 17-location paper results are replaced from the new run records; historical rescoring is not counted as completion.
- [ ] Other supporting outputs are reused/rescored where appropriate, with reasons for additional reruns.
- [ ] Actual dense/hashed behavior and total representation capacity are documented.
- [ ] Tiled original SuperF and registered multi-frame fusion are evaluated on matched outputs, or explicit blockers limit claims.
- [ ] Encoding bandwidth, update support, budgets, and convergence confounds are addressed.
- [ ] LR checkpoint selection is validated without using HR to select deployment outputs.
- [ ] A frozen new-geography evaluation and appropriate uncertainty/failure reporting are complete.
- [ ] A contiguous pilot reports total memory, end-to-end cost, seams, and failures.
- [ ] Both national products are delivered and assessed before claiming completion of that contribution.
- [ ] The dataset/evaluator package and reproducible paper results are available with documented access conditions.

### Final handoff from the implementing agent

Provide a short report listing changes made; experiments reused/rescored/rerun; exact code/data/evaluator versions; principal findings including negative results; artifact locations; remaining blockers; and which manuscript claims are now supported. Update affected paper tables and figures only from completed, traceable results.

### Background references for implementation inspection

- tiny-cuda-nn grid allocation: https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/grid.h
- tiny-cuda-nn grid indexing: https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/common_device.h
- SEN2NAIP: https://www.nature.com/articles/s41597-024-04214-y
- OpenSR-test: https://esaopensr.github.io/opensr-test/
- MuS2: https://www.nature.com/articles/s41597-023-02538-9

Inspect the project's pinned dependency versions rather than assuming the moving upstream `master` links describe the version used in existing experiments.
