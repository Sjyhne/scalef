Real / exported Sentinel-2 stacks (satburst-compatible layout)
================================================================

Each scene is a folder under this directory, with a nested run folder, e.g.::

  <scene_name>/scale_4_shift_1.0px_aug_none/

containing ``transform_log.json``, ``hr_ground_truth.png``, and LR ``sample_*.png``
(or NPZ paths when using raw reflectance).

Important: different scene folders may use **different LR and HR spatial sizes**
(resolution is read per scene from ``transform_log.json`` and ``hr_ground_truth.png``).
Cross-scene PSNR/LPIPS aggregates are therefore **not** fixed-geometry comparisons like a
single synthetic grid; treat them as per-patch metrics grouped for convenience.

Run the experiment suite on this root::

  python scripts/run_experiment_suite.py --dataset satburst_real

Or explicitly::

  python scripts/run_experiment_suite.py --dataset satburst_synth --satburst_data_root data_real

Only scenes that contain the requested ``scale_<df>_shift_<lr_shift>px_aug_<aug>/`` folder
are discovered (defaults match ``--df``, ``--lr_shift``, ``--aug`` on the suite).

Results are written under ``single_samples/satburst_real/<scene>/<run_name>/`` when using
``--dataset satburst_real``.

HashGrid ``--hash_base_resolution`` / ``--hash_max_resolution`` (and other hash flags) come from
``optimize.py`` defaults or whatever you pass on the command line; the suite uses fixed values in
``scripts/run_experiment_suite.py`` for the HashGrid experiment.

Large LR patches imply a larger HR coordinate grid (``df`` upsampling). **CUDA allocator fragmentation**
or **another process on the same device** can still cause failures when many ``optimize.py`` runs
start in sequence—even on high-VRAM GPUs.

The experiment suite **does not** enable ``expandable_segments`` unless you pass
``--cuda_expandable_segments`` (some stacks error on the first CUDA allocation with that setting).
You can also set ``PYTORCH_CUDA_ALLOC_CONF`` yourself before launching the suite.

Optional: ``--max_lr_side`` skips scenes whose reference LR max(H,W) in ``transform_log.json`` exceeds
the threshold (for shorter sweeps / size ablations), not as a VRAM requirement.
