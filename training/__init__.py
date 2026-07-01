"""Training loop, model factory, and dataset path helpers."""

from training.factory import (
    build_inr_model,
    build_input_projection_decoder_bundle,
    resolve_decoder_output_dim,
    resolve_hash_grid_resolutions,
    resolve_hash_max_resolution,
)
from training.loop import train_one_iteration
from training.paths import (
    WORLDSTRAT_DATASETS,
    discover_worldstrat_sample_ids,
    resolve_satburst_data_root,
    resolve_worldstrat_data_root,
    satburst_scene_dir,
    single_sample_output_dir,
)
from training.run_loop import TrainingHistory, run_training_loop

__all__ = [
    "WORLDSTRAT_DATASETS",
    "TrainingHistory",
    "build_inr_model",
    "build_input_projection_decoder_bundle",
    "discover_worldstrat_sample_ids",
    "resolve_decoder_output_dim",
    "resolve_hash_grid_resolutions",
    "resolve_hash_max_resolution",
    "resolve_satburst_data_root",
    "resolve_worldstrat_data_root",
    "run_training_loop",
    "satburst_scene_dir",
    "single_sample_output_dir",
    "train_one_iteration",
]
