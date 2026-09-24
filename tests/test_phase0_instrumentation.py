import argparse
from pathlib import Path

import numpy as np
import pytest
import torch

from optimize import (
    _json_safe,
    _model_parameter_counts,
    _summarize_training_instrumentation,
    _training_batch_query_counts,
)


def test_json_safe_converts_namespace_and_non_json_values():
    value = argparse.Namespace(
        path=Path("data/example"),
        scalar=np.int64(7),
        device=torch.device("cpu"),
        nested=("x", np.float32(1.5)),
    )

    assert _json_safe(value) == {
        "path": "data/example",
        "scalar": 7,
        "device": "cpu",
        "nested": ["x", pytest.approx(1.5)],
    }


def test_training_batch_query_counts_uses_actual_tensor_shapes():
    coords = torch.empty(3, 16, 20, 2)
    lr_target = torch.empty(3, 4, 5, 4)

    assert _training_batch_query_counts(coords, lr_target, 3) == {
        "lr_pixels": 60,
        "lr_elements": 240,
        "hr_pixels": 960,
        "hr_elements": 2880,
    }


def test_training_instrumentation_summary_reports_rates_and_averages():
    summary = _summarize_training_instrumentation(
        4,
        2.0,
        {"lr_pixels": 40, "lr_elements": 120, "hr_pixels": 640, "hr_elements": 1920},
    )

    assert summary["average_step_time_seconds"] == pytest.approx(0.5)
    assert summary["iterations_per_second"] == pytest.approx(2.0)
    assert summary["queried"]["average_per_step"]["lr_pixels"] == pytest.approx(10.0)
    assert summary["queried"]["average_per_step"]["hr_elements"] == pytest.approx(480.0)


def test_training_instrumentation_summary_handles_no_steps():
    summary = _summarize_training_instrumentation(0, 0.0, {"lr_pixels": 0})

    assert summary["average_step_time_seconds"] is None
    assert summary["iterations_per_second"] is None
    assert summary["queried"]["average_per_step"]["lr_pixels"] is None


def test_model_parameter_counts_distinguishes_frozen_parameters():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    model[1].weight.requires_grad_(False)

    assert _model_parameter_counts(model) == {"total": 13, "trainable": 10}
