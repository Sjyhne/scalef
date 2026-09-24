#!/usr/bin/env python3
"""Build the CPU-only, reproducible paper-result export.

Input JSON files are treated as accepted snapshots: they are validated, copied
byte-for-byte to ``paper/results``, recorded in a SHA256 manifest, and rendered
to the explicitly generated ``ScaleF_Overleaf/generated`` tree.

Examples
--------
    python scripts/build_paper_results.py --dry-run
    python scripts/build_paper_results.py --allow-partial
    python scripts/build_paper_results.py --input path/to/accepted_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = ROOT / "single_samples" / "sweep_results"
DEFAULT_RESULTS_DIR = ROOT / "paper" / "results"
DEFAULT_OVERLEAF_DIR = ROOT / "ScaleF_Overleaf"

CORE_INPUTS = (
    "paper_ablations.json",
    "paper_fourier_fair.json",
    "paper_fourier_fair_s2.json",
    "bench_complete_patches_size_ladder_nested_common_footprint.v2.json",
    "bench_complete_patches_nest_lr64_lr512align_v2.json",
    "bench_complete_patches_nest_lr128_lr512align_v2.json",
    "bench_complete_patches_nest_lr256_lr512align_v2.json",
    "bench_complete_patches_nest_lr512_lr512align_v2.json",
)
OPTIONAL_PATTERNS = (
    "fixed_k_summary.json",
    "encoding_size_summary.json",
    "gsd_ladder_summary.json",
    "b0_17_lr512align_v2_summary.json",
)
PROVENANCE_INPUTS = (
    (ROOT / "eval" / "confirmatory_eval_manifest.v2.json", "confirmatory_eval_manifest.v2.json"),
    (ROOT / "eval" / "spatial_alignment.json", "spatial_alignment.json"),
)
SOFTWARE_FREEZE = {
    "run_base_git_commit": "4438ba37433853533779ef1ea0c665ad124d395b",
    "run_git_dirty": True,
    "python": "3.11.15",
    "pytorch": "2.12.1+cu130",
    "cuda": "13.0",
    "tinycudann": "2.0",
    "gpu": "NVIDIA H100 80GB HBM3",
}
SOURCE_FILES = (
    ROOT / "optimize.py",
    ROOT / "s2_dataset.py",
    ROOT / "data.py",
    ROOT / "models" / "inr.py",
    ROOT / "training" / "factory.py",
)
METRICS = ("lpips", "psnr", "ssim", "training_time_s")
LATEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


class ValidationError(ValueError):
    """A summary cannot safely be promoted to the paper freeze."""


@dataclass
class InputArtifact:
    source: Path
    name: str
    data: dict[str, Any]
    kind: str
    role: str
    issues: list[str] = field(default_factory=list)
    partial: bool = False
    source_bytes: bytes = b""


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_escape(value: object) -> str:
    """Escape text for a normal LaTeX cell (not math mode)."""
    return "".join(LATEX_REPLACEMENTS.get(char, char) for char in str(value))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_keys(mapping: dict[str, Any], keys: Iterable[str], context: str) -> list[str]:
    return [f"{context}: missing {key}" for key in keys if key not in mapping]


def _rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "rows", "table"):
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def classify(name: str, data: dict[str, Any]) -> tuple[str, str]:
    low = name.lower()
    family = str(data.get("family", "")).lower()
    if "size_ladder_nested" in low or (
        data.get("mode") == "nested_from_lr512" and isinstance(data.get("table"), list)
    ):
        return "nested_ladder", "Nested same-footprint AOI-size summary"
    if "nest_lr" in low and isinstance(data.get("rows"), list):
        return "nested_rows", "Per-tile nested same-footprint runs"
    if "fourier" in low or any("F_aoi" in str(row.get("variant", "")) for row in _rows(data)):
        return "fourier", "Matched Fourier/hash encoding-by-scale summary"
    if "ablation" in low:
        return "ablations", "Current paper ablation summary"
    if re.search(r"fixed[-_]?k", low) or family == "fixed_k":
        return "fixed_k", "Fixed-tile K/coverage confirmatory summary"
    if "encoding" in low or family in {"encoding", "encoding_size"}:
        return "encoding", "Encoding confirmatory summary"
    return "generic", "Accepted confirmatory summary"


def _explicit_partial(data: dict[str, Any]) -> bool:
    status = str(data.get("status", "")).lower()
    return data.get("partial") is True or status in {"partial", "incomplete"}


def _metric(row: dict[str, Any], key: str) -> Any:
    """Read a metric from flat or confirmatory endpoint-oriented rows."""
    if key in row:
        return row[key]
    metrics = row.get("metrics")
    if isinstance(metrics, dict) and key in metrics:
        return metrics[key]
    for endpoint in ("stopped", "fixed_5000", "fixed_2000"):
        block = row.get(endpoint)
        if isinstance(block, dict) and key in block:
            return block[key]
    return None


def _validate_metric_rows(rows: list[dict[str, Any]], id_keys: tuple[str, ...]) -> list[str]:
    issues: list[str] = []
    if not rows:
        return ["summary has no result rows"]
    for index, row in enumerate(rows):
        context = f"row {index}"
        if not any(row.get(key) not in (None, "") for key in id_keys):
            issues.append(f"{context}: missing identifier ({'/'.join(id_keys)})")
        if row.get("error"):
            issues.append(f"{context}: source reports an error")
            continue
        if not _is_number(_metric(row, "lpips")):
            issues.append(f"{context}: missing/non-finite lpips")
        if not _is_number(_metric(row, "training_time_s")):
            issues.append(f"{context}: missing/non-finite training_time_s")
    return issues


def validate_artifact(path: Path) -> InputArtifact:
    try:
        content = path.read_bytes()
        data = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"{path}: top-level JSON must be an object")

    kind, role = classify(path.name, data)
    issues: list[str] = []
    rows = _rows(data)
    status = str(data.get("status", "")).lower()
    if status in {"rejected", "exploratory", "draft"} or data.get("accepted") is False:
        raise ValidationError(f"{path}: status={status or 'not accepted'} cannot enter paper freeze")
    if kind in {"ablations", "fourier"}:
        issues.extend(_require_keys(data, ("cities", "variants", "results"), path.name))
        issues.extend(_validate_metric_rows(rows, ("variant",)))
        for index, row in enumerate(rows):
            issues.extend(_require_keys(row, ("city", "variant"), f"row {index}"))
    elif kind == "nested_ladder":
        issues.extend(_require_keys(data, ("mode", "table", "by_size"), path.name))
        seen: set[int] = set()
        for index, row in enumerate(rows):
            issues.extend(
                _require_keys(
                    row,
                    (
                        "lr_side",
                        "n_projects",
                        "n_parents",
                        "n_tiles",
                        "mean_lpips_project",
                        "mean_lpips_bilinear_project",
                    ),
                    f"row {index}",
                )
            )
            if isinstance(row.get("lr_side"), int):
                seen.add(row["lr_side"])
        missing = {64, 128, 256, 512} - seen
        if missing:
            issues.append(f"missing nested LR sides: {sorted(missing)}")
    elif kind == "nested_rows":
        issues.extend(_require_keys(data, ("side", "n_tiles", "rows", "config"), path.name))
        issues.extend(_validate_metric_rows(rows, ("tile_id",)))
        expected = data.get("n_tiles")
        if isinstance(expected, int) and len(rows) != expected:
            issues.append(f"rows={len(rows)} but n_tiles={expected}")
        for index, row in enumerate(rows):
            required = ("tile_id", "parent_city") if row.get("error") else (
                "tile_id",
                "parent_city",
                "metrics_path",
            )
            issues.extend(_require_keys(row, required, f"row {index}"))
    else:
        issues.extend(_validate_metric_rows(rows, ("variant", "run", "label", "config")))

    # Duplicate experimental cells almost always indicate accidental concatenation.
    keys: set[tuple[Any, ...]] = set()
    for row in rows:
        key = tuple(
            row.get(k)
            for k in ("city", "variant", "config", "seed", "tile_id", "run", "run_name")
        )
        if any(value is not None for value in key):
            if key in keys:
                issues.append(f"duplicate result cell: {key}")
            keys.add(key)

    return InputArtifact(
        source=path.resolve(),
        name=path.name,
        data=data,
        kind=kind,
        role=role,
        issues=sorted(set(issues)),
        partial=_explicit_partial(data) or bool(issues),
        source_bytes=content,
    )


def validate_collection(artifacts: list[InputArtifact], allow_partial: bool) -> None:
    names = [artifact.name for artifact in artifacts]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValidationError(f"duplicate canonical filenames: {', '.join(duplicates)}")

    ablation = next((a for a in artifacts if a.kind == "ablations"), None)
    if ablation:
        present = {
            (row.get("city"), row.get("variant"))
            for row in _rows(ablation.data)
            if not row.get("error")
        }
        cities = [str(city) for city in ablation.data.get("cities", [])]
        for variant in ablation.data.get("variants", []):
            expected_cities = ["asker"] if str(variant).startswith("S_aoi") else cities
            for city in expected_cities:
                if (city, variant) not in present:
                    ablation.issues.append(f"missing declared cell: {city}/{variant}")

    fourier = [a for a in artifacts if a.kind == "fourier"]
    if fourier:
        variants = {
            str(row.get("variant"))
            for artifact in fourier
            for row in _rows(artifact.data)
            if not row.get("error")
        }
        expected = {
            *(f"H_aoi{side}" for side in (64, 128, 256, 512)),
            *(
                f"F_aoi{side}_s{scale}"
                for side in (64, 128, 256, 512)
                for scale in (2, 5, 10)
            ),
        }
        missing = sorted(expected - variants)
        if missing:
            fourier[0].issues.append(f"missing canonical Fourier/hash cells: {missing}")

    expected_configs = {
        "fixed_k": {
            "lr128_k1",
            "lr128_k2",
            "lr128_k4",
            "lr128_k8",
            "lr128_full",
        },
        "encoding": {
            *(f"lr{side}_hash" for side in (64, 128, 256, 512)),
            *(
                f"lr{side}_fourier_s{scale}"
                for side in (64, 128, 256, 512)
                for scale in (2, 5, 10)
            ),
        },
    }
    for artifact in artifacts:
        if artifact.kind in expected_configs:
            present = {
                str(row.get("config") or row.get("variant") or row.get("run"))
                for row in _rows(artifact.data)
                if not row.get("error")
            }
            missing = sorted(expected_configs[artifact.kind] - present)
            if missing:
                artifact.issues.append(f"missing canonical configurations: {missing}")

    for artifact in artifacts:
        artifact.issues = sorted(set(artifact.issues))
        artifact.partial = artifact.partial or bool(artifact.issues)
    partial = [artifact for artifact in artifacts if artifact.partial]
    if partial and not allow_partial:
        details = "\n".join(
            f"  {artifact.name}: {'; '.join(artifact.issues) or 'marked partial'}"
            for artifact in partial
        )
        raise ValidationError(
            "partial/incomplete summaries require --allow-partial:\n" + details
        )


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if _is_number(value) else r"\textemdash"


def _partial_banner(partial: bool) -> str:
    return (
        "% PARTIAL RESULT: generated with --allow-partial; do not cite as complete.\n"
        r"\noindent\colorbox{yellow!25}{\strut\textbf{PARTIAL RESULT}}\par"
        "\n"
        if partial
        else ""
    )


def _table_wrapper(label: str, caption: str, header: str, body: list[str], partial: bool) -> str:
    status = " [PARTIAL]" if partial else ""
    return (
        "% AUTO-GENERATED by scripts/build_paper_results.py; DO NOT EDIT.\n"
        + _partial_banner(partial)
        + "\\begin{table}[t]\n"
        + "  \\centering\n"
        + f"  \\caption{{{caption}{status}}}\n"
        + f"  \\label{{{label}}}\n"
        + "  \\footnotesize\n"
        + f"  \\begin{{tabular}}{{{header.split('|', 1)[0]}}}\n"
        + "    \\toprule\n"
        + f"    {header.split('|', 1)[1]} \\\\\n"
        + "    \\midrule\n"
        + "\n".join(f"    {line} \\\\" for line in body)
        + "\n    \\bottomrule\n"
        + "  \\end{tabular}\n"
        + "\\end{table}\n"
    )


def aggregate_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not row.get("error") and _is_number(_metric(row, "lpips")):
            identifier = row.get("variant") or row.get("config") or row.get("run") or row.get("label")
            grouped.setdefault(str(identifier or "unknown"), []).append(row)
    result = []
    for variant, group in sorted(grouped.items()):
        result.append(
            {
                "variant": variant,
                "n": len(group),
                **{
                    metric: mean(_metric(row, metric) for row in group if _is_number(_metric(row, metric)))
                    if any(_is_number(_metric(row, metric)) for row in group)
                    else None
                    for metric in METRICS
                },
                "lpips_bilinear": mean(
                    _metric(row, "lpips_bilinear")
                    for row in group
                    if _is_number(_metric(row, "lpips_bilinear"))
                )
                if any(_is_number(_metric(row, "lpips_bilinear")) for row in group)
                else None,
            }
        )
    return result


def render_ablations(artifact: InputArtifact) -> str:
    body = [
        " & ".join(
            (
                latex_escape(row["variant"]),
                str(row["n"]),
                _fmt(row["lpips"]),
                _fmt(row["lpips_bilinear"]),
                _fmt(row["training_time_s"], 1),
            )
        )
        for row in aggregate_variants(_rows(artifact.data))
    ]
    return _table_wrapper(
        "tab:generated-ablations",
        "Validated paper ablations (means over available sites).",
        "lrrrr|Variant & $n$ & LPIPS $\\downarrow$ & Bilinear & Time (s)",
        body,
        artifact.partial,
    )


def render_fourier(artifacts: list[InputArtifact]) -> str:
    rows = [row for artifact in artifacts for row in _rows(artifact.data) if not row.get("error")]
    by_variant = {str(row.get("variant")): row for row in rows}
    body: list[str] = []
    missing = False
    for side in (64, 128, 256, 512):
        fourier = [
            by_variant.get(f"F_aoi{side}_s{scale}") for scale in (2, 5, 10)
        ]
        fourier = [row for row in fourier if row and _is_number(row.get("lpips"))]
        hash_row = by_variant.get(f"H_aoi{side}")
        if not fourier or not hash_row:
            missing = True
        best = min(fourier, key=lambda row: row["lpips"]) if fourier else {}
        scale_match = re.search(r"_s(\d+)$", str(best.get("variant", "")))
        body.append(
            " & ".join(
                (
                    str(side),
                    _fmt(best.get("lpips")),
                    scale_match.group(1) if scale_match else r"\textemdash",
                    _fmt((hash_row or {}).get("lpips")),
                    _fmt((hash_row or best).get("lpips_bilinear")),
                    _fmt(best.get("training_time_s"), 1),
                    _fmt((hash_row or {}).get("training_time_s"), 1),
                )
            )
        )
    partial = any(artifact.partial for artifact in artifacts) or missing
    return _table_wrapper(
        "tab:generated-fair-fourier",
        "Matched full-field Fourier and hash results by LR field side.",
        "rrrrrrr|LR & Best $F$ & Scale & Hash & Bilinear & $t_F$ & $t_H$",
        body,
        partial,
    )


def _nested_times(nested_rows: list[InputArtifact]) -> dict[int, float]:
    values: dict[int, float] = {}
    for artifact in nested_rows:
        side = artifact.data.get("side")
        times = [
            row["training_time_s"]
            for row in _rows(artifact.data)
            if not row.get("error") and _is_number(row.get("training_time_s"))
        ]
        if isinstance(side, int) and times:
            values[side] = mean(times)
    return values


def render_nested(summary: InputArtifact, nested_rows: list[InputArtifact]) -> str:
    times = _nested_times(nested_rows)
    body = []
    for row in sorted(_rows(summary.data), key=lambda item: item.get("lr_side", 0)):
        side = row.get("lr_side")
        tile_time = times.get(side)
        serial_time = tile_time * (512 // side) ** 2 if tile_time is not None and side else None
        gpu_s_km2 = serial_time / 26.2144 if serial_time is not None else None
        km2_gpu_h = 3600.0 / gpu_s_km2 if gpu_s_km2 else None
        body.append(
            " & ".join(
                (
                    str(side),
                    str(row.get("n_tiles", r"\textemdash")),
                    _fmt(row.get("mean_lpips_project")),
                    _fmt(row.get("mean_lpips_bilinear_project")),
                    _fmt(gpu_s_km2, 1),
                    _fmt(km2_gpu_h, 0),
                )
            )
        )
    partial = summary.partial or any(artifact.partial for artifact in nested_rows)
    return _table_wrapper(
        "tab:generated-nested-ladder",
        "Nested common-footprint quality and projected optimization throughput.",
        "rrrrrr|LR & $n$ windows & LPIPS $\\downarrow$ & Bilinear & GPU-s/km$^2$ & km$^2$/GPU-h",
        body,
        partial,
    )


def render_generic(artifact: InputArtifact, label: str) -> str:
    body = []
    for row in aggregate_variants(_rows(artifact.data)):
        body.append(
            " & ".join(
                (
                    latex_escape(row["variant"]),
                    str(row["n"]),
                    _fmt(row["lpips"]),
                    _fmt(row["psnr"], 2),
                    _fmt(row["ssim"]),
                    _fmt(row["training_time_s"], 1),
                )
            )
        )
    return _table_wrapper(
        f"tab:generated-{label}",
        latex_escape(artifact.role) + ".",
        "lrrrrr|Configuration & $n$ & LPIPS & PSNR & SSIM & Time (s)",
        body,
        artifact.partial,
    )


def _save_quality_compute_plot(
    summary: InputArtifact,
    nested_rows: list[InputArtifact],
    destination: Path,
    partial: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = _nested_times(nested_rows)
    points = []
    for row in _rows(summary.data):
        side = row.get("lr_side")
        if side in times and _is_number(row.get("mean_lpips_project")):
            points.append((times[side] * (512 // side) ** 2, row["mean_lpips_project"], side))
    if not points:
        raise ValidationError("same-area plot: no nested quality/time pairs")
    figure, axis = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    axis.plot([p[0] for p in points], [p[1] for p in points], "o-", color="#2455a4")
    for compute, quality, side in points:
        axis.annotate(f"LR{side}", (compute, quality), xytext=(5, 4), textcoords="offset points")
    axis.set_xlabel("Serial GPU-seconds per LR512 footprint")
    axis.set_ylabel("Project-mean LPIPS (lower is better)")
    axis.grid(alpha=0.25)
    if partial:
        axis.text(
            0.5,
            0.5,
            "PARTIAL",
            transform=axis.transAxes,
            fontsize=34,
            alpha=0.16,
            ha="center",
            va="center",
            rotation=25,
        )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _run_image(row: dict[str, Any], root: Path, filename: str) -> Path:
    metrics = row.get("metrics_path")
    if not isinstance(metrics, str):
        raise ValidationError(f"{row.get('tile_id')}: missing metrics_path")
    return root / metrics.rsplit("/", 1)[0] / filename


def _select_spot_rows(nested_rows: list[InputArtifact]) -> tuple[dict[int, dict[str, Any]], int, int]:
    by_side = {int(a.data["side"]): _rows(a.data) for a in nested_rows if isinstance(a.data.get("side"), int)}
    if 64 not in by_side:
        raise ValidationError("nested spot panel requires LR64 rows")
    indexes = {
        side: {
            (
                row.get("parent_city"),
                row.get("patch_row"),
                row.get("patch_col"),
                _nested_index(row.get("tile_id"), "y"),
                _nested_index(row.get("tile_id"), "x"),
            ): row
            for row in rows
            if not row.get("error")
        }
        for side, rows in by_side.items()
    }
    for row64 in by_side[64]:
        if row64.get("error"):
            continue
        y64 = _nested_index(row64.get("tile_id"), "y")
        x64 = _nested_index(row64.get("tile_id"), "x")
        if y64 is None or x64 is None:
            continue
        base = (row64.get("parent_city"), row64.get("patch_row"), row64.get("patch_col"))
        selected = {64: row64}
        for side in (128, 256):
            key = (*base, (y64 * 64) // side, (x64 * 64) // side)
            if key in indexes.get(side, {}):
                selected[side] = indexes[side][key]
        candidates512 = [
            row
            for row in by_side.get(512, [])
            if (row.get("parent_city"), row.get("patch_row"), row.get("patch_col")) == base
            and not row.get("error")
        ]
        if candidates512:
            selected[512] = candidates512[0]
        if set(selected) == {64, 128, 256, 512}:
            return selected, y64 * 64, x64 * 64
    raise ValidationError("could not find one complete geographic spot across LR64/128/256/512")


def _nested_index(tile_id: Any, axis: str) -> int | None:
    match = re.search(rf"_{axis}(\d+)(?:_|$)", str(tile_id))
    return int(match.group(1)) if match else None


def _save_spot_panel(
    nested_rows: list[InputArtifact], root: Path, destination: Path, partial: bool
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    selected, global_y, global_x = _select_spot_rows(nested_rows)
    panels: list[tuple[str, Image.Image]] = []
    row64 = selected[64]
    for label, filename in (
        ("Ground truth", "ground_truth.png"),
        ("Bilinear", "bilinear_baseline.png"),
    ):
        panels.append((label, Image.open(_run_image(row64, root, filename)).convert("RGB")))
    for side in (64, 128, 256, 512):
        image = Image.open(_run_image(selected[side], root, "model_output_aligned.png")).convert("RGB")
        offset_y = (global_y % side) * 4
        offset_x = (global_x % side) * 4
        panels.append(
            (f"ScaleF LR{side}", image.crop((offset_x, offset_y, offset_x + 256, offset_y + 256)))
        )
    size = 256
    title_h = 32
    banner_h = 20 if partial else 0
    canvas = Image.new("RGB", (len(panels) * size, size + title_h + banner_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if partial:
        draw.rectangle((0, 0, canvas.width, banner_h), fill="#ffe58a")
        draw.text((8, 5), "PARTIAL RESULT", fill="#8a4b00", font=font)
    for index, (label, image) in enumerate(panels):
        canvas.paste(
            image.resize((size, size), Image.Resampling.LANCZOS),
            (index * size, title_h + banner_h),
        )
        draw.text((index * size + 8, banner_h + 10), label, fill="black", font=font)
    canvas.save(destination)


def _write_bytes(path: Path, content: bytes, dry_run: bool) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _write_text(path: Path, content: str, dry_run: bool) -> None:
    _write_bytes(path, content.encode("utf-8"), dry_run)


def build_manifest(
    artifacts: list[InputArtifact],
    generated: list[Path],
    results_dir: Path,
    root: Path,
    partial: bool,
    provenance: list[tuple[Path, Path]] | None = None,
) -> dict[str, Any]:
    def relative(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path.resolve())

    files = []
    for artifact in artifacts:
        canonical = results_dir / artifact.name
        files.append(
            {
                "name": artifact.name,
                "role": artifact.role,
                "kind": artifact.kind,
                "status": "partial" if artifact.partial else "complete",
                "validation_issues": artifact.issues,
                "source": relative(artifact.source),
                "source_bytes": len(artifact.source_bytes),
                "source_sha256": sha256_bytes(artifact.source_bytes),
                "source_mtime_utc": datetime.fromtimestamp(
                    artifact.source.stat().st_mtime, timezone.utc
                ).isoformat(),
                "canonical": relative(canonical),
                "canonical_sha256": sha256_bytes(artifact.source_bytes),
            }
        )
    generated_files = [
        {
            "path": relative(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in generated
        if path.is_file()
    ]
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial" if partial else "complete",
        "purpose": "Validated canonical paper results and CPU-generated exports.",
        "generator": "scripts/build_paper_results.py",
        "software": {
            **SOFTWARE_FREEZE,
            "source_files": {
                relative(path): sha256_file(path) for path in SOURCE_FILES if path.is_file()
            },
        },
        "files": files,
        "provenance": [
            {
                "source": relative(source),
                "canonical": relative(canonical),
                "sha256": sha256_file(canonical),
                "bytes": canonical.stat().st_size,
            }
            for source, canonical in (provenance or [])
            if canonical.is_file()
        ],
        "generated": generated_files,
    }


def discover_inputs(source_dir: Path) -> list[Path]:
    paths = [source_dir / name for name in CORE_INPUTS if (source_dir / name).is_file()]
    for pattern in OPTIONAL_PATTERNS:
        paths.extend(sorted(source_dir.glob(pattern)))
    return list(dict.fromkeys(path.resolve() for path in paths))


def run_pipeline(
    inputs: list[Path],
    *,
    root: Path,
    results_dir: Path,
    overleaf_dir: Path,
    allow_partial: bool,
    dry_run: bool,
) -> list[Path]:
    if not inputs:
        raise ValidationError("no input summaries found; pass --input")
    artifacts = [validate_artifact(path) for path in inputs]
    validate_collection(artifacts, allow_partial)
    partial = any(artifact.partial for artifact in artifacts)
    generated_root = overleaf_dir / "generated"
    tables_dir = generated_root / "tables"
    figures_dir = generated_root / "figures"
    outputs: list[Path] = []
    provenance: list[tuple[Path, Path]] = []

    for artifact in artifacts:
        target = results_dir / artifact.name
        _write_bytes(target, artifact.source_bytes, dry_run)
        outputs.append(target)
    for source, name in PROVENANCE_INPUTS:
        if source.is_file():
            target = results_dir / name
            _write_bytes(target, source.read_bytes(), dry_run)
            provenance.append((source, target))
            outputs.append(target)

    ablation = next((a for a in artifacts if a.kind == "ablations"), None)
    fourier = [a for a in artifacts if a.kind == "fourier"]
    nested_summary = next((a for a in artifacts if a.kind == "nested_ladder"), None)
    nested_rows = [a for a in artifacts if a.kind == "nested_rows"]

    table_specs: list[tuple[str, str]] = []
    if ablation:
        table_specs.append(("paper_ablations.tex", render_ablations(ablation)))
    if fourier:
        table_specs.append(("fair_fourier.tex", render_fourier(fourier)))
    if nested_summary:
        table_specs.append(("nested_size_ladder.tex", render_nested(nested_summary, nested_rows)))
    for artifact in artifacts:
        if artifact.kind in {"fixed_k", "encoding", "generic"}:
            label = re.sub(r"[^a-z0-9]+", "-", artifact.source.stem.lower()).strip("-")
            table_specs.append((f"{artifact.source.stem}.tex", render_generic(artifact, label)))
    for name, content in table_specs:
        path = tables_dir / name
        _write_text(path, content, dry_run)
        outputs.append(path)

    generated_readme = generated_root / "README.md"
    _write_text(
        generated_readme,
        "# Generated paper artifacts\n\n"
        "Everything below this directory is generated by "
        "`scripts/build_paper_results.py`; do not hand-edit it. Handwritten "
        "tables and figures remain in `../tables/` and `../figures/`.\n",
        dry_run,
    )
    outputs.append(generated_readme)

    if nested_summary and len(nested_rows) >= 4:
        quality_plot = figures_dir / "same_area_quality_compute.png"
        spot_panel = figures_dir / "nested_same_geographic_64_spot.png"
        if not dry_run:
            figures_dir.mkdir(parents=True, exist_ok=True)
            _save_quality_compute_plot(nested_summary, nested_rows, quality_plot, partial)
            _save_spot_panel(nested_rows, root, spot_panel, partial)
        outputs.extend((quality_plot, spot_panel))

    # Manifest is written last so generated checksums describe final files.
    manifest_path = results_dir / "MANIFEST.json"
    if not dry_run:
        manifest = build_manifest(
            artifacts, outputs, results_dir, root, partial, provenance=provenance
        )
        _write_text(manifest_path, json.dumps(manifest, indent=2) + "\n", False)
    outputs.append(manifest_path)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, dest="inputs")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--overleaf-dir", type=Path, default=DEFAULT_OVERLEAF_DIR)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    inputs = [path.resolve() for path in args.inputs] if args.inputs else discover_inputs(args.source_dir)
    try:
        outputs = run_pipeline(
            inputs,
            root=ROOT,
            results_dir=args.results_dir,
            overleaf_dir=args.overleaf_dir,
            allow_partial=args.allow_partial,
            dry_run=args.dry_run,
        )
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    verb = "Would write" if args.dry_run else "Wrote"
    print(f"{verb} {len(outputs)} artifacts:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
