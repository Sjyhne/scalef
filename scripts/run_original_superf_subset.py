#!/usr/bin/env python3
"""Train original SuperF on one interior LR64 window per Norwegian site.

The window is the center cell of the middle nested parent, not a cell chosen
by score. The extension result is the existing no-hold-out 5,000-step fit on
that same window. Original SuperF keeps its published defaults (2,000 steps,
seed 6). Fourier scale 10 is the published default; other scales write under
fourier_s<scale>/ and reuse the scale-10 SRData export.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SUPERF_ROOT = Path(os.environ.get("SUPERF_ROOT", "/tmp/superf"))
CITIES = ("asker", "bergen", "flekkefjord", "tromso", "trondheim", "vennesla")
ALIGN = ROOT / "eval/spatial_alignment.json"
FRAME_SCREEN = ROOT / "paper/results/frame_screen_nested_masks_only.json"
OUT_ROOT = ROOT / "paper/results/original_superf_norway_lr64"
PREFIX = "prod_full_nest64_nocut"


def _choose(city: str) -> tuple[Path, dict]:
    sample = ROOT / "single_samples" / city / "sample"
    pattern = re.compile(
        rf"{PREFIX}_{city}_nest64_(p\d+_\d+)_y(\d+)_x(\d+)$"
    )
    rows = []
    for path in sample.iterdir():
        match = pattern.match(path.name)
        if match and (path / "metrics.json").is_file():
            rows.append((match.group(1), int(match.group(2)), int(match.group(3)), path))
    if not rows:
        raise FileNotFoundError(f"{city}: no nest64 nocut runs")
    parents = sorted({row[0] for row in rows})
    parent = parents[len(parents) // 2]
    kids = [row for row in rows if row[0] == parent]
    cy = max(row[1] for row in kids) / 2
    cx = max(row[2] for row in kids) / 2
    kids.sort(key=lambda row: (abs(row[1] - cy) + abs(row[2] - cx), row[1], row[2]))
    run_dir = kids[0][3]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return run_dir, metrics


def _export(city: str, s2_dir: str, dest: Path) -> int:
    sys.path.insert(0, str(ROOT))
    from s2_dataset import S2NIBRevisitDataset
    from scripts.export_superf_srdata import export_dataset

    args = SimpleNamespace(
        s2_dir=s2_dir,
        hr_path=None,
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=0,
        lr_size=0,
        allow_no_hr=False,
        no_hr_harmonize=False,
        no_hr_spatial_align=False,
        spatial_alignment_path=str(ALIGN),
        dataset_device="cpu",
        frame_screen=str(FRAME_SCREEN),
        hr_harmonize_pixels="clear",
        s2_boa_offset="keep",
        mask_lr_clouds=True,
        lr_stats_pixels="clear",
    )
    dataset = S2NIBRevisitDataset(args, name=city, training_device="cpu")
    payload = export_dataset(dataset, dest)
    return int(payload["frame_count"])


def _scale_root(fourier_scale: float) -> Path:
    scale = float(fourier_scale)
    if scale == 10.0:
        return OUT_ROOT
    return OUT_ROOT / f"fourier_s{scale:g}"


def _train(srdata: Path, out_dir: Path, device: str, n_frames: int, fourier_scale: float) -> dict:
    """Run inside a process whose import path prefers the SuperF checkout."""
    if str(SUPERF_ROOT) not in sys.path:
        sys.path.insert(0, str(SUPERF_ROOT))
    os.chdir(SUPERF_ROOT)
    import numpy as np
    import torch
    from data import SRData
    from input_projections.utils import get_input_projection
    from models.inr import INR
    from models.utils import get_decoder
    from optimize import optimize_and_evaluate_sample

    torch.manual_seed(6)
    torch.cuda.manual_seed_all(6)
    np.random.seed(6)
    dev = torch.device(f"cuda:{device}")
    torch.cuda.set_device(dev)
    data = SRData(srdata, num_samples=n_frames, keep_in_memory=True, scale_factor=4, device=dev)
    projection = get_input_projection("fourier", 2, 256, dev, float(fourier_scale))
    decoder = get_decoder("mlp", 4, 256, 256)
    model = INR(
        projection,
        decoder,
        n_frames,
        use_gnll=False,
        use_base_frame=True,
        use_direct_param_T=True,
        use_separate_ud=False,
    ).to(dev)
    args = SimpleNamespace(
        iters=2000,
        learning_rate=2e-3,
        weight_decay=0.05,
        variance_reg=0.0,
        variance_smooth_reg=0.0,
        num_samples=n_frames,
        seed=6,
        input_projection="fourier",
        fourier_scale=float(fourier_scale),
        model="mlp",
        dataset="srdata",
        sample_id=srdata.name,
        network_depth=4,
        network_hidden_dim=256,
    )
    started = time.time()
    result = optimize_and_evaluate_sample(model, data, dev, 0, args, out_dir)
    model.eval()
    with torch.no_grad():
        coords = data.get_hr_coordinates().unsqueeze(0).to(dev)
        sample_id = torch.tensor([0], device=dev)
        output, _ = model(coords, sample_id, scale_factor=1, training=False)
        output = output * data.get_lr_std(0).to(dev) + data.get_lr_mean(0).to(dev)
    pred = output.squeeze(0).detach().cpu().numpy()
    np.save(out_dir / "pred_hwc.npy", np.clip(pred, 0, 1).astype(np.float32))
    result["wall_s"] = time.time() - started

    def _jsonable(value):
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return value

    (out_dir / "train_result.json").write_text(json.dumps(_jsonable(result), indent=2) + "\n")
    return result


def _float_lpips(city: str, s2_dir: str, pred_path: Path) -> float:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import lpips
    import numpy as np
    import torch
    from s2_dataset import S2NIBRevisitDataset

    args = SimpleNamespace(
        s2_dir=s2_dir,
        hr_path=None,
        df=4,
        scale_factor=4,
        hr_gsd_m=0.0,
        s2_native_gsd_m=10.0,
        num_samples=0,
        lr_size=0,
        allow_no_hr=False,
        no_hr_harmonize=False,
        no_hr_spatial_align=False,
        spatial_alignment_path=str(ALIGN),
        dataset_device="cpu",
        frame_screen=str(FRAME_SCREEN),
        hr_harmonize_pixels="clear",
        s2_boa_offset="keep",
        mask_lr_clouds=True,
        lr_stats_pixels="clear",
    )
    dataset = S2NIBRevisitDataset(args, name=city, training_device="cpu")
    hr = dataset.get_original_hr().detach().cpu().numpy()
    pred = np.load(pred_path)
    fn = lpips.LPIPS(net="vgg")

    def pack(image):
        tensor = torch.from_numpy(np.clip(image, 0, 1)).permute(2, 0, 1).unsqueeze(0).float()
        return tensor * 2 - 1

    with torch.no_grad():
        return float(fn(pack(pred), pack(hr)))


def run_one(city: str, device: str, fourier_scale: float) -> None:
    run_dir, metrics = _choose(city)
    s2_dir = metrics["provenance"]["args"]["s2_dir"]
    dest = _scale_root(fourier_scale) / city
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "extension_metrics_ref.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "s2_dir": s2_dir,
                "spatial_holdout": metrics.get("spatial_holdout"),
                "completed_iters": metrics.get("completed_iters"),
                "lpips": metrics.get("lpips"),
                "model_psnr": metrics.get("model_psnr"),
                "bilinear_psnr": metrics.get("bilinear_psnr"),
                "eval_alignment": metrics.get("eval_alignment"),
            },
            indent=2,
        )
        + "\n"
    )
    srdata = OUT_ROOT / city / "srdata"
    if (srdata / "hr_ground_truth.png").is_file():
        n_frames = len(list(srdata.glob("sample_*.png")))
    else:
        n_frames = _export(city, s2_dir, srdata)
    trained = (dest / "superf" / "pred_hwc.npy").is_file() and (dest / "superf" / "train_result.json").is_file()
    if not trained:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SUPERF_ROOT)
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--train-only",
                "--srdata",
                str(srdata),
                "--out-dir",
                str(dest / "superf"),
                "--device",
                device,
                "--frames",
                str(n_frames),
                "--fourier-scale",
                str(fourier_scale),
            ],
            cwd=SUPERF_ROOT,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
    result = json.loads((dest / "superf" / "train_result.json").read_text())
    result["float_hr_lpips"] = _float_lpips(city, s2_dir, dest / "superf" / "pred_hwc.npy")
    image = result.get("image_metrics") or {}
    summary = {
        "city": city,
        "device": device,
        "source_commit": "dff884c9ed61a6227737537c834a2604792af478",
        "budget": {"iters": 2000, "seed": 6, "model": "mlp", "fourier_scale": float(fourier_scale)},
        "n_frames": n_frames,
        "wall_s": result.get("wall_s"),
        "image_metrics": image,
        "float_hr_lpips": result.get("float_hr_lpips"),
        "extension_run": str(run_dir),
    }
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city")
    parser.add_argument("--device", default="1")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--srdata", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--fourier-scale", type=float, default=10.0)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()
    if args.train_only:
        _train(args.srdata, args.out_dir, args.device, args.frames, args.fourier_scale)
        return
    if args.city:
        run_one(args.city, args.device, args.fourier_scale)
        return
    if len(args.gpus) > 6:
        parser.error("at most 6 GPUs")
    if any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU ids must be non-negative")
    scale_root = _scale_root(args.fourier_scale)
    scale_root.mkdir(parents=True, exist_ok=True)
    pending = list(CITIES)
    free = [str(gpu) for gpu in args.gpus]
    active: dict[subprocess.Popen, tuple[str, str]] = {}
    failures = []
    while pending or active:
        while pending and free:
            city = pending.pop(0)
            device = free.pop(0)
            log = scale_root / f"{city}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--city",
                    city,
                    "--device",
                    device,
                    "--fourier-scale",
                    str(args.fourier_scale),
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            active[proc] = (city, device)
            print(f"started {city} on cuda:{device}", flush=True)
        finished = []
        for proc, (city, device) in active.items():
            code = proc.poll()
            if code is None:
                continue
            finished.append((proc, city, device, code))
        for proc, city, device, code in finished:
            del active[proc]
            free.append(device)
            print(f"done {city} rc={code}", flush=True)
            if code != 0:
                failures.append(city)
        if active:
            time.sleep(10)
    if failures:
        raise SystemExit(f"failed: {failures}")


if __name__ == "__main__":
    main()
