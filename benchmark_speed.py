#!/usr/bin/env python3
"""
Benchmark inference speed of the ScaleF INR model.

- CPU: float32 vs int8 dynamic quantization.
- GPU: float32 vs mixed precision (FP16/BF16 via autocast).

Usage:
  # CPU (int8 quantization)
  python benchmark_speed.py

  # GPU (float32 + mixed precision)
  python benchmark_speed.py --device cuda

  # Mixed precision dtype: --mixed_dtype bfloat16 or fp16
  python benchmark_speed.py --device cuda --mixed_dtype fp16

  # More warmup and repeats
  python benchmark_speed.py --warmup 50 --repeat 500
"""

import argparse
import time
import torch

from models.utils import get_decoder
from input_projections.utils import get_input_projection
from models.inr import INR


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark model inference speed (float vs quantized)")
    p.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda (quantization runs on CPU)")
    p.add_argument("--warmup", type=int, default=20, help="Warmup forward passes before timing")
    p.add_argument("--repeat", type=int, default=200, help="Number of timed forward passes")
    p.add_argument("--height", type=int, default=128, help="Spatial height of coordinate grid")
    p.add_argument("--width", type=int, default=128, help="Spatial width of coordinate grid")
    p.add_argument("--checkpoint", type=str, default=None, help="Optional path to checkpoint.pt to load weights")
    # Model config (match optimize.py defaults)
    p.add_argument("--model", type=str, default="mlp", choices=["mlp", "nir"])
    p.add_argument("--network_depth", type=int, default=4)
    p.add_argument("--network_hidden_dim", type=int, default=256)
    p.add_argument("--projection_dim", type=int, default=256)
    p.add_argument("--input_projection", type=str, default="fourier_10")
    p.add_argument("--fourier_scale", type=float, default=10.0)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--use_gnll", action="store_true")
    p.add_argument("--no_base_frame", action="store_true")
    p.add_argument("--no_direct_param_T", action="store_true")
    p.add_argument("--use_color_shift", action="store_true")
    p.add_argument("--use_separate_ud", action="store_true")
    p.add_argument("--mixed_dtype", type=str, default="auto",
                    choices=["auto", "fp16", "bfloat16", "fp8"],
                    help="Mixed precision on GPU: auto (bfloat16 if available else fp16), fp16, bfloat16, fp8 (try; often unsupported via autocast)")
    return p.parse_args()


def build_model(args, device):
    """Build INR model. Linear2DBias subclasses nn.Linear so quantize_dynamic can quantize it."""
    if args.input_projection.startswith("fourier_"):
        args.fourier_scale = float(args.input_projection.split("_")[1])
    proj_name = "fourier" if args.input_projection.startswith("fourier") else args.input_projection
    input_projection = get_input_projection(proj_name, 2, args.projection_dim, device, args.fourier_scale)
    decoder_input_dim = 2 if proj_name == "none" else args.projection_dim
    output_dim = 3 + args.num_samples * 3 if args.use_gnll and not args.use_separate_ud else 3
    decoder = get_decoder(
        args.model,
        args.network_depth,
        decoder_input_dim,
        args.network_hidden_dim,
        output_dim=output_dim,
    )
    model = INR(
        input_projection,
        decoder,
        args.num_samples,
        use_gnll=args.use_gnll,
        use_base_frame=not args.no_base_frame,
        use_direct_param_T=not args.no_direct_param_T,
        use_color_shift=args.use_color_shift,
        use_separate_ud=args.use_separate_ud,
    )
    return model.to(device)


def run_forward(model, coords, sample_id, device, autocast_dtype=None):
    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                out, _ = model(coords, sample_id, scale_factor=1, training=False)
        else:
            out, _ = model(coords, sample_id, scale_factor=1, training=False)
    return out


def benchmark(model, coords, sample_id, device, warmup, repeat, autocast_dtype=None):
    # Warmup
    for _ in range(warmup):
        run_forward(model, coords, sample_id, device, autocast_dtype=autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    # Timed runs
    start = time.perf_counter()
    for _ in range(repeat):
        run_forward(model, coords, sample_id, device, autocast_dtype=autocast_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / repeat  # ms per forward


def main():
    args = parse_args()
    device = torch.device(args.device)

    H, W = args.height, args.width
    coords = torch.rand(1, H, W, 2, device=device, dtype=torch.float32)
    sample_id = torch.tensor([0], device=device)

    print(f"Benchmark config: grid {H}x{W}, warmup={args.warmup}, repeat={args.repeat}, device={device}")
    print()

    # --- Float (no quantization) ---
    model_float = build_model(args, device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "model_state_dict" in ckpt:
            model_float.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model_float.load_state_dict(ckpt, strict=False)
        print("Loaded checkpoint into float model.")
    model_float.eval()
    ms_float = benchmark(model_float, coords, sample_id, device, args.warmup, args.repeat)
    print(f"  Float32: {ms_float:.2f} ms / forward")

    # --- Mixed precision (GPU only) ---
    if device.type == "cuda":
        if args.mixed_dtype == "auto":
            mixed_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            label = "BF16" if mixed_dtype == torch.bfloat16 else "FP16"
        elif args.mixed_dtype == "fp8":
            mixed_dtype = getattr(torch, "float8_e4m3fn", None)
            label = "FP8 (e4m3fn)"
            if mixed_dtype is None:
                print(f"  Mixed precision (FP8): skipped (PyTorch has no float8_e4m3fn)")
                mixed_dtype = None
        else:
            mixed_dtype = torch.bfloat16 if args.mixed_dtype == "bfloat16" else torch.float16
            label = "BF16" if mixed_dtype == torch.bfloat16 else "FP16"

        if mixed_dtype is not None:
            try:
                ms_mixed = benchmark(
                    model_float, coords, sample_id, device, args.warmup, args.repeat, autocast_dtype=mixed_dtype
                )
                print(f"  Mixed precision ({label}): {ms_mixed:.2f} ms / forward")
                speedup = ms_float / ms_mixed if ms_mixed > 0 else 0
                print(f"  Speedup (mixed vs float32): {speedup:.2f}x")
            except RuntimeError as e:
                if "float8" in label.lower() or "Float8" in str(e):
                    print(f"  Mixed precision ({label}): not supported")
                    print(f"    Reason: standard CUDA kernels (e.g. addmm) do not implement FP8.")
                    print(f"    Real FP8 needs H100+ and torch._scaled_mm / TransformerEngine, not autocast.")
                else:
                    raise

    # --- Int8 dynamic quantization (CPU only) ---
    if device.type == "cpu":
        model_quant = build_model(args, device)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
            sd = ckpt.get("model_state_dict", ckpt)
            model_quant.load_state_dict(sd, strict=False)
        model_quant.eval()
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model_quant = torch.ao.quantization.quantize_dynamic(
                    model_quant,
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
        except Exception as e:
            print(f"  Quantized (int8): skipped ({e})")
        else:
            ms_quant = benchmark(model_quant, coords, sample_id, device, args.warmup, args.repeat)
            print(f"  Quantized (int8): {ms_quant:.2f} ms / forward")
            speedup = ms_float / ms_quant if ms_quant > 0 else 0
            print(f"  Speedup (quantized vs float32): {speedup:.2f}x")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
