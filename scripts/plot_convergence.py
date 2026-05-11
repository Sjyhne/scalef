import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Plot PSNR-vs-iteration from metrics.jsonl files.")
    p.add_argument("runs", nargs="+", help="One or more sample directories (containing metrics.jsonl)")
    p.add_argument("--out", default="convergence.png", help="Output PNG path")
    p.add_argument("--title", default="Convergence", help="Plot title")
    args = p.parse_args()

    plt.figure(figsize=(10, 6))
    plotted = 0
    for run in args.runs:
        run_path = Path(run)
        log_path = run_path / "metrics.jsonl" if run_path.is_dir() else Path(run)
        rows = read_jsonl(log_path)
        if not rows:
            continue
        it = [r.get("iteration") for r in rows if "iteration" in r and "test_psnr" in r]
        psnr = [r.get("test_psnr") for r in rows if "iteration" in r and "test_psnr" in r]
        if not it:
            continue
        label = run_path.name if run_path.is_dir() else log_path.parent.name
        plt.plot(it, psnr, linewidth=2, label=label)
        plotted += 1

    plt.xlabel("Iteration")
    plt.ylabel("PSNR (dB)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend()
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()

