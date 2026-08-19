"""Reproducible train/validate runner for a local BFoV panoramic dataset.

The runner trains the lightweight residual adapter used by PanoSOT and then
evaluates a held-out sequence split with and without that adapter. It keeps
the backbone frozen, uses the runtime-matched feature layer 12, and defaults
to the conservative appearance-plus-feature-regularization recipe.

Expected input:
    DATA/train/train_real/seq_x/video.mp4
    DATA/train/train_real/seq_x/init.txt
    DATA/train/train_real/seq_x/groundtruth.txt
    DATA/train/train_sim/seq_y/...

If no explicit validation root is supplied, every fifth sequence is held out
deterministically.  The held-out set is never used to train the adapter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sequence_roots(train_root: Path) -> list[Path]:
    roots = [candidate for candidate in (
        train_root / "train_real",
        train_root / "train_sim",
        train_root / "real",
        train_root / "sim",
    ) if candidate.is_dir()]
    if roots:
        return roots
    return [train_root]


def _sequence_names(roots: list[Path]) -> list[str]:
    names: list[str] = []
    for root in roots:
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            if (path / "video.mp4").is_file() and (path / "groundtruth.txt").is_file():
                names.append(path.name)
    return sorted(set(names))


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(f'"{arg}"' if " " in arg else arg for arg in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root containing train_real/train_sim.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "ys_panotracking")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-per-sequence", type=int, default=80)
    parser.add_argument(
        "--delta", type=int, default=1,
        help="Frame gap for temporal pairs. One frame matches online tracking.",
    )
    parser.add_argument(
        "--sample-step", type=int, default=3,
        help="Stride used to enumerate temporal pairs before sampling.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-eval-sequences", type=int, default=0)
    parser.add_argument(
        "--hard-negatives", action="store_true",
        help="Enable batch hard-negative loss only after validation confirms a gain.",
    )
    parser.add_argument(
        "--spatial-negatives", action="store_true",
        help="Enable shifted-search negative loss only after validation confirms a gain.",
    )
    parser.add_argument("--skip-convert", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset}")

    train_root = dataset / "train" if (dataset / "train").is_dir() else dataset
    roots = _sequence_roots(train_root)
    if not any(root.is_dir() for root in roots):
        raise SystemExit(
            f"No train_real/train_sim roots found under {train_root}. "
            "Expected video.mp4 + init.txt + groundtruth.txt per sequence."
        )
    names = _sequence_names(roots)
    if len(names) < 2:
        raise SystemExit("At least two valid sequences are required for a train/validation split.")

    val_names = [name for index, name in enumerate(names) if index % 5 == 0]
    train_names = [name for name in names if name not in set(val_names)]
    if not train_names or not val_names:
        raise SystemExit("Could not create a non-empty deterministic train/validation split.")
    print(f"Train sequences: {len(train_names)}; validation sequences: {len(val_names)}")
    print("Validation subset:", ",".join(val_names))

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = output_root / "ys_local_bfov_adapter.pt"
    cache = output_root / "train_pairs.npz"
    converted = output_root / "eval_sequences"
    python = sys.executable

    train_command = [
        python, "tools/train_local_bfov_adapter.py",
        *(item for root in roots for item in ("--root", str(root))),
        "--subset", ",".join(train_names),
        "--output", str(adapter),
        "--cache", str(cache),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-samples", str(args.max_samples),
        "--max-per-sequence", str(args.max_per_sequence),
        "--delta", str(args.delta),
        "--sample-step", str(max(int(args.sample_step), 1)),
        "--feature-layer", "12",
        "--device", args.device,
        "--appearance-augment",
        "--feature-reg", "0.05",
        "--cache-dir", str(PROJECT_ROOT / ".cache" / "torch"),
    ]
    if args.hard_negatives:
        train_command.extend(["--hard-negative-weight", "0.20"])
    if args.spatial_negatives:
        train_command.extend(["--spatial-negative-weight", "0.20"])
    _run(train_command)

    if not args.skip_convert:
        _run([
            python, "tools/convert_train_dataset.py",
            "--train-root", str(train_root),
            "--out-dir", str(converted),
            "--splits", "real", "sim",
            "--count", "1000000",
            "--max-frames", "0",
            "--overwrite",
        ])

    eval_names = [f"{split}_{name}" for split in ("real", "sim") for name in val_names]
    if args.max_eval_sequences > 0:
        eval_names = eval_names[:args.max_eval_sequences]
    common = [
        python, "tools/batch_evaluate.py",
        "--data-root", str(converted),
        "--deep",
        "--feature-layer", "12",
        "--device", args.device,
        "--cache-dir", str(PROJECT_ROOT / ".cache" / "torch"),
        "--subset", ",".join(eval_names),
        "--save-pred",
        "--deep-fallback-ncc-arbitration",
        "--deep-fallback-ncc-arbitration-interval", "3",
        "--deep-probe-ncc-margin-arbitration",
        "--tiny-probe-growth-guard",
    ]
    _run([*common, "--output", str(output_root / "baseline.csv")])
    _run([*common, "--tracking-adapter", str(adapter), "--output", str(output_root / "adapter.csv")])
    print(f"\nAdapter: {adapter}")
    print(f"Baseline metrics: {output_root / 'baseline.json'}")
    print(f"Adapter metrics: {output_root / 'adapter.json'}")


if __name__ == "__main__":
    main()
