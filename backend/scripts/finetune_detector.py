"""Fine-tune the YOLOv8s road-damage detector on the unified RDD2022 dataset.

Preserves pretrained weights starting from `models/oracl4_yolov8s.pt` with
cosine learning rate decay (lr0=0.002) and early stopping patience.

Runs with safe multiprocessing worker defaults for Windows platforms.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 detector on RDD2022 dataset."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("models/oracl4_yolov8s.pt"),
        help="Pretrained weights path (default: models/oracl4_yolov8s.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/finetune/data.yaml"),
        help="Path to YOLO data.yaml (default: data/finetune/data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or cpu (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers (default: 4 for Windows safety)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to train on (default: 1.0, use 0.05 for timing probe)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="finetune_full",
        help="Run name under data/runs/ (default: finetune_full)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("data/runs"),
        help="Directory to save runs (default: data/runs)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.002,
        help="Initial learning rate (default: 0.002)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Cache images: ram or disk (default: None)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from data/runs/<name>/weights/last.pt",
    )
    return parser.parse_args(argv)


def run_finetune(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    weights_path = args.weights
    data_path = args.data.resolve()

    if not args.resume and not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' does not exist.")
        print("Please ensure models/oracl4_yolov8s.pt is in place.")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: data.yaml not found at '{data_path}'.")
        print("Please run backend.scripts.build_finetune_dataset first.")
        sys.exit(1)

    project_dir = args.project.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    cache_val: str | bool = False
    if args.cache:
        if args.cache.lower() in ("ram", "disk"):
            cache_val = args.cache.lower()
        elif args.cache.lower() in ("true", "1"):
            cache_val = True

    print("=" * 60)
    print("STARTING YOLOV8 DETECTOR FINE-TUNING")
    print(f"Weights:   {weights_path}")
    print(f"Dataset:   {data_path}")
    print(f"Epochs:    {args.epochs} (fraction={args.fraction})")
    print(f"Batch:     {args.batch}")
    print(f"Device:    {args.device}")
    print(f"Workers:   {args.workers}")
    print(f"Learning:  lr0={args.lr0}, cos_lr=True, patience={args.patience}")
    print(f"Output:    {project_dir / args.name}")
    print("=" * 60)

    if args.resume:
        last_pt = project_dir / args.name / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"Error: Cannot resume -- '{last_pt}' does not exist.")
            sys.exit(1)
        print(f"Resuming from {last_pt}...")
        model = YOLO(str(last_pt))
        model.train(resume=True)
    else:
        model = YOLO(str(weights_path))
        model.train(
            data=str(data_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            fraction=args.fraction,
            name=args.name,
            project=str(project_dir),
            lr0=args.lr0,
            cos_lr=True,
            patience=args.patience,
            cache=cache_val,
            exist_ok=True,
            val=True,
            save=True,
            plots=True,
            verbose=True,
        )

    best_pt = project_dir / args.name / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\nTraining completed successfully! Best weights: {best_pt}")
    else:
        print(f"\nRun ended. Check {project_dir / args.name}")


if __name__ == "__main__":
    run_finetune(parse_args())
