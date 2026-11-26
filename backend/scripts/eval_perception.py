"""Evaluate YOLO perception model weights on held-out test splits.

Evaluates weights on every `data/heldout/<source>/test` folder and reports
per-source metrics:
- mAP50
- mAP50-95
- Precision (mean)
- Recall (mean)
- D40 (pothole) per-class AP50 and AP50-95
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence

CLASSES = ["D00", "D10", "D20", "D40"]


def evaluate_source(
    model,
    source_name: str,
    test_dir: Path,
    device: str = "0",
    batch: int = 16,
    imgsz: int = 640,
) -> dict:
    """Evaluate YOLO model on a single source's test images/labels directory."""
    images_dir = test_dir / "images"
    labels_dir = test_dir / "labels"

    if not images_dir.exists() or not any(images_dir.iterdir()):
        return {
            "source": source_name,
            "images": 0,
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "d40_ap50": 0.0,
            "d40_map": 0.0,
        }

    # Generate a temporary data YAML pointing directly to this source's test set
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml_content = f"""path: {test_dir.resolve().as_posix()}
val: images
names:
  0: D00
  1: D10
  2: D20
  3: D40
"""
        f.write(yaml_content)
        temp_yaml = Path(f.name)

    try:
        metrics = model.val(
            data=str(temp_yaml),
            device=device,
            batch=batch,
            imgsz=imgsz,
            split="val",
            verbose=False,
            plots=False,
            save=False,
        )

        box = metrics.box
        mp = float(box.mp) if hasattr(box, "mp") else 0.0
        mr = float(box.mr) if hasattr(box, "mr") else 0.0
        map50 = float(box.map50) if hasattr(box, "map50") else 0.0
        map50_95 = float(box.map) if hasattr(box, "map") else 0.0

        # D40 is class index 3 (pothole)
        d40_ap50 = 0.0
        d40_map = 0.0
        if hasattr(box, "ap50") and len(box.ap50) > 3:
            d40_ap50 = float(box.ap50[3])
        if hasattr(box, "maps") and len(box.maps) > 3:
            d40_map = float(box.maps[3])

        num_images = len(list(images_dir.glob("*.*")))
        return {
            "source": source_name,
            "images": num_images,
            "precision": mp,
            "recall": mr,
            "map50": map50,
            "map50_95": map50_95,
            "d40_ap50": d40_ap50,
            "d40_map": d40_map,
        }
    finally:
        if temp_yaml.exists():
            temp_yaml.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO perception weights on held-out test splits."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to YOLO weights (.pt file) to evaluate",
    )
    parser.add_argument(
        "--heldout-dir",
        type=Path,
        default=Path("data/heldout"),
        help="Base directory containing heldout/<source>/test folders (default: data/heldout)",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Specific sources to evaluate. Defaults to all sources found under heldout-dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to run eval on (default: 0)",
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    weights_path = args.weights
    heldout_dir = args.heldout_dir

    if not weights_path.exists():
        print(f"Error: Weights file '{weights_path}' does not exist.")
        sys.exit(1)

    if not heldout_dir.exists():
        print(f"Error: Held-out directory '{heldout_dir}' does not exist.")
        sys.exit(1)

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # Discover test sources: data/heldout/<source>/test
    all_source_dirs = sorted([d.parent for d in heldout_dir.glob("*/test") if d.is_dir()])
    if not all_source_dirs:
        # Check if heldout_dir itself has subdirs
        all_source_dirs = sorted([d for d in heldout_dir.iterdir() if d.is_dir() and (d / "test").exists()])

    if args.sources:
        source_dirs = [d for d in all_source_dirs if d.name in args.sources]
    else:
        source_dirs = all_source_dirs

    if not source_dirs:
        print(f"No valid test sources found under {heldout_dir}.")
        sys.exit(1)

    print("=" * 80)
    print(f"PERCEPTION EVALUATION: {weights_path.name}")
    print(f"Weights: {weights_path.resolve()}")
    print(f"Device:  {args.device}")
    print(f"Sources: {[d.name for d in source_dirs]}")
    print("=" * 80)

    results = []
    for s_dir in source_dirs:
        source_name = s_dir.name
        test_dir = s_dir / "test"
        print(f"\nEvaluating source: {source_name} ({test_dir})...")
        res = evaluate_source(
            model=model,
            source_name=source_name,
            test_dir=test_dir,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
        )
        results.append(res)

    print("\n" + "=" * 90)
    print(f"EVALUATION RESULTS FOR {weights_path.name}")
    print("=" * 90)
    print(f"{'Source':<20} | {'Images':<8} | {'Precision':<10} | {'Recall':<8} | {'mAP50':<8} | {'mAP50-95':<9} | {'D40-AP50':<9} | {'D40-AP':<8}")
    print("-" * 90)
    for r in results:
        print(
            f"{r['source']:<20} | {r['images']:<8} | {r['precision']:<10.3f} | "
            f"{r['recall']:<8.3f} | {r['map50']:<8.3f} | {r['map50_95']:<9.3f} | "
            f"{r['d40_ap50']:<9.3f} | {r['d40_map']:<8.3f}"
        )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
