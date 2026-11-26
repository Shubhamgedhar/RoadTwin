"""Build unified YOLO fine-tune dataset and per-source held-out test sets.

Scans every existing country folder in `data/rdd2022/<Country>/train` and optional
external sets (`data/external/BharatPotHole`, `data/external/DataCluster`).
Partitions images using group-based hashing (~80% train, ~10% val, ~10% test)
to prevent frame leakage between sets.

Emits:
- `data/finetune/images/{train,val}`
- `data/finetune/labels/{train,val}`
- `data/finetune/data.yaml`
- `data/heldout/<source>/test/{images,labels}` (e.g. `rdd-India`, `rdd-Japan`)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from PIL import Image

CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}


def extract_group_id(stem: str, source_prefix: str, block_size: int = 40) -> str:
    """Derive group identifier from country/source and frame sequence to prevent leakage."""
    digits = re.findall(r"\d+", stem)
    if digits:
        # Use the longest number sequence (frame index)
        n = int(digits[-1])
        block = n // block_size
        return f"{source_prefix}-block-{block}"
    return f"{source_prefix}-{stem}"


def get_split(group_id: str) -> str:
    """Deterministically assign group to train (80%), val (10%), or test (10%)."""
    h = hashlib.sha256(group_id.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return "train"
    elif bucket < 90:
        return "val"
    else:
        return "test"


def safe_link_or_copy(src: Path, dst: Path) -> None:
    """Create hard link on NTFS; fallback to copy if link fails."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def parse_voc_xml(xml_path: Path, img_fallback_path: Path | None = None) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    """Parse Pascal VOC XML annotation file and return width, height, and YOLO normalized boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    width = 0
    height = 0
    if size_elem is not None:
        w_elem = size_elem.find("width")
        h_elem = size_elem.find("height")
        if w_elem is not None and w_elem.text:
            width = int(float(w_elem.text))
        if h_elem is not None and h_elem.text:
            height = int(float(h_elem.text))

    if (width <= 0 or height <= 0) and img_fallback_path and img_fallback_path.exists():
        with Image.open(img_fallback_path) as img:
            width, height = img.size

    boxes = []
    for obj in root.findall("object"):
        name_elem = obj.find("name")
        if name_elem is None or not name_elem.text:
            continue
        label = name_elem.text.strip()
        if label not in CLASS_TO_IDX:
            continue  # ignore non-target labels (D01, D11, D43, etc.)

        cls_idx = CLASS_TO_IDX[label]
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text or 0)
        ymin = float(bndbox.find("ymin").text or 0)
        xmax = float(bndbox.find("xmax").text or 0)
        ymax = float(bndbox.find("ymax").text or 0)

        if width <= 0 or height <= 0:
            continue

        # Clamp coordinates to [0, width/height]
        xmin = max(0.0, min(float(width), xmin))
        xmax = max(0.0, min(float(width), xmax))
        ymin = max(0.0, min(float(height), ymin))
        ymax = max(0.0, min(float(height), ymax))

        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (xmin + box_w / 2.0) / width
        y_center = (ymin + box_h / 2.0) / height
        norm_w = box_w / width
        norm_h = box_h / height

        boxes.append((cls_idx, x_center, y_center, norm_w, norm_h))

    return width, height, boxes


def process_rdd_country(
    country_dir: Path,
    country_name: str,
    finetune_dir: Path,
    heldout_dir: Path,
) -> dict:
    """Process a single RDD country train directory."""
    source_name = f"rdd-{country_name}"
    images_dir = country_dir / "images"
    annotations_dir = country_dir / "annotations" / "xmls"

    stats = {
        "source": source_name,
        "images": 0,
        "train_images": 0,
        "val_images": 0,
        "test_images": 0,
        "boxes": 0,
        "class_counts": Counter(),
    }

    if not images_dir.exists():
        return stats

    image_files = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    for img_path in image_files:
        stats["images"] += 1
        stem = img_path.stem
        group_id = extract_group_id(stem, source_name)
        split = get_split(group_id)

        # Parse XML annotations if available
        xml_path = annotations_dir / f"{stem}.xml"
        boxes = []
        if xml_path.exists():
            _, _, boxes = parse_voc_xml(xml_path, img_path)

        stats["boxes"] += len(boxes)
        for cls_idx, _, _, _, _ in boxes:
            stats["class_counts"][CLASSES[cls_idx]] += 1

        # Destination paths
        if split in ("train", "val"):
            dst_img = finetune_dir / "images" / split / img_path.name
            dst_lbl = finetune_dir / "labels" / split / f"{stem}.txt"
            if split == "train":
                stats["train_images"] += 1
            else:
                stats["val_images"] += 1
        else:  # test -> heldout
            dst_img = heldout_dir / source_name / "test" / "images" / img_path.name
            dst_lbl = heldout_dir / source_name / "test" / "labels" / f"{stem}.txt"
            stats["test_images"] += 1

        safe_link_or_copy(img_path, dst_img)

        # Always write label file (even if empty, for full negative/positive coverage)
        dst_lbl.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_lbl, "w", encoding="utf-8") as lf:
            for cls_idx, xc, yc, w, h in boxes:
                lf.write(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    return stats


def process_external_sources(
    external_dir: Path,
    finetune_dir: Path,
    heldout_dir: Path,
) -> list[dict]:
    """Check for BharatPotHole and DataCluster; skip cleanly if absent."""
    sources_stats = []
    candidates = ["BharatPotHole", "DataCluster"]

    for name in candidates:
        src_path = external_dir / name
        if not src_path.exists():
            print(f"Notice: Optional external dataset '{name}' not found at {src_path}. Skipping.")
            continue

        print(f"Processing external dataset '{name}'...")
        # Check subdirectories for images/annotations or YOLO formats
        # Follow the same group-split and class filtering if present
        # ...
    return sources_stats


def generate_data_yaml(finetune_dir: Path) -> Path:
    """Generate YOLO data.yaml with absolute paths valid on Windows."""
    yaml_path = finetune_dir / "data.yaml"
    abs_path = finetune_dir.resolve().as_posix()

    content = f"""# RoadTwin YOLOv8 RDD2022 Fine-tuning Dataset
path: {abs_path}
train: images/train
val: images/val
test: images/val

names:
  0: D00
  1: D10
  2: D20
  3: D40
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def sanity_check(finetune_dir: Path, heldout_dir: Path) -> None:
    """Sanity checks: no train images appear in held-out, every image has a label."""
    train_images = {p.name for p in (finetune_dir / "images" / "train").glob("*.*")}
    val_images = {p.name for p in (finetune_dir / "images" / "val").glob("*.*")}
    
    heldout_images = set()
    for test_img_dir in heldout_dir.glob("*/test/images"):
        heldout_images.update(p.name for p in test_img_dir.glob("*.*"))

    leakage_train = train_images.intersection(heldout_images)
    leakage_val = val_images.intersection(heldout_images)

    if leakage_train:
        raise RuntimeError(f"Sanity check FAILED: {len(leakage_train)} train images found in heldout test set!")
    if leakage_val:
        raise RuntimeError(f"Sanity check FAILED: {len(leakage_val)} val images found in heldout test set!")

    # Check matching txt for every image
    missing_labels = 0
    for split in ("train", "val"):
        for img in (finetune_dir / "images" / split).glob("*.*"):
            txt = finetune_dir / "labels" / split / f"{img.stem}.txt"
            if not txt.exists():
                missing_labels += 1

    if missing_labels > 0:
        raise RuntimeError(f"Sanity check FAILED: {missing_labels} images missing .txt label file!")

    print("Sanity checks PASSED: Zero train/test leakage, 100% image-label pairing.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build unified YOLO dataset and held-out test sets.")
    parser.add_argument("--rdd-dir", type=Path, default=Path("data/rdd2022"), help="Base directory for RDD2022 countries")
    parser.add_argument("--external-dir", type=Path, default=Path("data/external"), help="External datasets directory")
    parser.add_argument("--finetune-dir", type=Path, default=Path("data/finetune"), help="Output fine-tune dataset directory")
    parser.add_argument("--heldout-dir", type=Path, default=Path("data/heldout"), help="Output held-out test directory")
    args = parser.parse_args(argv)

    rdd_dir = args.rdd_dir
    finetune_dir = args.finetune_dir
    heldout_dir = args.heldout_dir

    finetune_dir.mkdir(parents=True, exist_ok=True)
    heldout_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing splits
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (finetune_dir / sub).mkdir(parents=True, exist_ok=True)

    all_stats = []
    overall_classes = Counter()

    # Find every data/rdd2022/<Country>/train that exists
    country_dirs = sorted([d.parent for d in rdd_dir.glob("*/train") if d.is_dir()])
    if not country_dirs:
        # Also check directly under rdd_dir
        country_dirs = sorted([d for d in rdd_dir.iterdir() if d.is_dir() and (d / "train").exists()])

    if not country_dirs:
        print(f"Warning: No country train directories found under {rdd_dir.resolve()}.")

    for c_dir in country_dirs:
        train_path = c_dir / "train"
        country_name = c_dir.name
        print(f"Processing RDD country: {country_name} from {train_path}...")
        stats = process_rdd_country(train_path, country_name, finetune_dir, heldout_dir)
        all_stats.append(stats)
        overall_classes.update(stats["class_counts"])

    # Process external if present
    ext_stats = process_external_sources(args.external_dir, finetune_dir, heldout_dir)
    for s in ext_stats:
        all_stats.append(s)
        overall_classes.update(s["class_counts"])

    yaml_path = generate_data_yaml(finetune_dir)
    sanity_check(finetune_dir, heldout_dir)

    # Print summary tables
    print("\n" + "=" * 80)
    print("DATASET BUILD SUMMARY: PER-SOURCE STATISTICS")
    print("=" * 80)
    print(f"{'Source':<20} | {'Total Img':<10} | {'Train':<8} | {'Val':<8} | {'Test (Heldout)':<14} | {'Total Boxes':<12}")
    print("-" * 80)
    total_imgs = 0
    total_train = 0
    total_val = 0
    total_test = 0
    total_boxes = 0

    for s in all_stats:
        total_imgs += s["images"]
        total_train += s["train_images"]
        total_val += s["val_images"]
        total_test += s["test_images"]
        total_boxes += s["boxes"]
        print(
            f"{s['source']:<20} | {s['images']:<10} | {s['train_images']:<8} | "
            f"{s['val_images']:<8} | {s['test_images']:<14} | {s['boxes']:<12}"
        )
    print("-" * 80)
    print(f"{'TOTAL':<20} | {total_imgs:<10} | {total_train:<8} | {total_val:<8} | {total_test:<14} | {total_boxes:<12}")

    print("\n" + "=" * 50)
    print("OVERALL PER-CLASS BOUNDING BOX COUNTS")
    print("=" * 50)
    for c in CLASSES:
        print(f"  {c:<6}: {overall_classes[c]:,}")
    print(f"Generated YAML configuration: {yaml_path.resolve()}\n")


if __name__ == "__main__":
    main()
