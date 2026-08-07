"""PASCAL VOC download and conversion to YOLO format for YOLOv1 training."""

from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

# Official 20 PASCAL VOC classes (fixed order used by YOLOv1)
VOC_CLASSES: list[str] = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# Direct download URLs (Oxford VOC mirror)
VOC_URLS: dict[str, dict[str, str]] = {
    "2007": {
        "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "test": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    },
    "2012": {
        "trainval": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    },
}

# Default train/val splits used in most detection benchmarks
DEFAULT_TRAIN_SPLITS: list[tuple[str, str]] = [
    ("2007", "trainval"),
    ("2012", "trainval"),
]
DEFAULT_VAL_SPLITS: list[tuple[str, str]] = [
    ("2007", "test"),
]

def _voc_year_dir(raw_dir: Path, year: str) -> Path:
    """Return VOC{year} root directory.

    Official tarballs extract as ``VOCdevkit/VOC2007/``, not ``VOC2007/`` at the top level.
    """
    candidates = [
        raw_dir / "VOCdevkit" / f"VOC{year}",
        raw_dir / f"VOC{year}",
    ]
    for path in candidates:
        if (path / "JPEGImages").is_dir():
            return path
    return raw_dir / "VOCdevkit" / f"VOC{year}"


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a simple progress indicator."""
    print(f"  Downloading: {url}")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  Progress: {pct}%", end="", flush=True)

    urlretrieve(url, dest, reporthook=_report)
    print()


def _extract_tar(tar_path: Path, extract_dir: Path) -> None:
    """Extract a .tar archive."""
    import tarfile

    print(f"  Extracting: {tar_path.name}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)


def download_voc_archives(
    raw_dir: Path,
    years: list[str],
    splits: list[str] | None = None,
    force: bool = False,
) -> None:
    """Download and extract PASCAL VOC tar archives.

    Args:
        raw_dir: directory to store raw VOC folders (VOCdevkit/...)
        years: e.g. ["2007"] or ["2007", "2012"]
        splits: subset of splits to fetch ("trainval", "test"); None = all available
        force: re-download even if archive exists
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = raw_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        if year not in VOC_URLS:
            raise ValueError(f"Unsupported VOC year: {year}. Choose from {list(VOC_URLS)}")

        year_splits = splits or list(VOC_URLS[year].keys())
        for split in year_splits:
            if split not in VOC_URLS[year]:
                print(f"  Skipping {year}/{split} (not available for download)")
                continue

            voc_marker = raw_dir / f"VOC{year}" / "JPEGImages"
            if voc_marker.exists() and not force:
                print(f"  VOC{year} already extracted, skipping download.")
                continue

            url = VOC_URLS[year][split]
            tar_name = url.split("/")[-1]
            tar_path = archives_dir / tar_name

            if not tar_path.exists() or force:
                _download_with_progress(url, tar_path)
            else:
                print(f"  Archive exists: {tar_path.name}")

            _extract_tar(tar_path, raw_dir)


def _parse_voc_annotation(
    xml_path: Path,
) -> tuple[str, list[tuple[str, tuple[float, float, float, float]]], float, float]:
    """Parse a VOC XML annotation file.

    Returns:
        image_id, objects, image_width, image_height
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    image_id = Path(filename).stem

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    objects: list[tuple[str, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        # Skip difficult/truncated objects optionally — keep all for training signal
        name = obj.find("name").text.strip().lower()
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Clamp to image bounds
        xmin = max(0.0, min(xmin, img_w))
        ymin = max(0.0, min(ymin, img_h))
        xmax = max(0.0, min(xmax, img_w))
        ymax = max(0.0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            continue

        objects.append((name, (xmin, ymin, xmax, ymax)))

    return image_id, objects, img_w, img_h


def _voc_to_yolo_line(
    class_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: float,
    img_h: float,
) -> str:
    """Convert VOC absolute bbox to YOLO normalized line."""
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink image to save disk space; fall back to copy if symlink fails."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def _load_image_ids(split_file: Path) -> list[str]:
    """Load image IDs from a VOC ImageSets/Main/*.txt file."""
    if not split_file.exists():
        return []
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def convert_voc_split(
    raw_dir: Path,
    output_dir: Path,
    year: str,
    split: str,
    class_names: list[str],
    class_to_idx: dict[str, int],
    max_samples: int | None = None,
) -> int:
    """Convert one VOC split to YOLO layout (images/ + labels/).

    Returns:
        Number of images written.
    """
    voc_root = _voc_year_dir(raw_dir, year)
    images_src = voc_root / "JPEGImages"
    ann_dir = voc_root / "Annotations"
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"

    if not images_src.exists():
        raise FileNotFoundError(
            f"VOC{year} not found at {voc_root}. Run download first."
        )

    image_ids = _load_image_ids(split_file)
    if not image_ids:
        raise FileNotFoundError(f"Split file not found or empty: {split_file}")

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id in image_ids:
        if max_samples is not None and written >= max_samples:
            break

        xml_path = ann_dir / f"{image_id}.xml"
        img_path = images_src / f"{image_id}.jpg"

        if not xml_path.exists() or not img_path.exists():
            continue

        _, objects, img_w, img_h = _parse_voc_annotation(xml_path)

        yolo_lines: list[str] = []
        for name, (xmin, ymin, xmax, ymax) in objects:
            if name not in class_to_idx:
                continue
            cls_id = class_to_idx[name]
            yolo_lines.append(
                _voc_to_yolo_line(cls_id, xmin, ymin, xmax, ymax, img_w, img_h)
            )

        # Skip images with no objects in selected classes
        if not yolo_lines:
            continue

        _link_or_copy(img_path, out_images / f"{image_id}.jpg")
        label_path = out_labels / f"{image_id}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n")

        written += 1
        if written % 500 == 0:
            print(f"    Converted {written} images...")

    return written


def prepare_voc_dataset(
    output_dir: str | Path,
    years: list[str] | None = None,
    train_splits: list[tuple[str, str]] | None = None,
    val_splits: list[tuple[str, str]] | None = None,
    max_train_samples: int | None = 2000,
    max_val_samples: int | None = 500,
    selected_classes: list[str] | None = None,
    force_download: bool = False,
) -> dict[str, Any]:
    """Download PASCAL VOC and convert to YOLO format for training.

    Size reduction options:
      - years=["2007"] only          ~900 MB vs ~3 GB for 2007+2012
      - max_train_samples / max_val  cap number of images per split
      - selected_classes             train on a subset of classes only

    Args:
        output_dir: root output folder (creates train/ and val/ subdirs)
        years: VOC years to download
        train_splits: list of (year, split_name) for training
        val_splits: list of (year, split_name) for validation
        max_train_samples: cap training images (None = no limit)
        max_val_samples: cap validation images (None = no limit)
        selected_classes: subset of VOC_CLASSES (None = all 20)
        force_download: re-download archives

    Returns:
        Metadata dict (classes, counts, paths)
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"

    years = years or ["2007"]
    train_splits = train_splits or [
        (y, "trainval") for y in years if y in VOC_URLS
    ]
    val_splits = val_splits or DEFAULT_VAL_SPLITS

    # Resolve class list and index mapping
    if selected_classes:
        unknown = set(selected_classes) - set(VOC_CLASSES)
        if unknown:
            raise ValueError(f"Unknown classes: {unknown}")
        class_names = selected_classes
    else:
        class_names = VOC_CLASSES.copy()

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print("=== PASCAL VOC: download ===")
    print(f"  Years: {years}")
    download_voc_archives(raw_dir, years=years, force=force_download)

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"

    # Clean previous conversion to avoid stale symlinks
    for split_dir in (train_dir, val_dir):
        if split_dir.exists():
            shutil.rmtree(split_dir)

    print("\n=== PASCAL VOC: convert train ===")
    train_total = 0
    train_remaining = max_train_samples
    for year, split in train_splits:
        if year not in years:
            continue
        cap = train_remaining if train_remaining is not None else None
        n = convert_voc_split(
            raw_dir, train_dir, year, split, class_names, class_to_idx, cap
        )
        train_total += n
        print(f"  {year}/{split}: {n} images")
        if train_remaining is not None:
            train_remaining -= n
            if train_remaining <= 0:
                break

    print("\n=== PASCAL VOC: convert val ===")
    val_total = 0
    val_remaining = max_val_samples
    for year, split in val_splits:
        if year not in years and split != "test":
            # VOC2007 test is always valid even if only 2007 downloaded
            pass
        cap = val_remaining if val_remaining is not None else None
        try:
            n = convert_voc_split(
                raw_dir, val_dir, year, split, class_names, class_to_idx, cap
            )
        except FileNotFoundError as exc:
            print(f"  Skipping {year}/{split}: {exc}")
            continue
        val_total += n
        print(f"  {year}/{split}: {n} images")
        if val_remaining is not None:
            val_remaining -= n
            if val_remaining <= 0:
                break

    metadata = {
        "classes": class_names,
        "num_classes": len(class_names),
        "train_samples": train_total,
        "val_samples": val_total,
        "years": years,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "train_root": str(train_dir),
        "val_root": str(val_dir),
    }

    meta_path = output_dir / "dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Done ===")
    print(f"  Train: {train_total} images -> {train_dir}")
    print(f"  Val:   {val_total} images -> {val_dir}")
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Metadata: {meta_path}")

    if selected_classes and len(class_names) != 20:
        print(
            f"\n  NOTE: Set model.num_classes={len(class_names)} in config/train.yaml"
        )

    return metadata
