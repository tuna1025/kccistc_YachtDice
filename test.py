"""Train a YOLO11 dice detector from the Roboflow export in this project.

Run this script from the `torch_env` conda environment used in the class
notebooks:

    conda activate torch_env
    pip install ultralytics
    python test.py

The supplied dataset contains `export/images` and `export/labels`, rather
than pre-made train/valid/test directories.  This script creates deterministic
80/10/10 image lists in `dataset_splits/` and gives those lists to Ultralytics.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "Dice.v2-medium-color.yolov11"
IMAGES_DIR = DATASET_DIR / "export" / "images"
LABELS_DIR = DATASET_DIR / "export" / "labels"
SPLITS_DIR = PROJECT_DIR / "dataset_splits"
DATA_YAML = SPLITS_DIR / "dice_data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 on the dice dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 selects automatically.")
    parser.add_argument("--device", default=None, help="GPU id (for example 0), cpu, or leave blank for automatic selection.")
    parser.add_argument("--model", default="yolo11n.pt", help="Starting model weights.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for data splitting.")
    return parser.parse_args()


def image_label_pairs() -> list[Path]:
    """Return only images that have their matching YOLO label file."""
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError(
            "Dataset folders were not found. Expected:\n"
            f"  {IMAGES_DIR}\n  {LABELS_DIR}"
        )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(path for path in IMAGES_DIR.iterdir() if path.suffix.lower() in image_extensions)
    paired_images = [path for path in images if (LABELS_DIR / f"{path.stem}.txt").exists()]

    if not paired_images:
        raise RuntimeError("No image/label pairs found in export/images and export/labels.")
    if len(paired_images) != len(images):
        print(f"Warning: skipping {len(images) - len(paired_images)} image(s) without a label file.")
    return paired_images


def write_dataset_yaml(seed: int) -> Path:
    """Create split lists and a YOLO data YAML without modifying source images."""
    images = image_label_pairs()
    random.Random(seed).shuffle(images)

    total = len(images)
    train_end = int(total * 0.8)
    valid_end = train_end + int(total * 0.1)
    splits = {"train": images[:train_end], "val": images[train_end:valid_end], "test": images[valid_end:]}

    SPLITS_DIR.mkdir(exist_ok=True)
    for split_name, split_images in splits.items():
        (SPLITS_DIR / f"{split_name}.txt").write_text(
            "\n".join(str(path.resolve()) for path in split_images) + "\n",
            encoding="utf-8",
        )

    source_config = yaml.safe_load((DATASET_DIR / "data.yaml").read_text(encoding="utf-8"))
    data_config = {
        "train": str((SPLITS_DIR / "train.txt").resolve()),
        "val": str((SPLITS_DIR / "val.txt").resolve()),
        "test": str((SPLITS_DIR / "test.txt").resolve()),
        "nc": source_config["nc"],
        "names": source_config["names"],
    }
    DATA_YAML.write_text(yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    print(f"Dataset split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Generated data config: {DATA_YAML}")
    return DATA_YAML


def main() -> None:
    args = parse_args()
    data_yaml = write_dataset_yaml(args.seed)

    model = YOLO(args.model)
    train_options = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(PROJECT_DIR / "runs"),
        "name": "dice_yolo11",
        "exist_ok": True,
        "seed": args.seed,
        "patience": 20,
        "plots": True,
    }
    if args.device is not None:
        train_options["device"] = args.device

    results = model.train(**train_options)
    print(f"Training finished. Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
