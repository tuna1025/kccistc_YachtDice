"""프로젝트의 두 주사위 데이터셋을 합쳐 YOLO11 모델을 학습한다.

첫 번째 데이터셋은 export/images와 export/labels만 있으므로 80/10/10으로
분할한다. 두 번째 Roboflow 데이터셋은 기존 train/valid/test 분할을 유지한다.
각 분할을 합친 목록 파일을 만든 뒤 학습에 사용한다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from ultralytics import YOLO


# train.py 파일이 있는 프로젝트 최상위 폴더
PROJECT_DIR = Path(__file__).resolve().parent
# 기존 데이터셋과 새로 추가한 데이터셋의 폴더 위치
ORIGINAL_DATASET_DIR = PROJECT_DIR / "Dice.v2-medium-color.yolov11"
NEW_DATASET_DIR = PROJECT_DIR / "yolo-dice.v9i.yolov11"
# 실행할 때 생성되는 통합 데이터 목록과 data.yaml 저장 위치
SPLITS_DIR = PROJECT_DIR / "dataset_splits"
DATA_YAML = SPLITS_DIR / "combined_dice_data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO11 on the combined dice datasets.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 selects automatically.")
    parser.add_argument("--device", default=0, help="GPU id (0), cpu, or another device.")
    parser.add_argument("--model", default="yolo11s.pt", help="Starting model weights.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the original dataset split.")
    # 주피터가 자동으로 전달하는 --f=... 커널 인자는 무시한다.
    args, _ = parser.parse_known_args()
    return args


def paired_images(images_dir: Path, labels_dir: Path) -> list[Path]:
    """동일한 파일명의 YOLO 라벨(txt)이 있는 이미지만 반환한다."""
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected dataset folders:\n  {images_dir}\n  {labels_dir}")

    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    pairs = [path for path in images if (labels_dir / f"{path.stem}.txt").is_file()]
    if not pairs:
        raise RuntimeError(f"No image/label pairs found in {images_dir}")
    if len(pairs) != len(images):
        print(f"Warning: {images_dir} skipped {len(images) - len(pairs)} image(s) without labels.")
    return pairs


def validate_class_config() -> dict:
    """두 데이터셋의 클래스 번호와 이름이 정확히 같은지 확인한다."""
    original = yaml.safe_load((ORIGINAL_DATASET_DIR / "data.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((NEW_DATASET_DIR / "data.yaml").read_text(encoding="utf-8"))
    if original["nc"] != new["nc"] or original["names"] != new["names"]:
        raise ValueError(
            "The datasets have different class mappings and cannot be combined safely.\n"
            f"Original: {original['names']}\nNew: {new['names']}"
        )
    return original


def write_combined_dataset_yaml(seed: int) -> Path:
    """원본을 복사하지 않고 통합 train/val/test 이미지 목록을 만든다."""
    config = validate_class_config()

    # export만 있는 기존 데이터셋은 학습 80%, 검증 10%, 테스트 10%로 나눈다.
    original_images = paired_images(
        ORIGINAL_DATASET_DIR / "export" / "images",
        ORIGINAL_DATASET_DIR / "export" / "labels",
    )
    random.Random(seed).shuffle(original_images)
    train_end = int(len(original_images) * 0.8)
    val_end = train_end + int(len(original_images) * 0.1)
    splits = {
        "train": original_images[:train_end],
        "val": original_images[train_end:val_end],
        "test": original_images[val_end:],
    }

    # 새 Roboflow 데이터셋의 기존 분할은 유지한 채 해당 목록에 추가한다.
    for split_name, folder_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        splits[split_name].extend(
            paired_images(
                NEW_DATASET_DIR / folder_name / "images",
                NEW_DATASET_DIR / folder_name / "labels",
            )
        )
        random.Random(seed).shuffle(splits[split_name])

    SPLITS_DIR.mkdir(exist_ok=True)
    for split_name, images in splits.items():
        (SPLITS_DIR / f"{split_name}.txt").write_text(
            "\n".join(str(path.resolve()) for path in images) + "\n",
            encoding="utf-8",
        )

    data_config = {
        "train": str((SPLITS_DIR / "train.txt").resolve()),
        "val": str((SPLITS_DIR / "val.txt").resolve()),
        "test": str((SPLITS_DIR / "test.txt").resolve()),
        "nc": config["nc"],
        "names": config["names"],
    }
    DATA_YAML.write_text(yaml.safe_dump(data_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Combined split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    return DATA_YAML


def main() -> None:
    args = parse_args()
    # 두 데이터셋을 합친 data.yaml을 먼저 생성한다.
    data_yaml = write_combined_dataset_yaml(args.seed)

    # 사전 학습된 YOLO 모델을 불러와 통합 데이터셋으로 추가 학습한다.
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(PROJECT_DIR / "runs"),
        name="dice_yolo11",
        exist_ok=True,
        seed=args.seed,
        patience=20,
        plots=True,
    )
    print(f"Training finished. Best weights: {Path(results.save_dir) / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
