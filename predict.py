"""학습이 끝난 YOLO11 주사위 모델로 이미지, 폴더 또는 웹캠을 테스트한다.

예시
----
# 이미지 한 장 테스트
python predict.py --source "C:\\path\\to\\dice.jpg"

# 테스트 이미지 폴더 전체 테스트
python predict.py --source "Dice.v2-medium-color.yolov11\\export\\images"

# 기본 웹캠 테스트 (q를 누르면 종료)
python predict.py --source 0 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


# 이 파일이 있는 프로젝트 폴더와 기본 학습 가중치 경로
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "dice_yolo11" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    """터미널 또는 주피터에서 전달한 테스트 옵션을 읽는다."""
    parser = argparse.ArgumentParser(description="Test the trained YOLO11 dice model.")
    parser.add_argument(
        "--source",
        default=str(PROJECT_DIR / "Dice.v2-medium-color.yolov11" / "export" / "images"),
        help="Image path, image folder, video path, URL, or webcam number (0).",
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the trained best.pt file.")
    parser.add_argument("--conf", type=float, default=0.40, help="Minimum confidence score (0.0 to 1.0).")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=0, help="GPU id (0), cpu, or another device.")
    parser.add_argument("--show", action="store_true", help="Show prediction results in a window.")
    # 주피터의 --f=... 커널 인자는 무시한다.
    args, _ = parser.parse_known_args()
    return args


def normalize_source(source: str) -> str | int:
    """'0'처럼 입력된 웹캠 번호는 정수로 바꾼다."""
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)

    # 학습을 끝내기 전에는 best.pt가 없으므로, 이해하기 쉬운 오류를 출력한다.
    if not weights.is_file():
        raise FileNotFoundError(
            f"학습 가중치를 찾을 수 없습니다: {weights}\n"
            "먼저 test.py로 학습을 완료했는지 확인하거나 --weights에 best.pt 경로를 지정하세요."
        )

    # 학습된 모델을 불러와 예측한다. 결과 이미지/영상은 runs/predict/dice_test에 저장된다.
    model = YOLO(str(weights))
    results = model.predict(
        source=normalize_source(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        show=args.show,
        project=str(PROJECT_DIR / "runs" / "predict"),
        name="dice_test",
        exist_ok=True,
    )

    print(f"테스트 완료: {len(results)}개 결과")
    print(f"저장 위치: {PROJECT_DIR / 'runs' / 'predict' / 'dice_test'}")


if __name__ == "__main__":
    main()
