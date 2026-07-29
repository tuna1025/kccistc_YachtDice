"""학습된 YOLO 모델로 주사위 눈값 5개를 리스트로 반환한다.

다른 코드에서 사용하는 예시:

    from dice_inference import DiceRecognizer

    recognizer = DiceRecognizer()
    dice_values = recognizer.predict(frame)  # 예: [1, 2, 3, 4, 6]
    if dice_values is not None:
        print(dice_values)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
# 학습이 끝난 뒤 생성되는 최고 성능 모델 가중치 파일
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "dice_yolo11" / "weights" / "best.pt"


class DiceRecognizer:
    """YOLO 모델을 한 번만 불러오고, 여러 웹캠 프레임을 반복 예측한다."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        confidence: float = 0.40,
        image_size: int = 640,
        device: int | str = 0,
    ) -> None:
        weights_path = Path(weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"학습 가중치를 찾을 수 없습니다: {weights_path}")

        # 모델은 여기서 한 번만 메모리에 올린다.
        # 웹캠 프레임마다 새로 불러오면 매우 느려지므로 recognizer 객체를 재사용해야 한다.
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def predict(self, frame: np.ndarray, expected_dice: int = 5) -> list[int] | None:
        """프레임에서 주사위 5개를 찾으면 오름차순 눈값 리스트를 반환한다.

        주사위가 정확히 expected_dice개가 아니면 아직 굴리는 중이거나 오인식일 수
        있으므로 None을 반환한다.
        """
        # frame은 cv2.VideoCapture().read()로 얻는 BGR 이미지다.
        # YOLO가 프레임에서 주사위 박스와 1~6 클래스를 예측한다.
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        # YOLO 클래스 번호(0~5)가 아니라 data.yaml에 정의된 실제 눈값 문자열(1~6)을 사용한다.
        dice_values = [int(result.names[int(box.cls[0])]) for box in result.boxes]
        if len(dice_values) != expected_dice:
            # 굴리는 중이거나 일부 주사위가 가려졌을 수 있으므로 점수 계산을 하지 않는다.
            return None

        # 요트다이스 점수 계산에는 화면상 위치가 필요 없으므로 숫자 오름차순으로 정렬한다.
        return sorted(dice_values)


def parse_args() -> argparse.Namespace:
    """파일 단독 실행 때 사용할 웹캠 테스트 옵션을 읽는다."""
    parser = argparse.ArgumentParser(description="Return five dice values from a webcam frame.")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="best.pt 경로")
    parser.add_argument("--conf", type=float, default=0.40, help="최소 신뢰도")
    parser.add_argument("--device", default=0, help="GPU 번호(0) 또는 cpu")
    args, _ = parser.parse_known_args()  # 주피터의 --f=... 인자는 무시한다.
    return args


def main() -> None:
    """웹캠 테스트: 5개가 검출된 프레임의 리스트를 콘솔에 출력한다."""
    args = parse_args()
    # DiceRecognizer는 루프 밖에서 한 번만 생성한다.
    recognizer = DiceRecognizer(args.weights, confidence=args.conf, device=args.device)
    # 기본 카메라(0)를 연다. 다른 카메라는 --camera 1처럼 지정할 수 있다.
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"웹캠 {args.camera}를 열 수 없습니다.")

    last_values: list[int] | None = None
    print("웹캠을 시작합니다. q를 누르면 종료합니다.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # 프레임을 전달하면 [1, 2, 3, 4, 6]처럼 정렬된 리스트 또는 None을 받는다.
            dice_values = recognizer.predict(frame)
            if dice_values is not None and dice_values != last_values:
                # 같은 값은 한 번만 출력한다. 게임에서는 이 위치에서 점수 계산 함수를 호출하면 된다.
                print(f"예측 주사위 값: {dice_values}")
                last_values = dice_values

            output = frame.copy()
            message = str(dice_values) if dice_values is not None else "주사위 5개를 찾는 중"
            cv2.putText(output, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Dice inference", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
