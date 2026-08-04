"""웹캠 YOLO 인식값과 요트다이스 점수 로직을 연결해 테스트한다.

실행:
    conda activate yacht_env
    python tests/camera_logic_test.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2

# tests 폴더에서 파일을 직접 실행해도 프로젝트의 핵심 모듈을 찾도록 한다.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dice_inference import DiceRecognizer
from dice_logic import calculate_scores


def printable_scores(scores: dict[str, int]) -> dict[str, int]:
    """콘솔에는 0점인 조합 점수를 제외해 보기 쉽게 만든다."""
    upper_categories = {"ones", "twos", "threes", "fours", "fives", "sixes", "choice"}
    return {name: score for name, score in scores.items() if name in upper_categories or score > 0}


def main() -> None:
    # YOLO 모델은 시작할 때 한 번만 불러온다.
    recognizer = DiceRecognizer(confidence=0.40, device=0)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    last_dice_values: list[int] | None = None
    last_scores: dict[str, int] | None = None
    print("주사위 5개를 화면에 놓으세요. q를 누르면 종료합니다.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # YOLO가 5개를 정확히 찾으면 [1, 2, 3, 4, 6] 같은 정렬된 리스트를 반환한다.
            dice_values = recognizer.predict(frame)
            if dice_values is not None:
                # 인식 리스트를 이번 턴의 요트다이스 점수 계산 함수에 전달한다.
                scores = calculate_scores(dice_values)

                # 동일한 프레임 결과가 반복될 때 콘솔이 너무 많이 출력되는 것을 막는다.
                if dice_values != last_dice_values:
                    print(f"\n인식 주사위: {dice_values}")
                    print("가능 점수:", printable_scores(scores))
                    last_dice_values = dice_values
                    last_scores = scores

            # 화면에는 가장 최근에 정상 인식한 주사위 리스트와 대표 점수를 표시한다.
            output = frame.copy()
            if last_dice_values is None:
                message = "5 dice are needed"
                detail = ""
            else:
                message = f"Dice: {last_dice_values}"
                detail = (
                    f"Choice: {last_scores['choice']}  "
                    f"Small: {last_scores['small_straight']}  "
                    f"Large: {last_scores['large_straight']}  "
                    f"Yacht: {last_scores['yacht']}"
                )

            cv2.putText(output, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(output, detail, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Camera + Yacht Dice Logic Test", output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
