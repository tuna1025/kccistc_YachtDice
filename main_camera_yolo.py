"""기존 UI에 카메라 YOLO 주사위 인식 결과를 연결하는 테스트 실행 파일.

기존 main.py는 수정하지 않는다.

실행:
    conda activate yacht_env
    python main_camera_yolo.py

조작:
    1: 1P를 현재 플레이어로 선택
    2: 2P를 현재 플레이어로 선택
    q: 프로그램 종료
"""

from __future__ import annotations

import cv2

import dice_ui
from dice_inference import DiceRecognizer
from dice_logic import LOWER_CATEGORIES, UPPER_CATEGORIES, calculate_scores


def empty_scores() -> dict[str, int]:
    """아직 인식값이 없는 플레이어의 빈 점수 표시용 딕셔너리를 만든다."""
    scores = {category: 0 for category in (*UPPER_CATEGORIES, *LOWER_CATEGORIES)}
    scores["upper_total"] = 0
    scores["upper_bonus"] = 0
    scores["grand_total"] = 0
    return scores


def turn_scores(dice_values: list[int] | None) -> dict[str, int]:
    """현재 인식한 주사위로 UI에 보여 줄 이번 턴의 점수 후보를 만든다.

    이 파일은 카메라 연동 테스트용이다. 실제 게임에서 선택한 점수를 누적하는
    기능은 아직 적용하지 않고, 현재 주사위로 얻을 수 있는 점수만 표시한다.
    """
    if dice_values is None:
        return empty_scores()

    scores = calculate_scores(dice_values)
    # 현재 값은 아직 점수표에 확정하지 않은 '후보'다.
    # 따라서 카테고리 행에는 후보 점수를 보여 주되, Subtotal/Total에는 합산하지 않는다.
    # 실제 게임에서 사용자가 카테고리를 선택했을 때만 별도 점수표에 기록해 합산한다.
    scores["upper_total"] = 0
    scores["upper_bonus"] = 0
    scores["grand_total"] = 0
    return scores


def main() -> None:
    # 모델은 시작할 때 한 번만 불러온 뒤 모든 웹캠 프레임에 재사용한다.
    recognizer = DiceRecognizer(confidence=0.40, device=0)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("웹캠을 열 수 없습니다.")

    active_player = 1
    last_dice_values: list[int] | None = None
    print("YOLO 카메라 UI 테스트를 시작합니다. 1/2: 플레이어 선택, q: 종료")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # 주사위 5개가 모두 검출되면 [1, 2, 3, 4, 6]처럼 정렬된 리스트를 받는다.
            detected_values = recognizer.predict(frame)
            if detected_values is not None:
                last_dice_values = detected_values

            # UI에는 최근에 정상 검출한 리스트를 사용한다.
            display_dice = last_dice_values if last_dice_values is not None else [0, 0, 0, 0, 0]
            current_scores = turn_scores(last_dice_values)
            scores_p1 = current_scores if active_player == 1 else empty_scores()
            scores_p2 = current_scores if active_player == 2 else empty_scores()

            ui_screen = dice_ui.create_full_ui(
                frame,
                display_dice,
                scores_p1,
                scores_p2,
                current_turn=1,
                active_player=active_player,
            )

            # 플레이어가 확인할 수 있도록 인식된 Python 리스트를 화면 상단에 직접 표시한다.
            if last_dice_values is None:
                message = "Detecting 5 dice..."
            else:
                message = f"Player {active_player} dice: {last_dice_values}"
            cv2.putText(ui_screen, message, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow("Yacht Dice - YOLO Camera Test", ui_screen)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("1"):
                active_player = 1
            if key == ord("2"):
                active_player = 2
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
