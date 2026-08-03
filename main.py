from __future__ import annotations

from collections import Counter, deque
import time

import cv2

import dice_logic
import dice_ui
from dice_inference import DiceRecognizer


WINDOW_NAME = "Yacht Dice Game"
CLEAR_FRAMES_REQUIRED = 5
# 최근 10개 인식 결과 중 같은 주사위 리스트가 7번 이상 나오면 확정한다.
VOTE_WINDOW_SIZE = 10
VOTES_REQUIRED = 7
def add_detection_vote(
    vote_history: deque[tuple[int, ...]],
    detected_values: list[int],
) -> tuple[list[int] | None, int]:
    """인식 결과를 투표에 추가하고 안정된 주사위 리스트와 최다 득표수를 반환한다."""
    vote_history.append(tuple(detected_values))
    most_common_values, vote_count = Counter(vote_history).most_common(1)[0]
    if vote_count >= VOTES_REQUIRED:
        return list(most_common_values), vote_count
    return None, vote_count


def score_totals(committed_scores: dict[str, int]) -> dict[str, int]:
    """Build the subtotal, bonus, and total using dice_logic only."""
    upper = dice_logic.calculate_upper_bonus(committed_scores)
    return {
        "upper_total": upper["upper_total"],
        "upper_bonus": upper["upper_bonus"],
        "grand_total": dice_logic.calculate_grand_total(committed_scores),
    }


def main() -> None:
    recognizer = DiceRecognizer(confidence=0.40, device=0)
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera 0 could not be opened.")

    # 플레이어가 바뀌거나 점수를 확정하면 비워지는 최근 인식 결과 목록이다.
    vote_history: deque[tuple[int, ...]] = deque(maxlen=VOTE_WINDOW_SIZE)

    state = {
        "active_player": 1,
        "committed_scores": {1: {}, 2: {}},
        "candidate_scores": {1: {}, 2: {}},
        "dice_values": None,
        "hovered_button": None,
        "awaiting_clear": False,
        "clear_frames": 0,
        "game_over": False,
        "effects": [],
        "game_result": None,
        "game_over_hovered": None,
        "quit_requested": False,
        "vote_count": 0,
    }

    def is_available(button) -> bool:
        """Only the active player's detected, unused score cells are buttons."""
        if button is None or state["game_over"] or state["awaiting_clear"]:
            return False
        player, key = button
        return (
            player == state["active_player"]
            and key in state["candidate_scores"][player]
            and key not in state["committed_scores"][player]
        )

    def on_mouse(event, x, y, flags, param) -> None:
        if state["game_over"]:
            result_ready = (
                state["game_result"] is not None
                and time.monotonic() >= state["game_result"]["started_at"] + 1.05
            )
            action = dice_ui.get_game_over_button_at(x, y) if result_ready else None
            if event == cv2.EVENT_MOUSEMOVE:
                state["game_over_hovered"] = action
            elif event == cv2.EVENT_LBUTTONUP and action == "quit":
                state["quit_requested"] = True
            elif event == cv2.EVENT_LBUTTONUP and action == "restart":
                vote_history.clear()
                state.update({
                    "active_player": 1,
                    "committed_scores": {1: {}, 2: {}},
                    "candidate_scores": {1: {}, 2: {}},
                    "dice_values": None,
                    "hovered_button": None,
                    "awaiting_clear": True,
                    "clear_frames": 0,
                    "game_over": False,
                    "effects": [],
                    "game_result": None,
                    "game_over_hovered": None,
                    "vote_count": 0,
                })
            return

        button = dice_ui.get_score_button_at(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            state["hovered_button"] = button if is_available(button) else None
            return

        if event != cv2.EVENT_LBUTTONUP or not is_available(button):
            return

        player, key = button
        score = state["candidate_scores"][player][key]
        bonus_before = dice_logic.calculate_upper_bonus(state["committed_scores"][player])["upper_bonus"]
        state["committed_scores"][player][key] = score
        bonus_after = dice_logic.calculate_upper_bonus(state["committed_scores"][player])["upper_bonus"]
        effect_time = time.monotonic()
        state["effects"].append({
            "type": "score", "player": player, "key": key,
            "score": score, "started_at": effect_time,
        })
        if key == "yacht" and score > 0:
            state["effects"].append({"type": "yacht", "started_at": effect_time})
        if bonus_before == 0 and bonus_after == dice_logic.BONUS_SCORE:
            state["effects"].append({"type": "bonus", "started_at": effect_time})
        state["candidate_scores"][player] = {}
        state["dice_values"] = None
        state["hovered_button"] = None
        state["vote_count"] = 0
        vote_history.clear()

        if all(len(state["committed_scores"][p]) == len(dice_ui.SCORING_KEYS) for p in (1, 2)):
            state["game_over"] = True
            total_p1 = dice_logic.calculate_grand_total(state["committed_scores"][1])
            total_p2 = dice_logic.calculate_grand_total(state["committed_scores"][2])
            if total_p1 > total_p2:
                winner_text = "1P win!"
            elif total_p2 > total_p1:
                winner_text = "2P win!"
            else:
                winner_text = "Draw!"
            if key == "yacht" and score > 0:
                celebration_delay = 2.6
            elif bonus_before == 0 and bonus_after == dice_logic.BONUS_SCORE:
                celebration_delay = 2.2
            else:
                celebration_delay = 0.9
            state["game_result"] = {
                "winner_text": winner_text,
                "score_p1": total_p1,
                "score_p2": total_p2,
                "started_at": effect_time + celebration_delay,
            }
        else:
            state["active_player"] = 2 if player == 1 else 1
            # Do not reuse dice left in view for the next player.
            state["awaiting_clear"] = True
            state["clear_frames"] = 0

        print(f"Player {player} confirmed {key}: {score}")

    print("Show five dice, then click a score cell. Press q to quit.")
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    try:
        while True:
            if state["quit_requested"]:
                break
            success, frame = camera.read()
            if not success:
                print("Camera frame could not be read.")
                break

            # Stop running YOLO after the game ends so the result animation stays smooth.
            detected_values = None if state["game_over"] else recognizer.predict(frame)

            if state["awaiting_clear"]:
                # 이전 플레이어의 주사위는 다음 플레이어 투표에 포함하지 않는다.
                vote_history.clear()
                state["vote_count"] = 0
                if detected_values is None:
                    state["clear_frames"] += 1
                else:
                    state["clear_frames"] = 0

                if state["clear_frames"] >= CLEAR_FRAMES_REQUIRED:
                    state["awaiting_clear"] = False
                    state["clear_frames"] = 0
            elif not state["game_over"] and detected_values is not None:
                # 한 프레임의 결과를 바로 쓰지 않고 최근 결과의 다수결로 안정화한다.
                stable_values, vote_count = add_detection_vote(vote_history, detected_values)
                state["vote_count"] = vote_count
                if stable_values is not None and stable_values != state["dice_values"]:
                    active_player = state["active_player"]
                    state["dice_values"] = stable_values
                    state["candidate_scores"][active_player] = dice_logic.calculate_scores(stable_values)

            committed_p1 = state["committed_scores"][1]
            committed_p2 = state["committed_scores"][2]
            animation_time = time.monotonic()
            effect_durations = {"score": 0.9, "yacht": 2.6, "bonus": 2.2}
            state["effects"] = [
                effect for effect in state["effects"]
                if animation_time - effect["started_at"] < effect_durations[effect["type"]]
            ]
            current_turn = min(12, 1 + (len(committed_p1) + len(committed_p2)) // 2)
            display_dice = state["dice_values"] or []

            ui_screen = dice_ui.create_full_ui(
                frame,
                display_dice,
                state["candidate_scores"][1],
                state["candidate_scores"][2],
                current_turn=current_turn,
                active_player=state["active_player"],
                committed_scores_p1=committed_p1,
                committed_scores_p2=committed_p2,
                hovered_button=state["hovered_button"],
                score_totals_p1=score_totals(committed_p1),
                score_totals_p2=score_totals(committed_p2),
                effects=state["effects"],
                animation_time=animation_time,
                game_result=state["game_result"],
                game_over_hovered=state["game_over_hovered"],
            )

            if state["game_over"]:
                status = "GAME OVER"
            elif state["awaiting_clear"]:
                status = f"Player {state['active_player']}: remove the previous dice"
            elif state["dice_values"] is None:
                status = (
                    f"Player {state['active_player']}: stabilizing "
                    f"{state['vote_count']}/{VOTES_REQUIRED}"
                )
            else:
                status = f"Player {state['active_player']} dice: {state['dice_values']}"
            cv2.putText(ui_screen, status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, ui_screen)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
