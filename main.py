import cv2

import dice_logic
import dice_ui


WINDOW_NAME = "Yacht Dice Game"


def main():
    cap = cv2.VideoCapture(0)
    state = {
        "active_player": 1,
        "committed_scores": {1: {}, 2: {}},
        "candidate_scores": {1: {}, 2: {}},
        "hovered_button": None,
    }

    def is_available(button):
        """A score button is enabled only for the active player and unused category."""
        if button is None:
            return False
        player, key = button
        return player == state["active_player"] and key not in state["committed_scores"][player]

    def on_mouse(event, x, y, flags, param):
        button = dice_ui.get_score_button_at(x, y)
        if event == cv2.EVENT_MOUSEMOVE:
            state["hovered_button"] = button if is_available(button) else None
        elif event == cv2.EVENT_LBUTTONUP and is_available(button):
            player, key = button
            state["committed_scores"][player][key] = state["candidate_scores"][player][key]
            state["active_player"] = 2 if player == 1 else 1
            state["hovered_button"] = None

    if not cap.isOpened():
        print("Error: camera could not be opened.")
        return

    print("Click an enabled score cell to confirm it. Press q to quit.")
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: frame could not be read.")
            break

        # Replace these sample rolls with the dice detector's live output when available.
        dice_p1 = [1, 2, 4, 5, 6]
        dice_p2 = [3, 3, 3, 5, 6]
        scores_p1 = dice_logic.calculate_scores(dice_p1)
        scores_p2 = dice_logic.calculate_scores(dice_p2)
        state["candidate_scores"][1] = scores_p1
        state["candidate_scores"][2] = scores_p2

        current_turn = 1 + (len(state["committed_scores"][1]) + len(state["committed_scores"][2])) // 2
        current_dice = dice_p1 if state["active_player"] == 1 else dice_p2
        ui_screen = dice_ui.create_full_ui(
            frame, current_dice, scores_p1, scores_p2,
            current_turn=current_turn,
            active_player=state["active_player"],
            committed_scores_p1=state["committed_scores"][1],
            committed_scores_p2=state["committed_scores"][2],
            hovered_button=state["hovered_button"],
        )
        cv2.imshow(WINDOW_NAME, ui_screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
