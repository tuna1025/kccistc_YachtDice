# main.py
import cv2
import dice_ui
import dice_logic

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: 웹캠을 열 수 없습니다.")
        return

    print("Yacht Dice 1P/2P 점수판 UI가 실행되었습니다. 종료하려면 'q'를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽어올 수 없습니다.")
            break

        # 1P 및 2P 주사위 눈금 예시 데이터
        dice_p1 = [1, 2, 4, 5, 6]
        dice_p2 = [3, 3, 3, 5, 6]  # 현재 롤링된 주사위

        # 각 플레이어별 점수 계산
        scores_p1 = dice_logic.calculate_scores(dice_p1)
        scores_p2 = dice_logic.calculate_scores(dice_p2)

        # 1P/2P 분할 점수판 UI 생성 (Turn 7/12, Active Player: 2P)
        ui_screen = dice_ui.create_full_ui(frame, dice_p2, scores_p1, scores_p2, current_turn=7, active_player=2)

        cv2.imshow("Yacht Dice Game", ui_screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()