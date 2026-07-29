# dice_ui.py
import cv2
import numpy as np

# --- 🎨 스타일 및 레이아웃 상수 정의 ---
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 960

CAM_HEIGHT = int(WINDOW_HEIGHT * 0.25)
DICE_ZONE_HEIGHT = int(WINDOW_HEIGHT * 0.16)
SCORE_BOARD_HEIGHT = WINDOW_HEIGHT - CAM_HEIGHT - DICE_ZONE_HEIGHT

# BGR 컬러 팔레트 (모던하고 부드러운 테마)
BG_COLOR = (248, 249, 250)
CARD_BG = (255, 255, 255)
SHADOW_COLOR = (220, 220, 220)
GRID_LINE_COLOR = (235, 235, 235)
BOLD_BORDER_COLOR = (50, 50, 50)
SPECIAL_ROW_BG = (225, 228, 232)  # Subtotal/Total 배경색
ACTIVE_P2_BG = (220, 240, 255)
TEXT_MAIN = (20, 20, 20)
TEXT_LIGHT = (20, 20, 20) # Special row 텍스트 색상 (어두운 배경에 흰색 대신)
TEXT_MUTED = (140, 140, 140)
ROW_ALT_COLOR = (247, 248, 249)

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1, corners=[True, True, True, True]):
    """모서리가 둥근 사각형을 그립니다. corners: [TL, TR, BL, BR]"""
    x1, y1 = pt1
    x2, y2 = pt2
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    tl, tr, bl, br = corners

    if thickness == -1:
        # 중앙 채우기
        cv2.rectangle(img, (x1 + radius, y1 + radius), (x2 - radius, y2 - radius), color, -1)
        # 변 채우기
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    # 모서리 그리기
    if tl: cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    else: cv2.rectangle(img, (x1, y1), (x1+radius, y1+radius), color, thickness)

    if tr: cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    else: cv2.rectangle(img, (x2-radius, y1), (x2, y1+radius), color, thickness)

    if bl: cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    else: cv2.rectangle(img, (x1, y2-radius), (x1+radius, y2), color, thickness)

    if br: cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    else: cv2.rectangle(img, (x2-radius, y2-radius), (x2, y2), color, thickness)

    # 선 그리기
    if thickness > 0:
        if tl and tr: cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        elif tl: cv2.line(img, (x1 + radius, y1), (x2, y1), color, thickness)
        elif tr: cv2.line(img, (x1, y1), (x2-radius, y1), color, thickness)
        else: cv2.line(img, (x1, y1), (x2, y1), color, thickness)

        if bl and br: cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        elif bl: cv2.line(img, (x1 + radius, y2), (x2, y2), color, thickness)
        elif br: cv2.line(img, (x1, y2), (x2 - radius, y2), color, thickness)
        else: cv2.line(img, (x1, y2), (x2, y2), color, thickness)

        if tl and bl: cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        elif tl: cv2.line(img, (x1, y1+radius), (x1, y2), color, thickness)
        elif bl: cv2.line(img, (x1, y1), (x1, y2-radius), color, thickness)
        else: cv2.line(img, (x1, y1), (x1, y2), color, thickness)

        if tr and br: cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        elif tr: cv2.line(img, (x2, y1+radius), (x2, y2), color, thickness)
        elif br: cv2.line(img, (x2, y1), (x2, y2-radius), color, thickness)
        else: cv2.line(img, (x2, y1), (x2, y2), color, thickness)

def draw_dice(img, number, x, y, size=60):
    """모던하고 깔끔한 디자인의 주사위를 그립니다."""
    
    if size > 25: # 큰 주사위에만 그림자 적용
        shadow_offset = 4
        cv2.rectangle(img, (x + shadow_offset, y + shadow_offset),
                      (x + size + shadow_offset, y + size + shadow_offset), SHADOW_COLOR, -1)
    
    cv2.rectangle(img, (x, y), (x + size, y + size), CARD_BG, -1)
    cv2.rectangle(img, (x, y), (x + size, y + size), (200, 200, 200), 1)

    dot_r = size // 11
    cx, cy = x + size // 2, y + size // 2
    dot_color = (40, 40, 40)
    
    positions = {
        1: [(cx, cy)], 2: [(x + size//4, y + size//4), (x + 3*size//4, y + 3*size//4)],
        3: [(x + size//4, y + size//4), (cx, cy), (x + 3*size//4, y + 3*size//4)],
        4: [(x + size//4, y + size//4), (x + 3*size//4, y + size//4), (x + size//4, y + 3*size//4), (x + 3*size//4, y + 3*size//4)],
        5: [(x + size//4, y + size//4), (x + 3*size//4, y + size//4), (cx, cy), (x + size//4, y + 3*size//4), (x + 3*size//4, y + 3*size//4)],
        6: [(x + size//4, y + size//4), (x + 3*size//4, y + size//4), (x + size//4, cy), (x + 3*size//4, cy), (x + size//4, y + 3*size//4), (x + 3*size//4, y + 3*size//4)]
    }
    if number in positions:
        for pos in positions[number]:
            cv2.circle(img, pos, dot_r, dot_color, -1)


def create_full_ui(frame, current_dice_values, scores_p1, scores_p2, current_turn=7, active_player=2):
    # --- 1. 상단 웹캠 영역 ---
    cam_zone = np.zeros((CAM_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    if frame is not None:
        h, w, _ = frame.shape
        target_aspect = WINDOW_WIDTH / CAM_HEIGHT
        frame_aspect = w / h
        if frame_aspect > target_aspect:
            new_w = int(h * target_aspect); x1 = (w - new_w) // 2
            cropped = frame[:, x1:x1+new_w]
        else:
            new_h = int(w / target_aspect); y1 = (h - new_h) // 2
            cropped = frame[y1:y1+new_h, :]
        cam_zone = cv2.resize(cropped, (WINDOW_WIDTH, CAM_HEIGHT))

    # --- 2. 중단 주사위 영역 ---
    dice_zone = np.full((DICE_ZONE_HEIGHT, WINDOW_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    cv2.putText(dice_zone, "CURRENT DICE", (25, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, TEXT_MUTED, 1)
    dice_area_width = (5 * 60) + (4 * 15)
    start_x = (WINDOW_WIDTH - dice_area_width) // 2
    for i, val in enumerate(current_dice_values):
        draw_dice(dice_zone, val, start_x + i * 75, (DICE_ZONE_HEIGHT - 60) // 2 + 10, size=60)

    # --- 3. 하단 점수판 영역 ---
    score_zone = np.full((SCORE_BOARD_HEIGHT, WINDOW_WIDTH, 3), BG_COLOR, dtype=np.uint8)
    bx1, by1, bx2, by2 = 15, 15, WINDOW_WIDTH - 15, SCORE_BOARD_HEIGHT - 20
    
    cv2.rectangle(score_zone, (bx1 + 5, by1 + 5), (bx2 + 5, by2 + 5), SHADOW_COLOR, -1)
    cv2.rectangle(score_zone, (bx1, by1), (bx2, by2), CARD_BG, -1)

    col_width = (bx2 - bx1) // 3
    cat_col_x, p1_x, p2_x = bx1, bx1 + col_width, bx1 + 2 * col_width

    if active_player == 1: draw_rounded_rect(score_zone, (p1_x, by1+1), (p2_x, by2-1), ACTIVE_P2_BG, radius=0)
    elif active_player == 2: draw_rounded_rect(score_zone, (p2_x, by1+1), (bx2-1, by2-1), ACTIVE_P2_BG, radius=0)

    header_y, header_line_y = by1 + 35, by1 + 55
    (w1, _), _ = cv2.getTextSize("1P", cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    (w2, _), _ = cv2.getTextSize("2P", cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
    cv2.putText(score_zone, f"Turn {current_turn}/12", (cat_col_x + 20, header_y), cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_MAIN, 1)
    cv2.putText(score_zone, "1P", (p1_x + (col_width - w1) // 2, header_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_MAIN, 1)
    cv2.putText(score_zone, "2P", (p2_x + (col_width - w2) // 2, header_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_MAIN, 2 if active_player == 2 else 1)
    cv2.line(score_zone, (bx1 + 1, header_line_y), (bx2 - 1, header_line_y), BOLD_BORDER_COLOR, 2)

    categories = [
        ("Aces", 'ones'), ("Deuces", 'twos'), ("Threes", 'threes'),
        ("Fours", 'fours'), ("Fives", 'fives'), ("Sixes", 'sixes'),
        ("Subtotal", 'upper_total', True), ("Choice", 'choice'), 
        ("4 of a Kind", 'four_of_a_kind'), ("Full House", 'full_house'),
        ("S. Straight", 'small_straight'), ("L. Straight", 'large_straight'), 
        ("Yacht", 'yacht'), ("Total", 'grand_total', True)
    ]
    
    y_pos, row_h = header_line_y, 32
    for i, item in enumerate(categories):
        is_special = len(item) == 3; label, key = item[0], item[1]
        
        row_y_start = y_pos
        text_y = row_y_start + 22
        
        if label == "Subtotal":
            cv2.rectangle(score_zone, (cat_col_x + 1, row_y_start), (bx2 - 1, row_y_start + row_h), SPECIAL_ROW_BG, -1)
            sub1, sub2 = f"{scores_p1.get('upper_total',0)}/63", f"{scores_p2.get('upper_total',0)}/63"
            (w1,_),_ = cv2.getTextSize(sub1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            (w2,_),_ = cv2.getTextSize(sub2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.putText(score_zone, "Subtotal", (cat_col_x + 20, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_MAIN, 1)
            cv2.putText(score_zone, sub1, (p1_x + (col_width - w1) // 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MAIN, 1)
            cv2.putText(score_zone, sub2, (p2_x + (col_width - w2) // 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MAIN, 1)
        elif label == "Total":
            cv2.rectangle(score_zone, (cat_col_x + 1, row_y_start), (bx2 - 1, by2 - 1), SPECIAL_ROW_BG, -1)
            total_h = by2 - row_y_start
            text_y = row_y_start + total_h // 2 + 5
            t1, t2 = str(scores_p1.get('grand_total', 0)), str(scores_p2.get('grand_total', 0))
            (w1,_),_ = cv2.getTextSize(t1, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2); (w2,_),_ = cv2.getTextSize(t2, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
            cv2.putText(score_zone, "Total", (cat_col_x + 20, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_MAIN, 2)
            cv2.putText(score_zone, t1, (p1_x + (col_width - w1) // 2, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_MAIN, 2)
            cv2.putText(score_zone, t2, (p2_x + (col_width - w2) // 2, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_MAIN, 2)
            y_pos += row_h; break
        else:
            if i < 6: draw_dice(score_zone, i + 1, cat_col_x + 15, row_y_start + 6, size=20)
            v1, v2 = str(scores_p1.get(key, '')), str(scores_p2.get(key, ''))
            (w1,_),_ = cv2.getTextSize(v1, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1);(w2,_),_ = cv2.getTextSize(v2, cv2.FONT_HERSHEY_DUPLEX, 0.55, 2 if active_player==2 else 1)
            cv2.putText(score_zone, label, (cat_col_x + 45, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_MAIN, 1)
            cv2.putText(score_zone, v1, (p1_x + (col_width - w1) // 2, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_MUTED if v1 == '' else TEXT_MAIN, 1)
            cv2.putText(score_zone, v2, (p2_x + (col_width - w2) // 2, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_MUTED if v2 == '' else TEXT_MAIN, 2 if active_player == 2 else 1)
        
        y_pos += row_h
        line_color = BOLD_BORDER_COLOR if is_special else GRID_LINE_COLOR
        cv2.line(score_zone, (bx1 + 1, y_pos), (bx2 - 1, y_pos), line_color, 2 if is_special else 1)
    
    final_y = min(y_pos, by2)
    cv2.line(score_zone, (p1_x, by1), (p1_x, final_y), BOLD_BORDER_COLOR, 2)
    cv2.line(score_zone, (p2_x, by1), (p2_x, final_y), BOLD_BORDER_COLOR, 2)

    cv2.rectangle(score_zone, (bx1, by1), (bx2, by2), BOLD_BORDER_COLOR, 3)

    full_screen = np.vstack((cam_zone, dice_zone, score_zone))
    return full_screen
