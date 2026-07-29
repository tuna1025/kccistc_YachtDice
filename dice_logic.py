# dice_logic.py

def calculate_scores(dice):
    """
    5개의 주사위 눈금 리스트(예: [1, 3, 3, 5, 6])를 받아
    야트 게임의 모든 족보별 점수를 딕셔너리 형태로 반환합니다.
    """
    scores = {}
    
    # 등장 횟수 카운트용 딕셔너리
    counts = {i: dice.count(i) for i in range(1, 7)}
    total_sum = sum(dice)
    
    # 1. 상단 영역 (Ones ~ Sixes)
    scores['ones'] = counts[1] * 1
    scores['twos'] = counts[2] * 2
    scores['threes'] = counts[3] * 3
    scores['fours'] = counts[4] * 4
    scores['fives'] = counts[5] * 5
    scores['sixes'] = counts[6] * 6
    
    # 상단 보너스 판단을 위한 합계 (1~6 총점)
    upper_total = sum([scores['ones'], scores['twos'], scores['threes'], 
                       scores['fours'], scores['fives'], scores['sixes']])
    scores['upper_bonus'] = 35 if upper_total >= 63 else 0
    
    # 2. 하단 영역 (Choice, 4 of a Kind, Full House, Straights, Yacht 등)
    scores['choice'] = total_sum
    
    # 4 of a Kind (같은 눈 4개 이상)
    is_four = any(count >= 4 for count in counts.values())
    scores['four_of_a_kind'] = total_sum if is_four else 0
    
    # Full House (3개 + 2개 조합)
    has_three = 3 in counts.values()
    has_two = 2 in counts.values()
    is_full_house = (has_three and has_two) or (5 in counts.values())
    scores['full_house'] = total_sum if is_full_house else 0
    
    # Small Straight (4개 연속)
    unique_sorted = sorted(list(set(dice)))
    is_s_straight = False
    s_patterns = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
    for p in s_patterns:
        if all(val in unique_sorted for val in p):
            is_s_straight = True
            break
    scores['small_straight'] = 15 if is_s_straight else 0
    
    # Large Straight (5개 연속)
    is_l_straight = unique_sorted in [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
    scores['large_straight'] = 30 if is_l_straight else 0
    
    # Yacht (5개 모두 같은 눈)
    is_yacht = 5 in counts.values()
    scores['yacht'] = 50 if is_yacht else 0
    
    # 총점 계산
    upper_sum = upper_total + scores['upper_bonus']
    lower_sum = sum([scores['choice'], scores['four_of_a_kind'], scores['full_house'], 
                     scores['small_straight'], scores['large_straight'], scores['yacht']])
    
    scores['grand_total'] = upper_sum + lower_sum
    
    return scores