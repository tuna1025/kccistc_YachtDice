"""요트다이스의 점수 계산 로직.

카메라 인식 결과 예시:
    dice_values = [1, 2, 3, 4, 6]

사용 예시:
    from yacht_dice_logic import available_scores

    scores = available_scores(dice_values)
    # 조건을 만족하는 항목만 점수와 함께 반환한다.
"""

from __future__ import annotations

from collections import Counter


# 화면 표시나 다른 팀원 코드에서 공통으로 사용할 카테고리 이름
CATEGORIES = (
    "ones",
    "twos",
    "threes",
    "fours",
    "fives",
    "sixes",
    "choice",
    "four_of_a_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yacht",
)


def validate_dice(dice_values: list[int]) -> list[int]:
    """1~6 범위의 주사위 값이 정확히 5개인지 확인하고 정렬한다."""
    if len(dice_values) != 5:
        raise ValueError("주사위 값은 반드시 5개여야 합니다.")
    if any(value not in range(1, 7) for value in dice_values):
        raise ValueError("각 주사위 값은 1부터 6 사이여야 합니다.")
    return sorted(dice_values)


def has_small_straight(unique_values: set[int]) -> bool:
    """서로 다른 연속 숫자 4개가 있으면 스몰 스트레이트(15점)다."""
    return any(set(sequence).issubset(unique_values) for sequence in ((1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 6)))


def has_large_straight(unique_values: set[int]) -> bool:
    """1~5 또는 2~6이 모두 있으면 라지 스트레이트(30점)다."""
    return unique_values in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6})


def available_scores(dice_values: list[int]) -> dict[str, int]:
    """조건을 만족하는 카테고리와 점수를 반환한다.

    윗줄(ones~sixes)과 초이스는 언제나 선택할 수 있으므로 0점도 반환한다.
    아래쪽 조합 카테고리는 조건을 만족할 때만 결과에 포함한다.
    """
    dice = validate_dice(dice_values)
    counts = Counter(dice)
    unique_values = set(dice)
    total = sum(dice)
    scores = {
        "ones": counts[1] * 1,
        "twos": counts[2] * 2,
        "threes": counts[3] * 3,
        "fours": counts[4] * 4,
        "fives": counts[5] * 5,
        "sixes": counts[6] * 6,
        "choice": total,
    }

    # 같은 눈이 4개 이상이면 모든 주사위 눈의 합을 포카드 점수로 사용한다.
    if max(counts.values()) >= 4:
        scores["four_of_a_kind"] = total

    # 서로 다른 눈이 두 종류이고, 한 종류가 2개 이상이면 풀하우스로 인정한다.
    # 5개가 모두 같은 요트도 풀하우스로 사용할 수 있도록 허용한다.
    if len(counts) == 1 or (len(counts) == 2 and min(counts.values()) >= 2):
        scores["full_house"] = total

    if has_small_straight(unique_values):
        scores["small_straight"] = 15
    if has_large_straight(unique_values):
        scores["large_straight"] = 30
    if len(counts) == 1:
        scores["yacht"] = 50

    return scores


def score_text(dice_values: list[int]) -> list[str]:
    """UI에 바로 표시하기 좋은 '카테고리: 점수점' 문자열 목록을 만든다."""
    korean_names = {
        "ones": "에이스",
        "twos": "듀스",
        "threes": "트리플",
        "fours": "쿼드",
        "fives": "펜타",
        "sixes": "헥사",
        "choice": "초이스",
        "four_of_a_kind": "포카드",
        "full_house": "풀하우스",
        "small_straight": "스몰 스트레이트",
        "large_straight": "라지 스트레이트",
        "yacht": "요트",
    }
    return [f"{korean_names[category]}: {score}점" for category, score in available_scores(dice_values).items()]


if __name__ == "__main__":
    # 간단한 단독 실행 예시
    example = [2, 3, 4, 5, 6]
    print(f"주사위: {example}")
    print("\n".join(score_text(example)))
