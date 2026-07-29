"""요트다이스 점수 계산 로직.

calculate_scores(dice)는 이번 턴 주사위로 선택 가능한 점수를 계산한다.
calculate_upper_bonus(selected_scores)는 게임 중 이미 선택한 상단 점수의
누적 합이 63점 이상인지 확인해 보너스 35점을 계산한다.
"""

from __future__ import annotations

from collections.abc import Mapping


UPPER_CATEGORIES = ("ones", "twos", "threes", "fours", "fives", "sixes")
LOWER_CATEGORIES = (
    "choice",
    "four_of_a_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yacht",
)
BONUS_THRESHOLD = 63
BONUS_SCORE = 35


def validate_dice(dice: list[int]) -> list[int]:
    """주사위 눈이 1~6 범위의 값 5개인지 확인하고 정렬한다."""
    if len(dice) != 5:
        raise ValueError("주사위 값은 반드시 5개여야 합니다.")
    if any(value not in range(1, 7) for value in dice):
        raise ValueError("각 주사위 값은 1부터 6 사이여야 합니다.")
    return sorted(dice)


def calculate_scores(dice: list[int]) -> dict[str, int]:
    """이번 턴 주사위로 각 카테고리에 기록할 수 있는 점수를 계산한다.

    조건을 만족하지 않는 조합 카테고리는 0점으로 반환한다. 상단 보너스는
    여러 턴에서 선택한 상단 점수의 누적값이 필요하므로 calculate_upper_bonus를
    별도로 호출해서 계산한다.
    """
    dice = validate_dice(dice)
    counts = {value: dice.count(value) for value in range(1, 7)}
    total = sum(dice)
    unique_values = set(dice)

    scores = {
        "ones": counts[1],
        "twos": counts[2] * 2,
        "threes": counts[3] * 3,
        "fours": counts[4] * 4,
        "fives": counts[5] * 5,
        "sixes": counts[6] * 6,
        "choice": total,
        "four_of_a_kind": total if max(counts.values()) >= 4 else 0,
        "full_house": 0,
        "small_straight": 0,
        "large_straight": 0,
        "yacht": 50 if max(counts.values()) == 5 else 0,
    }

    # 3개+2개 또는 같은 눈 5개를 풀하우스로 인정한다.
    count_values = list(counts.values())
    if (3 in count_values and 2 in count_values) or 5 in count_values:
        scores["full_house"] = total

    # 중복은 무시하고 연속된 숫자만 검사한다.
    small_patterns = ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6})
    if any(pattern.issubset(unique_values) for pattern in small_patterns):
        scores["small_straight"] = 15
    if unique_values in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}):
        scores["large_straight"] = 30

    return scores


def calculate_upper_bonus(selected_scores: Mapping[str, int | None]) -> dict[str, int]:
    """기록된 상단 점수의 누적 합과 63점 보너스를 계산한다.

    selected_scores 예시:
        {"ones": 3, "twos": 6, "threes": None, ...}

    아직 선택하지 않은 카테고리는 None 또는 누락 상태여도 된다.
    """
    upper_total = sum(selected_scores.get(category) or 0 for category in UPPER_CATEGORIES)
    upper_bonus = BONUS_SCORE if upper_total >= BONUS_THRESHOLD else 0
    return {
        "upper_total": upper_total,
        "upper_bonus": upper_bonus,
        "upper_with_bonus": upper_total + upper_bonus,
    }


def calculate_grand_total(selected_scores: Mapping[str, int | None]) -> int:
    """게임판에 기록된 모든 점수와 상단 보너스를 더해 총점을 반환한다."""
    upper = calculate_upper_bonus(selected_scores)
    lower_total = sum(selected_scores.get(category) or 0 for category in LOWER_CATEGORIES)
    return upper["upper_with_bonus"] + lower_total


if __name__ == "__main__":
    # 단독 실행 예시
    dice_values = [2, 3, 4, 5, 6]
    print("이번 턴 점수:", calculate_scores(dice_values))
    print("상단 보너스:", calculate_upper_bonus({"ones": 3, "twos": 6, "threes": 9, "fours": 12, "fives": 15, "sixes": 18}))

