"""PySide6로 구현한 Yacht Dice 데스크톱 UI.

OpenCV는 카메라 프레임 수집에, YOLO는 주사위 인식에만 사용한다.
화면 구성과 애니메이션은 모두 PySide6 위젯과 QPainter로 처리한다.
"""
from __future__ import annotations

from collections import Counter, deque
import math
import random
import sys
import time

import cv2
from PySide6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import dice_logic
from dice_inference import DiceRecognizer


CLEAR_FRAMES_REQUIRED = 5
VOTE_WINDOW_SIZE = 10
VOTES_REQUIRED = 7
UPPER_CATEGORIES = (
    ("Aces", "ones", 1),
    ("Deuces", "twos", 2),
    ("Threes", "threes", 3),
    ("Fours", "fours", 4),
    ("Fives", "fives", 5),
    ("Sixes", "sixes", 6),
)
LOWER_CATEGORIES = (
    ("Choice", "choice", "C"),
    ("4 of a Kind", "four_of_a_kind", "4"),
    ("Full House", "full_house", "FH"),
    ("S. Straight", "small_straight", "S"),
    ("L. Straight", "large_straight", "L"),
    ("Yacht", "yacht", "Y"),
)
ALL_CATEGORIES = UPPER_CATEGORIES + LOWER_CATEGORIES


def load_application_fonts() -> None:
    """Windows 오프스크린 렌더링에서도 한글과 영문 글꼴을 안정적으로 사용한다."""
    # 변경: 일부 Conda/Qt 조합은 시스템 글꼴을 자동 탐색하지 못해 직접 등록한다.
    for font_path in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
    ):
        QFontDatabase.addApplicationFont(font_path)


class DiceFace(QWidget):
    """숫자 대신 실제 주사위 눈을 그리는 현재 주사위 위젯."""

    PIP_LAYOUTS = {
        1: ((1, 1),),
        2: ((0, 0), (2, 2)),
        3: ((0, 0), (1, 1), (2, 2)),
        4: ((0, 0), (2, 0), (0, 2), (2, 2)),
        5: ((0, 0), (2, 0), (1, 1), (0, 2), (2, 2)),
        6: ((0, 0), (2, 0), (0, 1), (2, 1), (0, 2), (2, 2)),
    }

    def __init__(self, value: int | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.value = value
        self.setFixedSize(70, 70)

    def set_value(self, value: int | None) -> None:
        self.value = value
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(4, 4, -4, -4)
        painter.setPen(QPen(QColor("#D8C9A8"), 1))
        painter.setBrush(QColor("#FFFDF7"))
        painter.drawRoundedRect(rect, 15, 15)
        if self.value not in self.PIP_LAYOUTS:
            painter.setPen(QColor("#9B9A91"))
            painter.setFont(QFont("Segoe UI", 20, QFont.DemiBold))
            painter.drawText(rect, Qt.AlignCenter, "?")
            return
        pip_color = QColor("#B33A3A") if self.value in (1, 4) else QColor("#20231F")
        painter.setPen(Qt.NoPen)
        painter.setBrush(pip_color)
        positions = (rect.left() + 16, rect.center().x(), rect.right() - 16)
        for px, py in self.PIP_LAYOUTS[self.value]:
            painter.drawEllipse(QPoint(positions[px], positions[py]), 5, 5)


class CategoryBadge(QWidget):
    """점수 카테고리 앞에 붙는 작고 일관된 아이콘."""

    def __init__(self, value: int | str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.value = value
        self.setFixedSize(31, 31)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#20231F"))
        if isinstance(self.value, int):
            # 변경: 상단 카테고리는 참고 이미지의 검은 주사위 아이콘을 그대로 따른다.
            tile = self.rect().adjusted(4, 4, -4, -4)
            painter.drawRect(tile)
            painter.setBrush(QColor("#FFFFFF"))
            positions = (tile.left() + 5, tile.center().x(), tile.right() - 5)
            for px, py in DiceFace.PIP_LAYOUTS[self.value]:
                painter.drawEllipse(QPoint(positions[px], positions[py]), 2, 2)
            return

        # 변경: 하단 카테고리는 작은 검은 블록을 조합한 픽토그램으로 표현한다.
        patterns = {
            # Choice: 주사위 5의 눈처럼 네 모서리와 중앙에 배치한다.
            "C": ((4, 4), (20, 4), (12, 12), (4, 20), (20, 20)),
            "4": ((5, 6), (15, 6), (5, 16), (15, 16)),
            # Full House: 위 2개가 아래 3개의 사이에 오도록 반 칸 오른쪽으로 이동한다.
            "FH": ((8, 5), (18, 5), (3, 17), (13, 17), (23, 17)),
            # Small Straight: 대각선으로 이어지는 네 개의 블록만 사용한다.
            "S": ((3, 3), (9, 9), (15, 15), (21, 21)),
            # Large Straight: 이전 Yacht 아이콘의 배치를 그대로 사용한다.
            "L": ((4, 5), (20, 5), (8, 13), (16, 13), (12, 21)),
            # Yacht: 별의 다섯 꼭지점 위치에 블록을 배치한다.
            "Y": ((12, 2), (2, 10), (22, 10), (6, 22), (18, 22)),
        }
        for x, y in patterns[str(self.value)]:
            painter.drawRect(x, y, 5, 5)


class InferenceSignals(QObject):
    result = Signal(object)
    finished = Signal()


class InferenceTask(QRunnable):
    """YOLO 추론을 UI 스레드 밖에서 실행해 애니메이션이 멈추지 않게 한다."""

    def __init__(self, recognizer: DiceRecognizer, frame) -> None:
        super().__init__()
        self.recognizer = recognizer
        self.frame = frame
        self.signals = InferenceSignals()

    def run(self) -> None:
        try:
            self.signals.result.emit(self.recognizer.predict(self.frame))
        finally:
            self.signals.finished.emit()


class CelebrationOverlay(QWidget):
    """Yacht와 보너스 달성 시 화면 위에 그리는 비차단 축하 애니메이션."""

    def __init__(self, parent: QWidget, mode: str) -> None:
        super().__init__(parent)
        self.mode = mode
        self.started_at = time.monotonic()
        self.duration = 2.7 if mode == "yacht" else 2.2
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(parent.rect())
        randomizer = random.Random(42)
        self.particles = [
            (randomizer.random(), randomizer.random(), randomizer.choice((-1, 1)), randomizer.randint(4, 9))
            for _ in range(44)
        ]
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self.timer.start(16)
        self.show()
        self.raise_()

    @staticmethod
    def _ease_out_back(value: float) -> float:
        c1, c3 = 1.70158, 2.70158
        return 1 + c3 * (value - 1) ** 3 + c1 * (value - 1) ** 2

    def _advance(self) -> None:
        if time.monotonic() - self.started_at >= self.duration:
            self.deleteLater()
        else:
            self.update()

    def paintEvent(self, event) -> None:
        elapsed = time.monotonic() - self.started_at
        progress = min(1.0, elapsed / self.duration)
        fade = min(1.0, progress / 0.12) * min(1.0, (1.0 - progress) / 0.15)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(22, 22, 20, int((78 if self.mode == "yacht" else 38) * fade)))

        if self.mode == "yacht":
            colors = (QColor("#E6B74C"), QColor("#F7E4A8"), QColor("#3AA77D"), QColor("#FFFDF7"))
            painter.setPen(Qt.NoPen)
            for index, (x_seed, y_seed, direction, size) in enumerate(self.particles):
                x = int(x_seed * self.width() + direction * math.sin(progress * 7 + index) * 34)
                y = int((y_seed * self.height() + progress * self.height() * 0.8) % self.height())
                painter.setBrush(colors[index % len(colors)])
                painter.save()
                painter.translate(x, y)
                painter.rotate(progress * 360 * direction)
                painter.drawRect(-size // 2, -size // 4, size, max(3, size // 2))
                painter.restore()

        enter = min(1.0, progress / 0.20)
        y_center = self.height() // 2
        card_width = min(580, self.width() - 80)
        card_height = 164 if self.mode == "yacht" else 132
        y = int(-card_height + (y_center - card_height // 2 + card_height) * self._ease_out_back(enter))
        card = QRect((self.width() - card_width) // 2, y, card_width, card_height)
        painter.setPen(QPen(QColor("#E6B74C"), 3))
        # 변경: 축하 배너도 초록색 대신 흰색과 금색 조합을 사용한다.
        painter.setBrush(QColor("#FFFDF8"))
        painter.drawRoundedRect(card, 24, 24)
        painter.setPen(QColor("#20231F"))
        painter.setFont(QFont("Segoe UI", 34 if self.mode == "yacht" else 28, QFont.Black))
        title = "YACHT!" if self.mode == "yacht" else "+35 BONUS"
        painter.drawText(card.adjusted(0, 18, 0, -50), Qt.AlignCenter, title)
        painter.setPen(QColor("#8C6B22"))
        painter.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        subtitle = "FIVE OF A KIND · 50 POINTS" if self.mode == "yacht" else "UPPER SECTION TARGET REACHED"
        painter.drawText(card.adjusted(0, 74, 0, -10), Qt.AlignCenter, subtitle)


class YachtDiceWindow(QMainWindow):
    """게임 로직, 카메라, 점수 선택을 연결하는 메인 창."""

    def __init__(self, preview: bool = False) -> None:
        super().__init__()
        load_application_fonts()
        self.preview = preview
        self.setWindowTitle("Yacht Dice")
        self.setMinimumSize(1040, 720)
        self.resize(1180, 800)
        self.active_player = 1
        self.committed_scores = {1: {}, 2: {}}
        self.candidate_scores = {1: {}, 2: {}}
        self.dice_values: list[int] | None = None
        self.awaiting_clear = False
        self.clear_frames = 0
        self.vote_history: deque[tuple[int, ...]] = deque(maxlen=VOTE_WINDOW_SIZE)
        self.last_inference_at = 0.0
        self.inference_busy = False
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.camera = None
        self.recognizer = None
        self._animations: list[QParallelAnimationGroup] = []
        self._build_ui()

        if preview:
            self._load_preview_state()
            return

        # 변경: 모델과 카메라는 UI가 먼저 만들어진 뒤 준비한다.
        self.recognizer = DiceRecognizer(confidence=0.40, device=0)
        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("카메라 0을 열 수 없습니다.")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    def _build_ui(self) -> None:
        self.root = QWidget()
        self.root.setObjectName("root")
        self.setCentralWidget(self.root)
        page = QVBoxLayout(self.root)
        page.setContentsMargins(26, 22, 26, 24)
        page.setSpacing(18)

        # 변경: 상단은 브랜드와 현재 진행 상태만 두어 게임 화면의 초점을 유지한다.
        header = QHBoxLayout()
        brand_group = QVBoxLayout()
        brand_group.setSpacing(0)
        brand = QLabel("YACHT DICE")
        brand.setObjectName("brand")
        tagline = QLabel("CAMERA TABLE · TWO PLAYER MATCH")
        tagline.setObjectName("tagline")
        brand_group.addWidget(brand)
        brand_group.addWidget(tagline)
        header.addLayout(brand_group)
        header.addStretch()
        self.restart_button = QPushButton("NEW GAME")
        self.restart_button.setObjectName("newGame")
        self.restart_button.clicked.connect(self._restart_game)
        header.addWidget(self.restart_button)
        self.turn_pill = QLabel()
        self.turn_pill.setObjectName("turnPill")
        self.turn_pill.setAlignment(Qt.AlignCenter)
        header.addWidget(self.turn_pill)
        page.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(18)
        page.addLayout(content, 1)

        left = QVBoxLayout()
        left.setSpacing(16)
        content.addLayout(left, 7)
        camera_card = QFrame()
        camera_card.setObjectName("cameraCard")
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(12, 12, 12, 12)
        camera_layout.setSpacing(10)
        camera_head = QHBoxLayout()
        camera_title = QLabel("LIVE TABLE")
        camera_title.setObjectName("sectionTitle")
        self.status_label = QLabel()
        self.status_label.setObjectName("statusChip")
        camera_head.addWidget(camera_title)
        camera_head.addStretch()
        camera_head.addWidget(self.status_label)
        camera_layout.addLayout(camera_head)
        self.camera_label = QLabel("카메라를 준비하는 중입니다")
        self.camera_label.setObjectName("camera")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumHeight(390)
        self.camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout.addWidget(self.camera_label, 1)
        left.addWidget(camera_card, 1)

        dice_card = QFrame()
        dice_card.setObjectName("diceCard")
        dice_layout = QHBoxLayout(dice_card)
        dice_layout.setContentsMargins(18, 12, 18, 12)
        dice_text = QVBoxLayout()
        dice_title = QLabel("CURRENT DICE")
        dice_title.setObjectName("sectionTitle")
        self.dice_hint = QLabel("주사위 5개를 카메라에 보여주세요")
        self.dice_hint.setObjectName("muted")
        dice_text.addWidget(dice_title)
        dice_text.addWidget(self.dice_hint)
        dice_layout.addLayout(dice_text)
        dice_layout.addStretch()
        self.dice_widgets = [DiceFace() for _ in range(5)]
        for die in self.dice_widgets:
            dice_layout.addWidget(die)
        left.addWidget(dice_card)

        self.board = QFrame()
        self.board.setObjectName("scoreBoard")
        self.board.setMinimumWidth(430)
        content.addWidget(self.board, 4)
        board_layout = QVBoxLayout(self.board)
        # 변경: 표 내용이 외곽 테두리를 덮지 않도록 테두리 두께만큼 안쪽 여백을 둔다.
        board_layout.setContentsMargins(4, 4, 4, 4)
        board_layout.setSpacing(0)
        self.score_grid = QGridLayout()
        self.score_grid.setContentsMargins(0, 0, 0, 0)
        self.score_grid.setSpacing(0)
        # 변경: 참고 이미지처럼 카테고리, 1P, 2P 세 열을 같은 폭으로 구성한다.
        self.score_grid.setColumnStretch(0, 1)
        self.score_grid.setColumnStretch(1, 1)
        self.score_grid.setColumnStretch(2, 1)
        self.score_buttons: dict[tuple[int, str], QPushButton] = {}
        self.turn_header = QLabel()
        self.turn_header.setProperty("class", "tableHeader")
        self.turn_header.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.score_grid.addWidget(self.turn_header, 0, 0)
        self.player_headers: dict[int, QLabel] = {}
        for player in (1, 2):
            label = QLabel(f"{player}P")
            label.setProperty("class", "tableHeaderPlayer")
            label.setAlignment(Qt.AlignCenter)
            self.score_grid.addWidget(label, 0, player)
            self.player_headers[player] = label
        row = 1
        for label, key, badge in UPPER_CATEGORIES:
            self._add_score_row(row, label, key, badge)
            row += 1
        self.subtotal_labels = self._add_summary_row(row, "Subtotal", "", "subtotal")
        row += 1
        self.bonus_labels = self._add_summary_row(row, "+35 Bonus", "", "bonus")
        row += 1
        for label, key, badge in LOWER_CATEGORIES:
            self._add_score_row(row, label, key, badge)
            row += 1
        self.total_labels = self._add_summary_row(row, "Total", "", "total")
        # 변경: 점수 행이 남는 높이를 채우고 헤더가 불필요하게 늘어나지 않도록 한다.
        board_layout.addLayout(self.score_grid, 1)

        self.setStyleSheet("""
            QWidget#root { background: #F7F7F4; color: #20231F; font-family: 'Malgun Gothic', 'Segoe UI'; }
            QLabel#brand { color: #20231F; font-size: 28px; font-weight: 900; letter-spacing: 4px; }
            QLabel#tagline { color: #A77A20; font-size: 10px; font-weight: 700; letter-spacing: 2px; }
            QLabel#turnPill { background: #FFFFFF; color: #20231F; border: 2px solid #C89124; border-radius: 18px; min-width: 176px; min-height: 34px; font-size: 12px; font-weight: 900; }
            QFrame#cameraCard, QFrame#diceCard { background: #FFFFFF; border: 1px solid #DDD8CC; border-radius: 16px; }
            QLabel#sectionTitle { color: #20231F; font-size: 12px; font-weight: 900; letter-spacing: 2px; }
            QLabel#statusChip { background: #F5E9C4; color: #6F531B; border-radius: 11px; padding: 5px 10px; font-size: 10px; font-weight: 800; }
            QLabel#camera { background: #181A18; color: #C9C8C1; border-radius: 11px; }
            QLabel#muted { color: #81847D; font-size: 10px; }
            QFrame#scoreBoard { background: #FFFFFF; border: 4px solid #C89124; border-radius: 2px; }
            QPushButton#newGame { background: #FFFFFF; color: #20231F; border: 2px solid #C89124; border-radius: 9px; padding: 7px 12px; font-size: 10px; font-weight: 900; }
            QPushButton#newGame:hover { background: #F5E9C4; }
            QLabel[class="tableHeader"] { background: #FFFFFF; color: #20231F; border-bottom: 3px solid #C89124; padding-left: 16px; min-height: 53px; max-height: 53px; font-size: 18px; font-weight: 900; }
            QLabel[class="tableHeaderPlayer"] { background: #F5E9C4; color: #20231F; border-left: 3px solid #C89124; border-bottom: 3px solid #C89124; min-height: 53px; max-height: 53px; font-size: 20px; font-weight: 900; }
            QFrame[class="scoreName"] { background: #FFFFFF; border-top: 1px solid #BBC5BC; }
            QLabel[class="scoreLabel"] { color: #20231F; font-size: 11px; font-weight: 800; padding-left: 2px; }
            QPushButton[class="scoreCell"] { background: #FFFFFF; color: #8F9892; border: 0; border-left: 3px solid #C89124; border-top: 1px solid #BBC5BC; min-height: 37px; max-height: 37px; font-size: 13px; font-weight: 900; }
            QPushButton[class="scoreCell"][active="true"] { background: #F7EED1; }
            QPushButton[class="scoreCell"]:hover:enabled { background: #E8BE59; color: #20231F; }
            QPushButton[class="scoreCell"]:disabled { color: #909A94; }
            QFrame[class="summaryName"] { background: #F2E5BE; border-top: 1px solid #C89124; min-height: 36px; max-height: 36px; }
            QFrame[class="summaryName"][kind="bonus"] { border-top: 1px dashed #C89124; }
            QLabel[class="summaryTitle"] { color: #20231F; font-size: 12px; font-weight: 900; }
            QLabel[class="summaryHint"] { color: transparent; font-size: 0px; max-height: 0px; }
            QLabel[class="summaryValue"] { background: #F2E5BE; color: #20231F; border-left: 3px solid #C89124; border-top: 1px solid #C89124; min-height: 36px; max-height: 36px; font-size: 12px; font-weight: 900; }
            QLabel[class="summaryValue"][kind="bonus"] { border-top: 1px dashed #C89124; }
            QFrame[class="totalName"] { background: #F2E5BE; border-top: 3px solid #C89124; min-height: 71px; max-height: 71px; }
            QLabel[class="totalTitle"] { color: #20231F; font-size: 23px; font-weight: 900; }
            QLabel[class="totalHint"] { color: transparent; font-size: 0px; max-height: 0px; }
            QLabel[class="totalValue"] { background: #F2E5BE; color: #20231F; border-left: 3px solid #C89124; border-top: 3px solid #C89124; min-height: 71px; max-height: 71px; font-size: 21px; font-weight: 900; }
        """)
        self._refresh_ui()

    def _add_score_row(self, row: int, label: str, key: str, badge: int | str) -> None:
        name_frame = QFrame()
        name_frame.setProperty("class", "scoreName")
        name_layout = QHBoxLayout(name_frame)
        name_layout.setContentsMargins(10, 2, 7, 2)
        name_layout.setSpacing(6)
        name_layout.addWidget(CategoryBadge(badge))
        name = QLabel(label)
        name.setProperty("class", "scoreLabel")
        name_layout.addWidget(name, 1)
        self.score_grid.addWidget(name_frame, row, 0)
        for player in (1, 2):
            button = QPushButton("")
            button.setProperty("class", "scoreCell")
            button.clicked.connect(lambda checked=False, p=player, k=key: self._commit_score(p, k))
            self.score_grid.addWidget(button, row, player)
            self.score_buttons[player, key] = button

    def _add_summary_row(
        self, row: int, title: str, hint: str, kind: str,
    ) -> tuple[QLabel, QLabel]:
        frame = QFrame()
        frame.setProperty("class", "totalName" if kind == "total" else "summaryName")
        frame.setProperty("kind", kind)
        text_layout = QVBoxLayout(frame)
        text_layout.setContentsMargins(13, 5, 8, 5)
        text_layout.setSpacing(0)
        title_label = QLabel(title)
        title_label.setProperty("class", "totalTitle" if kind == "total" else "summaryTitle")
        title_label.setAlignment(Qt.AlignCenter)
        hint_label = QLabel(hint)
        hint_label.setProperty("class", "totalHint" if kind == "total" else "summaryHint")
        hint_label.setAlignment(Qt.AlignCenter)
        text_layout.addWidget(title_label)
        text_layout.addWidget(hint_label)
        self.score_grid.addWidget(frame, row, 0)
        values = []
        for player in (1, 2):
            value = QLabel()
            value.setProperty("class", "totalValue" if kind == "total" else "summaryValue")
            value.setProperty("kind", kind)
            value.setAlignment(Qt.AlignCenter)
            self.score_grid.addWidget(value, row, player)
            values.append(value)
        return values[0], values[1]

    def _load_preview_state(self) -> None:
        """카메라 없이도 디자인을 확인할 수 있는 샘플 상태."""
        self.committed_scores = {
            1: {"ones": 3, "twos": 6, "threes": 9, "choice": 22, "full_house": 30},
            2: {"ones": 2, "twos": 8, "fours": 12, "four_of_a_kind": 30},
        }
        self.dice_values = [2, 3, 4, 5, 6]
        self.candidate_scores[1] = dice_logic.calculate_scores(self.dice_values)
        self.camera_label.setText("LIVE CAMERA PREVIEW")
        self._refresh_ui()

    def _tick(self) -> None:
        ok, frame = self.camera.read()
        if not ok:
            self.status_label.setText("CAMERA DISCONNECTED")
            return
        self._show_frame(frame)
        # 변경: 추론은 별도 작업 스레드에서 한 번씩만 실행한다.
        if not self.inference_busy and time.monotonic() - self.last_inference_at >= 0.12:
            self.inference_busy = True
            self.last_inference_at = time.monotonic()
            task = InferenceTask(self.recognizer, frame.copy())
            task.signals.result.connect(self._process_detection)
            task.signals.finished.connect(self._inference_finished)
            self.thread_pool.start(task)

    def _inference_finished(self) -> None:
        self.inference_busy = False

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation,
        )
        self.camera_label.setPixmap(pixmap)

    def _process_detection(self, detected: list[int] | None) -> None:
        if self.awaiting_clear:
            self.vote_history.clear()
            self.clear_frames = self.clear_frames + 1 if detected is None else 0
            if self.clear_frames >= CLEAR_FRAMES_REQUIRED:
                self.awaiting_clear = False
                self.clear_frames = 0
            self._refresh_ui()
            return
        if detected is not None:
            self.vote_history.append(tuple(detected))
            values, count = Counter(self.vote_history).most_common(1)[0]
            if count >= VOTES_REQUIRED and list(values) != self.dice_values:
                self.dice_values = list(values)
                self.candidate_scores[self.active_player] = dice_logic.calculate_scores(self.dice_values)
        self._refresh_ui()

    def _commit_score(self, player: int, key: str) -> None:
        if (
            player != self.active_player
            or key not in self.candidate_scores[player]
            or key in self.committed_scores[player]
        ):
            return
        score = self.candidate_scores[player][key]
        bonus_before = dice_logic.calculate_upper_bonus(self.committed_scores[player])["upper_bonus"]
        self.committed_scores[player][key] = score
        bonus_after = dice_logic.calculate_upper_bonus(self.committed_scores[player])["upper_bonus"]
        self._refresh_ui()
        self._play_score_effect(self.score_buttons[player, key], score)
        if key == "yacht" and score > 0:
            CelebrationOverlay(self.root, "yacht")
        if bonus_before == 0 and bonus_after == dice_logic.BONUS_SCORE:
            QTimer.singleShot(400, lambda: CelebrationOverlay(self.root, "bonus"))
        self.candidate_scores[player] = {}
        self.dice_values = None
        self.vote_history.clear()
        if all(len(self.committed_scores[p]) == len(ALL_CATEGORIES) for p in (1, 2)):
            QTimer.singleShot(2800 if key == "yacht" and score else 900, self._finish_game)
        else:
            self.active_player = 2 if player == 1 else 1
            self.awaiting_clear = True
        self._refresh_ui()

    def _play_score_effect(self, target: QWidget, score: int) -> None:
        """선택한 점수가 셀 위로 가볍게 착지하는 효과."""
        label = QLabel(f"+{score}", self.root)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            "background:#E6B74C; color:#20231F; border-radius:16px; "
            "font-size:20px; font-weight:900;"
        )
        center = self.root.mapFromGlobal(target.mapToGlobal(target.rect().center()))
        end = QRect(0, 0, max(58, target.width() - 8), target.height() - 4)
        end.moveCenter(center)
        start = end.adjusted(-22, -18, 22, 18).translated(0, -16)
        label.setGeometry(start)
        opacity = QGraphicsOpacityEffect(label)
        label.setGraphicsEffect(opacity)
        label.show()
        label.raise_()
        geometry_animation = QPropertyAnimation(label, b"geometry")
        geometry_animation.setDuration(650)
        geometry_animation.setStartValue(start)
        geometry_animation.setEndValue(end)
        geometry_animation.setEasingCurve(QEasingCurve.OutBack)
        opacity_animation = QPropertyAnimation(opacity, b"opacity")
        opacity_animation.setDuration(900)
        opacity_animation.setKeyValueAt(0.0, 0.0)
        opacity_animation.setKeyValueAt(0.15, 1.0)
        opacity_animation.setKeyValueAt(0.72, 1.0)
        opacity_animation.setKeyValueAt(1.0, 0.0)
        group = QParallelAnimationGroup(self)
        group.addAnimation(geometry_animation)
        group.addAnimation(opacity_animation)
        group.finished.connect(label.deleteLater)
        group.finished.connect(lambda g=group: self._animations.remove(g) if g in self._animations else None)
        self._animations.append(group)
        group.start()

    def _refresh_ui(self) -> None:
        turn = 1 + (len(self.committed_scores[1]) + len(self.committed_scores[2])) // 2
        self.turn_header.setText(f"Turn {min(turn, 12)}/12")
        self.turn_pill.setText(f"PLAYER {self.active_player} TURN")
        if self.awaiting_clear:
            self.status_label.setText("REMOVE PREVIOUS DICE")
            self.dice_hint.setText("다음 플레이어를 위해 주사위를 치워주세요")
        elif self.dice_values is None:
            self.status_label.setText(f"SCANNING  {len(self.vote_history)}/{VOTES_REQUIRED}")
            self.dice_hint.setText("주사위 5개를 카메라에 보여주세요")
        else:
            self.status_label.setText("DICE LOCKED")
            self.dice_hint.setText("오른쪽 점수표에서 기록할 점수를 선택하세요")
        for index, die in enumerate(self.dice_widgets):
            die.set_value(self.dice_values[index] if self.dice_values else None)
        for (player, key), button in self.score_buttons.items():
            committed = self.committed_scores[player].get(key)
            candidate = self.candidate_scores[player].get(key)
            # 변경: 참고 이미지처럼 현재 플레이어의 점수 열 전체를 옅은 아이보리로 표시한다.
            button.setProperty("active", "true" if player == self.active_player else "false")
            button.style().unpolish(button)
            button.style().polish(button)
            if committed is not None:
                button.setText(str(committed))
                button.setStyleSheet("color:#20231F; font-weight:900;")
            elif player == self.active_player and candidate is not None:
                button.setText(str(candidate))
                button.setStyleSheet("color:#909A94; font-weight:900; font-style:italic;")
            else:
                button.setText("—")
                button.setStyleSheet("")
            button.setEnabled(
                committed is None
                and player == self.active_player
                and candidate is not None
                and not self.awaiting_clear
            )
        for index, player in enumerate((1, 2)):
            upper = dice_logic.calculate_upper_bonus(self.committed_scores[player])
            self.subtotal_labels[index].setText(f"{upper['upper_total']} / 63")
            self.bonus_labels[index].setText("+35" if upper["upper_bonus"] else "—")
            self.total_labels[index].setText(str(dice_logic.calculate_grand_total(self.committed_scores[player])))

    def _finish_game(self) -> None:
        scores = [dice_logic.calculate_grand_total(self.committed_scores[p]) for p in (1, 2)]
        winner = "DRAW" if scores[0] == scores[1] else f"PLAYER {1 if scores[0] > scores[1] else 2} WINS"
        QMessageBox.information(self, "Match Result", f"{winner}\n\n1P  {scores[0]}   :   {scores[1]}  2P")

    def _restart_game(self) -> None:
        self.active_player = 1
        self.committed_scores = {1: {}, 2: {}}
        self.candidate_scores = {1: {}, 2: {}}
        self.dice_values = None
        self.awaiting_clear = not self.preview
        self.clear_frames = 0
        self.vote_history.clear()
        self._refresh_ui()

    def closeEvent(self, event) -> None:
        # 변경: 창 종료 시 타이머, 카메라, 백그라운드 추론 작업을 정리한다.
        if hasattr(self, "timer"):
            self.timer.stop()
        if self.camera is not None:
            self.camera.release()
        self.thread_pool.clear()
        self.thread_pool.waitForDone(1200)
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    preview = "--preview" in sys.argv
    window = YachtDiceWindow(preview=preview)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
