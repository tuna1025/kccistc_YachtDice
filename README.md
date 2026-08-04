# Yacht Dice

YOLO11로 웹캠의 주사위 5개를 인식하고 PySide6 점수표에 후보 점수를 표시하는
2인용 요트 다이스 게임입니다. 점수 선택과 애니메이션은 PySide6가 담당하고,
OpenCV는 카메라 프레임 수집에 사용합니다.


## 주요 파일

```text
pyside6_main.py       PySide6 게임 UI와 카메라 연동
dice_inference.py     YOLO 모델 로드 및 주사위 값 반환
dice_logic.py         요트 다이스 후보 점수·보너스·총점 계산
train.py              두 데이터셋을 사용한 YOLO11 학습
models/best.pt        실행에 사용하는 최종 모델
tests/                예측·카메라·이전 UI 실험 코드와 노트북
```
