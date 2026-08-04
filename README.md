# Yacht Dice

**YOLO11s와 웹캠을 활용한 실시간 주사위 인식 요트 다이스 게임**

[![프로젝트 시연 영상](https://img.youtube.com/vi/oDf_O-nPHRc/0.jpg)](https://www.youtube.com/shorts/oDf_O-nPHRc)
이미지를 클릭하면 시연 영상으로 이동합니다.
<img src="https://github.com/user-attachments/assets/25a3a0c8-24fc-466c-8ed1-ea4e27cbec75" width="220">
<img src="https://github.com/user-attachments/assets/2eb59610-ee83-4056-aa44-497724f55ffc" width="500">


## 1. 프로젝트 개요

본 프로젝트는 웹캠으로 주사위 5개의 숫자를 실시간 인식하고, 요트 다이스 점수를 자동으로 계산하는 2인용 게임입니다.

YOLO11s가 인식한 결과를 게임 로직에 전달해 후보 점수를 표시하며, 사용자가 선택한 점수만 총점에 반영됩니다. 카메라와 점수표 UI는 PySide6로 구성했습니다.

## 2. 개발 환경 및 도구

`Python` `YOLO11s` `Ultralytics` `PyTorch` `CUDA` `OpenCV` `PySide6` `Git/GitHub`

- 객체 탐지: YOLO11s
- 영상 입력: OpenCV
- GUI: PySide6
- GPU 가속: PyTorch, CUDA
- 입력 장치: USB 웹캠
- 배포: PyInstaller

## 3. 주요 기능

- 웹캠을 통한 주사위 5개 실시간 인식
- 인식 결과를 정렬된 리스트로 반환
- 프레임 투표를 이용한 인식값 안정화
- 요트 다이스 항목별 후보 점수 자동 계산
- 선택한 점수만 총점에 반영
- 상단 합계 63점 이상 시 보너스 35점 적용
- 2인 플레이, 턴 전환 및 최종 승자 표시
- Yacht 및 보너스 획득 애니메이션

## 4. 기술 구현

### ■ 주사위 인식

주사위 숫자 1부터 6까지를 클래스로 정의하고 YOLO11s를 학습했습니다. 주사위가 정확히 5개 탐지된 경우에만 정렬된 리스트를 게임 로직에 전달합니다.

### ■ 인식 안정화

순간적인 오인식을 방지하기 위해 최근 최대 10개의 인식 결과 중 같은 조합이 7번 이상 나오면 최종 결과로 반영합니다.

### ■ 점수 계산

YOLO가 반환한 리스트로 Choice, Full House, Straight, Yacht 등의 후보 점수를 계산합니다. 후보 점수와 확정 점수를 분리해 사용자가 선택한 점수만 총점에 포함합니다.

### ■ PySide6 UI

카메라 화면은 왼쪽, 1P·2P 점수표는 오른쪽에 배치했습니다. YOLO 추론은 작업 스레드에서 실행하여 인식 중에도 UI가 멈추지 않도록 구현했습니다.

## 5. 어려웠던 점 및 해결 방법

### ■ 비슷한 숫자 오인식

4, 5, 6처럼 눈금 배치가 비슷한 숫자를 자주 혼동했습니다. 다양한 종류의 주사위와 각도, 배경 등의 데이터를 추가하고 클래스별 데이터 수를 보강하여 개선했습니다.

사용한 데이터셋
- https://public.roboflow.com/object-detection/dice
- https://universe.roboflow.com/workspace-spezm/dice-0sexk

### ■ 인식 결과 깜빡임

프레임마다 결과가 바뀌는 문제를 해결하기 위해 프레임 투표 방식을 적용했습니다.

### ■ UI 멈춤 현상

YOLO 추론을 UI 스레드와 분리하고, 동시에 하나의 추론 작업만 실행하도록 제한했습니다.

## 6. 프로젝트를 통해 배운 점

- YOLO 모델 학습과 데이터 라벨링 과정
- Precision, Recall, mAP 및 혼돈행렬 분석
- OpenCV 영상과 PySide6 UI 연동
- 실시간 추론을 위한 스레드 및 성능 최적화
- 인식, 게임 로직, UI를 모듈로 분리하는 설계 방법
