# 테스트 및 이전 구현

현재 배포용 프로그램은 프로젝트 루트의 `pyside6_main.py`입니다. 이 폴더에는
모델과 게임 로직을 확인하기 위해 사용했던 테스트·실험 코드를 보관합니다.

## 파일 구성

- `camera_logic_test.py`: 웹캠 인식값과 점수 계산 연결 테스트
- `predict.py`: 이미지, 폴더, 영상 및 웹캠 YOLO 예측 테스트
- `hybrid_webcam.py`: YOLO 검출과 OpenCV 눈금 계산을 결합한 실험
- `legacy/`: 이전 OpenCV 기반 UI와 카메라 연결 코드
- `notebooks/`: CUDA 환경 확인과 원본 데이터셋 학습 노트북
- `notes/`: 이전 구현 과정의 작업 메모

스크립트는 프로젝트 루트에서 실행합니다.

```powershell
python tests/camera_logic_test.py
python tests/predict.py --source 0 --show
python tests/hybrid_webcam.py
python tests/legacy/main.py
```
