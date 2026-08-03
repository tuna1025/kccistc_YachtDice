"""YOLO로 주사위를 찾고 OpenCV로 흰 눈금 개수를 세는 웹캠 테스트.

실행:
    python hybrid_webcam.py

종료:
    카메라 화면에서 q 키를 누른다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_DIR / "runs" / "dice_yolo11" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect dice with YOLO and count white pips with OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam number (default: 0).")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained best.pt.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence.")
    parser.add_argument("--device", default=0, help="GPU id (0) or cpu.")
    parser.add_argument("--white-value", type=int, default=170, help="Brightness threshold for white pips (0-255).")
    parser.add_argument("--debug", action="store_true", help="Show the pip mask for the first detected die.")
    # Jupyter kernel arguments such as --f=... are ignored.
    args, _ = parser.parse_known_args()
    return args


def count_white_pips(crop: np.ndarray, white_value: int) -> tuple[int, np.ndarray]:
    """Count bright, low-saturation and roughly circular blobs in one die crop."""
    # A small inset ignores bright edges/reflections on the die boundary.
    height, width = crop.shape[:2]
    inset_x, inset_y = int(width * 0.08), int(height * 0.08)
    inner = crop[inset_y : height - inset_y, inset_x : width - inset_x]
    if inner.size == 0:
        return 0, np.zeros((1, 1), dtype=np.uint8)

    # White has high brightness (V) and low saturation (S) in HSV colour space.
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, white_value), (180, 125, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = inner.shape[0] * inner.shape[1]
    minimum_area, maximum_area = crop_area * 0.0015, crop_area * 0.08
    pip_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter else 0

        # Reject tiny noise, large reflections and long non-circular shapes.
        if minimum_area <= area <= maximum_area and circularity >= 0.35:
            pip_count += 1

    return pip_count, mask


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    model = YOLO(str(weights))
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open webcam {args.camera}.")

    print("Webcam started. Press q in the video window to quit.")
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            # YOLO is used only for each die's bounding box. Its 1~6 class is
            # displayed for comparison; the final pip value comes from OpenCV.
            result = model(frame, conf=args.conf, device=args.device, verbose=False)[0]
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                pips, mask = count_white_pips(crop, args.white_value)
                yolo_value = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                pip_label = str(pips) if 1 <= pips <= 6 else "?"
                label = f"OpenCV: {pip_label} | YOLO: {yolo_value} ({confidence:.0%})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                if args.debug:
                    cv2.imshow("pip mask (first detected die)", mask)
                    args.debug = False

            cv2.imshow("YOLO + OpenCV dice test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
