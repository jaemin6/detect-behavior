from ultralytics import YOLO
import cv2
from enum import Enum

# =========================
# 설정
# =========================
MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../data/raw/FD_In_H11H21H31_0001_20210112_09.mp4"
CONF_TH = 0.4

FALL_RATIO_TH = 0.9
FALLEN_RATIO_TH = 0.75
DROP_Y_TH = 40
FALL_CONFIRM_FRAMES = 15

DISPLAY_SCALE = 0.5

# =========================
# 상태 정의
# =========================
class FallState(Enum):
    STANDING = 0
    FALLING = 1
    FALLEN = 2

state = FallState.STANDING
fall_counter = 0
prev_center_y = None

# =========================
# 모델 / 영상
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0

cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)

# =========================
# 메인 루프
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = model(frame, conf=CONF_TH, device="cpu", verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ratio = h / w

            # -------------------------
            # 상태 머신 로직
            # -------------------------
            if prev_center_y is not None:
                dy = cy - prev_center_y
            else:
                dy = 0

            if state == FallState.STANDING:
                if ratio < FALL_RATIO_TH and dy > DROP_Y_TH:
                    state = FallState.FALLING
                    fall_counter = 1

            elif state == FallState.FALLING:
                if ratio < FALLEN_RATIO_TH:
                    fall_counter += 1
                    if fall_counter >= FALL_CONFIRM_FRAMES:
                        state = FallState.FALLEN
                        print(f"[FALL CONFIRMED] frame {frame_idx}")
                else:
                    state = FallState.STANDING
                    fall_counter = 0

            elif state == FallState.FALLEN:
                if ratio > FALL_RATIO_TH:
                    state = FallState.STANDING
                    fall_counter = 0

            prev_center_y = cy

            # -------------------------
            # 시각화
            # -------------------------
            color = (0, 255, 0)
            if state == FallState.FALLING:
                color = (0, 255, 255)
            elif state == FallState.FALLEN:
                color = (0, 0, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f"{state.name} | ratio:{ratio:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # -------------------------
    # 화면 출력 (축소)
    # -------------------------
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Fall Detection", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("[DONE] Processing finished")
