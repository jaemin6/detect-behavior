from ultralytics import YOLO
import cv2
from enum import Enum
from collections import deque

# =========================
# 설정 (★ 중요 튜닝값)
# =========================
MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../data/raw/FD_In_H11H21H32_0018_20201016_14.mp4"
CONF_TH = 0.4

# 넘어짐 정의 확실히
STAND_RATIO = 1.4
FALLING_RATIO = 1.3
FALLEN_RATIO = 1.05

DY_FALLING_TH = 20
DY_FALLEN_TH = 40
FALL_CONFIRM_FRAMES = 10

DISPLAY_SCALE = 0.5
HISTORY_LEN = 15   # 누적 판단용

# =========================
# 상태 정의
# =========================
class FallState(Enum):
    STANDING = 0
    FALLING = 1
    FALLEN = 2

state = FallState.STANDING
fall_counter = 0

# 누적 기록
y_history = deque(maxlen=HISTORY_LEN)
ratio_history = deque(maxlen=HISTORY_LEN)

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

            # =========================
            # 누적 기록
            # =========================
            y_history.append(cy)
            ratio_history.append(ratio)

            if len(y_history) < 5:
                continue

            dy = y_history[-1] - y_history[0]
            ratio_drop = ratio_history[0] - ratio

            # =========================
            # 상태 머신
            # =========================
            if state == FallState.STANDING:
                if ratio < STAND_RATIO:
                    state = FallState.FALLING
                    fall_counter = 1

            elif state == FallState.FALLING:
                fall_counter += 1

                if (
                    ratio < FALLEN_RATIO and
                    dy > DY_FALLEN_TH and
                    fall_counter >= FALL_CONFIRM_FRAMES
                ):
                    state = FallState.FALLEN
                    print(f"[FALL CONFIRMED] frame {frame_idx}")

                if ratio > STAND_RATIO:
                    state = FallState.STANDING
                    fall_counter = 0

            elif state == FallState.FALLEN:
                if ratio > STAND_RATIO:
                    state = FallState.STANDING
                    fall_counter = 0

            # =========================
            # 시각화
            # =========================
            color = (0, 255, 0)
            if state == FallState.FALLING:
                color = (0, 255, 255)
            elif state == FallState.FALLEN:
                color = (0, 0, 255)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f"{state.name} | ratio:{ratio:.2f} dy:{dy:.1f} cnt:{fall_counter}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    # =========================
    # 화면 출력
    # =========================
    display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Fall Detection", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Processing finished")
