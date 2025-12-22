from ultralytics import YOLO
import cv2
from collections import deque

from state_machine import FallStateMachine  # FSM import

# =========================
# 설정 (튜닝값)
# =========================
MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../data/raw/FD_In_H12H21H31_0016_20201230_11.mp4"
CONF_TH = 0.4

# 비율 기준
STAND_RATIO = 1.4
FALLING_RATIO = 1.3

DISPLAY_SCALE = 0.5
HISTORY_LEN = 15

# =========================
# 색상 정의 (BGR)
# =========================
COLOR_IDLE = (0, 255, 0)        # 초록
COLOR_FALLING = (0, 165, 255)   # 주황
COLOR_FALLEN = (0, 0, 255)      # 빨강
COLOR_RECOVERED = (255, 0, 0)   # 파랑

# =========================
# 누적 기록
# =========================
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
# State Machine 생성
# =========================
fsm = FallStateMachine(
    falling_frames_threshold=5,
    still_frames_threshold=15,
    movement_threshold=0.02
)

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

            # =========================
            # FSM 입력값
            # =========================
            model_pred = "falling" if ratio < FALLING_RATIO else "normal"
            movement_score = abs(dy)

            # =========================
            # FSM 업데이트
            # =========================
            state = fsm.update(model_pred, movement_score)

            if fsm.should_alert():
                print(f"[FALL CONFIRMED] frame {frame_idx}")

            # =========================
            # 상태 기반 색상 결정
            # =========================
            if state.name == "IDLE":
                color = COLOR_IDLE
            elif state.name == "FALLING":
                color = COLOR_FALLING
            elif state.name in ("FALLEN", "CONFIRMED_FALL"):
                color = COLOR_FALLEN
            elif state.name == "RECOVERED":
                color = COLOR_RECOVERED
            else:
                color = COLOR_IDLE

            # =========================
            # 시각화
            # =========================
            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2
            )

            cv2.circle(frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)

            cv2.putText(
                frame,
                f"{state.name} | ratio:{ratio:.2f} dy:{dy:.1f}",
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
