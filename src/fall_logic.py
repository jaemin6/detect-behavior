from ultralytics import YOLO
import cv2

MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../data/raw/FD_In_H11H21H31_0001_20210112_09.mp4"
CONF_TH = 0.4

# 낙상 판별 파라미터
FALL_RATIO_TH = 0.8
DROP_Y_TH = 50
WINDOW = 5

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
history = []
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # VideoWriter 초기화
    if out is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("../runs/fall_debug.mp4", fourcc, 30, (w, h))
        print("[INFO] VideoWriter initialized")

    frame_idx += 1

    results = model(frame, conf=CONF_TH, device="cpu", verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w_box = x2 - x1
            h_box = y2 - y1
            if w_box <= 0 or h_box <= 0:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ratio = h_box / w_box

            history.append({
                "frame": frame_idx,
                "center_y": cy,
                "ratio": ratio
            })

            # ===== 낙상 후보 판별 (프레임 단위) =====
            if len(history) > WINDOW:
                prev = history[-WINDOW]
                curr = history[-1]
                dy = curr["center_y"] - prev["center_y"]

                if curr["ratio"] < FALL_RATIO_TH and dy > DROP_Y_TH:
                    print(f"[FALL CANDIDATE] frame {curr['frame']}")
                    cv2.putText(
                        frame,
                        "FALL CANDIDATE",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )

            # 시각화
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ratio:{ratio:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    out.write(frame)

    if frame_idx % 100 == 0:
        print(f"[INFO] processed frame {frame_idx}")

cap.release()
if out:
    out.release()

print("[DONE] Processing finished")
