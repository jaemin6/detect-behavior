from ultralytics import YOLO
import cv2

# ========================
# 설정
# ========================
MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../data/raw/FD_In_H11H21H31_0001_20210112_09.mp4"
CONF_TH = 0.4

# ========================
# 모델 & 영상 로드
# ========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
history = []  # 중심점, 비율 저장

# ========================
# 프레임 처리
# ========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    results = model(frame, conf=CONF_TH, device="cpu", verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        # ⚠ 현재는 1명 기준 (fall_person 단일 객체)
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ratio = h / w

            history.append({
                "frame": frame_idx,
                "center_x": cx,
                "center_y": cy,
                "ratio": ratio
            })

            # ========================
            # 시각화
            # ========================
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)
            cv2.putText(
                frame,
                f"ratio: {ratio:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    cv2.imshow("Fall Detection Debug", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
