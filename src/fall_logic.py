from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")

# 영상 로드
cap = cv2.VideoCapture("data/raw/FD_In_H11H21H31_0001_20210112_09.mp4")

frame_idx = 0
history = []  # 중심점, 비율 저장용

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # YOLO 추론
    results = model(frame, conf=0.4, device="cpu", verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            # bbox 좌표 (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 너비 / 높이
            w = x2 - x1
            h = y2 - y1

            # 중심점
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # bbox 비율 (핵심)
            ratio = h / w

            history.append({
                "frame": frame_idx,
                "center_x": cx,
                "center_y": cy,
                "ratio": ratio
            })

            # 시각화 (확인용)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, (0,0,255), -1)
            cv2.putText(frame, f"ratio: {ratio:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    cv2.imshow("Fall Detection Debug", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
