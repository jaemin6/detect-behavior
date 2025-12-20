from ultralytics import YOLO
import cv2

# =========================
# 설정
# =========================
MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../data/raw/FD_In_H11H21H31_0001_20210112_09.mp4"

CONF_TH = 0.4

# 낙상 판단 임계값 (디버그용: 느슨하게)
FALL_RATIO_TH = 1.2   # 눕기 시작 판단
DROP_Y_TH = 20        # 중심점 급하강
WINDOW = 3            # 프레임 비교 간격

# =========================
# 모델 & 비디오 로드
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

frame_idx = 0
history = []

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

            w_box = x2 - x1
            h_box = y2 - y1
            if w_box <= 0 or h_box <= 0:
                continue

            # 중심점 & 비율
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            ratio = h_box / w_box

            history.append({
                "frame": frame_idx,
                "center_y": cy,
                "ratio": ratio
            })

            # =========================
            # 낙상 후보 판별
            # =========================
            fall_flag = False
            dy = 0

            if len(history) > WINDOW:
                prev = history[-WINDOW]
                dy = cy - prev["center_y"]

                if ratio < FALL_RATIO_TH and dy > DROP_Y_TH:
                    fall_flag = True
                    print(
                        f"[FALL CANDIDATE] frame={frame_idx}, "
                        f"ratio={ratio:.2f}, dy={dy:.1f}"
                    )

            # =========================
            # 시각화
            # =========================
            color = (0, 0, 255) if fall_flag else (0, 255, 0)

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
                f"ratio:{ratio:.2f} dy:{dy:.1f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Fall Detection Debug", frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# 종료 처리
# =========================
cap.release()
cv2.destroyAllWindows()
print("[DONE] Processing finished")
