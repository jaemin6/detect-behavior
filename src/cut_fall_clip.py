import cv2
import subprocess
import os

VIDEO_PATH = "../data/raw/FD_In_H11H21H31_0001_20210112_09.mp4"
OUTPUT_DIR = "../data/clips"
CLIP_SEC = 6                  # 전후 3초 → 총 6초

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
fall_time = None

print("[INFO] 영상 재생 중...")
print("[INFO] 낙상 순간에 'f' 키를 누르세요")
print("[INFO] 종료하려면 'q'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Fall Clip Cutter", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == ord('f'):
        fall_time = frame_idx / fps
        print(f"[INFO] 낙상 시점 기록: {fall_time:.2f}초")
        break

    if key == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

if fall_time is not None:
    start = max(0, fall_time - CLIP_SEC / 2)
    output_path = os.path.join(OUTPUT_DIR, "fall_clip.mp4")

    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", VIDEO_PATH,
        "-t", str(CLIP_SEC),
        "-c", "copy",
        output_path
    ]

    subprocess.run(cmd)
    print(f"[SUCCESS] 클립 생성 완료 → {output_path}")
else:
    print("[INFO] 낙상 시점이 기록되지 않았습니다.")
