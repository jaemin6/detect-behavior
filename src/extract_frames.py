import cv2
import os

video_dir = "data/raw_videos"
output_dir = "data/frames"
frame_rate = 0.5  # 초당 3장만 추출 (라벨링 부담 줄이기)

os.makedirs(output_dir, exist_ok=True)

videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

for video_name in videos:
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)

    video_output_path = os.path.join(output_dir, video_name.split('.')[0])
    os.makedirs(video_output_path, exist_ok=True)

    frame_idx = 0
    save_idx = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            save_path = os.path.join(video_output_path, f"{save_idx}.jpg")
            cv2.imwrite(save_path, frame)
            save_idx += 1

        frame_idx += 1

    cap.release()

print("모든 영상 → 프레임 추출 완료!")
