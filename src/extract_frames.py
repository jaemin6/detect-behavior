import cv2
import os
import shutil 
# glob 모듈을 사용하여 폴더 내 파일 개수를 확인하는 대신 os.listdir을 사용하겠습니다.

# --- 설정 변수 ---
video_dir = "../data/raw"
output_dir = "../data/frames"
frame_rate = 2 # 목표: 초당 2장씩 추출 
# -----------------

os.makedirs(output_dir, exist_ok=True)

videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi'))]

total_videos = len(videos)
print(f"총 {total_videos}개의 영상을 목표치(초당 {frame_rate}장)로 재추출합니다.")
print("-" * 30)

for i, video_name in enumerate(videos):
    video_path = os.path.join(video_dir, video_name)
    video_base_name = video_name.split('.')[0]
    video_output_path = os.path.join(output_dir, video_base_name)

    # =========================================================================
    # 💡 [핵심 추가/수정] 이미 추출된 영상 건너뛰기 로직
    # 폴더가 존재하고, 안에 파일이 하나라도 있다면 (이미 추출 완료되었다고 가정) 건너뜁니다.
    # 이렇게 하면 0번부터 95번까지의 영상을 재처리하지 않아 시간을 절약할 수 있습니다.
    if os.path.exists(video_output_path) and len(os.listdir(video_output_path)) > 0:
        existing_frames = len(os.listdir(video_output_path))
        print(f"[{i+1}/{total_videos}] ⏭️ 이미 추출된 영상 건너뛰기: {video_name} ({existing_frames} 프레임 존재)")
        print("-" * 30)
        continue # 다음 영상으로 이동
    # =========================================================================
    
    print(f"[{i+1}/{total_videos}] 🎥 영상 처리 시작: {video_name}")

    # --- 💡 출력 폴더 초기화 로직 (정확한 재추출/재시작을 위해) ---
    if os.path.exists(video_output_path):
        # 이미 폴더는 있지만, 위의 건너뛰기 로직을 통과했다면 (프레임이 0개였거나 에러로 종료된 경우) 초기화합니다.
        print(f" ⚠️ 기존 폴더 존재. 정확한 {frame_rate}장 추출을 위해 폴더를 초기화합니다.")
        try:
            shutil.rmtree(video_output_path)
        except Exception as e:
            print(f" ❌ 폴더 삭제 실패: {e}. 다음 영상으로 넘어갑니다.")
            continue
            
    os.makedirs(video_output_path, exist_ok=True)
    # ----------------------------------------------------
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f" ❌ 영상을 열 수 없습니다: {video_name}")
        continue
    
    frame_idx = 0
    save_idx = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        print(f" ⚠️ FPS 정보를 가져올 수 없습니다. 스킵: {video_name}")
        cap.release()
        continue
        
    # 프레임 저장 간격 계산 (예: fps=30, frame_rate=2 -> interval=15)
    interval = int(fps / frame_rate)
    if interval == 0:
        interval = 1 # 최소 1로 설정 (frame_rate > fps인 경우 매 프레임 저장)
        
    print(f"  (FPS: {fps:.2f}, 저장 간격: {interval} 프레임마다)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            # 파일명은 0.jpg, 1.jpg, ... 형식으로 저장됩니다.
            save_path = os.path.join(video_output_path, f"{save_idx}.jpg") 
            cv2.imwrite(save_path, frame)
            save_idx += 1

        frame_idx += 1

    cap.release()
    
    # 추출 완료 후, 프레임이 실제로 저장되었는지 확인하는 단계 (추가된 안정성 로직)
    if save_idx > 0:
        # 이 시점에서는 해당 영상의 추출이 완료된 것으로 간주합니다.
        print(f"  ✅ 추출 완료. 저장된 프레임 수: {save_idx}")
    else:
        # 영상이 너무 짧거나 문제가 있어서 프레임이 저장되지 않은 경우입니다.
        print(f"  ⚠️ 추출 완료되었으나 저장된 프레임이 없습니다. 영상 길이 확인 필요.")
        
    print("-" * 30)

print("모든 영상 → 프레임 추출 완료!")