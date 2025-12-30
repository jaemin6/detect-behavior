#  노인 낙상 감지 시스템 (Fall Detection Project)

본 프로젝트는 **영상 기반 노인 낙상(Fall) 행동 감지 시스템**을 구축하는 것을 목표로 한다.  
영상에서 사람의 행동을 분석하여 **낙상 상황을 자동으로 탐지**하고,  
향후 실시간 감지 및 알림 시스템으로 확장하는 것을 최종 목표로 한다.

---

##  프로젝트 개요

- **프로젝트명**: Fall Detection (노인 낙상 감지)
- **문제 정의**  
  고령자 낙상은 즉각적인 대응이 필요한 위험 상황이지만,  
  항상 사람이 직접 관찰하기 어렵다.  
  이를 해결하기 위해 **Computer Vision + Object Detection** 기반 자동 감지 시스템을 개발한다.

- **접근 방법**
  - CCTV / 영상 데이터 활용
  - 프레임 단위 이미지 추출
  - Roboflow를 이용한 수동 라벨링
  - YOLO 기반 모델 학습
  - 낙상 객체 탐지 성능 평가

---

##  프로젝트 폴더 구조

```bash
detect_behavior/
├── data/
│   ├── raw_videos/        # 원본 낙상 / 정상 행동 영상
│   ├── frames/            # 영상 → 프레임 추출 이미지
│   └── dataset/           # Roboflow에서 export한 데이터셋
│
├── models/                # 학습된 YOLO 모델 (.pt)
├── src/
│   ├── cut_fall_clip.py   # 영상 클립 분리
│   ├── extract_frames.py  # 프레임 추출 스크립트
│   └── train.py           # YOLO 학습 스크립트
│
├── img/                   # README용 이미지
├── requirements.txt
└── README.md
```


## 1. 데이터 수집 및 프레임 추출

낙상 행동이 포함된 영상 다수 수집

각 영상을 프레임 단위 이미지로 변환

영상별로 프레임 폴더 분리 관리

## 프레임 추출 예시

![612](https://github.com/user-attachments/assets/faa8449e-5d44-4bcb-a6eb-ff84c5c0b15a)

```
data/frames/
├── video_001/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
├── video_002/
└── ...
```
## 2. Roboflow를 이용한 라벨링
### Roboflow 사용 이유

웹 기반 UI로 직관적인 라벨링 가능

YOLO 형식 데이터셋 자동 생성

Train / Valid / Test 자동 분리

### 라벨링 클래스
클래스명	설명
fall_person	낙상 중이거나 쓰러진 사람

초기 주석 설정 문제로 클래스명이 변경되지 않는 이슈를 경험했고,
Annotation Group 및 Class 설정을 재정비하여 해결함.

### Roboflow 라벨링 화면, Bounding Box 라벨링 예시

  <img width="500" height="350" alt="스크린샷 2025-12-16 205137" src="https://github.com/user-attachments/assets/167a8221-194c-444f-93b2-330155ead345" />

## 3. 데이터셋 Export

Export Format: YOLOv8

이미지 크기 통일

Train / Validation / Test 자동 분할
```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
├── test/
└── data.yaml
```

## 4. YOLO 모델 학습

사용 모델: YOLOv8

학습 방식: Object Detection

Epochs 설정 후 학습 진행
```
yolo task=detect mode=train \
    model=yolov8n.pt \
    data=data.yaml \
    epochs=XX
```

### 학습 진행 로그

```
| 구분        | Epoch 1 (Before) | Epoch 50 (After) |
| --------- | ---------------- | ---------------- |
| mAP50     | **0.7474**       | **0.9948**       |
| Recall    | 0.6653           | **0.9955**       |
| Precision | 0.7197           | **0.9881**       |

```


## 5. 학습 결과 (Metrics)

Precision / Recall

mAP@0.5

Loss 감소 확인

 Training Results (results.png)
<img width="1200" height="600" alt="results" src="https://github.com/user-attachments/assets/18951597-1d71-4606-a6c7-f4108a545cc1" />



 Validation 예측 결과

![val_batch2_pred](https://github.com/user-attachments/assets/b2a6c0b2-4d7e-474a-830a-32df55520c3e)








