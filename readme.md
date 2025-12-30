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
  
  <img width="500" height="350" alt="스크린샷 2025-12-16 205137" src="https://github.com/user-attachments/assets/167a8221-194c-444f-93b2-330155ead345" />

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
