# 낙상 감지 분석 및 테스트 결과

본 문서는 `detect-behavior` 모델의 실제 작동 사례와 학습 결과를 정리

## 1. 실시간 감지 테스트 (Inference)

객체의 바운딩 박스 비율(Aspect Ratio) 변화를 통해 상태를 정의

### A. 정상 상태 (Standing)
- **특징**: 세로 비율이 가로보다 월등히 높음 (Ratio > 1.5)

<img width="415" height="444" alt="스크린샷 2025-12-22 110705" src="https://github.com/user-attachments/assets/ff0c703e-7c44-41f3-9cfa-21fc66936854" />


- **결과**: `STANDING`으로 안정적인 감지 유지

### B. 낙상 발생 (Falling)
- **특징**: 낙상 시 신체의 높이가 낮아지며 비율이 급격히 변화 (Ratio 1.29 이하)

<img width="415" height="666" alt="스크린샷 2025-12-22 113002" src="https://github.com/user-attachments/assets/b60d7390-55e6-49ee-b8bb-6f26e0119a8c" />


- **결과**: `FALLING` 상태 발생 및 `state_machine.py`를 통한 알람 트리거
