# state_machine.py

from enum import Enum
import time


class PersonState(Enum):
    STANDING = "standing"
    FALLING = "falling"
    FALLEN = "fallen"
    RECOVERED = "recovered"


class FallStateMachine:
    def __init__(self):
        self.state = PersonState.STANDING

        # 이벤트 플래그
        self.fall_event_occurred = False

        # 타임스탬프
        self.fall_time = None
        self.recover_time = None

        # 로그 저장
        self.logs = []

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        self.logs.append(log_msg)
        print(log_msg)

    def update(self, model_state: str):
        """
        model_state:
        - "standing"
        - "falling"
        - "fallen"
        """

        # 1️. STANDING → FALLING
        if self.state == PersonState.STANDING:
            if model_state == "falling":
                self.state = PersonState.FALLING
                self.log("상태 전이: STANDING → FALLING")

        # 2️. FALLING → FALLEN (낙상 발생 지점)
        elif self.state == PersonState.FALLING:
            if model_state == "fallen":
                self.state = PersonState.FALLEN

                if not self.fall_event_occurred:
                    self.fall_event_occurred = True
                    self.fall_time = time.time()
                    self.log("낙상 이벤트 발생 (FALL_DETECTED)")
                else:
                    self.log("이미 낙상 이벤트 발생 상태")

        # 3️. FALLEN → RECOVERED (다시 일어남)
        elif self.state == PersonState.FALLEN:
            if model_state == "standing":
                self.state = PersonState.RECOVERED
                self.recover_time = time.time()
                self.log("낙상 이후 회복 감지 (RECOVERED_AFTER_FALL)")

        # 4️. RECOVERED → STANDING
        elif self.state == PersonState.RECOVERED:
            if model_state == "standing":
                self.state = PersonState.STANDING
                self.log("상태 전이: RECOVERED → STANDING")

    def is_fall_alert_active(self):
        """
        알림은 낙상 발생 후 계속 유지
        """
        return self.fall_event_occurred

    def get_logs(self):
        return self.logs
