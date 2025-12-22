# state_machine.py

from enum import Enum
import time


class PersonState(Enum):
    STANDING = "standing"
    FALLING = "falling"
    FALLEN = "fallen"
    RECOVERED = "recovered"


class FallStateMachine:
    def __init__(
        self,
        falling_frames_threshold=5,
        fallen_frames_threshold=10
    ):
        self.state = PersonState.STANDING

        # threshold
        self.falling_frames_threshold = falling_frames_threshold
        self.fallen_frames_threshold = fallen_frames_threshold

        # frame counters
        self.falling_count = 0
        self.fallen_count = 0

        # 이벤트 플래그
        self.fall_event_occurred = False

        # 타임스탬프
        self.fall_time = None
        self.recover_time = None

        # 로그
        self.logs = []

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        self.logs.append(msg)
        print(msg)

    def update(self, model_state: str):
        """
        model_state: standing | falling | fallen
        """

        # ---------- STANDING ----------
        if self.state == PersonState.STANDING:
            if model_state == "falling":
                self.falling_count += 1
                if self.falling_count >= self.falling_frames_threshold:
                    self.state = PersonState.FALLING
                    self.log("상태 전이: STANDING → FALLING")
            else:
                self.falling_count = 0

        # ---------- FALLING ----------
        elif self.state == PersonState.FALLING:
            if model_state == "fallen":
                self.fallen_count += 1
                if self.fallen_count >= self.fallen_frames_threshold:
                    self.state = PersonState.FALLEN

                    if not self.fall_event_occurred:
                        self.fall_event_occurred = True
                        self.fall_time = time.time()
                        self.log("🚨 낙상 이벤트 발생 (FALL_DETECTED)")
            else:
                self.fallen_count = 0

        # ---------- FALLEN ----------
        elif self.state == PersonState.FALLEN:
            if model_state == "standing":
                self.state = PersonState.RECOVERED
                self.recover_time = time.time()
                self.log("📝 낙상 이후 회복 감지 (RECOVERED_AFTER_FALL)")

        # ---------- RECOVERED ----------
        elif self.state == PersonState.RECOVERED:
            if model_state == "standing":
                self.state = PersonState.STANDING
                self.log("상태 전이: RECOVERED → STANDING")

    def is_fall_alert_active(self):
        return self.fall_event_occurred
