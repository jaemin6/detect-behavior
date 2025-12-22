# src/state_machine.py

from enum import Enum
import time


class FallState(Enum):
    IDLE = "idle"
    FALLING = "falling"
    FALLEN = "fallen"
    CONFIRMED_FALL = "confirmed_fall"
    RECOVERED = "recovered"


class FallStateMachine:
    def __init__(
        self,
        falling_frames_threshold=5,
        still_frames_threshold=15,
        movement_threshold=0.02
    ):
        self.state = FallState.IDLE

        # thresholds
        self.falling_frames_threshold = falling_frames_threshold
        self.still_frames_threshold = still_frames_threshold
        self.movement_threshold = movement_threshold

        # counters
        self.falling_count = 0
        self.still_count = 0

        # event flags
        self.alert_sent = False
        self.fall_event_occurred = False

        # timestamps
        self.fall_time = None
        self.recover_time = None

        # logs
        self.logs = []

    def log(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[{timestamp}] {message}"
        self.logs.append(msg)
        print(msg)

    def reset_to_idle(self):
        self.state = FallState.IDLE
        self.falling_count = 0
        self.still_count = 0

    def update(self, model_pred: str, movement_score: float):
        """
        model_pred: "falling" | "normal"
        movement_score: float
        """

        # ---------------- IDLE ----------------
        if self.state == FallState.IDLE:
            if model_pred == "falling":
                self.state = FallState.FALLING
                self.falling_count = 1
                self.log("상태 전이: IDLE → FALLING")

        # ---------------- FALLING ----------------
        elif self.state == FallState.FALLING:
            if model_pred == "falling":
                self.falling_count += 1
            else:
                self.falling_count = 0

            if self.falling_count >= self.falling_frames_threshold:
                self.state = FallState.FALLEN
                self.log("상태 전이: FALLING → FALLEN")

        # ---------------- FALLEN ----------------
        elif self.state == FallState.FALLEN:
            if movement_score < self.movement_threshold:
                self.still_count += 1
            else:
                self.still_count = 0

            if self.still_count >= self.still_frames_threshold:
                self.state = FallState.CONFIRMED_FALL

                if not self.fall_event_occurred:
                    self.fall_event_occurred = True
                    self.fall_time = time.time()
                    self.log("낙상 확정 (CONFIRMED_FALL)")

        # ---------------- CONFIRMED_FALL ----------------
        elif self.state == FallState.CONFIRMED_FALL:
            if movement_score > self.movement_threshold * 3:
                self.state = FallState.RECOVERED
                self.recover_time = time.time()
                self.log("낙상 후 회복 감지 (RECOVERED)")

        # ---------------- RECOVERED ----------------
        elif self.state == FallState.RECOVERED:
            # 알림은 유지, 로그만 남기고 정상 상태로 복귀
            if movement_score > self.movement_threshold:
                self.reset_to_idle()
                self.log("상태 전이: RECOVERED → IDLE")

        return self.state

    def should_alert(self):
        if self.state == FallState.CONFIRMED_FALL and not self.alert_sent:
            self.alert_sent = True
            return True
        return False
