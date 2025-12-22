# src/state_machine.py

class FallStateMachine:
    def __init__(
        self,
        falling_frames_threshold=5,
        still_frames_threshold=15,
        movement_threshold=0.02
    ):
        self.state = "IDLE"

        self.falling_count = 0
        self.still_count = 0

        self.falling_frames_threshold = falling_frames_threshold
        self.still_frames_threshold = still_frames_threshold
        self.movement_threshold = movement_threshold

        self.alert_sent = False

    def reset(self):
        self.state = "IDLE"
        self.falling_count = 0
        self.still_count = 0
        self.alert_sent = False

    def update(self, model_pred, movement_score):
        """
        model_pred: str ("falling", "normal")
        movement_score: float
        """

        # -------------------
        # IDLE 상태
        # -------------------
        if self.state == "IDLE":
            if model_pred == "falling":
                self.state = "FALLING"
                self.falling_count = 1

        # -------------------
        # FALLING 상태
        # -------------------
        elif self.state == "FALLING":
            if model_pred == "falling":
                self.falling_count += 1
            else:
                self.falling_count = 0

            if self.falling_count >= self.falling_frames_threshold:
                self.state = "FALLEN"

        # -------------------
        # FALLEN 상태
        # -------------------
        elif self.state == "FALLEN":
            if movement_score < self.movement_threshold:
                self.still_count += 1
            else:
                self.still_count = 0

            # 계속 움직임이 없으면 확정
            if self.still_count >= self.still_frames_threshold:
                self.state = "CONFIRMED_FALL"

        # -------------------
        # CONFIRMED_FALL 상태
        # -------------------
        elif self.state == "CONFIRMED_FALL":
            # 회복 움직임이 감지되면
            if movement_score > self.movement_threshold * 3:
                self.reset()

        return self.state

    def should_alert(self):
        """
        알림을 보내야 하는지 여부
        """
        if self.state == "CONFIRMED_FALL" and not self.alert_sent:
            self.alert_sent = True
            return True
        return False
