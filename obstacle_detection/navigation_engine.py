import time
from statistics import median

class NavigationEngine:
    """
    Intelligent Navigation & State Machine for Vision Mate.
    Eliminates decision flickering using Exponential Moving Average (EMA) smoothing
    and multi-frame hysteresis debouncing. Generates actionable movement commands
    ('Move left', 'Move right', 'Stop', 'Continue straight').
    """

    STATE_CLEAR = "CLEAR"
    STATE_CAUTION = "CAUTION"
    STATE_STOP = "STOP"

    def __init__(self, speech_cooldown=3.0, confidence_thresh=0.5):
        # Distance thresholds with hysteresis gap to prevent fluctuating
        self.STOP_THRESH_ENTER = 1.15
        self.STOP_THRESH_EXIT = 1.45

        self.CAUTION_THRESH_ENTER = 2.20
        self.CAUTION_THRESH_EXIT = 2.60

        # Smoothed zone distances (EMA) – start at a large sentinel
        self.dist_left = 9.9
        self.dist_center = 9.9
        self.dist_right = 9.9

        # Smoothing factor (alpha): 0.25 = smooth, 1.0 = raw
        self.alpha = 0.25

        # Current stabilized state and last state (for voice‑only‑on‑change)
        self.state = self.STATE_CLEAR
        self.last_state = self.STATE_CLEAR

        # Candidate state and debounce frame counter
        self.candidate_state = self.STATE_CLEAR
        self.candidate_count = 0
        self.REQUIRED_DEBOUNCE_FRAMES = 5  # Must hold for 5 consecutive frames (~160 ms)

        # Speech throttling
        self.speech_cooldown = speech_cooldown
        self.last_speech_time = 0
        self.last_spoken_instruction = ""

        # Most critical obstacle info for HUD
        self.critical_obstacle = None

        # Confidence threshold for detections (0‑1 range)
        self.confidence_thresh = confidence_thresh

    def update(self, detected_obstacles, frame_w):
        """Update navigation zones from a list of obstacles.
        `detected_obstacles` is a list of dicts with keys:
        - ``label`` (str)
        - ``dist`` (float, metres)
        - ``box`` (tuple x,y,w,h)
        - ``confidence`` (float, optional)
        Returns a tuple ready for UI/HUD.
        """
        raw_left = 9.9
        raw_center = 9.9
        raw_right = 9.9

        # Collect distances per zone for robust median smoothing
        left_vals, center_vals, right_vals = [], [], []

        closest_obstacle = None
        min_overall_dist = 9.9

        for obs in detected_obstacles:
            # Filter low‑confidence detections – they cause spurious audio
            if obs.get('confidence', 1.0) < self.confidence_thresh:
                continue

            box = obs['box']
            dist = obs['dist']
            label = obs['label']

            # Ignore obstacles beyond walking relevance (>2.6 m)
            if dist > 2.6:
                continue

            box_cx = box[0] + box[2] / 2.0

            if box_cx < frame_w * 0.35:
                raw_left = min(raw_left, dist)
                left_vals.append(dist)
            elif box_cx > frame_w * 0.65:
                raw_right = min(raw_right, dist)
                right_vals.append(dist)
            else:
                raw_center = min(raw_center, dist)
                center_vals.append(dist)

            if dist < min_overall_dist:
                min_overall_dist = dist
                closest_obstacle = (label, dist)

        self.critical_obstacle = closest_obstacle

        # Helper – EMA over median of collected values (more robust than min)
        def _smooth(current, raw, vals):
            zone_raw = median(vals) if vals else raw
            if current >= 9.0:
                return zone_raw
            return round(self.alpha * zone_raw + (1 - self.alpha) * current, 2)

        self.dist_left = _smooth(self.dist_left, raw_left, left_vals)
        self.dist_center = _smooth(self.dist_center, raw_center, center_vals)
        self.dist_right = _smooth(self.dist_right, raw_right, right_vals)

        # ---------- State determination (same logic as before) ----------
        if self.state == self.STATE_STOP:
            if self.dist_center > self.STOP_THRESH_EXIT:
                target_state = self.STATE_CAUTION if self.dist_center < self.CAUTION_THRESH_EXIT else self.STATE_CLEAR
            else:
                target_state = self.STATE_STOP
        elif self.state == self.STATE_CAUTION:
            if self.dist_center < self.STOP_THRESH_ENTER:
                target_state = self.STATE_STOP
            elif self.dist_center > self.CAUTION_THRESH_EXIT:
                target_state = self.STATE_CLEAR
            else:
                target_state = self.STATE_CAUTION
        else:  # STATE_CLEAR
            if self.dist_center < self.STOP_THRESH_ENTER:
                target_state = self.STATE_STOP
            elif self.dist_center < self.CAUTION_THRESH_ENTER:
                target_state = self.STATE_CAUTION
            else:
                target_state = self.STATE_CLEAR

        # ---------- Debounce over consecutive frames ----------
        if target_state != self.state:
            if target_state == self.STATE_STOP:
                # Immediate safety trigger – no delay
                self.state = self.STATE_STOP
                self.candidate_count = 0
            elif target_state == self.candidate_state:
                self.candidate_count += 1
                if self.candidate_count >= self.REQUIRED_DEBOUNCE_FRAMES:
                    self.state = target_state
                    self.candidate_count = 0
            else:
                self.candidate_state = target_state
                self.candidate_count = 1
        else:
            self.candidate_count = 0

        return self.get_navigation_decision()

    def get_navigation_decision(self):
        """Translate the current state into a spoken instruction and HUD data.
        Returns ``(instruction, hud_status, hud_color, is_urgent)``.
        """
        obs_label = self.critical_obstacle[0] if self.critical_obstacle else "obstacle"

        if self.state == self.STATE_STOP:
            # Prefer the side with more clearance for a safe sidestep
            if self.dist_left > self.dist_right and self.dist_left > 1.8:
                instruction = f"Please shift left to avoid {obs_label}."
            elif self.dist_right >= self.dist_left and self.dist_right > 1.8:
                instruction = f"Please shift right to avoid {obs_label}."
            else:
                instruction = f"Stop! {obs_label} ahead – path blocked."
            hud_status = f"STOP! {obs_label.upper()} {self.dist_center}m"
            hud_color = (0, 0, 255)  # Red
            is_urgent = True
        elif self.state == self.STATE_CAUTION:
            if self.dist_left > self.dist_right and self.dist_left > 2.0:
                instruction = f"Caution: {obs_label} ahead. Slightly steer left."
            elif self.dist_right >= self.dist_left and self.dist_right > 2.0:
                instruction = f"Caution: {obs_label} ahead. Slightly steer right."
            else:
                instruction = f"Caution: {obs_label} {self.dist_center:.1f} m ahead. Slow down."
            hud_status = f"CAUTION: {obs_label.upper()} {self.dist_center}m"
            hud_color = (0, 255, 255)  # Yellow
            is_urgent = False
        else:
            instruction = "Path clear – continue forward."
            hud_status = "PATH CLEAR - PROCEED"
            hud_color = (0, 255, 0)  # Green
            is_urgent = False

        # Force a fresh spoken instruction on any state transition
        if self.state != self.last_state:
            self.last_spoken_instruction = ""
        self.last_state = self.state

        return instruction, hud_status, hud_color, is_urgent

    def should_speak(self, instruction, is_urgent):
        """Decide whether to emit speech.
        Urgent STOP messages are spoken immediately; otherwise we honour
        ``speech_cooldown`` and only repeat when the instruction changes.
        """
        now = time.time()
        elapsed = now - self.last_speech_time

        if is_urgent and instruction != self.last_spoken_instruction:
            self.last_speech_time = now
            self.last_spoken_instruction = instruction
            return True

        if elapsed >= self.speech_cooldown:
            if instruction != self.last_spoken_instruction or elapsed >= self.speech_cooldown * 2:
                self.last_speech_time = now
                self.last_spoken_instruction = instruction
                return True
        return False
