import time

class NavigationEngine:
    """
    Intelligent Navigation & State Machine for Vision Mate.
    Eliminates decision flickering using Exponential Moving Average (EMA) smoothing
    and multi-frame hysteresis debouncing.
    Generates actionable movement commands ('Move left', 'Move right', 'Stop', 'Continue straight').
    """

    STATE_CLEAR = "CLEAR"
    STATE_CAUTION = "CAUTION"
    STATE_STOP = "STOP"

    def __init__(self, speech_cooldown=3.0):
        # Distance thresholds with hysteresis gap to prevent fluctuating
        self.STOP_THRESH_ENTER = 1.15
        self.STOP_THRESH_EXIT  = 1.45

        self.CAUTION_THRESH_ENTER = 2.20
        self.CAUTION_THRESH_EXIT  = 2.60

        # Smoothed zone distances (EMA)
        self.dist_left = 9.9
        self.dist_center = 9.9
        self.dist_right = 9.9

        # Smoothing factor (alpha): 0.25 = smooth, 1.0 = raw
        self.alpha = 0.25

        # Current stabilized state
        self.state = self.STATE_CLEAR

        # Candidate state and debounce frame counter
        self.candidate_state = self.STATE_CLEAR
        self.candidate_count = 0
        self.REQUIRED_DEBOUNCE_FRAMES = 5  # Must hold for 5 consecutive frames (~160ms)

        # Speech throttling
        self.speech_cooldown = speech_cooldown
        self.last_speech_time = 0
        self.last_spoken_instruction = ""

        # Most critical obstacle info
        self.critical_obstacle = None

    def update(self, detected_obstacles, frame_w):
        """
        Takes list of detected obstacles: [{'label': ..., 'dist': ..., 'box': (x,y,w,h)}]
        and updates spatial zones, smoothed distances, and navigation state.
        """
        raw_left = 9.9
        raw_center = 9.9
        raw_right = 9.9

        closest_obstacle = None
        min_overall_dist = 9.9

        for obs in detected_obstacles:
            box = obs['box']
            dist = obs['dist']
            label = obs['label']

            # Only obstacles within actionable walking range (< 2.6m) affect movement
            if dist > 2.6:
                continue

            box_cx = box[0] + box[2] / 2.0

            if box_cx < frame_w * 0.35:
                raw_left = min(raw_left, dist)
            elif box_cx > frame_w * 0.65:
                raw_right = min(raw_right, dist)
            else:
                raw_center = min(raw_center, dist)

            if dist < min_overall_dist:
                min_overall_dist = dist
                closest_obstacle = (label, dist)

        self.critical_obstacle = closest_obstacle

        # 1. Apply Exponential Moving Average (EMA) distance smoothing
        if self.dist_left >= 9.0:
            self.dist_left = raw_left
        else:
            self.dist_left = round(self.alpha * raw_left + (1 - self.alpha) * self.dist_left, 2)

        if self.dist_center >= 9.0:
            self.dist_center = raw_center
        else:
            self.dist_center = round(self.alpha * raw_center + (1 - self.alpha) * self.dist_center, 2)

        if self.dist_right >= 9.0:
            self.dist_right = raw_right
        else:
            self.dist_right = round(self.alpha * raw_right + (1 - self.alpha) * self.dist_right, 2)

        # 2. Determine raw candidate state using hysteresis
        if self.state == self.STATE_STOP:
            # Harder to leave STOP state: center must be > STOP_THRESH_EXIT
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

        else: # STATE_CLEAR
            if self.dist_center < self.STOP_THRESH_ENTER:
                target_state = self.STATE_STOP
            elif self.dist_center < self.CAUTION_THRESH_ENTER:
                target_state = self.STATE_CAUTION
            else:
                target_state = self.STATE_CLEAR

        # 3. Debounce state switching over consecutive frames
        if target_state != self.state:
            if target_state == self.STATE_STOP:
                # Immediate safety trigger for STOP (zero delay for danger)
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
        """
        Synthesizes smoothed state into an actionable guidance command.
        Returns: (instruction_text, hud_status, hud_color, is_urgent)
        """
        obs_label = self.critical_obstacle[0] if self.critical_obstacle else "obstacle"

        if self.state == self.STATE_STOP:
            # Way ahead is blocked. Check if user can steer left or right!
            if self.dist_left > self.dist_right and self.dist_left > 1.8:
                instruction = f"Stop! {obs_label} close ahead. Move to your left."
            elif self.dist_right >= self.dist_left and self.dist_right > 1.8:
                instruction = f"Stop! {obs_label} close ahead. Move to your right."
            else:
                instruction = f"Stop! {obs_label} ahead. Path completely blocked."

            hud_status = f"STOP! {obs_label.upper()} {self.dist_center}m"
            hud_color = (0, 0, 255) # Red
            is_urgent = True

        elif self.state == self.STATE_CAUTION:
            # Approaching obstacle. Give proactive turn advice
            if self.dist_left > self.dist_right and self.dist_left > 2.0:
                instruction = f"Caution, approaching {obs_label}. Steer slightly left."
            elif self.dist_right >= self.dist_left and self.dist_right > 2.0:
                instruction = f"Caution, approaching {obs_label}. Steer slightly right."
            else:
                instruction = f"Caution, {obs_label} {self.dist_center} meters ahead. Slow down."

            hud_status = f"CAUTION: {obs_label.upper()} {self.dist_center}m"
            hud_color = (0, 255, 255) # Yellow
            is_urgent = False

        else:
            instruction = "Path clear ahead. Continue straight."
            hud_status = "PATH CLEAR - PROCEED"
            hud_color = (0, 255, 0) # Green
            is_urgent = False

        return instruction, hud_status, hud_color, is_urgent

    def should_speak(self, instruction, is_urgent):
        """
        Determines if speech should be dispatched based on cooldown and urgency.
        """
        now = time.time()
        elapsed = now - self.last_speech_time

        # Urgent STOP alerts speak immediately if state just changed
        if is_urgent and instruction != self.last_spoken_instruction:
            self.last_speech_time = now
            self.last_spoken_instruction = instruction
            return True

        # Regular periodic guidance
        if elapsed >= self.speech_cooldown:
            if instruction != self.last_spoken_instruction or elapsed >= self.speech_cooldown * 2:
                self.last_speech_time = now
                self.last_spoken_instruction = instruction
                return True

        return False
