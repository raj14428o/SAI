import os
import sys
import asyncio
import concurrent.futures
import cv2
import time
import datetime
import re

# Add Communication directory to sys.path so modules and voice are always importable
COMM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if COMM_DIR not in sys.path:
    sys.path.insert(0, COMM_DIR)

from modules.communication_tool import CommunicationTool
from voice.speech_output import SpeechOutput
from voice.speech_input import SpeechInput
from modules.daily_tools import (
    WeatherTool,
    ReminderTool,
    NotesTool,
    CalculatorTool,
    DeviceStatusTool
)

# Add opencv/Faces to path for FaceManager
FACES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "opencv", "Faces"))
if FACES_DIR not in sys.path:
    sys.path.append(FACES_DIR)

from face_manager import FaceManager

# Add obstacle_detection to path for DepthEstimator & NavigationEngine
OBSTACLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "obstacle_detection"))
if OBSTACLE_DIR not in sys.path:
    sys.path.append(OBSTACLE_DIR)

from depth_estimator import DepthEstimator
from navigation_engine import NavigationEngine


class AllyController:
    """
    Core conversational AI brain for Vision Mate.
    Supports flexible conversational intent parsing, open-ended question answering (Gemini 3.8 Flash),
    smooth speech pacing, daily productivity tools (Weather, Reminders, Notes, Calculator, Device Status),
    and real-time vision modes.
    """

    def __init__(self):
        self.voice_in = SpeechInput()
        self.voice_out = SpeechOutput()
        self.comm_tool = CommunicationTool()
        self.face_manager = FaceManager()
        self.depth_estimator = DepthEstimator()
        self.reminder_tool = ReminderTool()
        self.notes_tool = NotesTool()

    def _ask_voice(self, prompt: str, retries: int = 2) -> str:
        """Speak a prompt, wait a moment, listen for response, retry on failure."""
        self.voice_out.speak(prompt)
        time.sleep(0.8)  # Let audio finish completely before listening
        for attempt in range(retries):
            response = self.voice_in.listen_command(timeout=7, phrase_limit=12)
            if response:
                time.sleep(0.4)
                return response
            if attempt < retries - 1:
                self.voice_out.speak("Please say it again.")
                time.sleep(0.6)
        return ""

    def answer_conversational(self, query: str):
        """
        Answers any open-ended question or conversational remark freely.
        Queries Gemini 3.8 Flash with adequate time (18s timeout) and an automatic retry
        to ensure it only falls back if the request genuinely failed or could not connect.
        Falls back to a natural local conversational response if Gemini is unavailable.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    "You are Ally, a helpful, intelligent voice companion and vision assistant for Vision Mate. "
                    "The user is speaking to you via a microphone. Respond conversationally, naturally, and warmly "
                    "in 1 to 2 spoken sentences. Do not use markdown, asterisks, or lists.\n\n"
                    f"User: {query}"
                )

                model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

                def _call_gemini():
                    return client.models.generate_content(model=model_name, contents=prompt)

                # Attempt up to 2 times with an 18-second timeout each to ensure it genuinely failed
                for attempt in range(2):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_call_gemini)
                        try:
                            response = future.result(timeout=18.0)
                            if response and response.text:
                                text = response.text.strip().replace("*", "").replace("#", "").replace('"', '')
                                if text:
                                    print(f"[Gemini Spoken]: {text}")
                                    self.voice_out.speak(text)
                                    return
                        except concurrent.futures.TimeoutError:
                            print(f"[WARN] Gemini call timed out on attempt {attempt + 1}")
                            if attempt == 0:
                                time.sleep(0.5)
                                continue
                        except Exception as e:
                            print(f"[WARN] Gemini error on attempt {attempt + 1}: {e}")
                            if attempt == 0:
                                time.sleep(0.5)
                                continue
            except Exception as e:
                print(f"[WARN] Gemini client setup failed: {e}")

        # Natural local conversational fallback
        now = datetime.datetime.now()
        q = query.lower()

        if "time" in q:
            self.voice_out.speak(f"It is currently {now.strftime('%I:%M %p')}.")
        elif any(w in q for w in ("date", "today", "what day")):
            self.voice_out.speak(f"Today is {now.strftime('%A, %B %d, %Y')}.")
        elif any(w in q for w in ("hello", "hi", "hey", "good morning", "good evening", "good afternoon")):
            self.voice_out.speak("Hello! I am Vision Mate. How can I help you?")
        elif "how are you" in q:
            self.voice_out.speak("I am doing great and all sensors are ready. What would you like to do?")
        elif any(w in q for w in ("what can you do", "help", "features", "who are you", "your name")):
            self.voice_out.speak(
                "I can guide your walking path with depth sensors, identify people in front of you, "
                "draft and send emails, check the weather, manage reminders and notes, and answer questions."
            )
        elif "thank" in q:
            self.voice_out.speak("You're very welcome! Let me know if you need anything else.")
        else:
            self.voice_out.speak(
                "I didn't quite catch that or couldn't reach the answer right now. Could you please say that again?"
            )

    def identify_person_in_camera(self):
        """Grabs a frame from webcam and recognizes the person in front."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return "Camera could not be opened."

        for _ in range(5):
            ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return "Could not capture image from camera."

        results = self.face_manager.identify_faces(frame)
        if not results:
            return "I don't see any face directly in front of the camera."

        names = [r["name"] for r in results if r["name"] != "Unknown"]
        if names:
            return f"I see {', '.join(names)} in front of you."
        else:
            return "There is an unknown person in front of you."

    async def run_walking_mode(self, max_seconds=120):
        """
        Activates continuous walking navigation using MiDaS depth & SSD MobileNet.
        Provides stabilized, actionable movement instructions ('Move left', 'Stop', etc.).
        While navigating the user can interrupt with voice commands (e.g., "stop walking",
        "pause", "resume walking").  The loop will break on interruption, returning control
        to the main command processor so the user can request other actions (email, face ID,
        etc.).  To continue walking the user simply says a walking keyword again.
        """
        self.voice_out.speak("Walking mode active. Monitoring obstacles. Say ‘stop walking’ to pause.")
        time.sleep(1.0)

        frozen_model = os.path.join(OBSTACLE_DIR, "frozen_inference_graph.pb")
        config_file  = os.path.join(OBSTACLE_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        labels_file  = os.path.join(OBSTACLE_DIR, "coco.names")

        with open(labels_file, "rt") as f:
            class_labels = f.read().rstrip("\n").split("\n")

        model = cv2.dnn_DetectionModel(frozen_model, config_file)
        model.setInputSize(320, 320)
        model.setInputScale(1.0 / 127.5)
        model.setInputMean((127.5, 127.5, 127.5))
        model.setInputSwapRB(True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.voice_out.speak("Error opening camera for walking mode.")
            return

        start_time = time.time()
        nav_engine = NavigationEngine(speech_cooldown=3.0)

        try:
            while time.time() - start_time < max_seconds:
                ret, frame = cap.read()
                if not ret:
                    break

                # ---- Vision pipeline ----
                depth_map, depth_colored = self.depth_estimator.estimate_depth(frame)
                classIndex, confidence, bbox = model.detect(frame, confThreshold=0.48)

                frame_h, frame_w = frame.shape[:2]
                detected_obstacles = []

                if len(classIndex) != 0:
                    for classInd, conf, box in zip(classIndex.flatten(), confidence.flatten(), bbox):
                        label_name = class_labels[classInd - 1] if 0 < classInd <= len(class_labels) else "object"
                        dist_m, status, status_color = self.depth_estimator.get_object_distance(depth_map, box)
                        detected_obstacles.append({
                            "label": label_name,
                            "dist": dist_m,
                            "box": box,
                            "confidence": float(conf)
                        })
                        cv2.rectangle(frame, box, status_color, 2)
                        cv2.putText(frame, f"{label_name.upper()} {dist_m}m", (box[0], max(20, box[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                # ---- Navigation update & speech ----
                instruction, hud_status, hud_color, is_urgent = nav_engine.update(detected_obstacles, frame_w)
                if nav_engine.should_speak(instruction, is_urgent):
                    self.voice_out.speak(instruction, priority=is_urgent)

                # ---- HUD overlay ----
                cv2.rectangle(frame, (0, 0), (frame_w, 42), (20, 20, 20), -1)
                cv2.putText(frame, f"STATUS: {hud_status}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)
                cv2.rectangle(frame, (0, 42), (frame_w, 70), (40, 40, 40), -1)
                cv2.putText(frame, f"COMMAND: {instruction}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.imshow("Vision Mate - Walking Mode Navigation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # ---- Voice interruption check (non‑blocking) ----
                # Small timeout so navigation stays responsive
                interrupt = self.voice_in.listen_command(timeout=0.5, phrase_limit=12)
                if interrupt:
                    cmd = interrupt.lower()
                    if "stop walking" in cmd or "pause" in cmd:
                        self.voice_out.speak("Navigation paused. You can ask for other actions now.")
                        break  # exit walking loop, return to main command processor
                    # Optional: allow a quick "status" query
                    if "status" in cmd:
                        self.voice_out.speak(f"Current instruction: {instruction}.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.voice_out.speak("Walking mode exited.")
            time.sleep(0.5)

        """
        Activates continuous walking navigation using MiDaS depth & SSD MobileNet.
        Provides stabilized, actionable movement instructions ('Move left', 'Stop', etc.).
        """
        self.voice_out.speak("Walking mode active. Monitoring obstacles. Press 'q' or say stop to exit.")
        time.sleep(1.0)

        frozen_model = os.path.join(OBSTACLE_DIR, "frozen_inference_graph.pb")
        config_file  = os.path.join(OBSTACLE_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
        labels_file  = os.path.join(OBSTACLE_DIR, "coco.names")

        with open(labels_file, "rt") as f:
            class_labels = f.read().rstrip("\n").split("\n")

        model = cv2.dnn_DetectionModel(frozen_model, config_file)
        model.setInputSize(320, 320)
        model.setInputScale(1.0 / 127.5)
        model.setInputMean((127.5, 127.5, 127.5))
        model.setInputSwapRB(True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.voice_out.speak("Error opening camera for walking mode.")
            return

        start_time = time.time()
        nav_engine = NavigationEngine(speech_cooldown=3.0)

        try:
            while time.time() - start_time < max_seconds:
                ret, frame = cap.read()
                if not ret:
                    break

                depth_map, depth_colored = self.depth_estimator.estimate_depth(frame)
                classIndex, confidence, bbox = model.detect(frame, confThreshold=0.48)

                frame_h, frame_w = frame.shape[:2]
                detected_obstacles = []

                if len(classIndex) != 0:
                    for classInd, conf, box in zip(classIndex.flatten(), confidence.flatten(), bbox):
                        label_name = class_labels[classInd - 1] if 0 < classInd <= len(class_labels) else "object"
                        dist_m, status, status_color = self.depth_estimator.get_object_distance(depth_map, box)

                        detected_obstacles.append({
                            "label": label_name,
                            "dist": dist_m,
                            "box": box,
                            "color": status_color
                        })
                        cv2.rectangle(frame, box, status_color, 2)
                        cv2.putText(frame, f"{label_name.upper()} {dist_m}m", (box[0], max(20, box[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

                # Stabilized navigation update
                instruction, hud_status, hud_color, is_urgent = nav_engine.update(detected_obstacles, frame_w)

                # Speak actionable guidance
                if nav_engine.should_speak(instruction, is_urgent):
                    self.voice_out.speak(instruction, priority=is_urgent)

                # Top HUD
                cv2.rectangle(frame, (0, 0), (frame_w, 42), (20, 20, 20), -1)
                cv2.putText(frame, f"STATUS: {hud_status}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

                cv2.rectangle(frame, (0, 42), (frame_w, 70), (40, 40, 40), -1)
                cv2.putText(frame, f"COMMAND: {instruction}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                cv2.imshow("Vision Mate - Walking Mode Navigation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.voice_out.speak("Walking mode exited.")
            time.sleep(0.5)

    async def process_command(self):
        """
        Main loop for listening and acting.
        Uses flexible intent routing for walking, vision, email, daily tools (weather, reminders, notes),
        and open-ended conversation.
        """
        # Brief natural pause before opening microphone
        time.sleep(0.4)
        command = self.voice_in.listen_command(timeout=6, phrase_limit=12)

        if not command:
            # Silent timeout: don't annoy user if room is silent
            return

        # Pacing: brief 0.5s pause before answering so it feels natural and human
        time.sleep(0.5)

        cmd = command.lower().strip()

        # ── 1. WALKING / NAVIGATION INTENT ──────────────────────────────
        walking_keywords = ("walk", "walking", "navigat", "guide me", "lead the way", "look ahead", "watch my step")
        if any(w in cmd for w in walking_keywords):
            await self.run_walking_mode(max_seconds=180)
            return

        # ── 2. FACE IDENTIFICATION INTENT ───────────────────────────────
        face_keywords = ("who is", "who's", "who that", "who are", "identify person", "recognize", "see someone", "who in front")
        if any(w in cmd for w in face_keywords):
            self.voice_out.speak("Looking at the camera...")
            time.sleep(0.4)
            result = self.identify_person_in_camera()
            print(f"[Face Result]: {result}")
            self.voice_out.speak(result)
            return

        # ── 3. WEATHER TOOL INTENT ───────────────────────────────────────
        if any(w in cmd for w in ("weather", "temperature", "forecast", "is it raining", "how hot", "how cold")):
            self.voice_out.speak("Checking the weather...")
            # Try to extract city from phrase like "weather in Mumbai" or "temperature in Paris"
            match = re.search(r'(?:in|for|at)\s+([a-zA-Z\s]+)', cmd)
            city = match.group(1).strip() if match else None
            # Filter out common filler words from extracted city
            if city and city in ("the city", "my area", "here", "today", "tomorrow"):
                city = None
            weather_report = WeatherTool.get_weather(city)
            print(f"[Weather Report]: {weather_report}")
            self.voice_out.speak(weather_report)
            return

        # ── 4. CALENDAR & REMINDERS TOOL INTENT ──────────────────────────
        # Setting a reminder
        if any(w in cmd for w in ("remind me", "set a reminder", "add reminder", "create reminder", "new reminder")):
            # Extract content after trigger
            match = re.search(r'(?:remind me to|remind me|set a reminder to|set a reminder for|add reminder to|add reminder)\s+(.*)', cmd)
            content = match.group(1).strip() if match else ""
            if not content:
                content = self._ask_voice("What should I remind you to do?")
            
            if content:
                # Check if time was mentioned e.g. "at 5 pm" or "tomorrow"
                time_match = re.search(r'(?:at|on|tomorrow|in)\s+(\d+.*|[a-zA-Z]+.*)', content)
                time_str = time_match.group(0).strip() if time_match else None
                resp = self.reminder_tool.add_reminder(content, time_str)
                self.voice_out.speak(resp)
            else:
                self.voice_out.speak("Reminder cancelled.")
            return

        # Listing reminders / calendar
        elif any(w in cmd for w in ("my reminders", "what are my reminders", "check reminders", "show reminders", "read reminders", "calendar", "my schedule")):
            resp = self.reminder_tool.list_reminders()
            self.voice_out.speak(resp)
            return

        # Clearing reminders
        elif any(w in cmd for w in ("clear reminders", "delete reminders", "clear my reminders", "delete my reminders")):
            resp = self.reminder_tool.clear_reminders()
            self.voice_out.speak(resp)
            return

        # ── 5. NOTES & TO-DO LIST TOOL INTENT ────────────────────────────
        # Adding a note
        if any(w in cmd for w in ("take a note", "add a note", "save a note", "write a note", "make a note", "new note")):
            match = re.search(r'(?:take a note|add a note|save a note|write a note|make a note|new note)\s+(.*)', cmd)
            content = match.group(1).strip() if match else ""
            if not content:
                content = self._ask_voice("What should the note say?")
            
            if content:
                resp = self.notes_tool.add_note(content)
                self.voice_out.speak(resp)
            else:
                self.voice_out.speak("Note cancelled.")
            return

        # Reading notes
        elif any(w in cmd for w in ("read my notes", "read notes", "show my notes", "show notes", "what are my notes", "check notes")):
            resp = self.notes_tool.list_notes()
            self.voice_out.speak(resp)
            return

        # Clearing notes
        elif any(w in cmd for w in ("clear notes", "delete notes", "clear my notes", "delete my notes")):
            resp = self.notes_tool.clear_notes()
            self.voice_out.speak(resp)
            return

        # ── 6. CALCULATOR & MATH TOOL INTENT ─────────────────────────────
        if any(w in cmd for w in ("calculate", "plus", "minus", "times", "divided by", "multiplied by", "what is")) and any(c.isdigit() for c in cmd):
            result = CalculatorTool.calculate(cmd)
            if result:
                self.voice_out.speak(result)
                return

        # ── 7. BATTERY & DEVICE STATUS INTENT ────────────────────────────
        if any(w in cmd for w in ("battery", "battery status", "how much battery", "system status", "device status", "power status")):
            status = DeviceStatusTool.get_status()
            self.voice_out.speak(status)
            return

        # ── 8. SEND EMAIL INTENT ─────────────────────────────────────────
        email_keywords = ("send email", "write email", "compose email", "send an email", "send mail", "draft an email")
        if any(w in cmd for w in email_keywords):
            person = self._ask_voice("Whom should I send the email to?")
            if not person:
                self.voice_out.speak("Could not get a recipient name. Cancelling.")
                return

            to_email = self.comm_tool.find_contact(person)
            if not to_email:
                to_email = self._ask_voice(f"I don't have {person} saved. What is their email address?")
                if not to_email:
                    self.voice_out.speak("No email address provided. Cancelling.")
                    return

            purpose = self._ask_voice("What is the purpose of this email?")
            if not purpose:
                purpose = "General catch-up and check-in"

            self.voice_out.speak("Drafting the email. Please wait.")
            time.sleep(0.4)
            subject, body = self.comm_tool.generate_email_content(person, purpose)

            print(f"[Draft Subject]: {subject}\n[Draft Body]:\n{body}")
            self.voice_out.speak(f"I drafted an email with subject: {subject}. Would you like me to send it?")
            confirm = self._ask_voice("Say yes to send, or cancel to abort.")

            if "yes" in confirm.lower() or "send" in confirm.lower():
                self.voice_out.speak("Sending your email now.")
                response = await self.comm_tool.manage_communication(
                    action="send_email", to=to_email, subject=subject, body=body
                )
                self.voice_out.speak(response)
            else:
                self.voice_out.speak("Email cancelled.")
            return

        # ── 9. RETRAIN FACES INTENT ──────────────────────────────────────
        retrain_keywords = ("retrain", "update face", "update faces", "train face", "train faces", "learn faces", "new photos")
        if any(w in cmd for w in retrain_keywords):
            self.voice_out.speak("Retraining face recognizer on all photos...")
            time.sleep(0.3)
            msg = self.face_manager.retrain()
            self.voice_out.speak("Face model updated successfully.")
            return

        # ── 10. READ EMAIL INTENT ────────────────────────────────────────
        if "read email" in cmd or "check email" in cmd:
            self.voice_out.speak("Reading your latest emails...")
            response = await self.comm_tool.manage_communication(action="read_emails")
            print(response)
            self.voice_out.speak(response)
            return

        # ── 11. FIND CONTACT INTENT ──────────────────────────────────────
        if "find contact" in cmd or "search contact" in cmd:
            name = self._ask_voice("Who do you want to search for?")
            if not name:
                self.voice_out.speak("Could not get a name. Cancelling.")
                return

            response = await self.comm_tool.manage_communication(action="find_contact", name=name)
            print(response)
            self.voice_out.speak(response)
            return

        # ── 12. EXIT / QUIT INTENT ───────────────────────────────────────
        exit_keywords = ("stop assistant", "shut down", "goodbye", "bye", "exit system")
        if any(w in cmd for w in exit_keywords):
            self.voice_out.speak("Goodbye! Vision Mate shutting down.")
            raise SystemExit(0)

        # ── 13. OPEN-ENDED CONVERSATION / GENERAL KNOWLEDGE ──────────────
        self.answer_conversational(command)
