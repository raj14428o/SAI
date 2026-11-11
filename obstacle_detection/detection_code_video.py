import cv2
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

COMM_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "Communication"))
if COMM_DIR not in sys.path:
    sys.path.insert(0, COMM_DIR)

from depth_estimator import DepthEstimator
from navigation_engine import NavigationEngine
from voice.speech_output import SpeechOutput

frozen_model = os.path.join(BASE_DIR, "frozen_inference_graph.pb")
config_file  = os.path.join(BASE_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
labels_file  = os.path.join(BASE_DIR, "coco.names")

with open(labels_file, "rt") as f:
    classLabels = f.read().rstrip("\n").split("\n")

# Initialize SSD MobileNet object detector
model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Initialize MiDaS Depth Estimator & Navigation Engine
print("[INFO] Loading MiDaS depth estimator and navigation engine...")
depth_estimator = DepthEstimator()
nav_engine = NavigationEngine(speech_cooldown=3.0)
speaker = SpeechOutput()
print("[INFO] Models initialized. Voice guidance active!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

ret, frame = cap.read()
if not ret:
    print("[ERROR] Cannot read from camera.")
    exit()

frame_h, frame_w = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter("output_video.mp4", fourcc, 20, (frame_w, frame_h))

font = cv2.FONT_HERSHEY_SIMPLEX
speaker.speak("Navigation active. Path monitoring started.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Camera stream ended.")
            break

        # 1. Compute dense depth map using MiDaS
        depth_map, depth_colored = depth_estimator.estimate_depth(frame)

        # 2. Detect objects using SSD MobileNet
        classIndex, confidence, bbox = model.detect(frame, confThreshold=0.48)

        detected_obstacles = []

        if len(classIndex) != 0:
            for classInd, conf, box in zip(classIndex.flatten(), confidence.flatten(), bbox):
                if 0 < classInd <= len(classLabels):
                    label_name = classLabels[classInd - 1]
                else:
                    label_name = "object"

                # Calculate distance using MiDaS
                dist_m, status, status_color = depth_estimator.get_object_distance(depth_map, box)

                detected_obstacles.append({
                    "label": label_name,
                    "dist": dist_m,
                    "box": box,
                    "color": status_color
                })

                # Draw bounding box
                cv2.rectangle(frame, box, status_color, 2)
                tag = f"{label_name.upper()} {dist_m}m"
                text_size = cv2.getTextSize(tag, font, 0.5, 2)[0]
                text_y = max(20, box[1] - 8)
                cv2.rectangle(frame, (box[0], text_y - text_size[1] - 4), (box[0] + text_size[0] + 4, text_y + 4), status_color, -1)
                cv2.putText(frame, tag, (box[0] + 2, text_y), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 3. Update Navigation Engine with hysteresis debouncing & EMA smoothing
        instruction, hud_status, hud_color, is_urgent = nav_engine.update(detected_obstacles, frame_w)

        # 4. Speak actionable instruction when appropriate
        if nav_engine.should_speak(instruction, is_urgent):
            speaker.speak(instruction, priority=is_urgent)

        # 5. Top HUD Navigation Bar
        cv2.rectangle(frame, (0, 0), (frame_w, 42), (20, 20, 20), -1)
        cv2.putText(frame, f"STATUS: {hud_status}", (15, 28), font, 0.7, hud_color, 2, cv2.LINE_AA)

        # Actionable sub-text bar
        cv2.rectangle(frame, (0, 42), (frame_w, 70), (40, 40, 40), -1)
        cv2.putText(frame, f"COMMAND: {instruction}", (15, 62), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Zone meters HUD (Left / Center / Right)
        zones_hud = f"L: {nav_engine.dist_left}m | C: {nav_engine.dist_center}m | R: {nav_engine.dist_right}m"
        cv2.putText(frame, zones_hud, (frame_w - 260, 26), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 6. Depth Thumbnail in bottom right
        pip_w, pip_h = 160, 120
        pip_thumb = cv2.resize(depth_colored, (pip_w, pip_h))
        frame[frame_h - pip_h - 10:frame_h - 10, frame_w - pip_w - 10:frame_w - 10] = pip_thumb
        cv2.rectangle(frame, (frame_w - pip_w - 10, frame_h - pip_h - 10), (frame_w - 10, frame_h - 10), (255, 255, 255), 1)

        # Display and record
        video.write(frame)
        cv2.imshow("Vision Mate - Obstacle & Depth Navigation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Exiting detection loop...")
            break

except Exception as e:
    print("[ERROR] Exception occurred:", e)

finally:
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released successfully.")
