# Vision Mate — Vision & Audio Assistant (SAI)

Vision Mate is an assistive application designed for real-time visual assistance, spatial navigation, and voice communication. It brings together computer vision, monocular depth estimation (MiDaS), face recognition, and natural voice interaction using a single camera.

---

## Key Features

1. **Real-time Obstacle & MiDaS Depth Navigation (Walking Mode)**
   - Detects objects (people, chairs, obstacles) using SSD MobileNet v3.
   - Calculates metric distance using Intel MiDaS v2.1 Small ONNX.
   - Identifies directional zones (ahead, left, right).
   - Categorizes hazards:
     - **CRITICAL (< 1.2m)**: Immediate warning.
     - **NEAR (1.2m - 2.5m)**: Caution / direction advice.
     - **SAFE (> 2.5m)**: Path clear.
   - On-screen Picture-in-Picture depth heatmap HUD.
   - Non-blocking voice interruption: say *"stop walking"* or *"pause"* to pause navigation and ask other questions.

2. **Face Identification & Dynamic Photo Training**
   - Identify known people live through the camera.
   - Add new family, friends, or colleagues by creating a folder with their name in `known_people/`.
   - Ask by voice: *"Who is in front of me?"*, *"Who is that?"*, or *"Who is here?"*.
   - Retrain directly from folders via voice command or the menu.

3. **Natural Voice Interaction & General Questions**
   - Natural phrasing supported: *"start walking"*, *"let's walk"*, *"guide my way"*.
   - Powered by Gemini 3.8 Flash (`gemini-3.8-flash`) with a timeout safeguard and local fallback.
   - Answers everyday queries: *"What time is it?"*, date, weather, device status, and general questions.

4. **Everyday Productivity Tools**
   - **Live Weather:** *"What's the weather in Mumbai?"* or *"How hot is it outside?"*
   - **Calendar & Reminders:** *"Remind me to take medicine at 8 pm"*, *"What are my reminders?"*, *"Clear reminders"*.
   - **Notes & To-Do List:** *"Take a note: buy groceries"*, *"Read my notes"*, *"Clear notes"*.
   - **Calculator & Math:** *"What is 25 times 4?"*, *"Calculate 150 divided by 3"*.
   - **Battery & System Status:** *"What's the battery level?"*, *"Device status"*.

5. **Voice Email Drafting & Sending**
   - Voice commands: *"Send email to Mom about visiting next weekend"*.
   - Contact book (`contacts.json`) maps names to email addresses.
   - Automatic draft generation with subject and body.
   - Spoken draft review and confirmation before sending over Gmail SMTP.

---

## Project Structure

`SAI/` is the main project folder containing all modules, models, and scripts:

```
vision mate/
│
└── SAI/                                 # Main Project Folder
    ├── main.py                          # Unified Interactive CLI Launcher
    ├── requirements.txt                 # Python dependencies
    ├── .env                             # Environment variables & API keys
    ├── .env.example                     # Example environment template
    ├── README.md                        # Documentation & manual
    │
    ├── known_people/                    # Add photos of people here
    │   ├── Raj/                         #   photo1.jpg, photo2.jpg...
    │   └── John/                        #   photo1.jpg, photo2.jpg...
    │
    ├── Communication/                   # Voice & Communication Assistant
    │   ├── main.py                      # Voice assistant entry point
    │   ├── contacts.json                # Contact name-to-email mapping
    │   ├── notes.json                   # Saved notes
    │   ├── reminders.json               # Saved reminders
    │   ├── assistant/controller.py      # Voice command routing & walking mode loop
    │   ├── modules/
    │   │   ├── communication_tool.py    # Email SMTP & draft generation
    │   │   └── daily_tools.py           # Weather, reminders, notes, calculator, status
    │   └── voice/
    │       ├── speech_input.py          # Microphone speech-to-text
    │       └── speech_output.py         # Text-to-speech engine
    │
    ├── obstacle_detection/              # Obstacle Detection & Monocular Depth
    │   ├── detection_code_video.py      # Combined SSD MobileNet + MiDaS pipeline
    │   ├── depth_estimator.py           # Intel MiDaS v2.1 Small ONNX wrapper
    │   ├── navigation_engine.py         # Motion state machine & direction guidance
    │   ├── model-small.onnx             # MiDaS model weights
    │   ├── frozen_inference_graph.pb    # SSD MobileNet weights
    │   ├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
    │   └── coco.names                   # 80 COCO object classes
    │
    └── opencv/Faces/                    # Face Recognition & Dynamic Training
        ├── face_manager.py              # Dynamic trainer & identifier
        ├── face_recognition.py          # Standalone face recognition view
        ├── faces_train.py               # LBPH training script
        ├── haar_face.xml                # Haar Cascade face detector
        └── face_trained.yml             # Trained LBPH model weights
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 to 3.12 installed
- Working webcam and microphone

### 2. Install Dependencies
Open your terminal in the `SAI/` directory:

```powershell
cd "SAI"
pip install -r requirements.txt
```

*(Packages include `opencv-contrib-python`, `numpy`, `SpeechRecognition`, `pyttsx3`, `PyAudio`, `python-dotenv`, and `google-genai`)*

### 3. Configure Credentials (Optional for Email & Questions)
Edit or create `.env` in the `SAI/` folder (copied from `.env.example`):

```ini
# Gmail SMTP Configuration (for sending emails)
GMAIL_MAIL=your_gmail_address@gmail.com
GMAIL_APP_PASSWORD=your_16_character_app_password

# Gemini API Key (for email drafting and questions)
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini Model Selection
GEMINI_MODEL=gemini-3.8-flash
```

> **Note:** For Gmail, generate an App Password from your [Google Account Security Settings](https://myaccount.google.com/apppasswords).

---

## How to Run

Launch the application from the `SAI/` folder:

```powershell
cd "SAI"
python main.py
```

The menu options:

```text
============================================================
                 VISION MATE ASSISTANT
============================================================
1. Start Voice Assistant (Voice, Email, Walking Mode)
2. Start Live Obstacle & Depth Navigation
3. Start Face Recognition Live View
4. Retrain Face Recognizer (from known_people/)
5. Exit
============================================================
```

---

## User Manual & Features

### 1. Adding New People to Identify

#### Step 1: Add Photos
Create a folder with the person's name inside `SAI/known_people/` and place 5 to 10 photos of their face inside:

```text
SAI/known_people/
  ├── Alice/
  │    ├── photo1.jpg
  │    ├── photo2.jpg
  │    └── photo3.jpg
  └── Bob/
       ├── photo1.jpg
       └── photo2.jpg
```

#### Step 2: Retrain the Model
You have two options:
- **Option A (Menu):** Select option 4 (`Retrain Face Recognizer`) in `python main.py`.
- **Option B (Voice):** In voice mode, say:
  > *"Update faces"* or *"Retrain faces"*

The system will scan the folder, train the recognizer, and confirm when complete.

---

### 2. Voice Assistant Commands

Start option 1 from `main.py` (or run `python Communication/main.py`). Speak clearly into your microphone:

| Voice Command | Action |
|---|---|
| **`"Who is in front of me?"`** / **`"Who is that?"`** | Captures a camera frame, checks against known faces, and speaks their name. |
| **`"Start walking mode"`** / **`"Walk"`** | Launches the obstacle navigation feed with distance and direction alerts. |
| **`"Stop walking"`** / **`"Pause"`** | Pauses navigation and returns to normal voice mode so you can ask other questions. |
| **`"Send email to <Name>"`** | Starts the email composer. |
| **`"Weather in <City>"`** | Gives the current temperature and conditions. |
| **`"Remind me to <task> at <time>"`** | Saves a scheduled reminder. |
| **`"Take a note: <text>"`** | Saves a quick memo. |
| **`"Update faces"`** / **`"Retrain"`** | Retrains the face recognizer with newly added photos. |
| **`"Find contact <Name>"`** | Looks up the email address of a person in `contacts.json`. |
| **`"Stop assistant"`** / **`"Exit"`** | Shuts down the assistant. |

---

### 3. Voice Email

1. Add contacts in `SAI/Communication/contacts.json`:
   ```json
   {
     "mom": "mom@example.com",
     "john": "john.doe@company.com"
   }
   ```
2. Say: *"Send email to Mom"* (or *"Write email"*).
3. The assistant will ask: *"What is the purpose of this email?"*.
4. Speak your intent: *"Tell her I will arrive on Saturday around 5 PM and ask if she needs anything from the store"*.
5. A draft subject and body are prepared.
6. The assistant reads the draft and asks for confirmation:
   > *"I drafted an email with subject: Weekend Visit. Would you like me to send it?"*
7. Say: *"Yes"* to send the email immediately over Gmail SMTP.

---

### 4. Live Obstacle & Depth Navigation

Select option 2 from `main.py` (or say *"Start walking"* in Voice Mode):
- Opens the live camera feed with an on-screen guidance bar.
- Bounding boxes show object name, metric distance, and position:
  - `PERSON 0.9m [CENTER]`
  - `CHAIR 1.8m [LEFT]`
- The bottom right corner displays the live MiDaS depth heatmap.
- Clear, actionable movement instructions are provided:
  > *"Please shift left to avoid chair."*
  > *"Caution: obstacle ahead. Slightly steer right."*
  > *"Path clear – continue forward."*
- To pause or exit walking mode at any time, say *"stop walking"* or press `q` on your keyboard.

---

## Troubleshooting

- **Microphone not picking up speech:**
  Ensure your default microphone is selected in Windows Sound Settings.
- **Camera fails to open:**
  Check that no other application (such as Zoom, Teams, or a browser) is using your webcam.
- **Email send failed:**
  Verify that `GMAIL_MAIL` and `GMAIL_APP_PASSWORD` are correct in `.env`, and that an App Password is used instead of your standard account password.
- **Face not recognized:**
  Ensure photos in `known_people/<Name>/` have good lighting and clear frontal faces, then run the retrain command.
