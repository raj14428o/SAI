"""
Vision Mate — Unified AI Vision Assistant
"""
import os
import sys

# Automatically load environment variables from SAI/.env
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMM_DIR = os.path.join(BASE_DIR, "Communication")
OBSTACLE_DIR = os.path.join(BASE_DIR, "obstacle_detection")
FACES_DIR = os.path.join(BASE_DIR, "opencv", "Faces")

def print_banner():
    print("=" * 60)
    print("                 VISION MATE ASSISTANT")
    print("=" * 60)
    print("1. Start Voice Assistant (Voice, Email, Walking Mode)")
    print("2. Start Live Obstacle & Depth Navigation")
    print("3. Start Face Recognition Live View")
    print("4. Retrain Face Recognizer (from known_people/)")
    print("5. Exit")
    print("=" * 60)

def main():
    while True:
        print_banner()
        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            print("\nStarting Voice Assistant...")
            os.system(f'python "{os.path.join(COMM_DIR, "main.py")}"')
        elif choice == "2":
            print("\nStarting Obstacle & MiDaS Depth Navigation (Press 'q' in window to exit)...")
            os.system(f'python "{os.path.join(OBSTACLE_DIR, "detection_code_video.py")}"')
        elif choice == "3":
            print("\nStarting Face Recognition (Press 'q' in window to exit)...")
            os.system(f'python "{os.path.join(FACES_DIR, "face_recognition.py")}"')
        elif choice == "4":
            print("\nRetraining Face Recognizer from known_people/...")
            sys.path.append(FACES_DIR)
            from face_manager import FaceManager
            fm = FaceManager()
            msg = fm.retrain()
            print(msg)
            input("\nPress Enter to return to menu...")
        elif choice == "5" or choice.lower() in ("q", "exit"):
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please choose 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main()
