import speech_recognition as sr
import time

class SpeechInput:
    """
    Robust Speech-To-Text input handler using Google Speech Recognition.
    Configured with generous pause thresholds to allow natural speech cadence
    and avoid cutting off the user mid-sentence.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Generous pause threshold (1.4s) so pauses between words don't cut off speech
        self.recognizer.pause_threshold = 1.4
        # Minimum audio energy to consider for recording
        self.recognizer.energy_threshold = 280
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        self._calibrated = False

    def calibrate(self):
        """Calibrates microphone for ambient noise once on startup."""
        if not self._calibrated:
            try:
                with sr.Microphone() as source:
                    print(" [Microphone]: Calibrating for room noise (1 sec)...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
                    self._calibrated = True
                    print(f" [Microphone]: Ready (energy threshold: {int(self.recognizer.energy_threshold)})")
            except Exception as e:
                print(f"[WARN] Mic calibration error: {e}")

    def listen_command(self, timeout: int = 6, phrase_limit: int = 12) -> str:
        """
        Capture voice input with generous timeout and phrase limit.
        Returns recognized lowercase text, or empty string on silence/error.
        """
        if not self._calibrated:
            self.calibrate()

        try:
            with sr.Microphone() as source:
                print(" [Microphone]: Listening...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            return ""
        except Exception as e:
            print(f"[WARN] Microphone input error: {e}")
            return ""

        try:
            command = self.recognizer.recognize_google(audio)
            print(f"[Heard]: \"{command}\"")
            return command.lower().strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            print("[WARN] Google Speech Recognition service unreachable (network check needed).")
            return ""
