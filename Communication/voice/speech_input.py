import speech_recognition as sr

class SpeechInput:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def listen_command(self) -> str:
        """Capture voice input and return recognized text."""
        with sr.Microphone() as source:
            print(" Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration = 0.3)
            audio = self.recognizer.listen(source)
        
        try:
            command = self.recognizer.recognize_google(audio)
            print(f"You said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return ""
        except sr.RequestError:
            print("Speech recognition service failed.")
            return ""
