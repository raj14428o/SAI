import pyttsx3

class SpeechOutput:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 1.0)

    def speak(self, text: str):
        """Convert text to speech."""
        print(f" Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
