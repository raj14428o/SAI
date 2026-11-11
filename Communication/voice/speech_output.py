import os
import sys
import subprocess
import threading
import queue

class SpeechOutput:
    """
    Robust, non-blocking Text-To-Speech engine for Vision Mate.
    Bypasses broken legacy SAPI5 COM drivers by using Windows 11 native
    System.Speech synthesizer via an asynchronous background worker.
    """

    def __init__(self):
        self._speech_queue = queue.Queue(maxsize=3)
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _speech_worker(self):
        while True:
            text = self._speech_queue.get()
            if text is None:
                break
            try:
                # Clean text of quotes and special characters
                clean_text = text.replace('"', '').replace("'", '').replace('`', '').replace('$', '')
                ps_script = (
                    f"Add-Type -AssemblyName System.Speech; "
                    f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$synth.Rate = 1; "
                    f"$synth.Speak('{clean_text}')"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )
            except Exception as e:
                print(f"[WARN] TTS worker error: {e}")
            finally:
                self._speech_queue.task_done()

    def speak(self, text: str, priority: bool = False):
        """
        Speaks text asynchronously without blocking the video stream.
        """
        print(f"[Voice Output]: {text}")
        if priority:
            # Clear old queued speech if an urgent STOP command arrives
            with self._speech_queue.mutex:
                self._speech_queue.queue.clear()
        try:
            self._speech_queue.put_nowait(text)
        except queue.Full:
            pass
