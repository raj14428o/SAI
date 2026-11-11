import os
import json
import time
import datetime
import urllib.request
import urllib.parse
import re

class WeatherTool:
    """
    Fetches real-time weather information using free public APIs without API keys.
    """
    @staticmethod
    def get_weather(location=None):
        try:
            if location and location.strip():
                clean_loc = urllib.parse.quote(location.strip())
                url = f"https://wttr.in/{clean_loc}?format=%C,+%t+(humidity:+%h,+wind:+%w)"
            else:
                url = "https://wttr.in/?format=%l:+%C,+%t+(humidity:+%h)"

            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.88.1"})
            with urllib.request.urlopen(req, timeout=5) as response:
                result = response.read().decode("utf-8").strip()
                if location:
                    return f"The current weather in {location.title()} is {result}."
                else:
                    return f"The current local weather is {result}."
        except Exception as e:
            return f"Could not retrieve live weather information right now ({e}). Please check your internet connection."


class ReminderTool:
    """
    Manages calendar reminders stored in local JSON.
    """
    def __init__(self, file_path=None):
        if file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "..", "reminders.json")
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def _load(self):
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, reminders):
        try:
            with open(self.file_path, "w") as f:
                json.dump(reminders, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save reminders: {e}")

    def add_reminder(self, text, time_str=None):
        reminders = self._load()
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
        item = {
            "id": len(reminders) + 1,
            "text": text.strip(),
            "time": time_str.strip() if time_str else "Today",
            "created_at": created_at,
            "completed": False
        }
        reminders.append(item)
        self._save(reminders)
        if time_str:
            return f"Reminder saved: {text} at {time_str}."
        return f"Reminder saved: {text}."

    def list_reminders(self):
        reminders = self._load()
        active = [r for r in reminders if not r.get("completed", False)]
        if not active:
            return "You have no active reminders scheduled."
        
        items = []
        for i, r in enumerate(active, 1):
            t = r.get("time", "")
            items.append(f"{i}. {r['text']}" + (f" (scheduled for {t})" if t else ""))
        
        return "Here are your active reminders: " + "; ".join(items)

    def clear_reminders(self):
        self._save([])
        return "All your reminders have been cleared."


class NotesTool:
    """
    Quick to-do and memo note taking tool.
    """
    def __init__(self, file_path=None):
        if file_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "..", "notes.json")
        self.file_path = file_path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def _load(self):
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _save(self, notes):
        try:
            with open(self.file_path, "w") as f:
                json.dump(notes, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save notes: {e}")

    def add_note(self, text):
        notes = self._load()
        item = {
            "id": len(notes) + 1,
            "text": text.strip(),
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %I:%M %p")
        }
        notes.append(item)
        self._save(notes)
        return f"Note added: {text}."

    def list_notes(self):
        notes = self._load()
        if not notes:
            return "You do not have any saved notes."
        items = [f"{i}. {n['text']}" for i, n in enumerate(notes, 1)]
        return "Here are your saved notes: " + "; ".join(items)

    def clear_notes(self):
        self._save([])
        return "All notes have been cleared."


class CalculatorTool:
    """
    Safe evaluation of basic mathematical expressions and conversions.
    """
    @staticmethod
    def calculate(text):
        try:
            cleaned = text.lower()
            cleaned = cleaned.replace("times", "*").replace("multiplied by", "*")
            cleaned = cleaned.replace("divided by", "/").replace("over", "/")
            cleaned = cleaned.replace("plus", "+").replace("add", "+")
            cleaned = cleaned.replace("minus", "-").replace("subtract", "-")
            cleaned = cleaned.replace("squared", "**2").replace("to the power of", "**")
            
            # Extract numbers and basic math operators
            expr = "".join([c for c in cleaned if c in "0123456789+-*/(). "]).strip()
            if not expr:
                return None
            
            # Disallow dangerous constructs
            if "__" in expr or any(kw in expr for kw in ["import", "eval", "exec", "open", "sys", "os"]):
                return None

            result = eval(expr, {"__builtins__": None}, {})
            if isinstance(result, (int, float)):
                if isinstance(result, float) and result.is_integer():
                    result = int(result)
                elif isinstance(result, float):
                    result = round(result, 2)
                return f"The result is {result}."
        except Exception:
            pass
        return None


class DeviceStatusTool:
    """
    Provides hardware status: battery percentage, power state, and current local time.
    """
    @staticmethod
    def get_status():
        now = datetime.datetime.now()
        status_parts = [f"It is {now.strftime('%I:%M %p on %A, %B %d, %Y')}"]
        
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                pct = int(battery.percent)
                plugged = "plugged in and charging" if battery.power_plugged else "on battery power"
                status_parts.append(f"Battery is at {pct}%, {plugged}")
        except Exception:
            pass
            
        return ". ".join(status_parts) + "."
