import os
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

class CommunicationTool:
    """
    A communication helper class for Ally Vision Assistant.
    Handles sending emails and (later) reading/finding contacts.
    """

    def __init__(self, sender_email=None, app_password=None):
        """Initialize the communication tool with Gmail credentials."""
        self.sender_email = sender_email or os.getenv("GMAIL_MAIL")
        self.app_password = app_password or os.getenv("GMAIL_APP_PASSWORD")

        if not self.sender_email or not self.app_password:
            print("[WARN] Gmail credentials not found. Please set GMAIL_MAIL and GMAIL_APP_PASSWORD in your .env file.")
        else:
            print(f"[INFO] CommunicationTool ready for {self.sender_email}")

    def send_email(self, to: str, subject: str, body: str):
        """
        Send an email using Gmail SMTP.
        """
        if not self.sender_email or not self.app_password:
            return "[ERROR] Cannot send email - credentials missing."

        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = to
            msg['Subject'] = subject

            # Add timestamp
            body_with_time = f"{body}\n\nSent on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(body_with_time, 'plain'))

            # Connect to Gmail SMTP server
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(self.sender_email, self.app_password)
            server.sendmail(self.sender_email, to, msg.as_string())
            server.quit()

            print(f"[INFO] Email sent successfully to {to}")
            return f"[SUCCESS] Email sent successfully to {to}."

        except Exception as e:
            print(f"[ERROR] Error sending email: {e}")
            return f"[ERROR] Error sending email: {e}"

    def find_contact(self, name: str):
        """Search contact by name from contacts.json or environment."""
        clean_name = name.lower().strip()
        contacts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "contacts.json")
        if os.path.exists(contacts_file):
            try:
                import json
                with open(contacts_file, "r") as f:
                    contacts = json.load(f)
                for contact_name, email in contacts.items():
                    if clean_name in contact_name.lower():
                        return email
            except Exception as e:
                print(f"[WARN] Error reading contacts.json: {e}")

        # If it's already an email
        if "@" in name:
            return name.replace(" at ", "@").replace(" dot ", ".").strip()

        return None

    def generate_email_content(self, recipient_name: str, purpose: str):
        """
        Generates email subject and body content using Gemini AI or structured template.
        Supports Gemini 3.8 Flash with automatic fallback.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    f"Write a concise, professional and warm email to {recipient_name} regarding: {purpose}.\n"
                    "Output format strictly as:\n"
                    "SUBJECT: <subject line>\n"
                    "BODY: <email body>"
                )
                
                # Gemini 3.8 Flash model
                model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                if response and response.text:
                    text = response.text
                    if "SUBJECT:" in text and "BODY:" in text:
                        parts = text.split("BODY:")
                        subject = parts[0].replace("SUBJECT:", "").strip()
                        body = parts[1].strip()
                        return subject, body
            except Exception as e:
                print(f"[WARN] Gemini email generation failed, falling back to template: {e}")

        # Fallback template
        subject = f"Update regarding {purpose.capitalize()}"
        body = f"Hi {recipient_name.capitalize()},\n\nI am writing to you regarding {purpose}. Please let me know your thoughts.\n\nBest regards,\nRaj"
        return subject, body

    def read_emails(self):
        """Placeholder: will use Gmail API later."""
        return "( Simulated) Reading emails from inbox..."

    async def manage_communication(self, action: str, **kwargs):
        """
        Dispatches communication requests asynchronously.
        """
        if action == "send_email":
            to = kwargs.get("to")
            subject = kwargs.get("subject", "")
            body = kwargs.get("body", "")
            return self.send_email(to, subject, body)
        elif action == "generate_email":
            name = kwargs.get("name", "there")
            purpose = kwargs.get("purpose", "general update")
            return self.generate_email_content(name, purpose)
        elif action == "read_emails":
            return self.read_emails()
        elif action == "find_contact":
            name = kwargs.get("name", "")
            found = self.find_contact(name)
            if found:
                return f"Found email for {name}: {found}"
            return f"No contact email found for {name}."
        else:
            return f"Unknown communication action: {action}"
