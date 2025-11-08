import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

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
            print("⚠️ Gmail credentials not found. Please set GMAIL_MAIL and GMAIL_APP_PASSWORD in your .env file.")
        else:
            print(f" CommunicationTool ready for {self.sender_email}")

    def send_email(self, to: str, subject: str, body: str):
        """
        Send an email using Gmail SMTP.
        """
        if not self.sender_email or not self.app_password:
            return " Cannot send email — credentials missing."

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

            print(f" Email sent successfully to {to}")
            return f" Email sent successfully to {to}."

        except Exception as e:
            print(f" Error sending email: {e}")
            return f" Error sending email: {e}"

    def find_contact(self, name: str):
        """Placeholder: will use Google People API later."""
        return f"( Simulated) Searching for contact: {name}"

    def read_emails(self):
        """Placeholder: will use Gmail API later."""
        return "( Simulated) Reading emails from inbox..."
