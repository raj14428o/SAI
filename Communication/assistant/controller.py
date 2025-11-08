import asyncio
from modules.communication_tool import CommunicationTool
from voice.speech_output import SpeechOutput
from voice.speech_input import SpeechInput

class AllyController:
    def __init__(self):
        self.voice_in = SpeechInput()
        self.voice_out = SpeechOutput()
        self.comm_tool = CommunicationTool()

    async def process_command(self):
        """Main loop for listening and acting."""
        command = self.voice_in.listen_command()

        if not command:
            self.voice_out.speak("Sorry, I didn't catch that.")
            return

        if "send email" in command:
            self.voice_out.speak("Whom should I send the email to?")
            to = input("Enter recipient email: ")  # or ask again via voice

            self.voice_out.speak("What is the subject?")
            subject = input("Enter subject: ")

            self.voice_out.speak("What should I say?")
            body = input("Enter message: ")

            response = await self.comm_tool.manage_communication(
                action="send_email",
                to=to, subject=subject, body=body
            )
            self.voice_out.speak(response)

        elif "read email" in command:
            self.voice_out.speak("Reading your latest emails...")
            response = await self.comm_tool.manage_communication(action="read_emails")
            print(response)
            self.voice_out.speak("Here are your latest emails.")
            # Optionally, read only subject/senders aloud

        elif "find contact" in command:
            self.voice_out.speak("Who do you want to search?")
            name = input("Enter contact name: ")

            response = await self.comm_tool.manage_communication(action="find_contact", name=name)
            print(response)
            self.voice_out.speak(response.split("\n")[0])  # Read only first line

        else:
            self.voice_out.speak("Sorry, I don’t recognize that command.")
