import os
import sys
import asyncio

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

COMM_DIR = os.path.dirname(os.path.abspath(__file__))
if COMM_DIR not in sys.path:
    sys.path.insert(0, COMM_DIR)

from assistant.controller import AllyController

async def main():
    print("[INFO] Vision Assistant — Voice Mode Active")
    ally = AllyController()
    while True:
        await ally.process_command()

if __name__ == "__main__":
    asyncio.run(main())
