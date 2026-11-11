import os
import sys
import asyncio

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
