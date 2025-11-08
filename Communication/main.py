import asyncio
from assistant.controller import AllyController

async def main():
    print("🤖Vision Assistant — Voice Mode Active")
    ally = AllyController()
    while True:
        await ally.process_command()

if __name__ == "__main__":
    asyncio.run(main())
