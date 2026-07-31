# test_tts.py
import asyncio
from src.tts_service import VoiceAlertManager

async def test():
    manager = VoiceAlertManager()
    text = manager.RECOMMENDATIONS[0]
    print("[TEST] Starting TTS test...")
    url = await manager.generate_speech(text)
    print(f"[TEST] Done! Audio path returned: {url}")

if __name__ == "__main__":
    asyncio.run(test())