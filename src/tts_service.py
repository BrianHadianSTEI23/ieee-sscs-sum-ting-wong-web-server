# backend/tts_service.py
import asyncio
from typing import Final
import edge_tts
from pathlib import Path

AUDIO_CACHE_DIR = Path("./static/audio")
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class VoiceAlertManager:
    RECOMMENDATIONS: Final[list[str]] = [
        "Drowsiness detected. Please pull over at a safe location, take a 15-minute power nap, or drink cold water immediately.",
        "Warning, severe fatigue detected. Microsleep impairs reaction time as much as alcohol intoxication. Stop driving and rest.",
        "Sleepiness detected. Open your window for fresh air, stretch your body, and take a short break before continuing."
    ]

    def __init__(self, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0
        self._msg_index = 0
        self._lock = asyncio.Lock()

    def get_next_message(self) -> str:
        """Cycles through safety messages sequentially."""
        msg = self.RECOMMENDATIONS[self._msg_index]
        self._msg_index = (self._msg_index + 1) % len(self.RECOMMENDATIONS)
        return msg

    def should_trigger(self, current_time: float) -> bool:
        """Enforces cooldown so the server doesn't spam voice alerts."""
        if current_time - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = current_time
            return True
        return False

    async def generate_speech(self, text: str) -> str:
        """Generates an MP3 file using Microsoft Edge Neural TTS."""
        file_path = AUDIO_CACHE_DIR / "alert_latest.mp3"
        print(f"[TTS DEBUG] Generating speech for text: '{text[:30]}...'")
        
        try:
            async with self._lock:
                communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
                await communicate.save(str(file_path))
            
            print(f"[TTS DEBUG] Successfully saved audio to: {file_path.resolve()}")
            return "/static/audio/alert_latest.mp3"
        except Exception as e:
            print(f"[TTS ERROR] Failed to generate TTS audio: {e}")
            raise e