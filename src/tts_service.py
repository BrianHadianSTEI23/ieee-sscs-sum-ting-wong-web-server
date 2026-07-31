# backend/tts_service.py
import asyncio
import os
from typing import Optional
from pathlib import Path
from google import genai
from google.genai import types
import edge_tts

AUDIO_CACHE_DIR = Path("./static/audio")
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class VoiceAlertManager:
    def __init__(self, api_key: Optional[str] = None, cooldown_seconds: float = 10.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0.0
        self._lock = asyncio.Lock()

        # Initialize Google GenAI client (uses provided api_key or GEMINI_API_KEY environment variable)
        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_api_key:
            print("[TTS WARNING] No Gemini API key provided. Falling back to default static alerts.")
            self.client = None
        else:
            self.client = genai.Client(api_key=resolved_api_key)

        # Static fallback alerts if API call fails or API key is missing
        self._fallback_messages = [
            "Drowsiness detected. Please pull over at a safe location, take a 15-minute power nap, or drink cold water immediately.",
            "Warning, severe fatigue detected. Microsleep impairs reaction time as much as alcohol intoxication. Stop driving and rest.",
            "Sleepiness detected. Open your window for fresh air, stretch your body, and take a short break before continuing."
        ]
        self._msg_index = 0

    def should_trigger(self, current_time: float) -> bool:
        """Enforces cooldown so the server doesn't spam voice alerts."""
        if current_time - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = current_time
            return True
        return False

    async def generate_recommendation_ai(self) -> str:
        """Generates an urgent 1-paragraph (3-4 sentences) warning & recommendation using Gemini."""
        prompt = (
            "You are an urgent driver-safety alert system. "
            "Write exactly one concise paragraph (3 to 4 sentences) warning the driver about microsleep. "
            "Explain the immediate dangers or impacts of microsleeping, and give clear, "
            "actionable recommendations to quickly reduce or counteract microsleep symptoms. "
            "Keep the tone urgent, direct, and alerting."
        )

        if not self.client:
            return self._get_fallback_message()

        try:
            # Run the synchronous GenAI SDK call inside an async thread executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=200,
                    )
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"[GEMINI ERROR] Failed to generate AI prompt: {e}")
            return self._get_fallback_message()

    def _get_fallback_message(self) -> str:
        """Sequential fallback mechanism if API call fails."""
        msg = self._fallback_messages[self._msg_index]
        self._msg_index = (self._msg_index + 1) % len(self._fallback_messages)
        return msg

    async def generate_speech(self, text: str) -> str:
        """Generates an MP3 file using Microsoft Edge Neural TTS."""
        file_path = AUDIO_CACHE_DIR / "alert_latest.mp3"
        print(f"[TTS DEBUG] Generating speech for text: '{text[:40]}...'")

        try:
            async with self._lock:
                communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
                await communicate.save(str(file_path))

            print(f"[TTS DEBUG] Successfully saved audio to: {file_path.resolve()}")
            return "/static/audio/alert_latest.mp3"
        except Exception as e:
            print(f"[TTS ERROR] Failed to generate TTS audio: {e}")
            raise e

    async def analyze_driver_response(self, user_speech: str) -> dict:
        """Analyzes driver feedback after a microsleep alert and generates an appropriate response."""
        prompt = (
            "You are an active in-car AI safety assistant. A driver was just alerted for microsleep.\n"
            f"The driver responded: '{user_speech}'\n\n"
            "Task:\n"
            "1. Assess if the driver is taking appropriate action (e.g., pulling over, taking a break) "
            "or making a dangerous choice (e.g., claiming they are fine despite drowsy telemetry).\n"
            "2. Respond directly to the driver in 1 to 2 short, calm, and firm sentences.\n"
            "3. If they say they are pulling over, confirm and support them. "
            "If they deny being tired, firmly emphasize that microsleep is involuntary."
        )

        if not self.client:
            reply_text = "Understood. Please prioritize your safety and pull over if fatigue continues."
        else:
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.4,
                            max_output_tokens=100,
                        )
                    )
                )
                reply_text = response.text.strip()
            except Exception as e:
                print(f"[GEMINI ERROR] Driver response analysis failed: {e}")
                reply_text = "Please pull over safely when you can."

        # Generate audio response for the follow-up
        audio_url = await self.generate_speech(reply_text)
        return {"reply_text": reply_text, "audio_url": audio_url}