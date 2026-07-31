# server.py

import os
import cv2
import json
import time
import asyncio
import threading
import numpy as np
from typing import AsyncGenerator, Optional
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from .tts_service import VoiceAlertManager

from dotenv import load_dotenv

class DriverResponseRequest(BaseModel):
    user_speech: str

class Server:
    def __init__(self, host: str = "0.0.0.0", port: int = 65500):

        load_dotenv()

        self.host = host
        self.port = port
        
        # Thread-safe shared state buffers
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        
        self._gyro_lock = threading.Lock()
        self._latest_gyro: dict = {
            "connected": False,
            "age": None,
            "worst_gap": 0,
            "msg_count": 0,
            "pitch": 0.0, "roll": 0.0, "rate": 0.0, "accdev": 0.0,
            "gx": 0.0, "gy": 0.0, "gz": 0.0, "prate": 0.0,
            "is_drowsy": False  # Track user state
        }

        self.alert_manager = VoiceAlertManager(api_key=os.getenv("GEMINI_API_KEY"), cooldown_seconds=45.0)

        # Initialize FastAPI App
        self.app = FastAPI(title="Microsleep Detector Dashboard")
        
        # Mount static directory to serve generated MP3s
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        
        self._setup_routes()

    def update_frame(self, frame: np.ndarray) -> None:
        """Thread-safe method called by SleepDetectorEngine to push processed frames."""
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def update_gyro(self, data: dict) -> None:
        """Thread-safe method called by GyroClient to push telemetry data."""
        with self._gyro_lock:
            # Preserve is_drowsy state across telemetry updates
            drowsy_state = self._latest_gyro.get("is_drowsy", False)
            self._latest_gyro = data.copy()
            self._latest_gyro["is_drowsy"] = drowsy_state

    def set_drowsy_state(self, is_drowsy: bool) -> None:
        """Called by SleepDetectorEngine to set current user alertness state."""
        with self._gyro_lock:
            self._latest_gyro["is_drowsy"] = is_drowsy

    def _generate_mjpeg_stream(self):
        """Generator function that yields JPEG frames for MJPEG streaming."""
        while True:
            with self._frame_lock:
                if self._latest_frame is None:
                    frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame_to_send = self._latest_frame

            success, buffer = cv2.imencode('.jpg', frame_to_send)
            if not success:
                time.sleep(0.03)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # ~30 FPS throttle

    def _setup_routes(self):
        """Define FastAPI endpoints."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Returns embedded HTML dashboard with mobile browser Web Audio API handling."""
            return """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Microsleep Telemetry Dashboard</title>
                <style>
                    :root {
                        --bg-color: #0f1115;
                        --panel-bg: #161920;
                        --panel-border: #262b36;
                        --text-main: #e2e8f0;
                        --text-muted: #8a94a6;
                        --accent-blue: #38bdf8;
                        --status-online: #10b981;
                        --status-offline: #ef4444;
                        --status-alert: #f59e0b;
                    }

                    * { box-sizing: border-box; margin: 0; padding: 0; }

                    body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        background-color: var(--bg-color);
                        color: var(--text-main);
                        padding: 40px 20px;
                        line-height: 1.5;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        min-height: 100vh;
                    }

                    .dashboard-wrapper {
                        width: 100%;
                        max-width: 1000px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }

                    header {
                        width: 100%;
                        text-align: center;
                        margin-bottom: 28px;
                        border-bottom: 1px solid var(--panel-border);
                        padding-bottom: 20px;
                    }

                    h1 {
                        font-size: 1.4rem;
                        font-weight: 600;
                        letter-spacing: -0.02em;
                        color: var(--text-main);
                    }

                    .subtitle {
                        font-size: 0.85rem;
                        color: var(--text-muted);
                        margin-top: 4px;
                    }

                    .dashboard-grid {
                        display: flex;
                        justify-content: center;
                        align-items: stretch;
                        gap: 20px;
                        width: 100%;
                        flex-wrap: wrap;
                    }

                    .card {
                        background: var(--panel-bg);
                        border: 1px solid var(--panel-border);
                        border-radius: 8px;
                        overflow: hidden;
                    }

                    .video-card { flex: 0 0 auto; }

                    .video-card img {
                        display: block;
                        width: 640px;
                        height: 480px;
                        background-color: #000;
                    }

                    .telemetry-card {
                        width: 320px;
                        padding: 20px;
                        display: flex;
                        flex-direction: column;
                    }

                    .card-header {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 16px;
                        padding-bottom: 12px;
                        border-bottom: 1px solid var(--panel-border);
                    }

                    .card-title {
                        font-size: 0.85rem;
                        font-weight: 600;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        color: var(--text-muted);
                    }

                    .status-badge {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        font-size: 0.75rem;
                        font-weight: 600;
                        padding: 4px 10px;
                        border-radius: 20px;
                        letter-spacing: 0.03em;
                    }

                    .status-badge .indicator {
                        width: 6px;
                        height: 6px;
                        border-radius: 50%;
                    }

                    .status-badge.online {
                        background: rgba(16, 185, 129, 0.1);
                        color: var(--status-online);
                        border: 1px solid rgba(16, 185, 129, 0.2);
                    }

                    .status-badge.online .indicator {
                        background-color: var(--status-online);
                        box-shadow: 0 0 8px var(--status-online);
                    }

                    .status-badge.offline {
                        background: rgba(239, 68, 68, 0.1);
                        color: var(--status-offline);
                        border: 1px solid rgba(239, 68, 68, 0.2);
                    }

                    .status-badge.offline .indicator {
                        background-color: var(--status-offline);
                    }

                    .btn-audio {
                        width: 100%;
                        padding: 10px;
                        margin-bottom: 16px;
                        background: var(--panel-border);
                        color: var(--text-main);
                        border: 1px solid #3b4252;
                        border-radius: 6px;
                        font-size: 0.85rem;
                        font-weight: 600;
                        cursor: pointer;
                        transition: all 0.2s ease;
                    }

                    .btn-audio.active {
                        background: rgba(56, 189, 248, 0.15);
                        color: var(--accent-blue);
                        border-color: var(--accent-blue);
                    }

                    .alert-banner {
                        display: none;
                        background: rgba(245, 158, 11, 0.15);
                        border: 1px solid var(--status-alert);
                        color: var(--status-alert);
                        padding: 12px;
                        border-radius: 6px;
                        font-size: 0.8rem;
                        margin-bottom: 16px;
                        line-height: 1.4;
                    }

                    .alert-banner.visible { display: block; }

                    .metrics-group {
                        display: flex;
                        flex-direction: column;
                        gap: 12px;
                    }

                    .metric-row {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        font-size: 0.9rem;
                    }

                    .metric-label { color: var(--text-muted); }

                    .metric-value {
                        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                        font-weight: 500;
                        color: var(--text-main);
                    }

                    .divider {
                        height: 1px;
                        background-color: var(--panel-border);
                        margin: 4px 0;
                    }
                </style>
            </head>
            <body>
                <div class="dashboard-wrapper">
                    <header>
                        <h1>Microsleep Telemetry Dashboard</h1>
                        <div class="subtitle">Real-Time Vision & Inertial Sensor Monitoring</div>
                    </header>

                    <main class="dashboard-grid">
                        <div class="card video-card">
                            <img src="/video_feed" alt="Live Camera Feed">
                        </div>

                        <div class="card telemetry-card">
                            <div class="card-header">
                                <span class="card-title">IMU Telemetry</span>
                                <div id="status-badge" class="status-badge offline">
                                    <span class="indicator"></span>
                                    <span id="status-text">DISCONNECTED</span>
                                </div>
                            </div>

                            <button id="audio-toggle-btn" class="btn-audio" onclick="unlockMobileAudio()">
                                🔊 Enable Voice Alerts
                            </button>

                            <div id="voice-alert-banner" class="alert-banner">
                                <strong>⚠️ Voice Guidance Active:</strong>
                                <div id="voice-alert-text"></div>
                            </div>

                            <div class="metrics-group">
                                <div class="metric-row">
                                    <span class="metric-label">Pitch Angle</span>
                                    <span id="pitch" class="metric-value">0.0°</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">Roll Angle</span>
                                    <span id="roll" class="metric-value">0.0°</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">Total Angular Rate</span>
                                    <span id="rate" class="metric-value">0.0</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">Pitch Rate</span>
                                    <span id="prate" class="metric-value">0.0</span>
                                </div>

                                <div class="divider"></div>

                                <div class="metric-row">
                                    <span class="metric-label">Accel Deviation</span>
                                    <span id="accdev" class="metric-value">0.00</span>
                                </div>
                                <div class="metric-row">
                                    <span class="metric-label">Packets Received</span>
                                    <span id="msg_count" class="metric-value">0</span>
                                </div>
                            </div>
                        </div>
                    </main>
                </div>

                <script>
                    const globalAudio = new Audio();
                    let audioUnlocked = false;

                    function unlockMobileAudio() {
                        globalAudio.src = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=";
                        globalAudio.play().then(() => {
                            audioUnlocked = true;
                            const btn = document.getElementById('audio-toggle-btn');
                            btn.innerText = "✓ Voice Alerts Enabled";
                            btn.classList.add("active");
                        }).catch(err => {
                            console.error("Audio unlock blocked by browser:", err);
                        });
                    }

                    const evtSource = new EventSource("/api/gyro/stream");

                    evtSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);

                        document.getElementById('pitch').innerText = (data.pitch ?? 0).toFixed(1) + "°";
                        document.getElementById('roll').innerText = (data.roll ?? 0).toFixed(1) + "°";
                        document.getElementById('rate').innerText = (data.rate ?? 0).toFixed(1);
                        document.getElementById('prate').innerText = (data.prate ?? 0).toFixed(1);
                        document.getElementById('accdev').innerText = (data.accdev ?? 0).toFixed(2);
                        document.getElementById('msg_count').innerText = data.msg_count ?? 0;

                        const badgeEl = document.getElementById('status-badge');
                        const textEl = document.getElementById('status-text');

                        if (data.connected) {
                            badgeEl.className = "status-badge online";
                            textEl.innerText = "ONLINE";
                        } else {
                            badgeEl.className = "status-badge offline";
                            textEl.innerText = "DISCONNECTED";
                        }

                        // Play audio and show banner when alert is triggered
                        if (data.drowsy_alert && data.audio_url) {
                            const banner = document.getElementById('voice-alert-banner');
                            const bannerText = document.getElementById('voice-alert-text');
                            
                            bannerText.innerText = data.msg;
                            banner.classList.add("visible");

                            if (audioUnlocked) {
                                globalAudio.pause();
                                globalAudio.src = data.audio_url;
                                globalAudio.load();
                                globalAudio.play().catch(e => console.warn("Audio playback error:", e));
                            } else {
                                console.warn("Audio alert received but browser audio is locked. Click 'Enable Voice Alerts'.");
                            }

                            setTimeout(() => {
                                banner.classList.remove("visible");
                            }, 12000);
                        }
                    };

                    let isListening = false;

                    function listenToDriverOnce() {
                        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                        if (!SpeechRecognition) {
                            console.warn("Speech recognition not supported on this browser.");
                            return;
                        }

                        if (isListening) return;
                        
                        const recognition = new SpeechRecognition();
                        recognition.lang = 'en-US';
                        recognition.interimResults = false;
                        recognition.maxAlternatives = 1;

                        // Visual feedback indicator
                        const bannerText = document.getElementById('voice-alert-text');
                        bannerText.innerText = "🎙️ Listening... Speak now (e.g., 'Pulling over soon')";

                        recognition.onstart = () => { isListening = true; };

                        recognition.onresult = async (event) => {
                            const userSpeech = event.results[0][0].transcript;
                            console.log("Captured driver speech:", userSpeech);
                            bannerText.innerText = `You said: "${userSpeech}" (Analyzing...)`;

                            // Send transcript to backend
                            try:
                                const res = await fetch('/api/driver/response', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ user_speech: userSpeech })
                                });
                                const data = await res.json();

                                if (data.status === 'success' && data.audio_url) {
                                    bannerText.innerText = data.reply_text;
                                    globalAudio.src = data.audio_url;
                                    globalAudio.play();
                                }
                            } catch (err) {
                                console.error("Error sending response:", err);
                            }
                        };

                        recognition.onerror = (err) => {
                            console.warn("Speech recognition error or timeout:", err.error);
                        };

                        recognition.onend = () => { isListening = false; };

                        // Start 5-second listening window
                        recognition.start();
                    }

                    // Attach listener to audio element completion
                    globalAudio.onended = () => {
                        // Only trigger voice recognition if this audio was a drowsy warning
                        if (window.lastAlertTriggered) {
                            window.lastAlertTriggered = false; // Reset single-shot flag
                            listenToDriverOnce();
                        }
                    };
                </script>
            </body>
            </html>
            """

        @self.app.get("/video_feed")
        async def video_feed():
            return StreamingResponse(
                self._generate_mjpeg_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.get("/api/gyro/stream")
        async def stream_gyro_data() -> StreamingResponse:
            """Server-Sent Event (SSE) endpoint pushing telemetry and AI audio notifications."""
            async def event_generator() -> AsyncGenerator[str, None]:
                while True:
                    now = time.time()
                    with self._gyro_lock:
                        payload_data = self._latest_gyro.copy()
                    
                    is_drowsy = payload_data.get("is_drowsy", False)
                    pitch_angle = abs(payload_data.get("pitch", 0.0))
                    
                    # TRIGGER CONDITION: Drowsiness detected OR Pitch angle > 25.0°
                    should_alert_trigger = is_drowsy or (pitch_angle > 25.0)

                    if should_alert_trigger and self.alert_manager.should_trigger(now):
                        try:
                            # 1. Generate recommendation text using Gemini AI
                            msg_text = await self.alert_manager.generate_recommendation_ai()
                            
                            # 2. Convert text to speech MP3
                            audio_url = await self.alert_manager.generate_speech(msg_text)
                            
                            payload_data["drowsy_alert"] = True
                            payload_data["audio_url"] = f"{audio_url}?t={int(now)}"
                            payload_data["msg"] = msg_text
                            print(f"[SERVER] Triggered Voice Alert: '{msg_text[:35]}...'")
                        except Exception as e:
                            print(f"[SERVER ERROR] Failed to process voice alert: {e}")
                            payload_data["drowsy_alert"] = False
                    else:
                        payload_data["drowsy_alert"] = False

                    yield f"data: {json.dumps(payload_data)}\n\n"
                    await asyncio.sleep(0.1) # 10 Hz refresh rate

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        @self.app.post("/api/driver/response")
        async def handle_driver_response(payload: DriverResponseRequest):
            """Handles driver voice response captured by the frontend after an alert."""
            if not payload.user_speech.strip():
                return {"status": "ignored", "reason": "Empty input"}

            result = await self.alert_manager.analyze_driver_response(payload.user_speech)
            return {
                "status": "success",
                "reply_text": result["reply_text"],
                "audio_url": f"{result['audio_url']}?t={int(time.time())}"
            }

    def run_in_thread(self):
        thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": self.app,
                "host": self.host,
                "port": self.port,
                "log_level": "error"
            },
            daemon=True
        )
        thread.start()
        print(f"[*] Dashboard Web Server running at http://{self.host}:{self.port}")