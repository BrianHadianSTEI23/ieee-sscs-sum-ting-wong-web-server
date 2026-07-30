import cv2
import json
import time
import asyncio
import threading
import numpy as np
from typing import AsyncGenerator, Optional
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn


class Server:
    def __init__(self, host: str = "0.0.0.0", port: int = 65500):
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
            "gx": 0.0, "gy": 0.0, "gz": 0.0, "prate": 0.0
        }

        # Initialize FastAPI App
        self.app = FastAPI(title="Microsleep Detector Dashboard")
        self._setup_routes()

    def update_frame(self, frame: np.ndarray) -> None:
        """Thread-safe method called by SleepDetectorEngine to push processed frames."""
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def update_gyro(self, data: dict) -> None:
        """Thread-safe method called by GyroClient to push telemetry data."""
        with self._gyro_lock:
            self._latest_gyro = data.copy()

    def _generate_mjpeg_stream(self):
        """Generator function that yields JPEG frames for MJPEG streaming."""
        while True:
            with self._frame_lock:
                if self._latest_frame is None:
                    # Fallback black frame if engine is initializing
                    frame_to_send = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame_to_send = self._latest_frame

            # Encode frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame_to_send)
            if not success:
                time.sleep(0.03)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.03)  # Approx ~30 FPS throttle

    def _setup_routes(self):
        """Define FastAPI endpoints."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            """Returns simple embedded HTML dashboard."""
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
                    }

                    * {
                        box-sizing: border-box;
                        margin: 0;
                        padding: 0;
                    }

                    body {
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        background-color: var(--bg-color);
                        color: var(--text-main);
                        padding: 40px 20px;
                        line-height: 1.5;
                        display: flex;
                        flex-direction: column;
                        align-items: center; /* Centers the main layout container */
                        min-height: 100vh;
                    }

                    .dashboard-wrapper {
                        width: 100%;
                        max-width: 1000px; /* Constrains total layout width for centered look */
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }

                    header {
                        width: 100%;
                        text-align: center; /* Centered title and subtitle */
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
                        justify-content: center; /* Centers video & gyro cards side-by-side */
                        align-items: stretch;
                        gap: 20px;
                        width: 100%;
                        flex-wrap: wrap; /* Stacks gracefully on narrower screens */
                    }

                    .card {
                        background: var(--panel-bg);
                        border: 1px solid var(--panel-border);
                        border-radius: 8px;
                        overflow: hidden;
                    }

                    .video-card {
                        flex: 0 0 auto;
                    }

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
                        margin-bottom: 20px;
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

                    .metrics-group {
                        display: flex;
                        flex-direction: column;
                        gap: 14px;
                    }

                    .metric-row {
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        font-size: 0.9rem;
                    }

                    .metric-label {
                        color: var(--text-muted);
                    }

                    .metric-value {
                        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                        font-weight: 500;
                        color: var(--text-main);
                    }

                    .divider {
                        height: 1px;
                        background-color: var(--panel-border);
                        margin: 6px 0;
                    }
                </style>
            </head>
            <body>
                <div class="dashboard-wrapper">
                    <!-- Header Title (Centered) -->
                    <header>
                        <h1>Microsleep Telemetry Dashboard</h1>
                        <div class="subtitle">Real-Time Vision & Inertial Sensor Monitoring</div>
                    </header>

                    <!-- Centered Main Content -->
                    <main class="dashboard-grid">
                        <!-- Video Stream Feed -->
                        <div class="card video-card">
                            <img src="/video_feed" alt="Live Camera Feed">
                        </div>

                        <!-- Telemetry Panel -->
                        <div class="card telemetry-card">
                            <div class="card-header">
                                <span class="card-title">IMU Telemetry</span>
                                <div id="status-badge" class="status-badge offline">
                                    <span class="indicator"></span>
                                    <span id="status-text">DISCONNECTED</span>
                                </div>
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
                    const evtSource = new EventSource("/api/gyro/stream");

                    evtSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);

                        // Safe value updates
                        document.getElementById('pitch').innerText = (data.pitch ?? 0).toFixed(1) + "°";
                        document.getElementById('roll').innerText = (data.roll ?? 0).toFixed(1) + "°";
                        document.getElementById('rate').innerText = (data.rate ?? 0).toFixed(1);
                        document.getElementById('prate').innerText = (data.prate ?? 0).toFixed(1);
                        document.getElementById('accdev').innerText = (data.accdev ?? 0).toFixed(2);
                        document.getElementById('msg_count').innerText = data.msg_count ?? 0;

                        // Update Status Badge UI
                        const badgeEl = document.getElementById('status-badge');
                        const textEl = document.getElementById('status-text');

                        if (data.connected) {
                            badgeEl.className = "status-badge online";
                            textEl.innerText = "ONLINE";
                        } else {
                            badgeEl.className = "status-badge offline";
                            textEl.innerText = "DISCONNECTED";
                        }
                    };
                </script>
            </body>
            </html>
            """

        @self.app.get("/video_feed")
        async def video_feed():
            """MJPEG video stream endpoint."""
            return StreamingResponse(
                self._generate_mjpeg_stream(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )

        @self.app.get("/api/gyro/stream")
        async def stream_gyro_data() -> StreamingResponse:
            """Server-Sent Event (SSE) endpoint pushing live gyro telemetry."""
            async def event_generator() -> AsyncGenerator[str, None]:
                while True:
                    with self._gyro_lock:
                        payload = json.dumps(self._latest_gyro)
                    yield f"data: {payload}\n\n"
                    await asyncio.sleep(0.01) # 10 Hz refresh rate

            return StreamingResponse(event_generator(), media_type="text/event-stream")

    def run_in_thread(self):
        """Run FastAPI server non-blocking inside a daemon thread."""
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


# Example Integration
if __name__ == "__main__":
    server = Server()
    server.run_in_thread()

    # Keep main thread alive for testing simulation
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass