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
            <html>
            <head>
                <title>Microsleep Telemetry Dashboard</title>
                <style>
                    body { font-family: monospace; background: #121212; color: #00dcff; margin: 20px; }
                    .container { display: flex; gap: 20px; }
                    .video-panel { border: 2px solid #333; border-radius: 8px; overflow: hidden; }
                    .telemetry-panel { background: #1e1e1e; padding: 15px; border-radius: 8px; width: 300px; }
                    .stat { margin-bottom: 8px; font-size: 14px; }
                    .val { color: #fff; float: right; }
                    .badge { padding: 2px 6px; border-radius: 4px; font-weight: bold; }
                    .online { background: #00aa44; color: white; }
                    .offline { background: #aa0000; color: white; }
                </style>
            </head>
            <body>
                <h2>Microsleep Detector Dashboard</h2>
                <div class="container">
                    <div class="video-panel">
                        <img src="/video_feed" width="640" height="480" alt="Stream">
                    </div>
                    <div class="telemetry-panel">
                        <h3>Gyro Status</h3>
                        <div class="stat">Status: <span id="connected" class="badge offline">DISCONNECTED</span></div>
                        <div class="stat">Pitch: <span id="pitch" class="val">0.0°</span></div>
                        <div class="stat">Roll: <span id="roll" class="val">0.0°</span></div>
                        <div class="stat">Rate: <span id="rate" class="val">0.0</span></div>
                        <div class="stat">Pitch Rate: <span id="prate" class="val">0.0</span></div>
                        <hr style="border-color: #333;">
                        <div class="stat">Acc Dev: <span id="accdev" class="val">0.0</span></div>
                        <div class="stat">Msg Count: <span id="msg_count" class="val">0</span></div>
                    </div>
                </div>

                <script>
                    // Poll Gyro Data via SSE (Server-Sent Events)
                    const evtSource = new EventSource("/api/gyro/stream");
                    evtSource.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        document.getElementById('pitch').innerText = data.pitch.toFixed(1) + "°";
                        document.getElementById('roll').innerText = data.roll.toFixed(1) + "°";
                        document.getElementById('rate').innerText = data.rate.toFixed(1);
                        document.getElementById('prate').innerText = data.prate.toFixed(1);
                        document.getElementById('accdev').innerText = data.accdev.toFixed(2);
                        document.getElementById('msg_count').innerText = data.msg_count;

                        const connEl = document.getElementById('connected');
                        if(data.connected) {
                            connEl.innerText = "CONNECTED";
                            connEl.className = "badge online";
                        } else {
                            connEl.innerText = "DISCONNECTED";
                            connEl.className = "badge offline";
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
                    await asyncio.sleep(0.1) # 10 Hz refresh rate

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