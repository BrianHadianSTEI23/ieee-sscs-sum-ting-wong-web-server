# IEEE SSCS Arduino Contest

## Brief Description 
Driving while tired remains a major cause of road accidents, but traditional monitoring systems are often bulky, expensive, and slow to react. To solve this, we built a lightweight, low-cost microsleep detector that combines real-time computer vision with head-movement tracking to catch fatigue before it becomes dangerous. Using an ESP32-S3 microcontroller paired with an OV2640 camera and an MPU6500 IMU sensor, the device continuously analyzes facial cues and sudden head drops. Telemetry and live video are streamed over Wi-Fi to a fast Python and FastAPI backend, which instantly triggers audio alerts via onboard buzzer patterns whenever microsleep is detected.

## Prerequisites
1. Cloudflared (tutorial to [install](https://one.dash.cloudflare.com/))
```
Direct & Immediate Answer
> To install Cloudflare Tunnels (cloudflared) on Windows, log in to the Cloudflare Zero Trust Dashboard, go to Networks > Tunnels, create a new tunnel, and choose Windows as your environment. Download and run the official .msi installer, then open Command Prompt as an administrator and execute the provided cloudflared.exe service install <TUNNEL_TOKEN> command.

Installation Steps
> Create a Tunnel in the Dashboard
> Open the Cloudflare Zero Trust Dashboard and navigate to Networks > Tunnels.
> Click Add a tunnel and select Cloudflare.Type a name for your tunnel and save it.
> Pick Windows under the environment setup options.

Download and Install the Connector
> Download the Windows installer (cloudflared-windows-amd64.msi) provided on the dashboard page.Run the .msi file and step through the setup wizard to complete the installation on your machine.
> Verify the install by opening Command Prompt or PowerShell and typing cloudflared --version.

Run as a Windows Service
> Copy the unique service installation command containing your specific token from the dashboard.
> Open Command Prompt or PowerShell as an Administrator.
> Paste and run the command cloudflared.exe service install <TUNNEL_TOKEN>.
> Check the Cloudflare dashboard to confirm the status changes to Healthy and connected.
```

## How to run 

### Step 1

```bash
pip install fastapi mediapipe numpy opencv-python uvicorn zeroconf
```
or 
```bash
uv sync
```

### Step 2
On terminal 1
```bash
python main.py 
```
or
```bash
uv run main.py 
```

### Step 3
On terminal 2
```bash 
cloudflared tunnel --url http://localhost:8000
```