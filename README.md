# Ninjoucon 2026: Motion Capture Studio & Interactive Copy-Training

An AI-powered motion capture and interactive pose-copying training studio for martial arts, dance, and movement learning.

Built for **DevFest Hackathon 2026**.

The MotionLearn product frontend lives in `final-web-app/`. See the [MotionLearn runbook](deploy/README.md) for the product's local setup and Compute Engine deployment. The quick start below opens the original testing UI.

---

## 🥋 Key Features

1. **Teacher Video Ingestion (YouTube & MP4)**:
   - Upload any video file (`.mp4`, `.webm`, `.mov`) or paste any YouTube URL (downloaded and converted to browser-compatible H.264 via `yt-dlp` and `ffmpeg`).

2. **Automated Keypose & Trajectory Extraction**:
   - Analyzes movement velocity and kinetic energy peaks using Google MediaPipe Pose to segment videos into keyposes and DTW trajectories.

3. **Integrated Video Player**:
   - Key-pose navigation and synchronized playback in Analysis.
   - Built-in expandable OpenCV webcam feedback overlay.

4. **Real-time Pose-Copying & Adaptive Playback**:
   - The teacher video pauses at each keypose and **only plays forward when the student matches and holds the target pose**.
   - Displays real-time joint correction arrows (Euclidean error).
   - Dynamic Time Warping (DTW) movement scoring during transitions.

5. **Clean Video Recording for Motion Capture**:
   - The user's raw webcam feed is recorded cleanly (without overlaid arrows/skeletons) so it is ready for high-precision 3D mocap.
   - Grouped neatly under each teacher video's expandable subdirectory in the sidebar.

6. **3D Motion Capture Pipeline (Y Bot)**:
   - Extracts 33 full-body landmarks with OneEuro filtering, parent-relative limb holding, and joint angle stabilization.
   - Retargets motion onto the **Y Bot** armature in Blender headlessly, outputting animated `.glb`, `.blend`, and `.fbx` files.

7. **Synchronized 3D & Real-Life Playback**:
   - WebGL Three.js 3D viewer featuring side-by-side comparison with the student's real-life video, locked to a shared scrubber.
   - Camera view presets (Perspective, Front, Side, Upper body), speed controls, and one-click downloads.

8. **Test Motion Capture (Misc.)**:
   - Quick one-click standalone recording and testing of the 3D Motion Capture pipeline without requiring a teacher video.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (with virtual environment in `venv`)
- **Blender 4.0+** (accessible in PATH or default install location `C:\Program Files\Blender Foundation\Blender 4.x\blender.exe`)
- **ffmpeg** (accessible in PATH)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r Motion_Capture/requirements.txt
```

### 2. Launch the Web Application
Double-click `run_app.bat` or run:
```bash
python web_app/run_server.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📂 Project Structure

```
ninjoucon2026/
├── Motion_Capture/              # 3D motion capture and Blender retargeting pipeline
│   ├── extract_landmarks.py     # MediaPipe landmark extraction
│   ├── postprocess_landmarks.py # OneEuro filtering, limb stabilization & gap-filling
│   ├── blender_retarget.py      # Headless Blender retargeting to Y Bot
│   └── run_pipeline.py          # Unified pipeline runner (outputs .glb, .blend, .fbx)
├── web_app/                     # FastAPI studio server and Three.js frontend
│   ├── app.py                   # REST & WebSocket backend
│   ├── training_session.py      # Interactive copy-training engine & clean webcam recorder
│   ├── run_server.py            # Uvicorn launcher
│   ├── static/                  # HTML, CSS, Three.js modules, AvatarViewer
│   └── teacher_videos/          # Stored videos and extracted keyposes
├── utilities/                   # MediaPipe vision, gesture recognition & DTW scoring
│   ├── mediapipeline.py         # Real-time MediaPipe pose landmarker
│   ├── video_mediapipeline.py   # Video-stream MediaPipe processing
│   ├── dtw_scoring.py           # Euclidean joint distance & DTW scoring
│   ├── handrecognizer.py        # Gesture detection (Thumb_Up/Down)
│   └── preprocess.py            # Kinetic energy velocity keypose extractor
├── Y Bot.fbx                    # Target humanoid rig for motion capture
├── pose_landmarker_full.task    # Google MediaPipe Pose Landmarker task model
├── run_app.bat                  # One-click Windows startup script
├── requirements.txt             # Project dependencies
└── README.md
```

---

## 📜 DevFest Hackathon 2026 提出情報

DevFest Hackathon 2026 チーム提出用リポジトリです。作品の全コード・モデル・Webアプリケーションスタジオが含まれています。

### ライセンス
[MIT License](LICENSE)
