"""Launcher script for Ninjoucon Mocap Studio Web App."""

import os
from pathlib import Path
import sys
import uvicorn

HERE = Path(__file__).resolve().parent
WORKSPACE_DIR = HERE.parent

# Ensure the app and workspace directories are in sys.path
sys.path.insert(0, str(HERE))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"==================================================")
    print(f"  Ninjoucon Motion Capture Studio Web App")
    print(f"  Open in your browser: http://localhost:{port}")
    print(f"==================================================")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1, ws_max_size=512 * 1024)

