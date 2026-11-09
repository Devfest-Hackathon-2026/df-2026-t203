FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY final-web-app/package*.json ./
RUN npm ci
COPY final-web-app/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS dependencies
ARG BLENDER_VERSION=4.5.15
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils
COPY deploy/requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/api && /opt/api/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m venv /opt/landmarks && /opt/landmarks/bin/pip install --no-cache-dir \
       mediapipe==0.10.9 numpy==1.23.5 opencv-contrib-python==4.11.0.86 \
    && python -m venv /opt/skellyforge && /opt/skellyforge/bin/pip install --no-cache-dir \
       'skellyforge @ git+https://github.com/freemocap/skellyforge@b96272b1ef06546c1a55d404a5f4a92f1fad1b0f'
RUN mkdir -p /opt/blender /opt/models \
    && archive="blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
    && base=https://download.blender.org/release/Blender4.5 \
    && cd /tmp \
    && curl -fsSLO "$base/$archive" \
    && curl -fsS "$base/blender-${BLENDER_VERSION}.sha256" | awk -v name="$archive" '$2 == name || $2 == "*" name' > blender.sha256 \
    && sha256sum --check blender.sha256 \
    && tar -xJf "$archive" --strip-components=1 -C /opt/blender \
    && curl -fsSL https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task \
       -o /opt/models/gesture_recognizer.task \
    && landmarks_site="$(/opt/landmarks/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
    && curl -fsSL https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite \
       -o "$landmarks_site/mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite"

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 libportaudio2 libsm6 \
    libx11-6 libxcursor1 libxext6 libxfixes3 libxi6 libxkbcommon0 libxrandr2 libxrender1 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 motionlearn
COPY --from=dependencies /opt /opt
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY web_app/ web_app/
COPY utilities/ utilities/
COPY Motion_Capture/ Motion_Capture/
COPY ["pose_landmarker_full.task", "Y Bot.fbx", "Dojo.glb", "./"]
COPY --from=frontend /build/dist /app/dist
ENV PATH=/opt/api/bin:$PATH \
    MEDIAPIPE_PYTHON=/opt/landmarks/bin/python \
    SKELLYFORGE_PYTHON=/opt/skellyforge/bin/python \
    BLENDER_EXE=/opt/blender/blender \
    MOTIONLEARN_GESTURE_MODEL=/opt/models/gesture_recognizer.task \
    MOTIONLEARN_DATA_DIR=/data \
    MOTIONLEARN_DIST=/app/dist \
    PYTHONUNBUFFERED=1
RUN install -d -o motionlearn -g motionlearn /data \
    && /opt/blender/blender --version \
    && /opt/landmarks/bin/python -c 'import mediapipe as mp; assert hasattr(mp, "solutions")' \
    && /opt/skellyforge/bin/python -c 'import skellyforge'
USER motionlearn
RUN /opt/api/bin/python -c 'import sys; sys.path.insert(0, "web_app"); import app'
EXPOSE 8000
CMD ["python", "web_app/run_server.py"]
