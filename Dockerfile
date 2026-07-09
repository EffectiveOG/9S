# Runtime image for the Jarvis web dashboard / API.
#
# This installs only the base (web) dependencies from pyproject.toml. The
# vision/audio stacks (torch, opencv, mediapipe, pyaudio) need host hardware
# (camera/microphone/GPU) and are intentionally NOT installed here; those
# components simply fail to start in a headless container, which the core
# handles gracefully.

FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JARVIS_HOST=0.0.0.0 \
    JARVIS_PORT=8000

# Install base dependencies first (better layer caching).
COPY pyproject.toml README.md ./
COPY jarvis ./jarvis
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Application config (device/scene definitions).
COPY config ./config

EXPOSE 8000

# Set JARVIS_ADMIN_PASSWORD / JARVIS_SECRET_KEY at runtime (docker run -e ...).
CMD ["python", "-m", "jarvis"]
