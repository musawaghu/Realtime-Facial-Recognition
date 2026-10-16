# Jobsite check-in server (kiosk + foreman dashboard + sync), CPU build.
#
#   docker build -t jobsite-checkin .
#   docker run -p 8000:8000 -e ADMIN_PASSWORD=change-me -e TZ=America/Chicago -v jobsite-data:/data jobsite-checkin
#
# Optional PPE detector: mount the trained weights:  -v ./models:/app/models:ro
# See docs/DEPLOY.md for HTTPS (required for the camera) and site/central sync.
FROM python:3.11.10-slim-bookworm

# OpenCV runtime libraries; tzdata so TZ works (check-in times must be the site's local time)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (the default Linux wheel bundles ~2 GB of CUDA libraries); the
# requirements' torch==2.2.2 pin is then already satisfied by 2.2.2+cpu.
RUN pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user; bake the FaceNet weights into the image so the first
# check-in doesn't wait for a download.
RUN useradd --create-home app && mkdir -p /data /app/models && chown app /data /app/models
USER app
RUN python -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

COPY --chown=app app/ app/
COPY --chown=app benchmarks/results/lfw.json benchmarks/results/lfw.json

# TZ: set to the jobsite's time zone; attendance times and daily hours use local time
ENV DB_PATH=/data/jobsite.db \
    PYTHONUNBUFFERED=1 \
    TZ=UTC
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/kiosk', timeout=4)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
