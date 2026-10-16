"""All settings in one place."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Storage
DB_PATH = os.getenv("DB_PATH", str(ROOT / "jobsite.db"))

# Foreman dashboard login (HTTP Basic, username "foreman")
ADMIN_USER = "foreman"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Detection
CONFIDENCE_THRESHOLD = 0.9   # MTCNN face probability needed to count as a face
MIN_FACE_SIZE = 40           # pixels; smaller faces are ignored
FACE_MARGIN = 14             # pixels of context kept around each face crop


def _benchmark_threshold(key: str, default: float) -> float:
    """Use a threshold measured on LFW (benchmarks/lfw_eval.py) when available."""
    path = ROOT / "benchmarks" / "results" / "lfw.json"
    try:
        return float(json.loads(path.read_text())[key])
    except (OSError, KeyError, ValueError):
        return default


# Matching: cosine similarity between embeddings (higher = more alike)
MATCH_THRESHOLD = _benchmark_threshold("threshold_far_0.1", default=0.5)

# Liveness (head-turn challenge). Yaw = nose offset from the eye midpoint, in eye-widths.
# On 1,500 LFW faces |yaw| has median 0.10 and 95th percentile 0.29.
LIVENESS_FRONTAL_YAW = 0.25  # must face the camera this squarely before a challenge starts
LIVENESS_TURN_YAW = 0.35     # turn needed in the requested direction to pass
LIVENESS_TIMEOUT_S = 6
# A turned face matches less well than a frontal one, so identity during the turn uses the
# looser LFW threshold at 1% false accepts (the frontal frame already passed the strict one).
LIVENESS_MATCH_THRESHOLD = _benchmark_threshold("threshold_far_1", default=0.4)

# PPE check (hard hat + vest). Off automatically if the weights file is missing.
PPE_WEIGHTS = ROOT / "models" / "ppe_yolo11n.pt"   # made by scripts/train_ppe.py
PPE_CONF = 0.4               # detector confidence needed to count a helmet / vest

# Enrollment
MIN_ENROLL_CAPTURES = 3
MAX_ENROLL_CAPTURES = 5
RETENTION_DAYS = 365         # biometric data deleted after this long without a check-in
ENROLL_ABANDON_HOURS = 24    # unfinished enrollments older than this are purged
RETENTION_CHECK_S = 3600     # how often the server looks for expired data

# Check-in rules
EVENT_COOLDOWN_S = 60        # repeat scans within this window are ignored
MAX_SHIFT_HOURS = 16         # an "in" older than this is treated as a missed check-out

# PIN fallback
PIN_MAX_FAILURES = 5
PIN_LOCKOUT_S = 300

# Uploads
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

# Offline-first sync (see app/sync.py). On-site server: set CENTRAL_URL + SYNC_TOKEN.
SITE_ID = os.getenv("SITE_ID", "site-1")
CENTRAL_URL = os.getenv("CENTRAL_URL", "")
SYNC_TOKEN = os.getenv("SYNC_TOKEN", "")
# Plain http:// to a non-localhost CENTRAL_URL is refused unless explicitly allowed, which is only
# sensible on a private network (e.g. between docker compose containers)
SYNC_ALLOW_HTTP = os.getenv("SYNC_ALLOW_HTTP", "") == "1"
SYNC_INTERVAL_S = 30
SYNC_BATCH = 500
# Central server: which sites may push, and each site's token ("site-a:tok-a,site-b:tok-b")
SITE_TOKENS = dict(pair.split(":", 1) for pair in os.getenv("SITE_TOKENS", "").split(",") if ":" in pair)
