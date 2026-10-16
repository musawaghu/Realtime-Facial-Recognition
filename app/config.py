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


def _benchmark_threshold(default: float) -> float:
    """Use the threshold measured on LFW (benchmarks/lfw_eval.py) when available."""
    path = ROOT / "benchmarks" / "results" / "lfw.json"
    try:
        return float(json.loads(path.read_text())["threshold_far_0.1"])
    except (OSError, KeyError, ValueError):
        return default


# Matching: cosine similarity between embeddings (higher = more alike)
MATCH_THRESHOLD = _benchmark_threshold(default=0.5)

# Enrollment
MIN_ENROLL_CAPTURES = 3
MAX_ENROLL_CAPTURES = 5
RETENTION_DAYS = 365         # biometric data kept this long after enrollment

# Check-in rules
EVENT_COOLDOWN_S = 60        # repeat scans within this window are ignored
MAX_SHIFT_HOURS = 16         # an "in" older than this is treated as a missed check-out

# PIN fallback
PIN_MAX_FAILURES = 5
PIN_LOCKOUT_S = 300

# Uploads
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
