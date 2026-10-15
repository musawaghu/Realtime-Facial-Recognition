"""FastAPI backend for the jobsite check-in kiosk and the foreman dashboard.

Run:  uvicorn app.api:app --reload
Kiosk:     http://localhost:8000/kiosk       (open: it's the device at the gate)
Dashboard: http://localhost:8000/dashboard   (HTTP Basic: foreman / ADMIN_PASSWORD)
"""
import csv
import io
import secrets
import threading
import time
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from app import config, liveness
from app.db import Database

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Jobsite Check-in")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ============ Shared resources (overridable in tests) ============
@lru_cache
def get_db() -> Database:
    return Database()


_engine_lock = threading.Lock()   # one inference at a time; torch models aren't shared safely


@lru_cache
def get_engine():
    from app.face_engine import FaceEngine  # heavy import: only when a camera endpoint is used
    engine = FaceEngine()
    engine.gallery.load(*get_db().gallery())
    return engine


@lru_cache
def get_ppe():
    """The PPE detector, or None if it hasn't been trained (scripts/train_ppe.py)."""
    if not config.PPE_WEIGHTS.exists():
        return None
    from app.ppe import PPEChecker
    return PPEChecker()


def reload_gallery(db: Database, engine):
    ids, vectors = db.gallery()
    with _engine_lock:   # don't swap the matrix while a scan is matching against it
        engine.gallery.load(ids, vectors)


# ============ Security helpers ============
_basic = HTTPBasic()


def require_foreman(creds: HTTPBasicCredentials = Depends(_basic)) -> str:
    if not config.ADMIN_PASSWORD:
        raise HTTPException(503, "Set ADMIN_PASSWORD in .env to enable the dashboard")
    ok_user = secrets.compare_digest(creds.username.encode(), config.ADMIN_USER.encode())
    ok_pass = secrets.compare_digest(creds.password.encode(), config.ADMIN_PASSWORD.encode())
    if not (ok_user and ok_pass):
        raise HTTPException(401, "Wrong credentials", headers={"WWW-Authenticate": "Basic"})
    return creds.username


class RateLimiter:
    """Token bucket per client IP: `rate` requests/second, bursts up to `burst`."""

    def __init__(self, rate: float, burst: int):
        self.rate, self.burst = rate, burst
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def __call__(self, request: Request):
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1:
                raise HTTPException(429, "Too many requests, slow down")
            self.buckets[key] = (tokens - 1, now)


scan_limit = RateLimiter(rate=5, burst=10)        # kiosk sends ~2.5 frames/s
pin_limit = RateLimiter(rate=10 / 60, burst=5)    # 10 PIN tries per minute per device


def read_image(file: UploadFile) -> np.ndarray:
    """Decode an uploaded image into an RGB array, rejecting oversized or non-image uploads."""
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large")
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Not an image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def largest_face(faces):
    """At the gate the closest person (biggest face) is the one checking in."""
    return max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


def pretty_time(ts: str) -> str:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p").lstrip("0")


# ============ Pages ============
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/kiosk")


@app.get("/kiosk", include_in_schema=False)
def kiosk_page():
    return FileResponse(STATIC / "kiosk.html")


@app.get("/dashboard", include_in_schema=False)
def dashboard_page(_: str = Depends(require_foreman)):
    return FileResponse(STATIC / "dashboard.html")


# ============ Kiosk ============
@app.post("/api/scan", dependencies=[Depends(scan_limit)])
def scan(file: UploadFile = File(...), challenge: str | None = Form(None),
         db: Database = Depends(get_db), engine=Depends(get_engine), ppe=Depends(get_ppe)):
    """One camera frame from the kiosk.

    Without `challenge`: recognize the closest face. A recognized worker facing the camera gets
    a liveness challenge ({"status": "turn", "direction": "left"|"right", "challenge": token}).
    With `challenge`: keep sending frames until the worker has turned their head as asked;
    only then are they checked in or out. Hard hat / vest are checked on the frontal frame;
    missing PPE is reported back ("ppe") and logged, but never blocks the check-in.
    """
    rgb = read_image(file)
    with _engine_lock:
        faces = engine.analyze(rgb)
        face = largest_face(faces) if faces else None
        if challenge:
            return _continue_challenge(challenge, face, db, engine)
        if face is None:
            return {"status": "no_face"}
        worker_id, similarity = engine.gallery.match(face.embedding)
    if worker_id is None:
        return {"status": "unknown", "box": face.box}
    name = db.get_worker(worker_id)["name"]
    if (last := db.in_cooldown(worker_id)) is not None:
        return {"status": "cooldown", "name": name, "time": pretty_time(last), "box": face.box}
    if not liveness.is_frontal(face.landmarks):
        return {"status": "look_straight", "box": face.box}
    ch = challenges.create(worker_id, face.box, similarity)
    if ppe is not None:
        with _engine_lock:
            ch.ppe = ppe.check(rgb, face.box)
    return {"status": "turn", "challenge": ch.token, "direction": ch.direction_name, "name": name, "box": face.box}


challenges = liveness.ChallengeStore()


def _continue_challenge(token: str, face, db: Database, engine) -> dict:
    """Called with _engine_lock held."""
    ch, expired = challenges.take(token)
    if ch is None:
        return {"status": "challenge_failed", "reason": "unknown_challenge"}
    if expired:
        db.audit("kiosk", "liveness_fail", ch.worker_id, "timed out (no head turn)")
        return {"status": "challenge_failed", "reason": "timeout"}
    identity_ok = False
    if face is not None:
        # still the same worker, and not a better match for anyone else
        best_id, _ = engine.gallery.match(face.embedding)
        identity_ok = (best_id in (None, ch.worker_id)
                       and engine.gallery.score(face.embedding, ch.worker_id) >= config.LIVENESS_MATCH_THRESHOLD)
    result = liveness.step(ch, face, identity_ok)
    if result == "turn":
        return {"status": "turn", "challenge": token, "direction": ch.direction_name, "box": face.box}
    challenges.close(token)
    if result != "passed":
        db.audit("kiosk", "liveness_fail", ch.worker_id, result.replace("_", " "))
        return {"status": "challenge_failed", "reason": result}
    status, ts = db.record_scan(ch.worker_id, "face", ch.similarity, ppe=ch.ppe)
    return {"status": status, "name": db.get_worker(ch.worker_id)["name"], "time": pretty_time(ts),
            "box": face.box, "ppe": ch.ppe}


class PinCheckIn(BaseModel):
    worker_id: int = Field(ge=1)
    pin: str = Field(pattern=r"^\d{4,6}$")


@app.post("/api/pin", dependencies=[Depends(pin_limit)])
def pin_checkin(body: PinCheckIn, db: Database = Depends(get_db)):
    """Fallback when the face isn't recognized (glare, dirt, new beard...)."""
    result = db.verify_pin(body.worker_id, body.pin)
    if result == "locked":
        return {"status": "locked"}
    if result == "bad":
        return {"status": "bad_pin"}
    status, ts = db.record_scan(body.worker_id, "pin")
    return {"status": status, "name": db.get_worker(body.worker_id)["name"], "time": pretty_time(ts)}


# ============ Enrollment (foreman) ============
class EnrollStart(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    trade: str = Field(default="", max_length=40)
    pin: str = Field(pattern=r"^\d{4,6}$")
    consent: bool

    @field_validator("name", "trade")
    @classmethod
    def clean_text(cls, v: str) -> str:
        v = " ".join(v.split())   # trim and collapse whitespace / control characters
        if v and not v.isprintable():
            raise ValueError("invalid characters")
        return v


@app.post("/api/enroll/start")
def enroll_start(body: EnrollStart, db: Database = Depends(get_db), actor: str = Depends(require_foreman)):
    if not body.consent:
        raise HTTPException(400, "Written consent is required before collecting biometric data")
    if not body.name:
        raise HTTPException(422, "Name is required")
    worker_id = db.create_worker(body.name, body.trade, body.pin, actor=actor)
    return {"worker_id": worker_id}


def _pending_worker(db: Database, worker_id: int) -> dict:
    w = db.get_worker(worker_id)
    if w is None or w["deleted_at"]:
        raise HTTPException(404, "No such worker")
    if w["enrolled_at"]:
        raise HTTPException(409, "Worker is already enrolled")
    return w


@app.post("/api/enroll/{worker_id}/capture")
def enroll_capture(worker_id: int, file: UploadFile = File(...), db: Database = Depends(get_db),
                   engine=Depends(get_engine), _: str = Depends(require_foreman)):
    """Store one face embedding (one angle). Only the embedding is kept, never the photo."""
    _pending_worker(db, worker_id)
    existing = db.get_embeddings(worker_id)
    if len(existing) >= config.MAX_ENROLL_CAPTURES:
        raise HTTPException(409, "Already have enough captures")
    rgb = read_image(file)
    with _engine_lock:
        faces = engine.analyze(rgb)
    if len(faces) != 1:
        return {"status": "no_face" if not faces else "multiple", "captures": len(existing)}
    emb = faces[0].embedding
    # every capture must look like the same person as the first one
    if len(existing) and float(np.max(existing @ emb)) < config.MATCH_THRESHOLD:
        return {"status": "mismatch", "captures": len(existing)}
    return {"status": "captured", "captures": db.add_embedding(worker_id, emb)}


@app.post("/api/enroll/{worker_id}/finish")
def enroll_finish(worker_id: int, db: Database = Depends(get_db), engine=Depends(get_engine),
                  actor: str = Depends(require_foreman)):
    _pending_worker(db, worker_id)
    captures = db.get_embeddings(worker_id)
    if len(captures) < config.MIN_ENROLL_CAPTURES:
        raise HTTPException(409, f"Need at least {config.MIN_ENROLL_CAPTURES} captures")
    # warn if this face already belongs to someone (enrolled twice, or a buddy-punch attempt)
    mean = captures.mean(axis=0)
    with _engine_lock:
        dup_id, _ = engine.gallery.match(mean / np.linalg.norm(mean))
    db.finish_enrollment(worker_id, actor=actor)
    reload_gallery(db, engine)
    dup = db.get_worker(dup_id)["name"] if dup_id is not None else None
    return {"status": "enrolled", "possible_duplicate_of": dup}


# ============ Dashboard (foreman) ============
@app.get("/api/onsite")
def onsite(db: Database = Depends(get_db), _: str = Depends(require_foreman)):
    return db.on_site_now()


@app.get("/api/workers")
def workers(db: Database = Depends(get_db), _: str = Depends(require_foreman)):
    return db.list_workers()


@app.delete("/api/workers/{worker_id}")
def delete_worker(worker_id: int, db: Database = Depends(get_db), engine=Depends(get_engine),
                  actor: str = Depends(require_foreman)):
    """'Delete my data': biometric data (face + PIN) is erased; attendance stays for payroll."""
    if not db.delete_worker_biometrics(worker_id, actor=actor):
        raise HTTPException(404, "No such worker, or already deleted")
    reload_gallery(db, engine)   # they stop being recognized immediately
    return {"status": "deleted"}


def _range(start: date | None, end: date | None) -> tuple[date, date]:
    start = start or date.today()
    end = end or start
    if end < start or (end - start).days > 366:
        raise HTTPException(422, "Invalid date range")
    return start, end


@app.get("/api/hours")
def hours(start: date | None = None, end: date | None = None, db: Database = Depends(get_db),
          _: str = Depends(require_foreman)):
    return db.hours(*_range(start, end))


def csv_safe(value) -> str:
    """Stop spreadsheet formula injection: a name like '=HYPERLINK(...)' is stored as text."""
    s = str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


@app.get("/api/hours.csv")
def hours_csv(start: date | None = None, end: date | None = None, db: Database = Depends(get_db),
              actor: str = Depends(require_foreman)):
    start, end = _range(start, end)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["worker_id", "name", "trade", "date", "hours", "sessions", "still_on_site", "missing_checkout"])
    for r in db.hours(start, end):
        writer.writerow([r["worker_id"], csv_safe(r["name"]), csv_safe(r["trade"]), r["date"], r["hours"],
                         r["sessions"], r["open"], r["missing_checkout"]])
    db.audit(actor, "export_hours", None, f"{start}..{end}")
    filename = f"hours_{start}_{end}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/audit")
def audit(db: Database = Depends(get_db), _: str = Depends(require_foreman)):
    return db.list_audit()


# ============ Emergency muster (foreman) ============
@app.post("/api/muster")
def muster_start(db: Database = Depends(get_db), actor: str = Depends(require_foreman)):
    return db.get_muster(db.start_muster(actor=actor))


@app.get("/api/muster/current")
def muster_current(db: Database = Depends(get_db), _: str = Depends(require_foreman)):
    mid = db.latest_open_muster()
    return db.get_muster(mid) if mid else None


@app.get("/api/muster/{muster_id}")
def muster_get(muster_id: int, db: Database = Depends(get_db), _: str = Depends(require_foreman)):
    m = db.get_muster(muster_id)
    if m is None:
        raise HTTPException(404, "No such muster")
    return m


@app.post("/api/muster/{muster_id}/account/{worker_id}")
def muster_account(muster_id: int, worker_id: int, accounted: bool = True,
                   db: Database = Depends(get_db), _: str = Depends(require_foreman)):
    if not db.set_accounted(muster_id, worker_id, accounted):
        raise HTTPException(404, "Not on this roster, or the muster is closed")
    return db.get_muster(muster_id)


@app.post("/api/muster/{muster_id}/close")
def muster_close(muster_id: int, db: Database = Depends(get_db), actor: str = Depends(require_foreman)):
    if not db.close_muster(muster_id, actor=actor):
        raise HTTPException(404, "No such open muster")
    return db.get_muster(muster_id)
