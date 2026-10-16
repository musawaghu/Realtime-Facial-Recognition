"""Automatic deletion: biometric data goes after a year without a check-in, and unfinished
enrollments are purged; attendance stays."""
from datetime import date, datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient

from app import api, config
from app.db import Database
from app.face_engine import Gallery

T0 = datetime(2026, 1, 5, 7, 0)
YEAR = timedelta(days=config.RETENTION_DAYS)


def make(db, name, enrolled=True, created=T0):
    wid = db.create_worker(name, "", "1234", now=created)
    db.add_embedding(wid, np.eye(512, dtype=np.float32)[wid])
    if enrolled:
        db.finish_enrollment(wid, now=created)
    return wid


def worker(db, wid):
    return db.get_worker(wid)


def test_checkins_restart_the_retention_clock(tmp_path):
    db = Database(str(tmp_path / "r.db"))
    w = make(db, "A")
    assert worker(db, w)["retention_until"] == "2027-01-05 07:00:00"
    db.record_scan(w, "face", now=T0 + timedelta(days=200))
    assert worker(db, w)["retention_until"] == (T0 + timedelta(days=200) + YEAR).strftime("%Y-%m-%d %H:%M:%S")


def test_inactive_worker_is_purged_active_one_kept(tmp_path):
    db = Database(str(tmp_path / "r.db"))
    idle, active = make(db, "Idle"), make(db, "Active")
    db.record_scan(idle, "pin", now=T0 + timedelta(hours=1))
    db.record_scan(active, "pin", now=T0 + timedelta(days=300))
    assert db.purge_expired(now=T0 + timedelta(days=364)) == []          # not yet
    assert db.purge_expired(now=T0 + YEAR + timedelta(days=2)) == [idle]
    assert worker(db, idle)["deleted_at"] and not worker(db, active)["deleted_at"]
    assert len(db.get_embeddings(idle)) == 0 and db.verify_pin(idle, "1234") == "bad"
    audit = db.list_audit()[0]
    assert (audit["action"], audit["actor"]) == ("delete_biometrics", "retention")
    assert "no check-in" in audit["detail"]
    # attendance is kept for payroll
    assert db.hours(date(2026, 1, 5), date(2026, 1, 5), now=T0 + timedelta(days=400))[0]["name"] == "Idle"
    assert db.purge_expired(now=T0 + YEAR + timedelta(days=3)) == []     # nothing purged twice


def test_abandoned_enrollment_is_purged(tmp_path):
    db = Database(str(tmp_path / "r.db"))
    pending = make(db, "Half done", enrolled=False)
    fresh = make(db, "Just started", enrolled=False, created=T0 + timedelta(hours=20))
    purged = db.purge_expired(now=T0 + timedelta(hours=config.ENROLL_ABANDON_HOURS + 1))
    assert purged == [pending]
    assert worker(db, fresh)["deleted_at"] is None
    assert db.list_audit()[0]["detail"] == "enrollment never finished"


def test_run_retention_stops_recognition_immediately(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "r.db"))
    w = make(db, "A", created=datetime.now() - YEAR - timedelta(days=1))

    class Engine:
        gallery = Gallery(threshold=0.5)

    engine = Engine()
    engine.gallery.load(np.array([w]), np.eye(512, dtype=np.float32)[[w]])
    assert api.run_retention(db, engine) == [w]
    assert len(engine.gallery) == 0
    assert api.run_retention(db, None) == []                             # works without models loaded


def test_retention_job_runs_at_startup(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "r.db"))
    w = make(db, "A", created=datetime.now() - YEAR - timedelta(days=1))
    monkeypatch.setattr(api, "get_db", lambda: db)
    with TestClient(api.app):                                            # runs the lifespan
        for _ in range(50):
            if worker(db, w)["deleted_at"]:
                break
            import time
            time.sleep(0.05)
    assert worker(db, w)["deleted_at"]
