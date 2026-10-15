from datetime import date, datetime, timedelta

import numpy as np
import pytest

from app import config
from app.db import Database

T0 = datetime(2026, 10, 12, 7, 0, 0)   # a Monday, 7:00 AM


@pytest.fixture
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


def enrolled(db, name="J. Rivera", pin="1234"):
    wid = db.create_worker(name, "Carpenter", pin, now=T0 - timedelta(days=1))
    for i in range(3):
        v = np.zeros(512, dtype=np.float32)
        v[i] = 1
        db.add_embedding(wid, v)
    db.finish_enrollment(wid, now=T0 - timedelta(days=1))
    return wid


def test_scan_toggles_in_and_out(db):
    w = enrolled(db)
    assert db.record_scan(w, "face", 0.9, now=T0)[0] == "checked_in"
    assert db.record_scan(w, "face", 0.9, now=T0 + timedelta(hours=8))[0] == "checked_out"
    assert db.record_scan(w, "face", 0.9, now=T0 + timedelta(hours=9))[0] == "checked_in"


def test_repeat_scan_within_cooldown_is_ignored(db):
    w = enrolled(db)
    db.record_scan(w, "face", now=T0)
    status, ts = db.record_scan(w, "face", now=T0 + timedelta(seconds=config.EVENT_COOLDOWN_S - 1))
    assert status == "cooldown" and ts == "2026-10-12 07:00:00"
    assert len(db.on_site_now(now=T0 + timedelta(minutes=5))) == 1


def test_stale_checkin_starts_new_shift(db):
    w = enrolled(db)
    db.record_scan(w, "face", now=T0)                    # forgot to check out
    status, _ = db.record_scan(w, "face", now=T0 + timedelta(days=1))
    assert status == "checked_in"


def test_on_site_now(db):
    a, b = enrolled(db, "A"), enrolled(db, "B")
    db.record_scan(a, "face", now=T0)
    db.record_scan(b, "pin", now=T0)
    db.record_scan(b, "face", now=T0 + timedelta(hours=1))
    names = [w["name"] for w in db.on_site_now(now=T0 + timedelta(hours=2))]
    assert names == ["A"]
    # 17h later A's check-in is stale and no longer counts as on site
    assert db.on_site_now(now=T0 + timedelta(hours=17)) == []


def test_hours_closed_open_and_missing_checkout(db):
    a, b, c = enrolled(db, "A"), enrolled(db, "B"), enrolled(db, "C")
    db.record_scan(a, "face", now=T0)
    db.record_scan(a, "face", now=T0 + timedelta(hours=8, minutes=30))
    db.record_scan(b, "face", now=T0 + timedelta(hours=1))          # still on site
    db.record_scan(c, "face", now=T0 - timedelta(days=1))           # never checked out
    db.record_scan(c, "face", now=T0)

    now = T0 + timedelta(hours=4)
    rows = {(r["name"], r["date"]): r for r in db.hours(date(2026, 10, 11), date(2026, 10, 12), now=now)}
    assert rows[("A", "2026-10-12")]["hours"] == 8.5
    assert rows[("B", "2026-10-12")]["hours"] == 3.0 and rows[("B", "2026-10-12")]["open"]
    assert rows[("C", "2026-10-11")]["missing_checkout"] and rows[("C", "2026-10-11")]["hours"] == 0
    assert rows[("C", "2026-10-12")]["open"]


def test_night_shift_counts_on_checkin_day(db):
    w = enrolled(db)
    db.record_scan(w, "face", now=datetime(2026, 10, 12, 22, 0))
    db.record_scan(w, "face", now=datetime(2026, 10, 13, 6, 0))
    rows = db.hours(date(2026, 10, 12), date(2026, 10, 12), now=datetime(2026, 10, 14))
    assert [(r["date"], r["hours"]) for r in rows] == [("2026-10-12", 8.0)]


def test_delete_removes_biometrics_but_keeps_attendance(db):
    w = enrolled(db)
    db.record_scan(w, "face", now=T0)
    assert db.delete_worker_biometrics(w, now=T0 + timedelta(hours=1))
    ids, vecs = db.gallery()
    assert len(ids) == 0 and len(db.get_embeddings(w)) == 0
    assert db.verify_pin(w, "1234", now=T0) == "bad"
    assert db.hours(date(2026, 10, 12), date(2026, 10, 12), now=T0 + timedelta(hours=2))[0]["name"] == "J. Rivera"
    assert db.list_audit()[0]["action"] == "delete_biometrics"
    assert not db.delete_worker_biometrics(w)              # already deleted


def test_gallery_excludes_pending_workers(db):
    pending = db.create_worker("Pending", "", None)
    db.add_embedding(pending, np.ones(512, dtype=np.float32))
    w = enrolled(db)
    ids, vecs = db.gallery()
    assert set(ids) == {w} and vecs.shape == (3, 512)


def test_embedding_roundtrip_is_exact(db):
    w = db.create_worker("X", "", None)
    v = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    db.add_embedding(w, v)
    np.testing.assert_array_equal(db.get_embeddings(w)[0], v)


def test_pin_is_hashed_and_locks_out(db):
    w = enrolled(db, pin="4821")
    with db._conn() as c:
        stored = c.execute("SELECT pin_hash FROM workers WHERE id = ?", (w,)).fetchone()[0]
    assert b"4821" not in stored
    assert db.verify_pin(w, "4821", now=T0) == "ok"
    for _ in range(config.PIN_MAX_FAILURES - 1):
        assert db.verify_pin(w, "0000", now=T0) == "bad"
    assert db.verify_pin(w, "0000", now=T0) == "locked"
    assert db.verify_pin(w, "4821", now=T0 + timedelta(seconds=10)) == "locked"   # even the right PIN
    assert db.verify_pin(w, "4821", now=T0 + timedelta(seconds=config.PIN_LOCKOUT_S + 1)) == "ok"
    assert any(a["action"] == "pin_lockout" for a in db.list_audit())


def test_unknown_worker_pin_looks_like_wrong_pin(db):
    assert db.verify_pin(999, "1234") == "bad"


def test_muster_roster(db):
    a, b, c = enrolled(db, "A"), enrolled(db, "B"), enrolled(db, "C")
    db.record_scan(a, "face", now=T0)
    db.record_scan(b, "face", now=T0)
    mid = db.start_muster(now=T0 + timedelta(hours=2))
    m = db.get_muster(mid)
    assert {e["name"] for e in m["entries"]} == {"A", "B"} and m["missing"] == 2
    assert db.set_accounted(mid, a, True)
    assert not db.set_accounted(mid, c, True)              # C wasn't on site
    assert db.get_muster(mid)["missing"] == 1
    assert db.latest_open_muster() == mid
    assert db.close_muster(mid)
    assert not db.set_accounted(mid, b, True)              # closed rosters are frozen
    assert db.latest_open_muster() is None
