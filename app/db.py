"""SQLite storage: workers, face embeddings, check-in events, musters, audit log.

Timestamps are local time ISO strings ("YYYY-MM-DD HH:MM:SS"), because a jobsite
cares about its own clock (shift start, daily hours). Every function that depends
on the current time takes an optional `now` so tests can control it.
"""
import hashlib
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

import numpy as np

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    trade            TEXT NOT NULL DEFAULT '',
    pin_hash         BLOB,
    pin_salt         BLOB,
    pin_failures     INTEGER NOT NULL DEFAULT 0,
    pin_locked_until TEXT,
    consent_at       TEXT NOT NULL,
    retention_until  TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    enrolled_at      TEXT,              -- set once enough face captures are stored
    deleted_at       TEXT               -- biometric data removed; attendance kept
);
CREATE TABLE IF NOT EXISTS embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id INTEGER NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
    vector    BLOB NOT NULL             -- 512 float32 values, raw bytes (no pickle)
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id  INTEGER NOT NULL REFERENCES workers(id),
    kind       TEXT NOT NULL CHECK (kind IN ('in', 'out')),
    method     TEXT NOT NULL CHECK (method IN ('face', 'pin')),
    similarity REAL,
    hard_hat   INTEGER,                 -- PPE seen at check-in: 1 yes, 0 no, NULL not checked
    vest       INTEGER,
    ts         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_worker_ts ON events(worker_id, ts);
CREATE TABLE IF NOT EXISTS musters (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    closed_at  TEXT
);
CREATE TABLE IF NOT EXISTS muster_entries (
    muster_id    INTEGER NOT NULL REFERENCES musters(id),
    worker_id    INTEGER NOT NULL REFERENCES workers(id),
    accounted_at TEXT,
    PRIMARY KEY (muster_id, worker_id)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    actor     TEXT NOT NULL,
    action    TEXT NOT NULL,
    worker_id INTEGER,
    detail    TEXT NOT NULL DEFAULT ''
);

-- Central server only: attendance pushed by on-site servers. Never face data or PINs.
CREATE TABLE IF NOT EXISTS sites (
    id           TEXT PRIMARY KEY,
    last_sync_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS site_workers (
    site_id    TEXT NOT NULL REFERENCES sites(id),
    worker_id  INTEGER NOT NULL,            -- the id on that site's server
    name       TEXT NOT NULL,
    trade      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (site_id, worker_id)
);
CREATE TABLE IF NOT EXISTS site_events (
    site_id    TEXT NOT NULL REFERENCES sites(id),
    event_id   INTEGER NOT NULL,            -- the id on that site's server (makes re-sends harmless)
    worker_id  INTEGER NOT NULL,
    kind       TEXT NOT NULL CHECK (kind IN ('in', 'out')),
    method     TEXT NOT NULL,
    hard_hat   INTEGER,
    vest       INTEGER,
    ts         TEXT NOT NULL,
    PRIMARY KEY (site_id, event_id)
);
"""


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def _hash_pin(pin: str, salt: bytes) -> bytes:
    # scrypt is deliberately slow, so a stolen DB can't be brute-forced quickly
    return hashlib.scrypt(pin.encode(), salt=salt, n=2**14, r=8, p=1)


class Database:
    def __init__(self, path: str = config.DB_PATH):
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)
            # databases created before the PPE check / sync: add their columns
            cols = {r["name"] for r in c.execute("PRAGMA table_info(events)")}
            for col, kind in (("hard_hat", "INTEGER"), ("vest", "INTEGER"), ("synced_at", "TEXT")):
                if col not in cols:
                    c.execute(f"ALTER TABLE events ADD COLUMN {col} {kind}")
            c.execute("CREATE INDEX IF NOT EXISTS idx_events_unsynced ON events(id) WHERE synced_at IS NULL")

    @contextmanager
    def _conn(self):
        """A short-lived connection per operation; safe across FastAPI's worker threads."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    # ---------- audit ----------
    def audit(self, actor: str, action: str, worker_id: int | None = None,
              detail: str = "", now: datetime | None = None, conn=None):
        sql = "INSERT INTO audit_log (ts, actor, action, worker_id, detail) VALUES (?,?,?,?,?)"
        args = (_fmt(now or datetime.now()), actor, action, worker_id, detail)
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self._conn() as c:
                c.execute(sql, args)

    def list_audit(self, limit: int = 100) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in rows]

    # ---------- workers ----------
    def create_worker(self, name: str, trade: str, pin: str | None,
                      actor: str = "foreman", now: datetime | None = None) -> int:
        """Create a worker in 'pending' state. Caller must have collected consent."""
        now = now or datetime.now()
        salt = secrets.token_bytes(16) if pin else None
        pin_hash = _hash_pin(pin, salt) if pin else None
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO workers (name, trade, pin_hash, pin_salt, consent_at,
                                        retention_until, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (name, trade, pin_hash, salt, _fmt(now),
                 _fmt(now + timedelta(days=config.RETENTION_DAYS)), _fmt(now)))
            self.audit(actor, "enroll_start", cur.lastrowid, "consent recorded", now, conn=c)
            return cur.lastrowid

    def get_worker(self, worker_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                """SELECT id, name, trade, consent_at, retention_until, created_at,
                          enrolled_at, deleted_at FROM workers WHERE id = ?""",
                (worker_id,)).fetchone()
            return dict(row) if row else None

    def list_workers(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT w.id, w.name, w.trade, w.enrolled_at, w.deleted_at, w.retention_until,
                          (SELECT COUNT(*) FROM embeddings e WHERE e.worker_id = w.id) AS captures,
                          w.pin_hash IS NOT NULL AS has_pin
                   FROM workers w ORDER BY w.name""")
            return [dict(r) for r in rows]

    def delete_worker_biometrics(self, worker_id: int, actor: str = "foreman",
                                 now: datetime | None = None, reason: str = "") -> bool:
        """'Delete my data': remove face embeddings and PIN. Attendance rows stay for payroll."""
        now = now or datetime.now()
        with self._conn() as c:
            row = c.execute("SELECT deleted_at FROM workers WHERE id = ?", (worker_id,)).fetchone()
            if row is None or row["deleted_at"]:
                return False
            c.execute("DELETE FROM embeddings WHERE worker_id = ?", (worker_id,))
            c.execute("""UPDATE workers SET pin_hash = NULL, pin_salt = NULL, deleted_at = ?
                         WHERE id = ?""", (_fmt(now), worker_id))
            self.audit(actor, "delete_biometrics", worker_id, reason, now, conn=c)
            return True

    def purge_expired(self, now: datetime | None = None) -> list[int]:
        """Automatic deletion. Removes biometric data for:
          - workers whose retention date has passed (a year without a check-in), and
          - enrollments abandoned before finishing (face data for nobody).
        Returns the worker ids purged.
        """
        now = now or datetime.now()
        abandoned_before = _fmt(now - timedelta(hours=config.ENROLL_ABANDON_HOURS))
        with self._conn() as c:
            expired = [r["id"] for r in c.execute(
                "SELECT id FROM workers WHERE deleted_at IS NULL AND retention_until < ?", (_fmt(now),))]
            abandoned = [r["id"] for r in c.execute(
                """SELECT id FROM workers WHERE deleted_at IS NULL AND enrolled_at IS NULL
                   AND created_at < ?""", (abandoned_before,))]
        for wid in expired:
            self.delete_worker_biometrics(wid, actor="retention", now=now,
                                          reason=f"no check-in for {config.RETENTION_DAYS} days")
        for wid in set(abandoned) - set(expired):
            self.delete_worker_biometrics(wid, actor="retention", now=now, reason="enrollment never finished")
        return sorted(set(expired) | set(abandoned))
    # ---------- embeddings ----------
    def add_embedding(self, worker_id: int, vector: np.ndarray) -> int:
        """Store one capture; returns how many captures the worker now has."""
        with self._conn() as c:
            c.execute("INSERT INTO embeddings (worker_id, vector) VALUES (?, ?)",
                      (worker_id, vector.astype(np.float32).tobytes()))
            return c.execute("SELECT COUNT(*) FROM embeddings WHERE worker_id = ?",
                             (worker_id,)).fetchone()[0]

    def get_embeddings(self, worker_id: int) -> np.ndarray:
        with self._conn() as c:
            rows = c.execute("SELECT vector FROM embeddings WHERE worker_id = ? ORDER BY id",
                             (worker_id,)).fetchall()
        return np.array([np.frombuffer(r["vector"], dtype=np.float32) for r in rows]).reshape(-1, 512)

    def finish_enrollment(self, worker_id: int, actor: str = "foreman",
                          now: datetime | None = None):
        now = now or datetime.now()
        with self._conn() as c:
            c.execute("UPDATE workers SET enrolled_at = ? WHERE id = ?", (_fmt(now), worker_id))
            self.audit(actor, "enroll_finish", worker_id, "", now, conn=c)

    def gallery(self) -> tuple[np.ndarray, np.ndarray]:
        """All embeddings of enrolled, non-deleted workers: (worker_ids[M], vectors[M,512])."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT e.worker_id, e.vector FROM embeddings e
                   JOIN workers w ON w.id = e.worker_id
                   WHERE w.enrolled_at IS NOT NULL AND w.deleted_at IS NULL""").fetchall()
        ids = np.array([r["worker_id"] for r in rows], dtype=np.int64)
        vecs = np.array([np.frombuffer(r["vector"], dtype=np.float32) for r in rows],
                        dtype=np.float32).reshape(-1, 512)
        return ids, vecs

    # ---------- PIN fallback ----------
    def verify_pin(self, worker_id: int, pin: str, now: datetime | None = None) -> str:
        """Returns 'ok', 'bad' or 'locked'. Too many failures lock the worker's PIN briefly."""
        now = now or datetime.now()
        with self._conn() as c:
            w = c.execute("""SELECT pin_hash, pin_salt, pin_failures, pin_locked_until, enrolled_at,
                                    deleted_at FROM workers WHERE id = ?""", (worker_id,)).fetchone()
            if w is None or w["pin_hash"] is None or w["deleted_at"] or not w["enrolled_at"]:
                return "bad"  # same answer as a wrong PIN, so IDs can't be probed
            if w["pin_locked_until"] and _parse(w["pin_locked_until"]) > now:
                return "locked"
            if secrets.compare_digest(_hash_pin(pin, w["pin_salt"]), w["pin_hash"]):
                c.execute("UPDATE workers SET pin_failures = 0, pin_locked_until = NULL WHERE id = ?",
                          (worker_id,))
                return "ok"
            failures = w["pin_failures"] + 1
            if failures >= config.PIN_MAX_FAILURES:
                c.execute("UPDATE workers SET pin_failures = 0, pin_locked_until = ? WHERE id = ?",
                          (_fmt(now + timedelta(seconds=config.PIN_LOCKOUT_S)), worker_id))
                self.audit("kiosk", "pin_lockout", worker_id, f"{failures} failed attempts", now, conn=c)
                return "locked"
            c.execute("UPDATE workers SET pin_failures = ? WHERE id = ?", (failures, worker_id))
            return "bad"

    # ---------- check-in / check-out ----------
    def record_scan(self, worker_id: int, method: str, similarity: float | None = None,
                    now: datetime | None = None, ppe: dict | None = None) -> tuple[str, str]:
        """Flip the worker between in and out. Returns (status, timestamp).

        `ppe` is {"hard_hat": bool|None, "vest": bool|None} from the kiosk camera, if checked.
        Missing PPE at check-in is audit-logged for the foreman; it never blocks the check-in.

        status is 'checked_in', 'checked_out', or 'cooldown' (a repeat scan that was ignored).
        A previous 'in' older than MAX_SHIFT_HOURS counts as a missed check-out, so the
        next scan starts a new shift instead of closing a stale one.
        """
        now = now or datetime.now()
        with self._conn() as c:
            last = c.execute("SELECT kind, ts FROM events WHERE worker_id = ? ORDER BY ts DESC, id DESC LIMIT 1",
                             (worker_id,)).fetchone()
            if last:
                age = now - _parse(last["ts"])
                if age < timedelta(seconds=config.EVENT_COOLDOWN_S):
                    return "cooldown", last["ts"]
                stale = age > timedelta(hours=config.MAX_SHIFT_HOURS)
                kind = "out" if last["kind"] == "in" and not stale else "in"
            else:
                kind = "in"
            # every check-in or check-out restarts the retention clock
            c.execute("UPDATE workers SET retention_until = ? WHERE id = ? AND deleted_at IS NULL",
                      (_fmt(now + timedelta(days=config.RETENTION_DAYS)), worker_id))
            ppe = ppe or {}
            c.execute("""INSERT INTO events (worker_id, kind, method, similarity, hard_hat, vest, ts)
                         VALUES (?,?,?,?,?,?,?)""",
                      (worker_id, kind, method, similarity, ppe.get("hard_hat"), ppe.get("vest"), _fmt(now)))
            missing = [item for item, key in (("hard hat", "hard_hat"), ("vest", "vest")) if ppe.get(key) is False]
            if kind == "in" and missing:
                self.audit("kiosk", "ppe_missing", worker_id, "no " + ", no ".join(missing), now, conn=c)
            return ("checked_in" if kind == "in" else "checked_out"), _fmt(now)

    def in_cooldown(self, worker_id: int, now: datetime | None = None) -> str | None:
        """Timestamp of the worker's last event if it was within the cooldown window, else None."""
        now = now or datetime.now()
        with self._conn() as c:
            last = c.execute("SELECT ts FROM events WHERE worker_id = ? ORDER BY ts DESC, id DESC LIMIT 1",
                             (worker_id,)).fetchone()
        if last and now - _parse(last["ts"]) < timedelta(seconds=config.EVENT_COOLDOWN_S):
            return last["ts"]
        return None

    def on_site_now(self, now: datetime | None = None) -> list[dict]:
        """Workers whose latest event is a check-in within the max shift length."""
        now = now or datetime.now()
        cutoff = _fmt(now - timedelta(hours=config.MAX_SHIFT_HOURS))
        with self._conn() as c:
            rows = c.execute(
                """SELECT w.id, w.name, w.trade, e.ts AS since, e.method, e.hard_hat, e.vest
                   FROM events e JOIN workers w ON w.id = e.worker_id
                   WHERE e.id = (SELECT id FROM events WHERE worker_id = e.worker_id
                                 ORDER BY ts DESC, id DESC LIMIT 1)
                     AND e.kind = 'in' AND e.ts >= ?
                   ORDER BY e.ts""", (cutoff,))
            return [dict(r) for r in rows]

    def hours(self, start: date, end: date, now: datetime | None = None) -> list[dict]:
        """Hours per worker per day (by check-in date), for days start..end inclusive.

        Each row: worker_id, name, trade, date, hours, sessions, open (still on site),
        missing_checkout (a shift with no check-out; not counted, needs foreman review).
        """
        now = now or datetime.now()
        lo = _fmt(datetime.combine(start, datetime.min.time()))
        # look past `end` so night shifts that check out after midnight still pair up
        hi = _fmt(datetime.combine(end + timedelta(days=1), datetime.min.time())
                  + timedelta(hours=config.MAX_SHIFT_HOURS))
        with self._conn() as c:
            rows = c.execute(
                """SELECT e.worker_id, w.name, w.trade, e.kind, e.ts FROM events e
                   JOIN workers w ON w.id = e.worker_id
                   WHERE e.ts >= ? AND e.ts < ? ORDER BY e.worker_id, e.ts, e.id""",
                (lo, hi)).fetchall()

        max_shift = timedelta(hours=config.MAX_SHIFT_HOURS)
        days: dict[tuple[int, date], dict] = {}

        def bucket(r, d):
            key = (r["worker_id"], d)
            if key not in days:
                days[key] = {"worker_id": r["worker_id"], "name": r["name"], "trade": r["trade"],
                             "date": d.isoformat(), "hours": 0.0, "sessions": 0,
                             "open": False, "missing_checkout": False}
            return days[key]

        def close_open(open_in, end_ts, missed=False):
            """Pair a check-in with end_ts (its check-out), or flag it if none came."""
            t_in = _parse(open_in["ts"])
            d = t_in.date()
            if not (start <= d <= end):
                return
            b = bucket(open_in, d)
            if missed:
                b["missing_checkout"] = True
            elif end_ts is not None and end_ts - t_in <= max_shift:
                b["hours"] += (end_ts - t_in).total_seconds() / 3600
                b["sessions"] += 1
            elif end_ts is None and now - t_in <= max_shift:
                b["hours"] += (now - t_in).total_seconds() / 3600  # still on site
                b["sessions"] += 1
                b["open"] = True
            else:
                b["missing_checkout"] = True

        open_in = None
        prev_worker = None
        for r in rows:
            if r["worker_id"] != prev_worker:
                if open_in is not None:
                    close_open(open_in, None)
                open_in, prev_worker = None, r["worker_id"]
            if r["kind"] == "in":
                if open_in is not None:
                    close_open(open_in, None, missed=True)  # two check-ins in a row
                open_in = r
            elif open_in is not None:
                close_open(open_in, _parse(r["ts"]))
                open_in = None
        if open_in is not None:
            close_open(open_in, None)

        out = sorted(days.values(), key=lambda b: (b["date"], b["name"]))
        for b in out:
            b["hours"] = round(b["hours"], 2)
        return out

    # ---------- emergency muster ----------
    def start_muster(self, actor: str = "foreman", now: datetime | None = None) -> int:
        """Snapshot everyone on site into a roster the foreman checks off."""
        now = now or datetime.now()
        on_site = self.on_site_now(now)
        with self._conn() as c:
            mid = c.execute("INSERT INTO musters (started_at) VALUES (?)", (_fmt(now),)).lastrowid
            c.executemany("INSERT INTO muster_entries (muster_id, worker_id) VALUES (?, ?)",
                          [(mid, w["id"]) for w in on_site])
            self.audit(actor, "muster_start", None, f"muster {mid}, {len(on_site)} on site", now, conn=c)
            return mid

    def set_accounted(self, muster_id: int, worker_id: int, accounted: bool,
                      now: datetime | None = None) -> bool:
        now = now or datetime.now()
        with self._conn() as c:
            cur = c.execute(
                """UPDATE muster_entries SET accounted_at = ? WHERE muster_id = ? AND worker_id = ?
                   AND muster_id IN (SELECT id FROM musters WHERE closed_at IS NULL)""",
                (_fmt(now) if accounted else None, muster_id, worker_id))
            return cur.rowcount > 0

    def close_muster(self, muster_id: int, actor: str = "foreman", now: datetime | None = None) -> bool:
        now = now or datetime.now()
        with self._conn() as c:
            cur = c.execute("UPDATE musters SET closed_at = ? WHERE id = ? AND closed_at IS NULL",
                            (_fmt(now), muster_id))
            if cur.rowcount:
                missing = c.execute("""SELECT COUNT(*) FROM muster_entries
                                       WHERE muster_id = ? AND accounted_at IS NULL""",
                                    (muster_id,)).fetchone()[0]
                self.audit(actor, "muster_close", None, f"muster {muster_id}, {missing} unaccounted", now, conn=c)
            return cur.rowcount > 0

    def get_muster(self, muster_id: int) -> dict | None:
        with self._conn() as c:
            m = c.execute("SELECT * FROM musters WHERE id = ?", (muster_id,)).fetchone()
            if m is None:
                return None
            entries = c.execute(
                """SELECT w.id AS worker_id, w.name, w.trade, me.accounted_at
                   FROM muster_entries me JOIN workers w ON w.id = me.worker_id
                   WHERE me.muster_id = ? ORDER BY me.accounted_at IS NOT NULL, w.name""",
                (muster_id,)).fetchall()
        entries = [dict(e) for e in entries]
        return {**dict(m), "entries": entries,
                "missing": sum(e["accounted_at"] is None for e in entries)}

    def latest_open_muster(self) -> int | None:
        with self._conn() as c:
            row = c.execute("SELECT id FROM musters WHERE closed_at IS NULL ORDER BY id DESC LIMIT 1").fetchone()
            return row["id"] if row else None

    # ---------- sync: on-site server side ----------
    def unsynced_batch(self, limit: int) -> dict:
        """Oldest events not yet sent to the central server, plus the workers they refer to
        (name and trade only: never face data or PINs)."""
        with self._conn() as c:
            events = [dict(r) for r in c.execute(
                """SELECT id, worker_id, kind, method, hard_hat, vest, ts FROM events
                   WHERE synced_at IS NULL ORDER BY id LIMIT ?""", (limit,))]
            ids = sorted({e["worker_id"] for e in events})
            workers = [dict(r) for r in c.execute(
                f"SELECT id, name, trade FROM workers WHERE id IN ({','.join('?' * len(ids))})", ids)] if ids else []
        return {"workers": workers, "events": events}

    def mark_synced(self, event_ids: list[int], now: datetime | None = None):
        if not event_ids:
            return
        with self._conn() as c:
            c.executemany("UPDATE events SET synced_at = ? WHERE id = ?",
                          [(_fmt(now or datetime.now()), i) for i in event_ids])

    def unsynced_count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM events WHERE synced_at IS NULL").fetchone()[0]

    # ---------- sync: central server side ----------
    def ingest_site_batch(self, site_id: str, workers: list[dict], events: list[dict],
                          now: datetime | None = None) -> list[int]:
        """Store a site's batch. Re-sent events are ignored, so retries after a dropped
        connection are safe. Returns the event ids now stored (all of them, new or not)."""
        now = now or datetime.now()
        with self._conn() as c:
            c.execute("""INSERT INTO sites (id, last_sync_at) VALUES (?, ?)
                         ON CONFLICT(id) DO UPDATE SET last_sync_at = excluded.last_sync_at""",
                      (site_id, _fmt(now)))
            c.executemany("""INSERT INTO site_workers (site_id, worker_id, name, trade) VALUES (?,?,?,?)
                             ON CONFLICT(site_id, worker_id) DO UPDATE SET name = excluded.name,
                             trade = excluded.trade""",
                          [(site_id, w["id"], w["name"], w["trade"]) for w in workers])
            c.executemany("""INSERT OR IGNORE INTO site_events
                             (site_id, event_id, worker_id, kind, method, hard_hat, vest, ts)
                             VALUES (?,?,?,?,?,?,?,?)""",
                          [(site_id, e["id"], e["worker_id"], e["kind"], e["method"], e["hard_hat"],
                            e["vest"], e["ts"]) for e in events])
        return [e["id"] for e in events]

    def site_summaries(self, now: datetime | None = None) -> list[dict]:
        """Per site: when it last synced, and who is on site according to its latest data."""
        now = now or datetime.now()
        cutoff = _fmt(now - timedelta(hours=config.MAX_SHIFT_HOURS))
        with self._conn() as c:
            sites = [dict(r) for r in c.execute("SELECT id, last_sync_at FROM sites ORDER BY id")]
            for site in sites:
                site["on_site"] = [dict(r) for r in c.execute(
                    """SELECT w.worker_id AS id, w.name, w.trade, e.ts AS since, e.hard_hat, e.vest
                       FROM site_events e JOIN site_workers w
                         ON w.site_id = e.site_id AND w.worker_id = e.worker_id
                       WHERE e.site_id = ? AND e.kind = 'in' AND e.ts >= ?
                         AND e.event_id = (SELECT event_id FROM site_events
                                           WHERE site_id = e.site_id AND worker_id = e.worker_id
                                           ORDER BY ts DESC, event_id DESC LIMIT 1)
                       ORDER BY e.ts""", (site["id"], cutoff))]
                site["minutes_since_sync"] = round((now - _parse(site["last_sync_at"])).total_seconds() / 60, 1)
        return sites
