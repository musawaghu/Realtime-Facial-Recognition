"""Offline-first sync between an on-site server and the central server, in one process."""
from datetime import datetime, timedelta

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api, config, sync
from app.db import Database

TOKENS = {"site-a": "token-a", "site-b": "token-b"}
NOW = datetime.now().replace(microsecond=0)


@pytest.fixture
def central(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "central.db"))
    monkeypatch.setattr(config, "SITE_TOKENS", TOKENS)
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "pw")
    api.app.dependency_overrides = {api.get_db: lambda: db}
    yield TestClient(api.app), db
    api.app.dependency_overrides = {}


@pytest.fixture
def site(tmp_path):
    db = Database(str(tmp_path / "site.db"))
    a = db.create_worker("Ana Torres", "Carpenter", "1234")
    db.add_embedding(a, np.ones(512, dtype=np.float32))      # must never leave the site
    db.finish_enrollment(a)
    b = db.create_worker("Ben Okafor", "Electrician", "5678")
    db.finish_enrollment(b)
    db.record_scan(a, "face", 0.8, now=NOW - timedelta(hours=2), ppe={"hard_hat": True, "vest": None})
    db.record_scan(b, "pin", now=NOW - timedelta(hours=2))
    db.record_scan(b, "pin", now=NOW - timedelta(hours=1))
    return db, a, b


def client_for(site_db, central_client, site_id="site-a", token="token-a", calls=None):
    def post(url, json, headers):
        if calls is not None:
            calls.append(json)
        return central_client.post("/api/sync/events", json=json, headers=headers)
    return sync.SyncClient(site_db, post, central_url="https://central.example", site_id=site_id, token=token)


def test_push_sends_attendance_only_never_biometrics(site, central):
    site_db, a, b = site
    central_client, central_db = central
    calls = []
    c = client_for(site_db, central_client, calls=calls)
    assert c.push_once() == 3
    assert site_db.unsynced_count() == 0
    sent = calls[0]
    assert {k for w in sent["workers"] for k in w} == {"id", "name", "trade"}       # no PIN, no face data
    assert "1234" not in str(sent) and "vector" not in str(sent)
    summary = central_db.site_summaries()
    assert [s["id"] for s in summary] == ["site-a"]
    assert [(w["name"], w["hard_hat"]) for w in summary[0]["on_site"]] == [("Ana Torres", 1)]   # Ben checked out


def test_offline_keeps_queue_and_backs_off_then_catches_up(site, central):
    site_db, _, _ = site
    central_client, central_db = central
    online = False

    def flaky_post(url, json, headers):
        if not online:
            raise httpx.ConnectError("no route to host")
        return central_client.post("/api/sync/events", json=json, headers=headers)

    c = sync.SyncClient(site_db, flaky_post, central_url="https://central.example", site_id="site-a", token="token-a")
    assert c.push_once() == 0 and c.push_once() == 0
    assert site_db.unsynced_count() == 3 and "ConnectError" in c.status()["last_error"]
    assert c.next_delay() == config.SYNC_INTERVAL_S * 4                              # backing off
    site_db.record_scan(site[1], "face", now=NOW)                                    # check-ins keep working offline
    online = True
    assert c.push_once() == 4 and site_db.unsynced_count() == 0
    assert c.next_delay() == config.SYNC_INTERVAL_S


def test_resending_is_harmless(site, central):
    site_db, _, _ = site
    central_client, central_db = central
    batch = {"site_id": "site-a", **site_db.unsynced_batch(100)}
    for _ in range(2):   # e.g. the confirmation got lost and the site sent it again
        r = central_client.post("/api/sync/events", json=batch, headers={"Authorization": "Bearer token-a"})
        assert r.status_code == 200
    with central_db._conn() as conn:
        assert conn.execute("SELECT COUNT(*) FROM site_events").fetchone()[0] == 3


def test_large_backlog_goes_in_batches(site, central, monkeypatch):
    site_db, a, _ = site
    for h in range(7):
        site_db.record_scan(a, "pin", now=NOW + timedelta(hours=h + 1))
    monkeypatch.setattr(config, "SYNC_BATCH", 4)
    calls = []
    assert client_for(site_db, central[0], calls=calls).push_once() == 10
    assert [len(c["events"]) for c in calls] == [4, 4, 2]


@pytest.mark.parametrize("site_id,token,code", [
    ("site-a", "wrong", 401),
    ("site-a", "token-b", 401),       # another site's token
    ("site-z", "token-a", 401),       # unknown site
])
def test_sync_auth(site, central, site_id, token, code):
    batch = {"site_id": site_id, **site[0].unsynced_batch(10)}
    r = central[0].post("/api/sync/events", json=batch, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == code
    assert central[1].site_summaries() == []


def test_sync_rejects_missing_token_and_bad_payload(site, central):
    batch = {"site_id": "site-a", **site[0].unsynced_batch(10)}
    assert central[0].post("/api/sync/events", json=batch).status_code == 401
    batch["events"][0]["kind"] = "teleport"
    r = central[0].post("/api/sync/events", json=batch, headers={"Authorization": "Bearer token-a"})
    assert r.status_code == 422


def test_server_without_site_tokens_refuses_sync(site, central, monkeypatch):
    monkeypatch.setattr(config, "SITE_TOKENS", {})
    batch = {"site_id": "site-a", **site[0].unsynced_batch(10)}
    assert central[0].post("/api/sync/events", json=batch, headers={"Authorization": "Bearer x"}).status_code == 404


def test_tokens_only_over_https():
    assert sync.secure_url("https://central.example.com")
    assert sync.secure_url("http://localhost:8000")
    assert not sync.secure_url("http://central.example.com")
    with pytest.raises(ValueError):
        sync.SyncClient(None, None, central_url="http://central.example.com")


def test_status_endpoints(site, central):
    central_client, central_db = central
    assert central_client.get("/api/sync/status", auth=("foreman", "pw")).json() == {"enabled": False, "pending": 0}
    client_for(site[0], central_client).push_once()
    sites = central_client.get("/api/sites", auth=("foreman", "pw")).json()
    assert sites[0]["id"] == "site-a" and sites[0]["minutes_since_sync"] < 1
    assert central_client.get("/api/sites").status_code == 401


def test_old_database_gets_sync_column(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE events (id INTEGER PRIMARY KEY, worker_id INTEGER, kind TEXT,
                    method TEXT, similarity REAL, ts TEXT)""")
    conn.execute("INSERT INTO events VALUES (1, 1, 'in', 'pin', NULL, '2026-01-01 07:00:00')")
    conn.commit(); conn.close()
    db = Database(path)
    assert db.unsynced_count() == 1      # existing history gets synced too
