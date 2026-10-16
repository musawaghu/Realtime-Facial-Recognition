"""Offline-first sync: each jobsite runs its own server on the kiosk laptop, so check-in, the PIN
fallback and the emergency muster keep working with no internet. When a connection is available,
attendance is pushed to a central server so the office sees every site.

What leaves the site: event rows (who, in/out, how, PPE seen, when) and each worker's name and
trade. Never face embeddings or PINs; those stay on the site's machine.

Delivery is at-least-once: events are marked synced only after the central server confirms them,
and the central server ignores events it already has, so a push cut off mid-way just retries.

Configure a site server with SITE_ID, CENTRAL_URL and SYNC_TOKEN, and the central server with
SITE_TOKENS="site-a:token-a,site-b:token-b" (see .env.example).
"""
import logging
from datetime import datetime
from urllib.parse import urlparse

from app import config

log = logging.getLogger("jobsite.sync")


def secure_url(url: str) -> bool:
    """Tokens only travel over HTTPS (plain HTTP is allowed to localhost for development)."""
    u = urlparse(url)
    return u.scheme == "https" or (u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1"))


class SyncClient:
    def __init__(self, db, post, central_url: str = config.CENTRAL_URL, site_id: str = config.SITE_ID,
                 token: str = config.SYNC_TOKEN):
        """`post(url, json, headers)` returns a response with .status_code and .json(), e.g. httpx.post."""
        if not secure_url(central_url):
            raise ValueError("CENTRAL_URL must be https:// (http:// only for localhost)")
        self.db, self.post = db, post
        self.url = central_url.rstrip("/") + "/api/sync/events"
        self.site_id, self.token = site_id, token
        self.last_success: str | None = None
        self.last_error: str | None = None
        self.failures = 0

    def push_once(self) -> int:
        """Send everything pending, batch by batch. Returns how many events were confirmed.
        Network or server errors are recorded and the events stay queued for the next try."""
        sent = 0
        while True:
            batch = self.db.unsynced_batch(config.SYNC_BATCH)
            if not batch["events"]:
                break
            try:
                r = self.post(self.url, json={"site_id": self.site_id, **batch},
                              headers={"Authorization": f"Bearer {self.token}"})
                if r.status_code != 200:
                    raise RuntimeError(f"central server answered {r.status_code}")
                accepted = r.json()["accepted"]
            except Exception as e:   # offline, DNS, timeout, 5xx: keep the queue, try later
                self.failures += 1
                self.last_error = f"{datetime.now():%Y-%m-%d %H:%M:%S} {type(e).__name__}: {e}"
                log.warning("sync failed (%s pending): %s", self.db.unsynced_count(), e)
                break
            self.db.mark_synced(accepted)
            sent += len(accepted)
            self.failures = 0
            self.last_success = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
            if len(accepted) < len(batch["events"]):
                break   # partial acceptance: don't spin, try the rest next round
        return sent

    def next_delay(self) -> float:
        """Every SYNC_INTERVAL_S normally; back off (up to 10 min) while the link is down."""
        return min(config.SYNC_INTERVAL_S * 2 ** min(self.failures, 5), 600)

    def status(self) -> dict:
        return {"enabled": True, "site_id": self.site_id, "pending": self.db.unsynced_count(),
                "last_success": self.last_success, "last_error": self.last_error}

