"""API tests. The face models are replaced by a fake engine whose 'detected faces' we choose,
so these run in seconds and test the check-in logic, auth, validation and privacy rules."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api, config
from app.db import Database
from app.face_engine import DetectedFace, Gallery

AUTH = ("foreman", "test-pass")


def face_vec(person: int, jitter: float = 0.0) -> np.ndarray:
    """A synthetic 'face embedding' for person N (a basis vector, optionally perturbed)."""
    v = np.zeros(512, dtype=np.float32)
    v[person] = 1
    v[100 + person] = jitter
    return v / np.linalg.norm(v)


def landmarks(x0, size, yaw):
    """5 MTCNN-style points for a face box; nose shifted sideways by `yaw` eye-widths."""
    eye_w = 0.3 * size
    return np.array([[x0 + 0.35 * size, 0.4 * size], [x0 + 0.65 * size, 0.4 * size],
                     [x0 + 0.5 * size + yaw * eye_w, 0.55 * size],
                     [x0 + 0.38 * size, 0.75 * size], [x0 + 0.62 * size, 0.75 * size]])


class FakeEngine:
    def __init__(self):
        self.gallery = Gallery(threshold=0.5)
        self.next_faces: list[DetectedFace] = []

    def show(self, *vecs, sizes=None, yaw=0.0, offset=0):
        """Make the 'camera' see these faces on the next frame, with the head turned by `yaw`."""
        sizes = sizes or [100] * len(vecs)
        self.next_faces = [DetectedFace([offset, 0, offset + s, s], 0.99, landmarks(offset, s, yaw), v)
                           for v, s in zip(vecs, sizes)]

    def analyze(self, rgb):
        return self.next_faces


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "test-pass")
    db, engine = Database(str(tmp_path / "t.db")), FakeEngine()
    api.app.dependency_overrides = {api.get_db: lambda: db, api.get_engine: lambda: engine,
                                    api.get_ppe: lambda: None}   # PPE off unless a test turns it on
    # fresh rate limiters per test
    monkeypatch.setattr(api.scan_limit, "buckets", {})
    monkeypatch.setattr(api.pin_limit, "buckets", {})
    yield TestClient(api.app), db, engine
    api.app.dependency_overrides = {}


def jpeg() -> dict:
    ok, buf = cv2.imencode(".jpg", np.zeros((64, 64, 3), np.uint8))
    return {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}


def scan(client, challenge=None) -> dict:
    data = {"challenge": challenge} if challenge else None
    return client.post("/api/scan", files=jpeg(), data=data).json()


def face_checkin(client, engine, vec) -> dict:
    """Walk up facing the camera, then turn the way the kiosk asks."""
    engine.show(vec)
    r = scan(client)
    assert r["status"] == "turn", r
    engine.show(vec, yaw=0.5 if r["direction"] == "left" else -0.5)
    return scan(client, r["challenge"])


def enroll(client, engine, name="J. Rivera", person=1, pin="1234") -> int:
    r = client.post("/api/enroll/start", json={"name": name, "trade": "Carpenter", "pin": pin, "consent": True}, auth=AUTH)
    wid = r.json()["worker_id"]
    for j in (0.0, 0.2, 0.4):
        engine.show(face_vec(person, j))
        assert client.post(f"/api/enroll/{wid}/capture", files=jpeg(), auth=AUTH).json()["status"] == "captured"
    assert client.post(f"/api/enroll/{wid}/finish", auth=AUTH).json()["status"] == "enrolled"
    return wid


# ---------- auth ----------
@pytest.mark.parametrize("method,path", [
    ("get", "/dashboard"), ("get", "/api/onsite"), ("get", "/api/hours.csv"), ("post", "/api/muster"),
    ("delete", "/api/workers/1"), ("post", "/api/enroll/start"), ("get", "/api/audit")])
def test_foreman_routes_need_password(ctx, method, path):
    client, _, _ = ctx
    assert getattr(client, method)(path).status_code == 401
    assert getattr(client, method)(path, auth=("foreman", "wrong")).status_code == 401


def test_dashboard_disabled_without_admin_password(ctx, monkeypatch):
    client, _, _ = ctx
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "")
    assert client.get("/api/onsite", auth=("foreman", "")).status_code == 503


# ---------- enrollment ----------
def test_enrollment_requires_consent(ctx):
    client, db, _ = ctx
    r = client.post("/api/enroll/start", json={"name": "A", "pin": "1234", "consent": False}, auth=AUTH)
    assert r.status_code == 400 and db.list_workers() == []


@pytest.mark.parametrize("body", [
    {"name": "", "pin": "1234", "consent": True},
    {"name": "x" * 61, "pin": "1234", "consent": True},
    {"name": "A", "pin": "12", "consent": True},
    {"name": "A", "pin": "12ab", "consent": True},
    {"name": "A\x00B", "pin": "1234", "consent": True}])
def test_enrollment_validates_input(ctx, body):
    client, _, _ = ctx
    assert client.post("/api/enroll/start", json=body, auth=AUTH).status_code == 422


def test_enrollment_capture_rules(ctx):
    client, db, engine = ctx
    wid = client.post("/api/enroll/start", json={"name": "A", "pin": "1234", "consent": True}, auth=AUTH).json()["worker_id"]
    engine.show()
    assert client.post(f"/api/enroll/{wid}/capture", files=jpeg(), auth=AUTH).json()["status"] == "no_face"
    engine.show(face_vec(1), face_vec(2))
    assert client.post(f"/api/enroll/{wid}/capture", files=jpeg(), auth=AUTH).json()["status"] == "multiple"
    engine.show(face_vec(1))
    assert client.post(f"/api/enroll/{wid}/capture", files=jpeg(), auth=AUTH).json()["captures"] == 1
    engine.show(face_vec(2))   # someone else stepped in mid-enrollment
    assert client.post(f"/api/enroll/{wid}/capture", files=jpeg(), auth=AUTH).json()["status"] == "mismatch"
    assert client.post(f"/api/enroll/{wid}/finish", auth=AUTH).status_code == 409   # only 1 capture
    assert len(engine.gallery) == 0


def test_bad_upload_rejected(ctx):
    client, _, _ = ctx
    r = client.post("/api/scan", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_duplicate_enrollment_is_flagged(ctx):
    client, _, engine = ctx
    enroll(client, engine, "A", person=1)
    r = client.post("/api/enroll/start", json={"name": "A again", "pin": "9999", "consent": True}, auth=AUTH)
    wid = r.json()["worker_id"]
    for j in (0.1, 0.3, 0.5):
        engine.show(face_vec(1, j))
        client.post(f"/api/enroll/{wid}/capture", files=jpeg(), auth=AUTH)
    assert client.post(f"/api/enroll/{wid}/finish", auth=AUTH).json()["possible_duplicate_of"] == "A"


# ---------- kiosk ----------
def test_face_checkin_checkout_flow(ctx, monkeypatch):
    client, db, engine = ctx
    enroll(client, engine)
    r = face_checkin(client, engine, face_vec(1, 0.1))
    assert r["status"] == "checked_in" and r["name"] == "J. Rivera"
    engine.show(face_vec(1, 0.1))
    assert scan(client)["status"] == "cooldown"     # no head turn needed just to be told that
    assert [w["name"] for w in client.get("/api/onsite", auth=AUTH).json()] == ["J. Rivera"]

    monkeypatch.setattr(config, "EVENT_COOLDOWN_S", 0)
    assert face_checkin(client, engine, face_vec(1, 0.1))["status"] == "checked_out"
    assert client.get("/api/onsite", auth=AUTH).json() == []


def test_unknown_and_no_face(ctx):
    client, _, engine = ctx
    enroll(client, engine)
    engine.show(face_vec(7))
    assert client.post("/api/scan", files=jpeg()).json()["status"] == "unknown"
    engine.show()
    assert client.post("/api/scan", files=jpeg()).json()["status"] == "no_face"


def test_closest_face_is_the_one_checked_in(ctx):
    client, _, engine = ctx
    enroll(client, engine, "Near", person=1)
    enroll(client, engine, "Far", person=2)
    engine.show(face_vec(2), face_vec(1), sizes=[60, 200])
    assert client.post("/api/scan", files=jpeg()).json()["name"] == "Near"


def test_pin_fallback_and_lockout(ctx):
    client, _, engine = ctx
    wid = enroll(client, engine, pin="4821")
    assert client.post("/api/pin", json={"worker_id": wid, "pin": "0000"}).json()["status"] == "bad_pin"
    r = client.post("/api/pin", json={"worker_id": wid, "pin": "4821"}).json()
    assert r["status"] == "checked_in" and r["name"] == "J. Rivera"
    for _ in range(config.PIN_MAX_FAILURES - 1):
        client.post("/api/pin", json={"worker_id": wid, "pin": "0000"})
    # the 5th request per device hits the rate limit before the lockout can be reached here
    statuses = {client.post("/api/pin", json={"worker_id": wid, "pin": "0000"}).status_code for _ in range(3)}
    assert 429 in statuses


def test_pin_lockout_via_api(ctx, monkeypatch):
    client, _, engine = ctx
    monkeypatch.setattr(api.pin_limit, "burst", 100)
    wid = enroll(client, engine, pin="4821")
    for _ in range(config.PIN_MAX_FAILURES - 1):
        assert client.post("/api/pin", json={"worker_id": wid, "pin": "0000"}).json()["status"] == "bad_pin"
    assert client.post("/api/pin", json={"worker_id": wid, "pin": "0000"}).json()["status"] == "locked"
    assert client.post("/api/pin", json={"worker_id": wid, "pin": "4821"}).json()["status"] == "locked"


def test_scan_rate_limited(ctx):
    client, _, engine = ctx
    engine.show()
    codes = [client.post("/api/scan", files=jpeg()).status_code for _ in range(15)]
    assert codes[:10] == [200] * 10 and 429 in codes


# ---------- privacy ----------
def test_delete_my_data_stops_recognition(ctx):
    client, db, engine = ctx
    wid = enroll(client, engine)
    assert face_checkin(client, engine, face_vec(1))["status"] == "checked_in"
    assert client.delete(f"/api/workers/{wid}", auth=AUTH).json()["status"] == "deleted"
    engine.show(face_vec(1))
    assert client.post("/api/scan", files=jpeg()).json()["status"] == "unknown"
    assert client.post("/api/pin", json={"worker_id": wid, "pin": "1234"}).json()["status"] == "bad_pin"
    assert client.delete(f"/api/workers/{wid}", auth=AUTH).status_code == 404
    # attendance is still there for payroll
    assert client.get("/api/hours", auth=AUTH).json()[0]["name"] == "J. Rivera"


# ---------- dashboard ----------
def test_csv_export_escapes_formulas(ctx):
    client, db, engine = ctx
    wid = enroll(client, engine, name="=HYPERLINK(evil)")
    db.record_scan(wid, "pin")
    r = client.get("/api/hours.csv", auth=AUTH)
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert "'=HYPERLINK(evil)" in r.text
    assert db.list_audit()[0]["action"] == "export_hours"


def test_csv_safe():
    assert api.csv_safe("=1+1") == "'=1+1"
    assert api.csv_safe("@SUM") == "'@SUM"
    assert api.csv_safe("J. Rivera") == "J. Rivera"


def test_invalid_date_range(ctx):
    client, _, _ = ctx
    assert client.get("/api/hours?start=2026-10-12&end=2026-10-01", auth=AUTH).status_code == 422


def test_muster_flow(ctx):
    client, db, engine = ctx
    a = enroll(client, engine, "A", person=1)
    enroll(client, engine, "B", person=2)
    db.record_scan(a, "pin")
    assert client.get("/api/muster/current", auth=AUTH).json() is None
    m = client.post("/api/muster", auth=AUTH).json()
    assert [e["name"] for e in m["entries"]] == ["A"] and m["missing"] == 1
    m = client.post(f"/api/muster/{m['id']}/account/{a}", auth=AUTH).json()
    assert m["missing"] == 0
    assert client.get("/api/muster/current", auth=AUTH).json()["id"] == m["id"]
    assert client.post(f"/api/muster/{m['id']}/close", auth=AUTH).json()["closed_at"]
    assert client.post(f"/api/muster/{m['id']}/account/{a}", auth=AUTH).status_code == 404


# ---------- liveness ----------
def turn_for(direction: str) -> float:
    return 0.5 if direction == "left" else -0.5


def test_recognition_alone_does_not_check_in(ctx):
    client, db, engine = ctx
    enroll(client, engine)
    engine.show(face_vec(1))
    for _ in range(5):                       # a photo held still in front of the camera
        r = scan(client)
        assert r["status"] == "turn" and r["direction"] in ("left", "right") and r["name"] == "J. Rivera"
        assert scan(client, r["challenge"])["status"] == "turn"
    assert db.on_site_now() == []


def test_turning_the_wrong_way_does_not_pass(ctx):
    client, db, engine = ctx
    enroll(client, engine)
    engine.show(face_vec(1))
    r = scan(client)
    engine.show(face_vec(1), yaw=-turn_for(r["direction"]))
    assert scan(client, r["challenge"])["status"] == "turn"
    assert db.on_site_now() == []


def test_already_turned_face_gets_no_challenge(ctx):
    client, _, engine = ctx
    enroll(client, engine)
    engine.show(face_vec(1), yaw=0.6)        # e.g. a photo of the worker already turned
    assert scan(client)["status"] == "look_straight"


def test_challenge_times_out(ctx, monkeypatch):
    client, db, engine = ctx
    enroll(client, engine)
    clock = [1000.0]
    monkeypatch.setattr(api.challenges, "clock", lambda: clock[0])
    engine.show(face_vec(1))
    r = scan(client)
    clock[0] += config.LIVENESS_TIMEOUT_S + 1
    engine.show(face_vec(1), yaw=turn_for(r["direction"]))
    assert scan(client, r["challenge"]) == {"status": "challenge_failed", "reason": "timeout"}
    assert db.on_site_now() == [] and db.list_audit()[0]["action"] == "liveness_fail"


@pytest.mark.parametrize("swap", ["different_person", "lost_face", "jumped_box"])
def test_challenge_fails_if_the_face_changes(ctx, swap):
    client, db, engine = ctx
    enroll(client, engine, "A", person=1)
    enroll(client, engine, "B", person=2)
    engine.show(face_vec(1))
    r = scan(client)
    turn = turn_for(r["direction"])
    if swap == "different_person":           # B steps in and does the turn for A
        engine.show(face_vec(2), yaw=turn)
    elif swap == "lost_face":                # photo pulled away
        engine.show()
    else:                                    # a second, turned photo held somewhere else
        engine.show(face_vec(1), yaw=turn, offset=300)
    assert scan(client, r["challenge"]) == {"status": "challenge_failed", "reason": swap.replace("jumped_box", "lost_face")}
    assert db.on_site_now() == []
    assert db.list_audit()[0]["action"] == "liveness_fail"


def test_challenge_token_is_single_use(ctx):
    client, _, engine = ctx
    enroll(client, engine)
    engine.show(face_vec(1))
    r = scan(client)
    engine.show(face_vec(1), yaw=turn_for(r["direction"]))
    assert scan(client, r["challenge"])["status"] == "checked_in"
    assert scan(client, r["challenge"]) == {"status": "challenge_failed", "reason": "unknown_challenge"}


# ---------- PPE ----------
class FakePPE:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def check(self, rgb, face_box):
        self.calls += 1
        return self.result


def test_missing_ppe_warns_and_logs_but_checks_in(ctx):
    client, db, engine = ctx
    fake = FakePPE({"hard_hat": False, "vest": True})
    api.app.dependency_overrides[api.get_ppe] = lambda: fake
    wid = enroll(client, engine)
    r = face_checkin(client, engine, face_vec(1))
    assert r["status"] == "checked_in" and r["ppe"] == {"hard_hat": False, "vest": True}
    assert fake.calls == 1                         # checked once, on the frontal frame
    onsite = client.get("/api/onsite", auth=AUTH).json()
    assert onsite[0]["hard_hat"] == 0 and onsite[0]["vest"] == 1
    audit = db.list_audit()[0]
    assert audit["action"] == "ppe_missing" and audit["worker_id"] == wid and audit["detail"] == "no hard hat"


def test_full_ppe_is_not_logged(ctx):
    client, db, engine = ctx
    api.app.dependency_overrides[api.get_ppe] = lambda: FakePPE({"hard_hat": True, "vest": None})
    enroll(client, engine)
    r = face_checkin(client, engine, face_vec(1))
    assert r["ppe"] == {"hard_hat": True, "vest": None}      # vest unknown: torso not in view
    assert all(a["action"] != "ppe_missing" for a in db.list_audit())


def test_checkout_without_ppe_is_not_flagged(ctx, monkeypatch):
    client, db, engine = ctx
    monkeypatch.setattr(config, "EVENT_COOLDOWN_S", 0)
    enroll(client, engine)
    face_checkin(client, engine, face_vec(1))                 # in, PPE not checked
    api.app.dependency_overrides[api.get_ppe] = lambda: FakePPE({"hard_hat": False, "vest": False})
    assert face_checkin(client, engine, face_vec(1))["status"] == "checked_out"
    assert all(a["action"] != "ppe_missing" for a in db.list_audit())   # hat off on the way out is fine


def test_without_detector_ppe_is_unknown(ctx):
    client, _, engine = ctx
    enroll(client, engine)
    assert face_checkin(client, engine, face_vec(1))["ppe"] is None
    assert client.get("/api/onsite", auth=AUTH).json()[0]["hard_hat"] is None
