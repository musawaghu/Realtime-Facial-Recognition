import numpy as np

from app import config, liveness
from app.face_engine import DetectedFace, Gallery


def pts(nose_x, eye_y=40.0):
    """Eyes at x=35 and x=65 (eye width 30), nose at nose_x."""
    return np.array([[35, eye_y], [65, eye_y], [nose_x, 55], [38, 75], [62, 75]], dtype=np.float32)


def test_yaw_sign_and_scale():
    assert liveness.yaw(pts(50)) == 0
    assert liveness.yaw(pts(65)) == 0.5        # nose toward image right = turned to their own left
    assert np.isclose(liveness.yaw(pts(41)), -0.3)


def test_flat_photo_tilted_keeps_yaw():
    """Turning a flat photo about its vertical axis squeezes x uniformly: yaw doesn't change."""
    lm = pts(53)
    squeezed = lm.copy()
    squeezed[:, 0] = 50 + (lm[:, 0] - 50) * 0.6
    assert np.isclose(liveness.yaw(lm), liveness.yaw(squeezed))


def test_degenerate_landmarks():
    assert liveness.yaw(np.zeros((5, 2))) == 0.0


def test_frontal_threshold():
    assert liveness.is_frontal(pts(50 + 30 * config.LIVENESS_FRONTAL_YAW))
    assert not liveness.is_frontal(pts(50 + 30 * (config.LIVENESS_FRONTAL_YAW + 0.01)))


def test_continuity():
    assert liveness.continuous([0, 0, 100, 100], [20, 10, 125, 110])
    assert not liveness.continuous([0, 0, 100, 100], [60, 0, 160, 100])    # jumped sideways
    assert not liveness.continuous([0, 0, 100, 100], [0, 0, 50, 50])      # shrank a lot
    assert not liveness.continuous([0, 0, 100, 100], [0, 0, 200, 200])    # grew a lot


def test_step_passes_only_in_requested_direction():
    face = lambda nose_x: DetectedFace([0, 0, 100, 100], 0.99, pts(nose_x), np.zeros(512))
    ch = liveness.Challenge(1, liveness.LEFT, [0, 0, 100, 100], 0.8, 0.0)
    assert liveness.step(ch, face(50), True) == "turn"
    assert liveness.step(ch, face(35), True) == "turn"           # turned right instead
    assert liveness.step(ch, face(65), True) == "passed"         # yaw 0.5 to their left
    assert liveness.step(ch, face(65), False) == "different_person"
    assert liveness.step(ch, None, True) == "lost_face"


def test_store_expiry_and_cleanup():
    t = [0.0]
    store = liveness.ChallengeStore(clock=lambda: t[0])
    a = store.create(1, [0, 0, 10, 10], 0.9)
    assert store.take(a.token) == (a, False)
    t[0] = config.LIVENESS_TIMEOUT_S + 1
    assert store.take(a.token) == (a, True)
    assert store.take(a.token) == (None, False)
    b = store.create(2, [0, 0, 10, 10], 0.9)
    t[0] += config.LIVENESS_TIMEOUT_S + 1
    store.create(3, [0, 0, 10, 10], 0.9)          # creating purges expired challenges
    assert b.token not in store.open


def test_direction_is_random():
    store = liveness.ChallengeStore()
    assert {store.create(1, [0, 0, 1, 1], 1.0).direction for _ in range(64)} == {liveness.LEFT, liveness.RIGHT}


def test_gallery_score_per_worker():
    g = Gallery(threshold=0.5)
    v = np.eye(512, dtype=np.float32)
    g.load(np.array([5, 9, 5]), v[[0, 1, 2]])
    assert g.score(v[2], 5) == 1.0 and g.score(v[2], 9) == 0.0
    assert g.score(v[0], 7) == 0.0          # not enrolled
