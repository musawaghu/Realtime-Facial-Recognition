"""Head-turn liveness challenge: proves a live person is at the kiosk, not a photo.

After a worker is recognized facing the camera, the kiosk asks them to turn their head to
their LEFT or RIGHT (random). The server watches the following frames and passes the
challenge only if the SAME person, tracked without gaps, turns the requested way in time.

Why a photo fails: head turn is measured as how far the nose sits from the midpoint between
the eyes, in eye-widths ("yaw"). Tilting or rotating a flat photo squeezes eyes and nose
together, so that ratio barely changes; only a real 3-D head moves the nose sideways.

Limits: a video of the worker turning, or two prepared photos swapped quickly, can still pass.
Stopping those needs depth/IR cameras or a trained anti-spoofing model.
"""
import secrets
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app import config

LEFT, RIGHT = 1, -1   # sign of yaw when the person turns to THEIR left / right


def yaw(landmarks: np.ndarray) -> float:
    """Head turn from MTCNN's 5 landmarks (eyes, nose, mouth corners), in raw camera
    (unmirrored) coordinates. ~0 facing the camera; positive = turned to their own left."""
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_width = float(np.linalg.norm(right_eye - left_eye))
    if eye_width < 1e-6:
        return 0.0
    return float((nose[0] - (left_eye[0] + right_eye[0]) / 2) / eye_width)


def is_frontal(landmarks: np.ndarray) -> bool:
    return abs(yaw(landmarks)) <= config.LIVENESS_FRONTAL_YAW


def continuous(prev_box, box) -> bool:
    """Same face as the last frame: center moved < half a face width, size changed < 40%.
    Swapping a photo or a different person stepping in breaks this."""
    pw, bw = prev_box[2] - prev_box[0], box[2] - box[0]
    pc = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
    bc = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    return float(np.linalg.norm(bc - pc)) < 0.5 * pw and 0.6 < bw / max(pw, 1) < 1 / 0.6


@dataclass
class Challenge:
    worker_id: int
    direction: int            # LEFT or RIGHT
    box: list[int]            # last seen face box, for continuity
    similarity: float         # recognition score from the frontal frame
    started: float
    token: str = field(default_factory=lambda: secrets.token_urlsafe(16))

    @property
    def direction_name(self) -> str:
        return "left" if self.direction == LEFT else "right"


class ChallengeStore:
    """Open challenges, in memory (they live for a few seconds)."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.open: dict[str, Challenge] = {}
        self.lock = threading.Lock()

    def create(self, worker_id: int, box: list[int], similarity: float) -> Challenge:
        ch = Challenge(worker_id, secrets.choice([LEFT, RIGHT]), box, similarity, self.clock())
        with self.lock:
            # drop expired ones so abandoned challenges don't pile up
            now = self.clock()
            self.open = {t: c for t, c in self.open.items() if not self._expired(c, now)}
            self.open[ch.token] = ch
        return ch

    def _expired(self, ch: Challenge, now: float) -> bool:
        return now - ch.started > config.LIVENESS_TIMEOUT_S

    def take(self, token: str) -> tuple[Challenge | None, bool]:
        """Look up a challenge. Returns (challenge, expired). Expired ones are removed."""
        with self.lock:
            ch = self.open.get(token)
            if ch is None:
                return None, False
            if self._expired(ch, self.clock()):
                del self.open[token]
                return ch, True
            return ch, False

    def close(self, token: str):
        with self.lock:
            self.open.pop(token, None)


def step(ch: Challenge, face, identity_ok: bool) -> str:
    """Advance a challenge with the next frame's face (or None).

    Returns 'passed', 'turn' (keep going), 'lost_face' or 'different_person'.
    """
    if face is None or not continuous(ch.box, face.box):
        return "lost_face"
    if not identity_ok:
        return "different_person"
    ch.box = face.box
    if yaw(face.landmarks) * ch.direction >= config.LIVENESS_TURN_YAW:
        return "passed"
    return "turn"
