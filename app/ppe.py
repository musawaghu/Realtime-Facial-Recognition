"""PPE check at the kiosk: is the person checking in wearing a hard hat and a hi-vis vest?

A YOLO11n detector (trained by scripts/train_ppe.py on the Construction-PPE dataset) finds
helmets, vests and "no_helmet" heads anywhere in the frame. `assess` then decides which of
those belong to the person checking in, using their face box from the recognition step:

  hard hat  a helmet box sitting on top of their face  -> True
            a no_helmet box on their head, or no helmet -> False
  vest      a vest box on the torso below their face   -> True
            torso visible but no vest                  -> False
            camera doesn't show enough below the chin  -> None (unknown, never "missing")

Policy (decided with the site owner): missing PPE never blocks check-in. The worker is
warned on screen and the foreman sees it on the dashboard and in the audit log.
"""
from dataclasses import dataclass

from app import config


@dataclass
class Detection:
    name: str          # class name, e.g. "helmet", "vest", "no_helmet"
    conf: float
    box: list[float]   # [x1, y1, x2, y2]


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def assess(detections: list[Detection], face_box, frame_height: int) -> dict:
    """Which PPE does the person with this face wear? Returns {"hard_hat": bool, "vest": bool|None}."""
    fx1, fy1, fx2, fy2 = face_box
    fw, fh = fx2 - fx1, fy2 - fy1
    fcx = (fx1 + fx2) / 2

    def on_head(box):
        # centered over the face, bottom edge between ~0.8 face-heights above the face top
        # and the eyes (a helmet's brim sits at the forehead)
        cx, _ = _center(box)
        return abs(cx - fcx) < 0.6 * fw and fy1 - 0.8 * fh < box[3] < fy1 + 0.6 * fh

    def on_torso(box):
        # centered under the face, top edge between the chin and ~1.5 face-heights below it
        cx, _ = _center(box)
        return abs(cx - fcx) < 1.5 * fw and fy2 - 0.3 * fh < box[1] < fy2 + 1.5 * fh

    def best(name, where):
        scores = [d.conf for d in detections if d.name == name and d.conf >= config.PPE_CONF and where(d.box)]
        return max(scores, default=0.0)

    helmet, bare_head = best("helmet", on_head), best("no_helmet", lambda b: on_head(b) or _contains(b, face_box))
    hard_hat = helmet > bare_head

    torso_visible = frame_height - fy2 >= 1.2 * fh
    vest = best("vest", on_torso) > 0 if torso_visible else None
    return {"hard_hat": hard_hat, "vest": vest}


def _contains(outer, inner) -> bool:
    """outer box contains the center of inner box."""
    cx, cy = _center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


class PPEChecker:
    """Loads the trained detector once; `check` returns assess() for one frame."""

    def __init__(self, weights=config.PPE_WEIGHTS):
        import torch
        from ultralytics import YOLO   # heavy import, only when PPE is enabled
        self.model = YOLO(str(weights))
        self.device = "0" if torch.cuda.is_available() else "cpu"

    def detect(self, rgb) -> list[Detection]:
        # Ultralytics expects BGR numpy input
        result = self.model.predict(rgb[:, :, ::-1], imgsz=480, conf=config.PPE_CONF, device=self.device, verbose=False)[0]
        names = result.names
        return [Detection(names[int(c)], float(p), [float(v) for v in b])
                for b, p, c in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist())]

    def check(self, rgb, face_box) -> dict:
        return assess(self.detect(rgb), face_box, rgb.shape[0])
