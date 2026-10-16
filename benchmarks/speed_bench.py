"""Speed benchmark: where the time goes, and what batching / vectorizing buys.

Measures on 640x480 frames:
  - detection (MTCNN) ms
  - embedding ms for 4 faces: one-at-a-time vs one batched forward pass
  - matching ms against a 500-worker x 5-angle gallery: Python loop vs one matrix product
  - full pipeline FPS for 1 face, with detection on every frame vs every 3rd frame (desktop demo)
  - /api/scan round trip (in-process: JPEG decode + pipeline + DB write), what the kiosk waits for

Frames are built from LFW images (run lfw_eval.py first), so no webcam is needed.
Pass --camera to time live webcam frames instead for the pipeline numbers.

Run:  python benchmarks/speed_bench.py [--camera]
"""
import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.face_engine import FaceEngine, Gallery  # noqa: E402

LFW = ROOT / "benchmarks" / "data" / "lfw"
RESULTS = ROOT / "benchmarks" / "results"


def timed(fn, repeats):
    fn()  # warm-up
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t) / repeats * 1000  # ms


def lfw_faces(n):
    people = sorted(p for p in LFW.iterdir() if p.is_dir())[:n]
    return [np.asarray(Image.open(next(p.glob("*.jpg"))).convert("RGB")) for p in people]


def frame_with(faces):
    """A 640x480 frame with LFW faces (scaled to 160px) side by side, like a camera view."""
    frame = np.full((480, 640, 3), 90, np.uint8)
    for i, f in enumerate(faces):
        small = cv2.resize(f, (160, 160))
        frame[160:320, i * 160:(i + 1) * 160] = small
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", action="store_true", help="use webcam frames for pipeline timing")
    args = parser.parse_args()
    if not LFW.exists():
        sys.exit("Run benchmarks/lfw_eval.py first (it downloads the LFW images used here).")

    engine = FaceEngine()
    faces = lfw_faces(4)
    one = frame_with(faces[:1])
    four = frame_with(faces)
    if args.camera:
        cap = cv2.VideoCapture(0)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            sys.exit("Could not read from the webcam")
        one = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB)

    r = {"device": engine.device, "cpu": platform.processor(), "frame": "640x480",
         "source": "webcam" if args.camera else "LFW faces composed into frames"}

    # Detection
    r["detect_ms"] = round(timed(lambda: engine.detect(one), 30), 1)

    # Embedding: per face vs batched
    boxes4, _, _ = engine.detect(four)
    r["faces_in_batch_frame"] = int(len(boxes4))
    r["encode_one_by_one_ms"] = round(timed(lambda: [engine.encode(four, boxes4[[i]]) for i in range(len(boxes4))], 20), 1)
    r["encode_batched_ms"] = round(timed(lambda: engine.encode(four, boxes4), 20), 1)

    # Matching: Python loop (old main.py) vs vectorized Gallery
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((2500, 512)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = np.repeat(np.arange(500), 5)
    probe = vecs[1234]
    gallery = Gallery(threshold=0.5)
    gallery.load(ids, vecs)

    def loop_match():
        best, best_id = -1.0, None
        for wid, v in zip(ids, vecs):
            s = float(np.dot(probe, v))
            if s > best:
                best, best_id = s, wid
        return best_id

    r["gallery"] = "500 workers x 5 angles"
    r["match_loop_ms"] = round(timed(loop_match, 20), 2)
    r["match_vectorized_ms"] = round(timed(lambda: gallery.match(probe), 200), 3)

    # Full pipeline, 1 face
    pipeline_ms = timed(lambda: engine.analyze(one), 30)
    r["pipeline_ms"] = round(pipeline_ms, 1)
    r["fps_detect_every_frame"] = round(1000 / pipeline_ms, 1)
    # every 3rd frame runs the models; other frames only draw (cost ~0)
    r["fps_detect_every_3rd_frame_model_bound"] = round(3000 / pipeline_ms, 1)

    # /api/scan round trip in-process
    from fastapi.testclient import TestClient
    from app import api
    from app.db import Database
    with tempfile.TemporaryDirectory() as tmp:
        db = Database(str(Path(tmp) / "bench.db"))
        api.app.dependency_overrides = {api.get_db: lambda: db, api.get_engine: lambda: engine}
        api.scan_limit.rate = api.scan_limit.burst = 10_000
        client = TestClient(api.app)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(cv2.resize(one, (480, 360)), cv2.COLOR_RGB2BGR))
        files = {"file": ("f.jpg", jpg.tobytes(), "image/jpeg")}
        r["api_scan_ms"] = round(timed(lambda: client.post("/api/scan", files=files), 20), 1)
        api.app.dependency_overrides = {}

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "speed.json").write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
