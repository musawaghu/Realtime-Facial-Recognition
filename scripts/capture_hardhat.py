"""Capture face embeddings with and without a hard hat, for benchmarks/hardhat_eval.py real.

    python scripts/capture_hardhat.py p01
    python scripts/capture_hardhat.py p02 --conditions nohat hat hat_outdoor

For each condition, the person looks at the webcam and moves their head a little while 30 frames
are taken. Like the app, only face embeddings are saved, never photos, to
benchmarks/data/hardhat/<id>.npz (git-ignored). Use an ID, not a name. Delete the file to
delete their data.

Tip: capture "nohat" first; the eval enrolls each person from their first 3 nohat captures.
For a realistic test, wear the hat the way you would on site (brim level, chin strap if any).
"""
import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import liveness  # noqa: E402
from app.face_engine import FaceEngine  # noqa: E402

OUT = ROOT / "benchmarks" / "data" / "hardhat"
FRAMES = 30
FRAME_GAP_S = 0.33
WINDOW = "Hard hat capture"

CONSENT = """
This captures face templates (numbers computed from your face, not photos) with and
without a hard hat, to measure how well check-in works for people wearing one. They are
stored only on this computer, used only for this test, and deleted on request.
"""

INSTRUCTIONS = {
    "nohat": "NO hard hat",
    "hat": "WEARING a hard hat",
}


def label(frame, lines):
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (12, 34 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(frame, text, (12, 34 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def capture_condition(engine, cap, condition):
    """Wait for SPACE, then take FRAMES frames. Returns (embeddings, yaws, frames tried) or None to quit."""
    what = INSTRUCTIONS.get(condition, condition)
    while True:
        ok, frame = cap.read()
        if not ok:
            sys.exit("Webcam read failed")
        label(frame, [f"Next: {what}", "Get ready, then press SPACE (Q to quit)"])
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return None

    embs, yaws = [], []
    for n in range(FRAMES):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _, lms = engine.detect(rgb)
        if len(boxes) == 1:   # exactly one face, or we can't be sure whose it is
            embs.append(engine.encode(rgb, boxes)[0])
            yaws.append(liveness.yaw(lms[0]))
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label(frame, [f"{what}: {n + 1}/{FRAMES}", "Look at the camera, move your head a little"])
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(1)
        time.sleep(FRAME_GAP_S)
    print(f"  {condition}: face found in {len(embs)}/{FRAMES} frames")
    return np.array(embs, dtype=np.float32).reshape(-1, 512), np.array(yaws, dtype=np.float32), FRAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="an ID like p01, not a name")
    parser.add_argument("--conditions", nargs="+", default=["nohat", "hat"])
    parser.add_argument("--replace", action="store_true", help="overwrite this participant's existing file")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", args.participant):
        sys.exit("Participant ID: 1-20 letters, digits, _ or - (use an ID, not a name)")
    if not all(re.fullmatch(r"[a-z0-9_]{1,20}", c) for c in args.conditions):
        sys.exit("Conditions: lowercase letters, digits and _")
    out = OUT / f"{args.participant}.npz"
    if out.exists() and not args.replace:
        sys.exit(f"{out} exists; pass --replace to overwrite")

    print(CONSENT)
    if input("Does the participant agree? (y/n): ").strip().lower() != "y":
        sys.exit("No consent: nothing captured")

    engine = FaceEngine()
    cap = cv2.VideoCapture(0)
    data = {"consent_at": np.array(datetime.now().isoformat(timespec="seconds"))}
    try:
        for condition in args.conditions:
            result = capture_condition(engine, cap, condition)
            if result is None:
                sys.exit("Quit: nothing saved")
            data[f"{condition}__emb"], data[f"{condition}__yaw"], data[f"{condition}__frames"] = (
                result[0], result[1], np.array(result[2]))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)
    print(f"Saved embeddings (no photos) to {out}")


if __name__ == "__main__":
    main()
