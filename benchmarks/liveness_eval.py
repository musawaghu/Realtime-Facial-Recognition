"""Does a turned head still match the person's enrollment? (liveness identity check)

During the head-turn challenge the kiosk re-checks identity on the TURNED face, using the
looser LFW threshold at 1% false accepts. This measures how often a genuinely turned face
(|yaw| >= LIVENESS_TURN_YAW) still matches 3 frontal photos of the same person, i.e. how often
an honest worker would fail the challenge because their turned face stopped matching.

Also reports the yaw distribution, which the frontal/turn thresholds were chosen from.

Run:  python benchmarks/lfw_eval.py first (downloads LFW), then python benchmarks/liveness_eval.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config, liveness  # noqa: E402
from app.face_engine import FaceEngine  # noqa: E402

LFW = ROOT / "benchmarks" / "data" / "lfw"
RESULTS = ROOT / "benchmarks" / "results"


def main():
    if not LFW.exists():
        sys.exit("Run benchmarks/lfw_eval.py first (it downloads LFW).")
    engine = FaceEngine()
    people = [p for p in sorted(LFW.iterdir()) if len(list(p.glob("*.jpg"))) >= 4]
    print(f"{len(people)} people with 4+ photos; embedding ...")

    # yaw + embedding for every photo of those people (subject = face nearest the center)
    data = defaultdict(list)
    for k, person in enumerate(people):
        for path in sorted(person.glob("*.jpg")):
            rgb = np.asarray(Image.open(path).convert("RGB"))
            boxes, _, lms = engine.detect(rgb)
            if not len(boxes):
                continue
            i = int(np.argmin(np.linalg.norm((boxes[:, :2] + boxes[:, 2:]) / 2 - 125, axis=1)))
            emb = engine.encode(rgb, boxes[[i]])[0]
            data[person.name].append((liveness.yaw(lms[i]), emb))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(people)} people")

    all_yaw = np.array([y for photos in data.values() for y, _ in photos])
    sims = []
    for photos in data.values():
        frontal = [e for y, e in photos if abs(y) <= config.LIVENESS_FRONTAL_YAW]
        turned = [e for y, e in photos if abs(y) >= config.LIVENESS_TURN_YAW]
        if len(frontal) < 3:
            continue
        enrolled = np.stack(frontal[:3])          # like the kiosk: several frontal-ish captures
        sims += [float(np.max(enrolled @ t)) for t in turned]
    sims = np.array(sims)

    result = {
        "photos": int(len(all_yaw)),
        "abs_yaw_percentiles_50_90_95_99": np.percentile(np.abs(all_yaw), [50, 90, 95, 99]).round(3).tolist(),
        "frontal_yaw_max": config.LIVENESS_FRONTAL_YAW,
        "turn_yaw_min": config.LIVENESS_TURN_YAW,
        "share_of_photos_frontal": round(float(np.mean(np.abs(all_yaw) <= config.LIVENESS_FRONTAL_YAW)), 3),
        "turned_faces_tested": int(len(sims)),
        "turned_match_rate_at_liveness_threshold": round(float(np.mean(sims >= config.LIVENESS_MATCH_THRESHOLD)), 4),
        "liveness_match_threshold": config.LIVENESS_MATCH_THRESHOLD,
        "turned_match_rate_at_strict_threshold": round(float(np.mean(sims >= config.MATCH_THRESHOLD)), 4),
        "strict_threshold": config.MATCH_THRESHOLD,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "liveness.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
