"""How much does a hard hat hurt recognition?

  python benchmarks/hardhat_eval.py synthetic
      PROXY. LFW's 6,000 pairs with a drawn-on hard hat (brim + shadow, placed from each face's
      landmarks). Compares: no hat on either photo / enrolled without, checking in with a hat /
      hat on both. Needs lfw_eval.py to have downloaded LFW.

  python benchmarks/hardhat_eval.py real
      REAL. Embeddings of consenting people captured with and without a hard hat by
      scripts/capture_hardhat.py. Small sample, but real hats, light and posture.

Both report, at the threshold the kiosk actually uses (config.MATCH_THRESHOLD), how often a
genuine check-in is accepted and how many impostor comparisons are falsely accepted, plus how
often a face is detected at all and how often it reads as facing the camera (liveness needs that).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import config, liveness  # noqa: E402
from benchmarks.hardhat import draw_hard_hat, real_report  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"
CAPTURES = ROOT / "benchmarks" / "data" / "hardhat"


def nearest_center(boxes, size=250):
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    return int(np.argmin(np.linalg.norm(centers - size / 2, axis=1)))


def condition_metrics(scores, labels, folds, threshold):
    from benchmarks.lfw_eval import best_threshold, tar_at_far
    fold_acc = []
    for f in range(10):
        t = best_threshold(scores[folds != f], labels[folds != f])
        fold_acc.append(((scores[folds == f] >= t) == labels[folds == f]).mean())
    genuine, impostor = scores[labels == 1], scores[labels == 0]
    return {
        "accuracy_10fold": round(float(np.mean(fold_acc)), 4),
        "accepted_rate_at_app_threshold": round(float(np.mean(genuine >= threshold)), 4),
        "false_accepts_at_app_threshold": int(np.sum(impostor >= threshold)),
        "impostor_pairs": int(len(impostor)),
        "tar_at_far_0.1": round(tar_at_far(scores, labels, 1e-3)[0], 4),
    }


def synthetic():
    from app.face_engine import FaceEngine
    from benchmarks.lfw_eval import DATA, read_pairs
    if not (DATA / "lfw").exists():
        sys.exit("Run benchmarks/lfw_eval.py first (it downloads LFW).")
    pairs, labels, folds = read_pairs()
    engine = FaceEngine()
    unique = sorted({p for pair in pairs for p in pair})
    print(f"Embedding {len(unique)} images with and without a hard hat ...")

    bare, hat = {}, {}
    stats = {"bare_detected": 0, "hat_detected": 0, "bare_frontal": 0, "hat_frontal": 0}
    for k, path in enumerate(unique):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        boxes, _, lms = engine.detect(rgb)
        if not len(boxes):   # can't place a hat without landmarks; same fallback as lfw_eval
            box = np.array([[75.0, 70.0, 175.0, 190.0]])
            bare[path] = hat[path] = engine.encode(rgb, box)[0]
            continue
        i = nearest_center(boxes)
        stats["bare_detected"] += 1
        stats["bare_frontal"] += liveness.is_frontal(lms[i])
        bare[path] = engine.encode(rgb, boxes[[i]])[0]

        hatted = draw_hard_hat(rgb, lms[i])
        hboxes, _, hlms = engine.detect(hatted)
        if len(hboxes):
            j = nearest_center(hboxes)
            stats["hat_detected"] += 1
            stats["hat_frontal"] += liveness.is_frontal(hlms[j])
            hat[path] = engine.encode(hatted, hboxes[[j]])[0]
        else:                # detector missed the hatted face: embed the original box anyway
            hat[path] = engine.encode(hatted, boxes[[i]])[0]
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(unique)}")

    t = config.MATCH_THRESHOLD
    conditions = {
        "no_hat": [float(bare[a] @ bare[b]) for a, b in pairs],
        "enrolled_bare_checkin_with_hat": [float(bare[a] @ hat[b]) for a, b in pairs],
        "hat_on_both": [float(hat[a] @ hat[b]) for a, b in pairs],
    }
    n = stats["bare_detected"]
    result = {
        "kind": "SYNTHETIC PROXY: LFW with a drawn hard hat",
        "app_threshold": t,
        "images": len(unique),
        "face_detected_rate": {"no_hat": round(n / len(unique), 4), "hat": round(stats["hat_detected"] / max(n, 1), 4)},
        "reads_as_frontal_rate": {"no_hat": round(stats["bare_frontal"] / max(n, 1), 4),
                                  "hat": round(stats["hat_frontal"] / max(stats["hat_detected"], 1), 4)},
        "conditions": {name: condition_metrics(np.array(s), labels, folds, t) for name, s in conditions.items()},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hardhat_synthetic.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def real():
    files = sorted(CAPTURES.glob("*.npz"))
    if len(files) < 2:
        sys.exit(f"Need captures from at least 2 people in {CAPTURES} (scripts/capture_hardhat.py).")
    participants, capture_stats = {}, {}
    for f in files:
        z = np.load(f)
        conds = sorted({k.split("__")[0] for k in z.files if k.endswith("__emb")})
        participants[f.stem] = {c: z[f"{c}__emb"] for c in conds}
        for c in conds:
            s = capture_stats.setdefault(c, {"frames": 0, "detected": 0, "frontal": 0})
            s["frames"] += int(z[f"{c}__frames"])
            s["detected"] += len(z[f"{c}__emb"])
            s["frontal"] += int(np.sum(np.abs(z[f"{c}__yaw"]) <= config.LIVENESS_FRONTAL_YAW))
    result = {
        "kind": "REAL: consenting people captured with and without a hard hat",
        "app_threshold": config.MATCH_THRESHOLD,
        "people": len(files),
        "face_detected_rate": {c: round(s["detected"] / max(s["frames"], 1), 4) for c, s in capture_stats.items()},
        "reads_as_frontal_rate": {c: round(s["frontal"] / max(s["detected"], 1), 4) for c, s in capture_stats.items()},
        "conditions": real_report(participants, config.MATCH_THRESHOLD),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hardhat_real.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "synthetic":
        synthetic()
    elif mode == "real":
        real()
    else:
        sys.exit("usage: python benchmarks/hardhat_eval.py synthetic|real")
