"""Face verification accuracy on LFW (Labeled Faces in the Wild), standard 10-fold protocol.

6,000 image pairs (3,000 same person, 3,000 different). Each image goes through the SAME
pipeline the kiosk uses (MTCNN detect -> crop -> FaceNet embedding), and a pair's score is
the cosine similarity of the two embeddings.

Reports 10-fold accuracy, ROC AUC, TAR at FAR 0.1% / 1%, and writes the threshold at
FAR 0.1% to benchmarks/results/lfw.json, which app/config.py uses as MATCH_THRESHOLD.

Run:  python benchmarks/lfw_eval.py
Data comes from scikit-learn's figshare mirror (the UMass host is often down); files are
checked against SHA-256 before use.
"""
import hashlib
import json
import platform
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.face_engine import FaceEngine  # noqa: E402

DATA = ROOT / "benchmarks" / "data"
RESULTS = ROOT / "benchmarks" / "results"

# (filename, url, sha256) — same mirror and checksums scikit-learn uses
FILES = [
    ("lfw.tgz", "https://ndownloader.figshare.com/files/5976018",
     "055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0"),
    ("pairs.txt", "https://ndownloader.figshare.com/files/5976006",
     "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url, checksum in FILES:
        path = DATA / name
        if not path.exists():
            print(f"Downloading {name} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except OSError as e:
                sys.exit(f"Download failed ({e}). Get {name} from {url} manually and put it in {DATA}")
        if sha256(path) != checksum:
            path.unlink()
            sys.exit(f"{name} failed its checksum; deleted. Re-run to download again.")
    if not (DATA / "lfw").exists():
        print("Extracting images ...")
        with tarfile.open(DATA / "lfw.tgz") as tar:
            # only plain files/folders that stay inside DATA (blocks path traversal)
            safe = [m for m in tar.getmembers() if (m.isfile() or m.isdir())
                    and (DATA / m.name).resolve().is_relative_to(DATA.resolve())]
            tar.extractall(DATA, members=safe)


def read_pairs() -> tuple[list[tuple[Path, Path]], np.ndarray, np.ndarray]:
    """Parse pairs.txt: 10 folds x (300 same-person lines, then 300 different-person lines)."""
    lines = (DATA / "pairs.txt").read_text().split("\n")
    n_folds, n_per = map(int, lines[0].split())
    img = lambda name, i: DATA / "lfw" / name / f"{name}_{int(i):04d}.jpg"
    pairs, labels, folds = [], [], []
    for row, line in enumerate(lines[1:1 + n_folds * n_per * 2]):
        p = line.split("\t")
        if len(p) == 3:
            pairs.append((img(p[0], p[1]), img(p[0], p[2])))
            labels.append(1)
        else:
            pairs.append((img(p[0], p[1]), img(p[2], p[3])))
            labels.append(0)
        folds.append(row // (n_per * 2))
    return pairs, np.array(labels), np.array(folds)


def embed_all(engine: FaceEngine, paths: list[Path]) -> tuple[dict, int]:
    """Embed each unique image. LFW images are 250x250 with the subject centered, so when
    several faces are found we take the one nearest the center. If none is found, we fall
    back to a fixed center box (counted and reported)."""
    embeddings, misses = {}, 0
    center = np.array([125, 125])
    start = time.time()
    for k, path in enumerate(paths):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        boxes, _, _ = engine.detect(rgb)
        if len(boxes):
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            box = boxes[[np.argmin(np.linalg.norm(centers - center, axis=1))]]
        else:
            misses += 1
            box = np.array([[75.0, 70.0, 175.0, 190.0]])
        embeddings[path] = engine.encode(rgb, box)[0]
        if (k + 1) % 500 == 0:
            rate = (k + 1) / (time.time() - start)
            print(f"  {k + 1}/{len(paths)} images ({rate:.1f}/s)")
    return embeddings, misses


def best_threshold(scores, labels):
    """Threshold that maximizes accuracy on the given pairs."""
    candidates = np.unique(scores)
    accs = [((scores >= t) == labels).mean() for t in candidates]
    return float(candidates[int(np.argmax(accs))])


def tar_at_far(scores, labels, far):
    """Lowest threshold whose false-accept rate is <= far, and the true-accept rate there."""
    fpr, tpr, thr = roc_curve(labels, scores)
    i = np.searchsorted(fpr, far, side="right") - 1
    return float(tpr[i]), float(thr[i])


def main():
    download()
    pairs, labels, folds = read_pairs()
    engine = FaceEngine()
    unique = sorted({p for pair in pairs for p in pair})
    print(f"Embedding {len(unique)} images on {engine.device} ...")
    t0 = time.time()
    emb, misses = embed_all(engine, unique)
    elapsed = time.time() - t0

    scores = np.array([float(emb[a] @ emb[b]) for a, b in pairs])

    # Standard protocol: pick the threshold on 9 folds, measure accuracy on the 10th
    fold_acc = []
    for f in range(10):
        t = best_threshold(scores[folds != f], labels[folds != f])
        fold_acc.append(((scores[folds == f] >= t) == labels[folds == f]).mean())
    tar_1e3, thr_1e3 = tar_at_far(scores, labels, 1e-3)
    tar_1e2, thr_1e2 = tar_at_far(scores, labels, 1e-2)

    result = {
        "dataset": "LFW original images, 10-fold pairs.txt (6000 pairs)",
        "accuracy_mean": round(float(np.mean(fold_acc)), 4),
        "accuracy_std": round(float(np.std(fold_acc)), 4),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 5),
        "tar_at_far_0.1": round(tar_1e3, 4),
        "threshold_far_0.1": round(thr_1e3, 4),
        "tar_at_far_1": round(tar_1e2, 4),
        "threshold_far_1": round(thr_1e2, 4),
        "impostor_pairs": int((labels == 0).sum()),
        "detection_misses": misses,
        "images": len(unique),
        "seconds_per_image": round(elapsed / len(unique), 4),
        "device": engine.device,
        "cpu": platform.processor(),
        "model": "MTCNN + InceptionResnetV1 (vggface2), facenet-pytorch 2.6",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lfw.json").write_text(json.dumps(result, indent=2))
    np.save(RESULTS / "lfw_scores.npy", np.stack([scores, labels, folds]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
