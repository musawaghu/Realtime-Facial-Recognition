"""Train the PPE detector (hard hat + hi-vis vest) and measure it.

Fine-tunes YOLO11n (Ultralytics' smallest current model, pretrained on COCO) on the
Construction-PPE dataset: 1,416 real jobsite photos, labeled helmet, vest, Person, no_helmet and
more. The dataset downloads from Ultralytics' GitHub releases into benchmarks/data/.

Writes:
  models/ppe_yolo11n.pt           the trained weights the kiosk loads (git-ignored)
  benchmarks/results/ppe.json     precision / recall / mAP on the held-out TEST split

Run:  python scripts/train_ppe.py [--epochs 40]
      python scripts/train_ppe.py --eval-only      # re-measure existing weights
CPU training takes a few hours; it can be stopped and resumed with --resume.

License note: Ultralytics and the Construction-PPE dataset are AGPL-3.0.
"""
import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "benchmarks" / "data"
RUNS = DATA / "ppe_runs"
WEIGHTS = ROOT / "models" / "ppe_yolo11n.pt"
RESULTS = ROOT / "benchmarks" / "results" / "ppe.json"
CLASSES_WE_USE = ["helmet", "no_helmet", "vest", "Person"]


def dataset_yaml() -> Path:
    """The bundled Construction-PPE config, pointed at benchmarks/data instead of ~/datasets."""
    import ultralytics
    import yaml
    src = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "construction-ppe.yaml"
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    cfg["path"] = str(DATA / "construction-ppe")
    if not (DATA / "construction-ppe" / "images").exists():
        # download here ourselves; Ultralytics' auto-download ignores `path` and uses ~/../datasets
        from ultralytics.utils.downloads import download
        download(cfg["download"], dir=DATA)
    cfg.pop("download")
    out = DATA / "construction-ppe.yaml"
    DATA.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def train(epochs: int, resume: bool):
    from ultralytics import YOLO
    data = dataset_yaml()
    last = RUNS / "train" / "weights" / "last.pt"
    base = ROOT / "models" / "yolo11n.pt"   # official COCO-pretrained weights, fetched into models/
    if not resume and not base.exists():
        from ultralytics.utils.downloads import attempt_download_asset
        base.parent.mkdir(exist_ok=True)
        attempt_download_asset(str(base))
    model = YOLO(str(last) if resume else str(base))
    model.train(data=str(data), epochs=epochs, imgsz=640, batch=16, device="cpu", workers=4,
                patience=15, project=str(RUNS), name="train", exist_ok=True, resume=resume, seed=0,
                plots=False)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(RUNS / "train" / "weights" / "best.pt", WEIGHTS)
    print(f"Saved {WEIGHTS}")


def evaluate():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(dataset_yaml()), split="test", imgsz=640, device="cpu",
                        project=str(RUNS), name="test", exist_ok=True, plots=False)
    names = model.names
    per_class = {}
    for i, c in enumerate(metrics.box.ap_class_index):
        name = names[int(c)]
        if name in CLASSES_WE_USE:
            p, r, ap50, ap = metrics.box.class_result(i)
            per_class[name] = {"precision": round(float(p), 3), "recall": round(float(r), 3),
                               "mAP50": round(float(ap50), 3), "mAP50_95": round(float(ap), 3)}

    # speed on a kiosk-sized frame (480x360)
    frame = np.zeros((360, 480, 3), np.uint8)
    model.predict(frame, imgsz=480, device="cpu", verbose=False)       # warm-up
    t = time.perf_counter()
    for _ in range(20):
        model.predict(frame, imgsz=480, device="cpu", verbose=False)
    ms = (time.perf_counter() - t) / 20 * 1000

    result = {
        "model": "YOLO11n fine-tuned on Construction-PPE (Ultralytics 8.3.253)",
        "split": "test (141 images, never used in training)",
        "classes": per_class,
        "all_classes_mAP50": round(float(metrics.box.map50), 3),
        "inference_ms_480x360_cpu": round(ms, 1),
        "cpu": platform.processor(),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if not args.eval_only:
        train(args.epochs, args.resume)
    if not WEIGHTS.exists():
        sys.exit(f"No weights at {WEIGHTS}; train first")
    evaluate()


if __name__ == "__main__":
    main()
