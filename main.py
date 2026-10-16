"""Desktop demo: live recognition in an OpenCV window, using the same engine and database as the kiosk.

    python main.py              # recognize
    python main.py --register   # enroll someone from the webcam
    python main.py --clear-db   # delete the local database
"""
import argparse
import time
from pathlib import Path

import cv2

from app import config
from app.db import Database
from app.face_engine import FaceEngine

CAMERA_INDEX = 0
WINDOW_NAME = "Face Recognition"
DETECT_EVERY = 3   # run the models on every 3rd frame; reuse the boxes/labels in between


def names_by_id(db: Database) -> dict[int, str]:
    return {w["id"]: w["name"] for w in db.list_workers()}


def register_from_webcam(engine: FaceEngine, db: Database):
    """Capture several angles; each capture is stored as its own embedding (no averaging)."""
    print("\n=== Registration ===")
    print("Press 'c' to capture (3-5 times, turning your head a little each time)")
    print("Press 's' to save, 'q' to quit without saving")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    captured = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _, _ = engine.detect(rgb)
        for x1, y1, x2, y2 in boxes.astype(int):
            color = (0, 255, 0) if len(boxes) == 1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Captures: {len(captured)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and len(boxes) == 1:
            captured.append(engine.encode(rgb, boxes)[0])
            print(f"Captured {len(captured)}")
        elif key == ord("s") and len(captured) >= config.MIN_ENROLL_CAPTURES:
            break
        elif key == ord("s"):
            print(f"Need at least {config.MIN_ENROLL_CAPTURES} captures")
        elif key == ord("q"):
            captured = []
            break
    cap.release()
    cv2.destroyAllWindows()

    if not captured:
        print("Registration cancelled")
        return
    if input("Does this person consent to storing their face template? (y/n): ").strip().lower() != "y":
        print("No consent; nothing saved")
        return
    name = input("Name: ").strip()[:60]
    if not name:
        return
    worker_id = db.create_worker(name, "", None, actor="desktop")
    for emb in captured[:config.MAX_ENROLL_CAPTURES]:
        db.add_embedding(worker_id, emb)
    db.finish_enrollment(worker_id, actor="desktop")
    engine.gallery.load(*db.gallery())
    print(f"Registered {name} (worker #{worker_id})")


def run_recognition(engine: FaceEngine, db: Database):
    print("\n=== Recognition === ('q' quit, 'r' register)")
    names = names_by_id(db)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    labels = []          # [(box, text, color)] from the last detection frame
    frame_idx, fps, last = 0, 0.0, time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % DETECT_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            labels = []
            for face in engine.analyze(rgb):
                worker_id, sim = engine.gallery.match(face.embedding)
                known = worker_id is not None
                text = f"{names.get(worker_id, '?') if known else 'Unknown'} ({sim:.2f})"
                labels.append((face.box, text, (0, 255, 0) if known else (0, 0, 255)))
        frame_idx += 1

        for (x1, y1, x2, y2), text, color in labels:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # measured FPS (smoothed), not a claim
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.putText(frame, f"{fps:.1f} FPS | enrolled: {len(engine.gallery)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            cap.release()
            cv2.destroyAllWindows()
            register_from_webcam(engine, db)
            names = names_by_id(db)
            cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Face recognition desktop demo")
    parser.add_argument("--register", action="store_true", help="Start in registration mode")
    parser.add_argument("--clear-db", action="store_true", help="Delete the local database")
    args = parser.parse_args()

    if args.clear_db:
        if input(f"Delete {config.DB_PATH}? (y/n): ").lower() == "y":
            Path(config.DB_PATH).unlink(missing_ok=True)
            print("Database deleted")
        return

    db = Database()
    engine = FaceEngine()
    engine.gallery.load(*db.gallery())
    print(f"Device: {engine.device} | enrolled workers: {len(engine.gallery)} | threshold: {config.MATCH_THRESHOLD}")
    try:
        if args.register:
            register_from_webcam(engine, db)
        else:
            run_recognition(engine, db)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
