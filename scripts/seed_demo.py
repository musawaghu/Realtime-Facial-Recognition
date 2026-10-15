"""Fill the database with made-up workers and two weeks of check-ins so the dashboard has
something to show. These workers have NO face data: only a PIN (1111), so anyone trying the
demo can use the kiosk's PIN fallback with their worker number.

Run:  python scripts/seed_demo.py
"""
import random
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402
from app.db import Database  # noqa: E402

CREW = [("Ana Torres", "Carpenter"), ("Ben Okafor", "Electrician"), ("Chris Novak", "Laborer"),
        ("Dana Kim", "Plumber"), ("Eli Brooks", "Ironworker"), ("Faye Lin", "Laborer"),
        ("Gus Moreno", "Operator"), ("Hana Sato", "Electrician"), ("Ivan Petrov", "Carpenter"),
        ("Jade Rivera", "Concrete"), ("Kofi Mensah", "Laborer"), ("Lena Ortiz", "Safety")]


def main():
    db = Database()
    if db.list_workers():
        sys.exit(f"{config.DB_PATH} already has workers. Delete it first (python main.py --clear-db).")

    rng = random.Random(7)
    start = date.today() - timedelta(days=14)
    ids = []
    for name, trade in CREW:
        wid = db.create_worker(name, trade, "1111", actor="seed", now=datetime.combine(start, time(6)))
        db.finish_enrollment(wid, actor="seed", now=datetime.combine(start, time(6)))
        ids.append(wid)

    now = datetime.now()
    for offset in range(15):
        day = start + timedelta(days=offset)
        if day.weekday() >= 5:      # no weekend work
            continue
        for wid in ids:
            if rng.random() < 0.08:  # occasional absence
                continue
            t_in = datetime.combine(day, time(6, 30)) + timedelta(minutes=rng.randint(0, 45))
            t_out = datetime.combine(day, time(15, 0)) + timedelta(minutes=rng.randint(0, 75))
            method = "pin" if rng.random() < 0.1 else "face"
            if t_in > now:
                continue
            db.record_scan(wid, method, None if method == "pin" else round(rng.uniform(0.6, 0.85), 3), now=t_in)
            forgot = rng.random() < 0.03
            if t_out <= now and not forgot:
                db.record_scan(wid, method, None, now=t_out)
    print(f"Seeded {len(ids)} workers (PIN 1111, worker numbers {ids[0]}-{ids[-1]}) into {config.DB_PATH}")


if __name__ == "__main__":
    main()
