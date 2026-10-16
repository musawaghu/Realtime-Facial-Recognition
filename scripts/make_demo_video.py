"""Build the demo video from your recorded clips (see docs/DEMO_SCRIPT.md).

    python scripts/make_demo_video.py --github https://github.com/<you>/<repo>

Order: title -> the problem -> your clips from demo/raw/ (captioned) -> dashboard tour
(screenshots of a separate, made-up demo database) -> measured numbers -> outro.
Output: demo/jobsite-checkin-demo.mp4 (1920x1080, 30 fps, no audio).

Needs: imageio-ffmpeg (bundled ffmpeg) and Microsoft Edge or Google Chrome for the screenshots.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
BUILD = DEMO / "build"
# the dashboard tour uses its own throwaway database, never your real one
os.environ["DB_PATH"] = str(BUILD / "dashboard-tour.db")
sys.path.insert(0, str(ROOT))

import imageio_ffmpeg  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

W, H, FPS = 1920, 1080, 30
BG, PANEL, TEXT, MUTED, ACCENT = "#111418", "#1b2027", "#f2f4f7", "#9aa4b2", "#ffb020"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ENCODE = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS), "-an"]

SHOTS = [
    ("01_enroll", "Enrollment: written consent first. 5 angles stored as numbers, never photos."),
    ("02_checkin", "Check-in: recognized, then a random head turn proves a live person is there."),
    ("03_spoof", "A photo can't pass: tilting a flat picture doesn't turn a head."),
    ("04_ppe", "No hard hat? Still checked in, warned on the spot, flagged for the foreman."),
    ("05_pin", "Glare, dirt, new beard? A PIN on a glove-sized keypad always works."),
]


# ---------- drawing ----------
def font(size, bold=False):
    names = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    for name in names:
        for folder in (Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/dejavu")):
            if (folder / name).exists():
                return ImageFont.truetype(str(folder / name), size)
    return ImageFont.load_default()


def card(title, lines=(), kicker=None) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 14], fill=ACCENT)
    y = 330 - 40 * len(lines)
    if kicker:
        d.text((160, y - 70), kicker.upper(), font=font(34, True), fill=ACCENT)
    d.text((160, y), title, font=font(92, True), fill=TEXT)
    y += 150
    for line in lines:
        d.text((160, y), line, font=font(44), fill=MUTED if not line.startswith("•") else TEXT)
        y += 72
    return img


def caption_bar(text) -> Image.Image:
    """Transparent 1920x1080 overlay with a caption band at the bottom."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, H - 150, W, H], fill=(17, 20, 24, 225))
    d.rectangle([0, H - 150, 12, H], fill=ACCENT)
    size = 46
    while size > 28 and d.textlength(text, font=font(size, True)) > W - 140:   # shrink to fit
        size -= 2
    d.text((70, H - 75 - size // 2 - 8), text, font=font(size, True), fill=TEXT)
    return img


def with_caption(screenshot: Path, text) -> Image.Image:
    img = Image.open(screenshot).convert("RGB").resize((W, H))
    img.paste(caption_bar(text), (0, 0), caption_bar(text))
    return img


# ---------- ffmpeg ----------
def run(args):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args], check=True)


def still(img: Image.Image, seconds: float, out: Path):
    png = out.with_suffix(".png")
    img.save(png)
    fade_out = max(seconds - 0.4, 0)
    run(["-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", str(png),
         "-vf", f"fade=in:st=0:d=0.3,fade=out:st={fade_out}:d=0.4", *ENCODE, str(out)])


def clip(raw: Path, caption: str, out: Path):
    cap = out.with_suffix(".caption.png")
    caption_bar(caption).save(cap)
    run(["-i", str(raw), "-i", str(cap), "-filter_complex",
         f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x111418,"
         f"fps={FPS},setsar=1[v];[v][1:v]overlay=0:0", *ENCODE, str(out)])


def concat(parts: list[Path], out: Path):
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(out)])


# ---------- dashboard tour ----------
def browser() -> str | None:
    for p in (os.environ.get("BROWSER", ""),
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              shutil.which("google-chrome") or "", shutil.which("chromium") or ""):
        if p and Path(p).exists():
            return p
    return None


def fill_demo_database():
    """Made-up crew (no faces) with a realistic 'right now': people on site, PPE gaps, an open
    muster, and two other sites reporting in."""
    from scripts import seed_demo
    from app.db import Database
    Path(os.environ["DB_PATH"]).unlink(missing_ok=True)
    seed_demo.main()
    db = Database()
    now = datetime.now()
    # Replace today's seeded history with a crew that is mid-shift right now, so the tour looks
    # like a working day whatever time the script runs.
    with db._conn() as c:
        c.execute("DELETE FROM events WHERE ts >= ?", (now.strftime("%Y-%m-%d 00:00:00"),))
    workers = [w["id"] for w in db.list_workers()]
    ppe_for = {2: {"hard_hat": False, "vest": True}, 5: {"hard_hat": True, "vest": False}}
    for i, wid in enumerate(workers[:9]):   # on site now, arrived 1-2.5 h ago
        pin = i == 3                        # PIN check-ins have no camera frame, so no PPE check
        db.record_scan(wid, "pin" if pin else "face", None if pin else 0.8, now=now - timedelta(minutes=150 - 11 * i),
                       ppe=None if pin else ppe_for.get(i, {"hard_hat": True, "vest": True}))
    for i, wid in enumerate(workers[9:]):   # came in early and already left
        t_in = now - timedelta(hours=6, minutes=10 * i)
        if t_in.date() == now.date():
            db.record_scan(wid, "face", 0.8, now=t_in, ppe={"hard_hat": True, "vest": True})
            db.record_scan(wid, "face", 0.8, now=now - timedelta(minutes=40 - 5 * i))
    mid = db.start_muster(now=now - timedelta(minutes=2))
    for wid in [e["worker_id"] for e in db.get_muster(mid)["entries"]][:5]:
        db.set_accounted(mid, wid, True, now=now - timedelta(minutes=1))
    other_sites = {"riverside-tower": ["Maya Chen", "Luis Ortega", "Priya Nair", "Tom Becker"],
                   "eastgate-school": ["Omar Haddad", "Grace Kim"]}
    for site, names in other_sites.items():
        workers_ = [{"id": i + 1, "name": n, "trade": "Laborer"} for i, n in enumerate(names)]
        events = [{"id": i + 1, "worker_id": i + 1, "kind": "in", "method": "face", "hard_hat": True,
                   "vest": i != 1, "ts": (now - timedelta(minutes=80 - 9 * i)).strftime("%Y-%m-%d %H:%M:%S")}
                  for i in range(len(names))]
        db.ingest_site_batch(site, workers_, events, now=now - timedelta(minutes=1 if site == "riverside-tower" else 14))
    return db


def dashboard_tour(port=8790) -> list[tuple[Path, str]]:
    exe = browser()
    if not exe:
        print("No Edge/Chrome found: skipping the dashboard tour")
        return []
    if datetime.now().hour < 3:
        print("Note: just after midnight the crew's check-ins fall on yesterday, so 'today' looks "
              "quiet. Build after 3 AM for the most realistic tour.")
    import uvicorn
    from app import api
    db = fill_demo_database()
    api.app.dependency_overrides = {api.get_db: lambda: db, api.require_foreman: lambda: "foreman"}
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    shots = [("onsite", "Foreman dashboard: who's on site right now, and who's missing PPE."),
             ("muster", "Emergency muster: one click lists everyone on site to tick off."),
             ("hours", "Daily hours per worker, with a CSV export for payroll."),
             ("sites", "Offline-first: each site runs its own server and syncs to the office when online.")]
    out = []
    with tempfile.TemporaryDirectory() as profile:
        for tab, caption in shots:
            png = BUILD / f"dash_{tab}.png"
            # render at 1280x720 so the dashboard fills the frame, then scale up to 1080p
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1280,720",
                            f"--user-data-dir={profile}", "--virtual-time-budget=5000",
                            f"--screenshot={png}", f"http://127.0.0.1:{port}/dashboard#{tab}"],
                           check=True, capture_output=True, timeout=60)
            out.append((png, caption))
    server.should_exit = True
    return out


def numbers_card() -> Image.Image:
    def load(name):
        try:
            return json.loads((ROOT / "benchmarks" / "results" / name).read_text())
        except OSError:
            return {}
    lfw, live, hat, ppe = load("lfw.json"), load("liveness.json"), load("hardhat_synthetic.json"), load("ppe.json")
    lines = []
    if lfw:
        lines.append(f"•  {lfw['accuracy_mean']:.2%} face verification accuracy on LFW (6,000 pairs)")
        lines.append(f"•  {lfw['tar_at_far_0.1']:.1%} of genuine check-ins accepted at a 0.1% false-accept rate")
    if live:
        lines.append(f"•  {live['turned_faces_tested']}/{live['turned_faces_tested']} turned heads still matched (liveness)")
    if hat:
        c = hat["conditions"]["enrolled_bare_checkin_with_hat"]
        lines.append(f"•  {c['accepted_rate_at_app_threshold']:.1%} accepted with a hard hat on (synthetic test)")
    if ppe:
        lines.append(f"•  Hard hat recall {ppe['classes']['helmet']['recall']:.0%}, vest {ppe['classes']['vest']['recall']:.0%}"
                     f" · {ppe['inference_ms_480x360'].get('gpu', ppe['inference_ms_480x360']['cpu'])} ms per frame")
    return card("Measured, not claimed", lines, kicker="Benchmarks")


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", default="github.com/musawaghu/Realtime-Facial-Recognition")
    parser.add_argument("--raw", type=Path, default=DEMO / "raw", help="folder with your recorded clips")
    parser.add_argument("--out", type=Path, default=DEMO / "jobsite-checkin-demo.mp4")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    parts = []

    def add_still(name, img, seconds):
        path = BUILD / f"{len(parts):02d}_{name}.mp4"
        still(img, seconds, path)
        parts.append(path)

    add_still("title", card("Jobsite Check-in", ["Face check-in for construction sites:",
                                                 "accurate hours, PPE checks, one-click emergency headcount."],
                            kicker="Portfolio project"), 4)
    add_still("problem", card("The problem", [
        "•  Buddy punching: paying for hours nobody worked",
        "•  Evacuations: paper sign-in sheets can't say who's still inside",
        "•  Hard hats and vests checked by eye, if at all"], kicker="Why"), 6)

    found = 0
    for name, caption in SHOTS:
        raw = next(iter(sorted(args.raw.glob(f"{name}.*"))), None)
        if raw is None:
            print(f"(skipping {name}: no clip in {args.raw})")
            continue
        path = BUILD / f"{len(parts):02d}_{name}.mp4"
        clip(raw, caption, path)
        parts.append(path)
        found += 1

    for png, caption in dashboard_tour():
        add_still(png.stem, with_caption(png, caption), 4.5)
    add_still("numbers", numbers_card(), 7)
    add_still("outro", card("Jobsite Check-in", [args.github, "", "Python · FastAPI · PyTorch (MTCNN, FaceNet, YOLO11) · SQLite · Docker"],
                            kicker="Code and benchmarks"), 4)

    concat(parts, args.out)
    print(f"Built {args.out} from {len(parts)} segments ({found} of your clips)")


if __name__ == "__main__":
    main()
