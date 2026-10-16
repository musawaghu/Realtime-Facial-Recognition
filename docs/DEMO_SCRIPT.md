# Demo video: what to record

The finished video is about 2 minutes:

**title → the problem → your 4–5 recorded clips (captioned) → dashboard tour → measured numbers → outro**

You record only the clips with your face (about 10 minutes, including retakes).
`scripts/make_demo_video.py` builds everything else and stitches it together.

## Setup (once)

1. Use a fresh demo database so the dashboard looks real but has no one's face but yours:
   ```powershell
   .venv\Scripts\activate
   $env:DB_PATH = "demo.db"; $env:ADMIN_PASSWORD = "demo"
   python scripts/seed_demo.py
   uvicorn app.api:app
   ```
2. Kiosk in one browser window, dashboard in another, both full screen (F11), zoom 100%.
3. Screen recorder: **Win + Alt + R** (Xbox Game Bar) records the active window. OBS works too.
   Record at 1080p if you can. Any resolution works; the script letterboxes to 1920×1080.
4. Good light on your face, plain background, and a hard hat within reach.

## Shots

Save each clip into `demo/raw/` with **exactly** these names. A missing clip is skipped, so you
can build the video with just some of them.

| File | ~Length | What to do on screen | Caption the script adds |
|---|---|---|---|
| `01_enroll.mp4` | 20 s | Dashboard → Workers → **Enroll a worker**. Tick consent, type your name, PIN `4821`, Next. Follow the 5 pose prompts until it closes. | Enrollment: written consent first. 5 angles stored as numbers, never photos. |
| `02_checkin.mp4` | 15 s | Kiosk. Walk into frame, face the camera, turn your head the way the arrow says, and hold on the green "Checked in" screen. **Wear the hard hat.** | Check-in: recognized, then a random head turn proves a live person is there. |
| `03_spoof.mp4` | 15 s | Kiosk. Hold up a **printed photo** of yourself (or a photo on your phone). Tilt it when the arrow appears. Let it time out ("Didn't catch that"). | A photo can't pass: tilting a flat picture doesn't turn a head. |
| `04_ppe.mp4` | 12 s | After shot 2, wait a minute and check **out** (not recorded). Wait another minute, take the hard hat **off**, then record checking **in** again. Hold on the yellow "No hard hat detected" warning. (Each scan needs the 60 s cooldown, and the PPE check runs only on check-in. Needs the trained PPE model.) | No hard hat? Still checked in, warned on the spot, flagged for the foreman. |
| `05_pin.mp4` *(optional)* | 10 s | Kiosk → **Use PIN** → a demo worker number (1–12) → `1111`. | Glare, dirt, new beard? A PIN on a glove-sized keypad always works. |

Tips:
- Start each recording a second before you act and stop a second after the result shows. The
  script doesn't trim.
- If a take goes wrong, just record it again under the same name.
- No narration is needed. The script drops the audio, and captions carry the story. Add a
  voiceover or music afterwards in any editor if you want.

## Build the video

```powershell
python scripts/make_demo_video.py --github https://github.com/musawaghu/Realtime-Facial-Recognition
```

It builds:
- the title, problem and numbers cards (numbers come from `benchmarks/results/*.json`)
- the dashboard tour, from screenshots of a separate demo database
- your clips, with captions

The output is `demo/jobsite-checkin-demo.mp4`. `demo/` is git-ignored because the clips contain
your face, so upload the finished video yourself (YouTube unlisted, LinkedIn, or a GitHub release)
and link it from the README.
