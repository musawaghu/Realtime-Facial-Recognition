# Manual testing guide

The automated tests (`pytest`, `pytest -m slow`) cover the logic with stubbed or recorded inputs.
This guide covers what only a person with a webcam can check. It takes about 30 minutes.
Tick each item; anything that behaves differently from "Expect" is a bug worth noting.

## 0. Setup (once)

```powershell
cd Realtime-Facial-Recognition
.venv\Scripts\activate
```

1. Add a dashboard password to `.env` (the file currently only has the old Postgres keys):
   ```
   ADMIN_PASSWORD=pick-something
   ```
2. Use a separate database for testing so your real data stays clean:
   ```powershell
   $env:DB_PATH = "test-run.db"
   python scripts/seed_demo.py        # 12 made-up workers (PIN 1111) + 2 weeks of history
   uvicorn app.api:app
   ```
3. Open two browser windows side by side:
   - Kiosk: http://localhost:8000/kiosk
   - Dashboard: http://localhost:8000/dashboard (user `foreman`, your password)

   Allow camera access when the browser asks. Camera access works only on `localhost` or HTTPS.

The first face scan takes a few seconds while the models load. After that, each scan is fast.

## 1. Enrollment

On the dashboard: **Workers → Enroll a worker**.

| Step | Expect |
|---|---|
| Click "Next" without ticking consent | "Consent is required…" and nothing is created |
| Tick consent, name "Test Me", PIN `4821`, Next | Camera opens; the prompt says to take off your hard hat |
| Follow the 5 pose prompts (straight, left, right, chin up, chin down) | Dots fill in one by one; it finishes by itself after 5 |
| Let someone else step in mid-capture (optional) | "That doesn't look like the same person" and the capture isn't counted |
| Workers tab | "Test Me", status "enrolled · 5 angles", and a worker number (#) |

## 2. Face check-in with the liveness check

At the kiosk:

| Step | Expect |
|---|---|
| Walk up, face the camera | "Hi Test Me…" and an arrow: "Turn your head to your LEFT" (or RIGHT) |
| Turn your head the way it asks | Green "Checked in · 9:41 AM", Test Me |
| Stay in front of the camera | "Already recorded" (60 s cooldown), no head turn asked |
| Dashboard → On site | Test Me is listed, "How: Face" |
| Wait 60 s, check in again with a head turn | Blue "Checked out"; Test Me leaves the On site list |
| Turn the *wrong* way when asked | Keeps showing the prompt; no check-in |
| Ignore the prompt for 6 s | "Didn't catch that…"; the dashboard's Audit log shows "liveness fail, timed out" |
| Walk up already looking sideways | "Look straight at the camera" first |

## 3. Spoof attempts (should all fail)

Print or display a photo of the enrolled face.

| Attempt | Expect |
|---|---|
| Printed photo held still | The turn prompt appears but never passes; it times out |
| Tilt or rotate the printed photo | Still never passes (tilting a flat photo doesn't change the measured head turn) |
| Photo on a phone screen | Same: never passes |
| Swap between two photos (straight, then turned) | Fails: "lost face" or a jump in the face box shows up in the Audit log |
| Video of you turning your head (phone) | **May pass** about half the time. This is a known limitation, listed in the README |

## 4. PPE check (needs a trained detector: `python scripts/train_ppe.py`)

| Step | Expect |
|---|---|
| Check in with a hard hat on | Normal green confirmation; On site shows "✓ hat" (vest "—" if the camera can't see your torso) |
| Check in without a hard hat | Still checked in, plus a yellow "⚠ No hard hat detected…" for about 5 s |
| Dashboard | "On site without full PPE" count goes up; Audit log has "ppe missing, no hard hat" |
| Stand back so your torso is in view, wearing a hi-vis vest | On site shows "✓" for both |
| Check out without a hat | No warning (only check-in is checked) |

## 5. PIN fallback

| Step | Expect |
|---|---|
| Kiosk → **Use PIN** → worker number of "Test Me" → `4821` | Checked in or out, same confirmation screen |
| Wrong PIN | "Worker number or PIN not recognized" |
| Wrong PIN 5 times | "Locked for 5 minutes"; even the right PIN is refused; Audit log has "pin lockout". If "Too many tries — wait a minute" appears first, that's the separate per-device rate limit (5 quick tries, then 10 a minute); wait a few seconds between tries |
| Demo workers: their number + `1111` | Works (they have no face data, PIN only) |

## 6. Foreman dashboard

| Step | Expect |
|---|---|
| Hours tab: last 2 weeks → Show | Daily hours per worker; "missed check-out" tags on a few seeded days |
| Export CSV | Downloads `hours_<from>_<to>.csv`, which opens cleanly in Excel |
| Muster → Start emergency muster | Everyone on site is listed with "Unaccounted for: N of N" |
| Tick people off | The count goes down; ticked names are struck through |
| Close muster | Back to the start button; Audit log has "muster close, N unaccounted" |
| Workers → Delete data for Test Me | Status becomes "data deleted"; the kiosk now treats your face as unknown and the PIN stops working; your hours stay on the Hours tab |

## 7. Hard-hat measurements (needs 2+ consenting people)

```powershell
python scripts/capture_hardhat.py p01                              # you
python scripts/capture_hardhat.py p02                              # a friend
python scripts/capture_hardhat.py p03 --conditions nohat hat hat_outdoor   # optional: outside, in sun
python benchmarks/hardhat_eval.py real
```
Each run asks for consent, then you press SPACE and look at the camera while 30 frames are
taken, first without and then with the hard hat. Only embeddings are saved, never photos.
The eval writes `benchmarks/results/hardhat_real.json`; send me the numbers and I'll put them in
the README.

## 8. Clean up

```powershell
Remove-Item test-run.db
Remove-Item Env:DB_PATH
```
