# Jobsite Check-in: face recognition for construction site attendance

Workers check in at the gate with their face. Foremen always know who's on site, payroll gets
accurate hours, and in an emergency the headcount is one click away.

## The problem

- **Buddy punching.** On paper sign-in sheets or shared badges, workers clock in for absent
  coworkers, and contractors pay for hours nobody worked.
- **Emergency headcount.** During an evacuation the foreman has to know who is on site.
  A sign-in sheet doesn't tell you who forgot to sign out.
- **Jobsite conditions.** Gloves, dirt, glare and patchy internet. Check-in has to be
  hands-free, and there has to be a fallback when the camera can't see you.

## Two users

| | Uses | Does |
|---|---|---|
| **Worker** | Kiosk at the gate (tablet or laptop + camera) | Walks up, gets recognized, turns their head the way the screen asks (a liveness check), and sees "Checked in · 6:58 AM". If recognition fails, taps **Use PIN** and enters worker number + PIN on a large keypad. |
| **Foreman** | Web dashboard | Sees who's on site now, daily hours with CSV export for payroll, an **emergency muster roster** to tick off at the assembly point, enrolls workers with consent, deletes a worker's data on request. |

## How it works

```
Kiosk (browser)                       FastAPI server                         SQLite
getUserMedia → JPEG every 400 ms ──▶  /api/scan                              workers
                                       MTCNN: detect faces + landmarks         embeddings (3–5 per worker)
                                       take the closest (largest) face         events (in/out, face/pin)
                                       FaceNet: 512-d embedding                musters + entries
                                       cosine similarity vs every stored       audit_log
                                       embedding, best per worker
                                       ≥ threshold → toggle in/out ─────────▶
```

- **Detection:** MTCNN (facenet-pytorch). **Recognition:** InceptionResnetV1 pretrained on VGGFace2.
- **Several embeddings per worker.** Enrollment captures 5 angles (straight, left, right, up, down)
  and stores each one. A worker's score is their *best* matching angle. This is more robust than
  averaging into a single vector.
- **Vectorized matching.** All stored embeddings sit in one matrix, so matching is one matrix
  product plus a per-worker max (`np.maximum.reduceat`).
- **Batched embedding.** All faces in a frame go through FaceNet in one forward pass.
- **Check-in rules.** Each recognized scan flips in ↔ out. Repeat scans within 60 s are ignored.
  A check-in older than 16 h counts as a missed check-out, gets flagged for the foreman, and isn't
  paid automatically.
- **Threshold from data.** The match threshold isn't hand-picked. `benchmarks/lfw_eval.py` sets it
  to the similarity where the false-accept rate on LFW is 0.1%.
- **Liveness: random head turn.** Being recognized isn't enough to check in. Once a worker is
  recognized facing the camera, the kiosk asks them to turn their head LEFT or RIGHT (random), and
  check-in happens only after the server sees that turn. The server measures the turn ("yaw") as
  how far the nose sits from the midpoint between the eyes, in eye-widths, using the 5 landmarks
  MTCNN already returns, so no extra model is needed. Tilting a flat photo squeezes the eyes and
  nose together, so it can't fake a turn. The challenge fails if:
  - the face disappears,
  - the face jumps or changes size (a swapped photo),
  - the turned face stops matching the worker, or matches someone else better,
  - or 6 seconds pass.

  A worker who is already turned when recognized is asked to look straight first. Otherwise a
  static photo of a turned face would pass half the time. Failures are written to the audit log.
  The thresholds come from LFW: on 1,500 faces |yaw| has median 0.10 and 95th percentile 0.29.
  "Facing straight" means ≤ 0.25, and a turn means ≥ 0.35 in the asked direction.

## Measured results

All numbers come from the scripts in `benchmarks/`, run on an **AMD Ryzen 7 3700X, CPU only
(no GPU)**. Raw output is in `benchmarks/results/`.

**Verification accuracy: LFW, standard 10-fold protocol, 6,000 pairs**

| Metric | Result |
|---|---|
| Accuracy (10-fold) | **99.43% ± 0.37%** |
| ROC AUC | 0.9995 |
| True-accept rate at 0.1% false-accept rate | **98.9%** (threshold 0.446, used by the app) |
| True-accept rate at 1% false-accept rate | 99.4% (threshold 0.392) |
| Images where MTCNN found no face | 1 of 7,701 |

The 0.1% figure comes from 3,000 impostor pairs, so it rests on very few false accepts. Treat it
as an order of magnitude, not a guarantee. On a real site the risk of matching the wrong person
also grows with the number of enrolled workers.

**Speed: 640×480 frames**

| Step | Time |
|---|---|
| Face detection (MTCNN) | 23.5 ms |
| Embedding 4 faces one at a time → batched in one pass | 177.6 ms → **81.8 ms** (2.2× faster) |
| Matching against 500 workers × 5 angles: Python loop → vectorized | 2.84 ms → **0.18 ms** (16× faster) |
| Full pipeline, 1 face | 67.8 ms → **14.7 FPS** with the models on every frame |
| Desktop demo, models on every 3rd frame | ~44 FPS model-bound (calculated), so it keeps up with a 30 FPS webcam |
| Kiosk `/api/scan` (JPEG decode + pipeline + DB write, in-process) | **69.7 ms** |

**Liveness: does a turned head still match?** (`benchmarks/liveness_eval.py`, LFW)

| Metric | Result |
|---|---|
| Turned faces (yaw ≥ 0.35) still matching 3 frontal photos of the same person | **124 / 124**, at both the liveness threshold (0.392) and the strict check-in threshold (0.446) |
| Photos that read as facing straight (\|yaw\| ≤ 0.25) | 92% of 6,733. The rest would first be asked to look straight |

124 is a small sample. Zero failures in 124 means the true pass rate is very likely at least
97.6% (rule of three), not that it is 100%.

The kiosk sends a frame every 400 ms. From a face entering the frame to the confirmation
screen is about one scan interval plus about 70 ms of server time, plus network latency when
the server isn't on the same machine.

## Privacy by design

Face data is biometric data. Laws like Illinois' BIPA require written consent, a retention
schedule, and deletion. The system is built around that:

- **Consent first.** Enrollment can't start until the foreman confirms the worker read the notice
  and agreed. The server rejects enrollment without `consent: true`, and the consent time is stored.
- **Embeddings only, never photos.** Frames are processed in memory and discarded. The database
  holds 512 numbers per angle, stored as raw float32 (no pickle).
- **Retention date** on every worker (enrollment + 1 year).
- **Delete my data.** Erases face embeddings and PIN immediately, and the kiosk stops recognizing
  them. Attendance rows stay, because payroll needs them.
- **Audit log** of enrollments, deletions, PIN lockouts, musters and payroll exports.
- **PIN security.** PINs are hashed with scrypt and lock for 5 minutes after 5 wrong tries. An
  unknown worker number gets the same answer as a wrong PIN, so IDs can't be probed.
- **Hardening.** Per-IP rate limits on the kiosk endpoints, 2 MB upload cap, input validation,
  CSV formula-injection escaping, and HTTP Basic auth on every foreman endpoint.

## Run it

Requires **Python 3.11** (facenet-pytorch needs torch 2.2 / numpy < 2).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env            # then set ADMIN_PASSWORD in .env
python scripts/seed_demo.py       # optional: made-up crew + 2 weeks of history (no faces; PIN 1111)
uvicorn app.api:app
```

- Kiosk: http://localhost:8000/kiosk
- Dashboard: http://localhost:8000/dashboard (user `foreman`, password from `.env`)

The browser only allows camera access on `localhost` or HTTPS.

There is also a desktop OpenCV demo using the same engine and database: `python main.py`
(`--register` to enroll from the webcam).

### Tests and benchmarks

```bash
pytest                              # 61 fast tests (models stubbed): check-in rules, liveness, hours, PIN lockout, privacy, auth
python benchmarks/lfw_eval.py       # downloads LFW (~180 MB), writes benchmarks/results/lfw.json
pytest -m slow                      # real-model tests on LFW photos (incl. head-turn direction)
python benchmarks/speed_bench.py    # writes benchmarks/results/speed.json
python benchmarks/liveness_eval.py  # do turned faces still match? writes benchmarks/results/liveness.json
```

## Limitations and next steps

- **The liveness check stops photos, not determined attackers.** A printed photo or a still
  image on a phone can't pass. A video of the worker turning their head can pass (about half the
  time, since the direction is random), and so can two prepared photos swapped very smoothly.
  Stopping those needs depth/IR cameras or a trained anti-spoofing model. The PIN fallback
  also skips the liveness check, since it is a separate factor.
- **Hard hats aren't measured yet.** The numbers above are on LFW, not on people wearing
  hard hats in sun glare. Next: a small consented capture set, with and without headwear.
- **CPU only so far.** Every figure above was measured without a GPU.
- Next: PPE check (hard hat and vest) before check-in, automatic deletion when the retention
  date passes, an offline queue for sites with patchy internet, and deployment.
