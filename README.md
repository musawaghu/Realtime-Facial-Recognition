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
| **Worker** | Kiosk at the gate (tablet or laptop + camera) | Walks up, gets recognized, sees "Checked in · 6:58 AM". If recognition fails, taps **Use PIN** and enters worker number + PIN on a large keypad. |
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
pytest                              # 44 fast tests (models stubbed): check-in rules, hours, PIN lockout, privacy, auth
python benchmarks/lfw_eval.py       # downloads LFW (~180 MB), writes benchmarks/results/lfw.json
pytest -m slow                      # real-model smoke test on LFW photos
python benchmarks/speed_bench.py    # writes benchmarks/results/speed.json
```

## Limitations and next steps

- **No liveness check yet.** A photo held up to the camera can check someone in. Next: a
  random head-turn challenge using the MTCNN landmarks the engine already returns.
- **Hard hats aren't measured yet.** The numbers above are on LFW, not on people wearing
  hard hats in sun glare. Next: a small consented capture set, with and without headwear.
- **CPU only so far.** Every figure above was measured without a GPU.
- Next: PPE check (hard hat and vest) before check-in, automatic deletion when the retention
  date passes, an offline queue for sites with patchy internet, and deployment.
