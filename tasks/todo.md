# Jobsite Check-in — Phases 1–2

- [x] Step 0: security cleanup (.gitignore, purge .env/faces.db from history) — force push pending your OK
- [x] Step 1: refactor into app/ package (config, face_engine, db) + desktop demo
- [x] Step 2: LFW benchmark + speed benchmark; MATCH_THRESHOLD read from lfw.json (0.446)
- [x] Step 3: SQLite data model (workers, embeddings, events, musters, audit)
- [x] Step 4: FastAPI endpoints (scan, PIN, enroll, dashboard, muster, delete)
- [x] Step 5: browser kiosk + foreman dashboard + demo seed script
- [x] Step 6: honest README with measured numbers
- [x] Verification: 44 fast tests, 2 slow real-model tests, API smoke test via curl, layout screenshots
- [ ] Manual: enroll your own face at /dashboard and check in at /kiosk (needs a webcam + you)

## Review
- Real bug caught by the speed benchmark: numpy int32 box coords broke JSON in /api/scan.
  The stubbed API tests used Python ints, so they missed it; the slow test now asserts plain ints.
- FAR 0.1% is estimated from only 3,000 impostor pairs — noted in the README.
- Not yet covered by automated tests: the browser JS (kiosk keypad, enrollment capture loop).
- Next round: liveness (head-turn using landmarks), hard-hat accuracy test, PPE, retention job,
  offline queue, deployment.

# Round 2 — Liveness (random head turn)

- [x] Calibrate yaw on 1,500 LFW faces (median 0.10, p95 0.29); check sign by eye (George_Clooney_0002)
- [x] app/liveness.py: yaw, frontal check, box continuity, challenge store with 6 s timeout
- [x] /api/scan: recognition issues a challenge; check-in only after the requested turn
- [x] Identity re-checked on the turned face (LFW 1%-FAR threshold), audit log on failure
- [x] Kiosk prompt (arrow + text), faster sampling during the challenge
- [x] Tests: 17 new fast tests (photo held still, wrong direction, timeout, swaps, token reuse), real-photo sign test
- [ ] benchmarks/liveness_eval.py result in README (running)
- [ ] Manual: try it with a webcam, plus a printed photo and a phone photo
