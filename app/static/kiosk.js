// Kiosk: sends a camera frame to the server ~2.5 times a second and shows the result.
// A recognized worker must then turn their head as asked (liveness check) before check-in.
const SCAN_INTERVAL_MS = 400;
const CHALLENGE_INTERVAL_MS = 250; // sample faster while watching the head turn
const CONFIRM_MS = 3000;
const FRAME_WIDTH = 480;           // downscale before upload: faster, and plenty for a face at the gate

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const statusEl = document.getElementById("status");
const confirmEl = document.getElementById("confirm");
const frame = document.createElement("canvas");

let paused = false;                // true while a confirmation or the keypad is showing
let unknownStreak = 0;
let challenge = null;              // token of the open head-turn challenge, if any
const turnPrompt = document.getElementById("turn-prompt");

// ---------- clock ----------
const clock = document.getElementById("clock");
const tick = () => clock.textContent = new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
tick(); setInterval(tick, 1000);

// ---------- camera ----------
async function startCamera() {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
    await video.play();
    setStatus("Step up to the camera");
    scanLoop();
  } catch (e) {
    setStatus("Camera unavailable — use PIN");
  }
}

function setStatus(text) { statusEl.textContent = text; }

function drawBox(box) {
  overlay.width = frame.width; overlay.height = frame.height;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!box) return;
  ctx.strokeStyle = "#ffb020"; ctx.lineWidth = 4;
  ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]);
}

function grabFrame() {
  frame.width = FRAME_WIDTH;
  frame.height = Math.round(video.videoHeight * FRAME_WIDTH / video.videoWidth);
  frame.getContext("2d").drawImage(video, 0, 0, frame.width, frame.height);
  return new Promise(resolve => frame.toBlob(resolve, "image/jpeg", 0.85));
}

async function scanOnce() {
  const form = new FormData();
  form.append("file", await grabFrame(), "frame.jpg");
  if (challenge) form.append("challenge", challenge);
  const res = await fetch("/api/scan", { method: "POST", body: form });
  if (res.status === 429) return;                    // rate-limited: just try again next tick
  if (!res.ok) throw new Error(res.status);
  const r = await res.json();
  drawBox(r.box);
  if (r.status !== "turn") endChallenge();

  if (r.status === "turn") {
    unknownStreak = 0;
    challenge = r.challenge;
    showTurnPrompt(r.direction, r.name);
  } else if (r.status === "challenge_failed") {
    setStatus("Didn’t catch that — look at the camera and try again");
    await new Promise(res => setTimeout(res, 1500));
  } else if (r.status === "look_straight") {
    setStatus("Look straight at the camera");
  } else if (r.status === "checked_in" || r.status === "checked_out") {
    unknownStreak = 0;
    showConfirm(r.status === "checked_in" ? "in" : "out",
                r.status === "checked_in" ? `Checked in · ${r.time}` : `Checked out · ${r.time}`, r.name);
  } else if (r.status === "cooldown") {
    showConfirm("warn", `Already recorded · ${r.time}`, r.name);
  } else if (r.status === "unknown") {
    unknownStreak++;
    setStatus(unknownStreak >= 4 ? "Not recognized — tap “Use PIN”" : "Hold still…");
  } else {
    unknownStreak = 0;
    setStatus("Step up to the camera");
  }
}

async function scanLoop() {
  while (true) {
    if (!paused && video.readyState >= 2) {
      try { await scanOnce(); } catch { setStatus("Connection problem — retrying"); }
    }
    await new Promise(r => setTimeout(r, challenge ? CHALLENGE_INTERVAL_MS : SCAN_INTERVAL_MS));
  }
}

function showTurnPrompt(direction, name) {
  turnPrompt.className = `turn-prompt show ${direction}`;
  document.getElementById("turn-arrow").textContent = "⬅";   // CSS mirrors it for "right"
  document.getElementById("turn-text").textContent = `Turn your head to your ${direction.toUpperCase()}`;
  if (name) setStatus(`Hi ${name} — quick check that you’re really here`);
}

function endChallenge() {
  challenge = null;
  turnPrompt.className = "turn-prompt";
}

function showConfirm(kind, title, name) {
  paused = true;
  drawBox(null);
  confirmEl.className = `confirm show ${kind}`;
  document.getElementById("confirm-title").textContent = title;
  document.getElementById("confirm-name").textContent = name;
  setTimeout(() => {
    confirmEl.className = "confirm";
    setStatus("Step up to the camera");
    paused = false;
  }, CONFIRM_MS);
}

// ---------- PIN fallback ----------
const keypad = document.getElementById("keypad");
const entry = document.getElementById("keypad-entry");
const prompt = document.getElementById("keypad-prompt");
const msg = document.getElementById("keypad-msg");
let step = "id", digits = "", workerId = null;

function renderEntry() { entry.textContent = step === "pin" ? "•".repeat(digits.length) : digits; }

function openKeypad() {
  paused = true; endChallenge(); step = "id"; digits = ""; workerId = null;
  prompt.textContent = "Enter your worker number"; msg.textContent = "";
  renderEntry(); keypad.classList.add("show");
}

function closeKeypad() { keypad.classList.remove("show"); paused = false; }

async function submitKeypad() {
  if (step === "id") {
    if (!digits) return;
    workerId = Number(digits); step = "pin"; digits = "";
    prompt.textContent = "Enter your PIN"; msg.textContent = ""; renderEntry();
    return;
  }
  if (digits.length < 4) { msg.textContent = "PIN is 4–6 digits"; return; }
  const res = await fetch("/api/pin", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ worker_id: workerId, pin: digits }),
  });
  digits = ""; renderEntry();
  if (res.status === 429) { msg.textContent = "Too many tries — wait a minute"; return; }
  const r = await res.json();
  if (r.status === "bad_pin") { msg.textContent = "Worker number or PIN not recognized"; return; }
  if (r.status === "locked") { msg.textContent = "Locked for 5 minutes — see your foreman"; return; }
  keypad.classList.remove("show");
  if (r.status === "cooldown") showConfirm("warn", `Already recorded · ${r.time}`, r.name);
  else showConfirm(r.status === "checked_in" ? "in" : "out",
                   `${r.status === "checked_in" ? "Checked in" : "Checked out"} · ${r.time}`, r.name);
}

const keys = document.getElementById("keys");
for (const k of ["1", "2", "3", "4", "5", "6", "7", "8", "9", "⌫", "0", "OK"]) {
  const b = document.createElement("button");
  b.textContent = k;
  if (k === "OK") b.className = "primary";
  b.onclick = () => {
    if (k === "⌫") digits = digits.slice(0, -1);
    else if (k === "OK") return submitKeypad();
    else if (digits.length < 6) digits += k;
    renderEntry();
  };
  keys.appendChild(b);
}
document.getElementById("pin-btn").onclick = openKeypad;
document.getElementById("keypad-cancel").onclick = closeKeypad;

startCamera();
