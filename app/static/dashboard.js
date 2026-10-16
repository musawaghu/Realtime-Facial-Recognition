// Foreman dashboard. All data is inserted with textContent (never innerHTML), so names can't inject markup.
const $ = id => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : "Invalid input");
  }
  return res.json();
}

function cell(text, cls) {
  const td = document.createElement("td");
  td.textContent = text ?? "";
  if (cls) td.className = cls;
  return td;
}

function tag(text, cls) {
  const s = document.createElement("span");
  s.className = `tag ${cls || ""}`;
  s.textContent = text;
  return s;
}

function fillTable(tbody, rows, makeRow, emptyText) {
  tbody.replaceChildren();
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = cell(emptyText, "empty");
    td.colSpan = 10;
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }
  for (const r of rows) tbody.appendChild(makeRow(r));
}

const time = ts => new Date(ts.replace(" ", "T")).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
const today = () => new Date().toLocaleDateString("en-CA");   // YYYY-MM-DD in local time

// ---------- tabs ----------
const loaders = {};
let current = "onsite";
document.querySelectorAll("#tabs button").forEach(b => b.onclick = () => {
  current = b.dataset.tab;
  document.querySelectorAll("#tabs button").forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll("main section").forEach(s => s.hidden = s.id !== `tab-${current}`);
  history.replaceState(null, "", `#${current}`);
  loaders[current]?.();
});

// ---------- on site ----------
loaders.onsite = async () => {
  const [onsite, hours] = await Promise.all([api("/api/onsite"), api(`/api/hours?start=${today()}`)]);
  $("onsite-count").textContent = onsite.length;
  $("today-count").textContent = hours.length;
  $("missing-count").textContent = hours.filter(h => h.missing_checkout).length;
  $("ppe-count").textContent = onsite.filter(w => w.hard_hat === 0 || w.vest === 0).length;
  fillTable($("onsite-body"), onsite, w => {
    const tr = document.createElement("tr");
    tr.append(cell(w.name), cell(w.trade), cell(time(w.since)), cell(w.method === "pin" ? "PIN" : "Face"), ppeCell(w));
    return tr;
  }, "Nobody is checked in right now.");
};
setInterval(() => current === "onsite" && loaders.onsite().catch(() => {}), 5000);

// PPE seen by the kiosk camera at check-in: 1 yes, 0 no, null not checked (PIN, no detector, torso out of view)
function ppeCell(w) {
  const td = cell("");
  if (w.hard_hat === 0) td.appendChild(tag("no hard hat", "warn"));
  if (w.vest === 0) td.appendChild(tag("no vest", "warn"));
  if (!td.childNodes.length) td.textContent = w.hard_hat === 1 && w.vest === 1 ? "✓" : w.hard_hat === 1 ? "✓ hat" : "—";
  return td;
}

// ---------- hours ----------
// default to the last 7 days: payroll is usually weekly
$("hours-start").value = new Date(Date.now() - 6 * 864e5).toLocaleDateString("en-CA");
$("hours-end").value = today();
loaders.hours = async () => {
  const q = `start=${$("hours-start").value}&end=${$("hours-end").value}`;
  try {
    const rows = await api(`/api/hours?${q}`);
    fillTable($("hours-body"), rows, r => {
      const tr = document.createElement("tr");
      const notes = cell("");
      if (r.open) notes.appendChild(tag("on site now"));
      if (r.missing_checkout) notes.appendChild(tag("missed check-out", "warn"));
      tr.append(cell(r.date), cell(r.name), cell(r.trade), cell(r.hours.toFixed(2), "num"), cell(r.sessions, "num"), notes);
      return tr;
    }, "No attendance in this range.");
  } catch (e) { alertRow($("hours-body"), e.message); }
};
$("hours-load").onclick = loaders.hours;
$("hours-csv").onclick = () => {
  location.href = `/api/hours.csv?start=${$("hours-start").value}&end=${$("hours-end").value}`;
};

function alertRow(tbody, text) { fillTable(tbody, [], null, text); }

// ---------- muster ----------
let musterId = null;
function renderMuster(m) {
  musterId = m && !m.closed_at ? m.id : null;
  $("muster-idle").hidden = !!musterId;
  $("muster-active").hidden = !musterId;
  if (!musterId) return;
  $("muster-missing").textContent = `${m.missing} of ${m.entries.length}`;
  $("muster-started").textContent = `Started ${time(m.started_at)}`;
  const list = $("muster-roster");
  list.replaceChildren();
  for (const e of m.entries) {
    const li = document.createElement("li");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !!e.accounted_at;
    box.onchange = async () => renderMuster(await api(`/api/muster/${musterId}/account/${e.worker_id}?accounted=${box.checked}`, { method: "POST" }));
    const label = document.createElement("span");
    label.textContent = `${e.name}${e.trade ? " — " + e.trade : ""}`;
    li.className = e.accounted_at ? "done" : "";
    li.append(box, label);
    list.appendChild(li);
  }
  if (!m.entries.length) {
    const li = document.createElement("li");
    li.textContent = "Nobody was checked in when the muster started.";
    list.appendChild(li);
  }
}
loaders.muster = async () => renderMuster(await api("/api/muster/current"));
$("muster-start").onclick = async () => renderMuster(await api("/api/muster", { method: "POST" }));
$("muster-close").onclick = async () => {
  if (confirm("Close this muster? The roster will be saved in the audit log.")) {
    await api(`/api/muster/${musterId}/close`, { method: "POST" });
    renderMuster(null);
  }
};

// ---------- workers ----------
loaders.workers = async () => {
  const workers = await api("/api/workers");
  fillTable($("workers-body"), workers, w => {
    const tr = document.createElement("tr");
    const status = cell("");
    if (w.deleted_at) status.appendChild(tag("data deleted", "bad"));
    else if (w.enrolled_at) status.appendChild(tag(w.captures ? `enrolled · ${w.captures} angles` : "PIN only"));
    else status.appendChild(tag("enrollment unfinished", "warn"));
    const actions = cell("");
    if (!w.deleted_at) {
      const del = document.createElement("button");
      del.textContent = "Delete data";
      del.onclick = async () => {
        if (!confirm(`Delete ${w.name}'s face data and PIN? Attendance records are kept for payroll.`)) return;
        await api(`/api/workers/${w.id}`, { method: "DELETE" });
        loaders.workers();
      };
      actions.appendChild(del);
    }
    tr.append(cell(w.id), cell(w.name), cell(w.trade), status,
              cell(w.deleted_at ? "—" : w.retention_until.slice(0, 10)), actions);
    return tr;
  }, "No workers yet. Click “Enroll a worker”.");
};

// ---------- sync status (on-site server) ----------
const ago = ts => {
  const m = Math.round((Date.now() - new Date(ts.replace(" ", "T"))) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
};
async function refreshSync() {
  const s = await api("/api/sync/status");
  const chip = $("sync-chip");
  chip.hidden = !s.enabled;
  if (!s.enabled) return;
  const behind = s.pending > 0 && s.last_error && (!s.last_success || s.last_error > s.last_success);
  chip.className = `tag ${behind ? "warn" : ""}`;
  chip.textContent = behind ? `Offline · ${s.pending} waiting to sync`
    : s.last_success ? `Synced ${ago(s.last_success)}` : "Sync starting…";
  chip.title = s.last_error || "";
}
refreshSync().catch(() => {});
setInterval(() => refreshSync().catch(() => {}), 30000);

// ---------- all sites (central server) ----------
loaders.sites = async () => {
  const sites = await api("/api/sites");
  const list = $("sites-list");
  list.replaceChildren();
  for (const site of sites) {
    const card = document.createElement("div");
    card.className = "card site-card";
    const h = document.createElement("h3");
    h.append(site.id, tag(`${site.on_site.length} on site`));
    if (site.minutes_since_sync > 10) h.appendChild(tag(`no sync for ${Math.round(site.minutes_since_sync)} min`, "warn"));
    const sub = document.createElement("div");
    sub.className = "muted";
    sub.textContent = `Last sync ${ago(site.last_sync_at)}`;
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.style.marginTop = "10px";
    const table = document.createElement("table");
    const tbody = document.createElement("tbody");
    fillTable(tbody, site.on_site, w => {
      const tr = document.createElement("tr");
      tr.append(cell(w.name), cell(w.trade), cell(time(w.since)), ppeCell(w));
      return tr;
    }, "Nobody on site.");
    table.appendChild(tbody); wrap.appendChild(table);
    card.append(h, sub, wrap);
    list.appendChild(card);
  }
};
// show the tab only on a central server that has received data
api("/api/sites").then(s => { $("sites-tab").hidden = !s.length; }).catch(() => {});

// ---------- audit ----------
loaders.audit = async () => {
  fillTable($("audit-body"), await api("/api/audit"), a => {
    const tr = document.createElement("tr");
    tr.append(cell(a.ts), cell(a.actor), cell(a.action.replaceAll("_", " ")), cell(a.worker_id), cell(a.detail));
    return tr;
  }, "Nothing logged yet.");
};

// ---------- enrollment ----------
const POSES = ["Look straight at the camera", "Turn slightly to your left", "Turn slightly to your right",
               "Tilt your chin up a little", "Tilt your chin down a little"];
const dialog = $("enroll");
let enrollId = null, captures = 0, stream = null, capturing = false;

$("enroll-open").onclick = () => {
  for (const id of ["e-name", "e-trade", "e-pin"]) $(id).value = "";
  $("e-consent").checked = false;
  enrollId = null;
  $("e-error1").textContent = $("e-error2").textContent = "";
  $("enroll-step1").hidden = false; $("enroll-step2").hidden = true;
  dialog.showModal();
};

$("e-next").onclick = async () => {
  $("e-error1").textContent = "";
  if (!$("e-consent").checked) { $("e-error1").textContent = "Consent is required before capturing a face."; return; }
  if (!/^\d{4,6}$/.test($("e-pin").value)) { $("e-error1").textContent = "PIN must be 4–6 digits."; return; }
  try {
    const r = await api("/api/enroll/start", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("e-name").value, trade: $("e-trade").value, pin: $("e-pin").value, consent: true }),
    });
    enrollId = r.worker_id; captures = 0;
  } catch (e) { $("e-error1").textContent = e.message; return; }
  $("enroll-step1").hidden = true; $("enroll-step2").hidden = false;
  stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
  $("e-video").srcObject = stream;
  renderCapture();
  capturing = true;
  captureLoop();
};

function renderCapture() {
  $("e-pose").textContent = captures < POSES.length ? POSES[captures] : "All angles captured";
  $("e-dots").replaceChildren(...POSES.map((_, i) => {
    const d = document.createElement("span");
    if (i < captures) d.className = "on";
    return d;
  }));
  $("e-finish").disabled = captures < 3;
}

async function captureLoop() {
  const video = $("e-video"), canvas = document.createElement("canvas");
  while (capturing && captures < POSES.length) {
    await new Promise(r => setTimeout(r, 900));   // time to change pose
    if (!capturing || video.readyState < 2) continue;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const form = new FormData();
    form.append("file", await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.9)), "capture.jpg");
    try {
      const r = await api(`/api/enroll/${enrollId}/capture`, { method: "POST", body: form });
      captures = r.captures;
      $("e-error2").textContent = { no_face: "No face found — move closer", multiple: "Only one person in frame, please",
        mismatch: "That doesn't look like the same person", captured: "" }[r.status] ?? "";
      renderCapture();
    } catch (e) { $("e-error2").textContent = e.message; }
  }
  if (captures >= POSES.length) finishEnrollment();
}

function stopCamera() {
  capturing = false;
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
}

async function finishEnrollment() {
  stopCamera();
  try {
    const r = await api(`/api/enroll/${enrollId}/finish`, { method: "POST" });
    dialog.close();
    if (r.possible_duplicate_of) alert(`Enrolled. Warning: this face closely matches ${r.possible_duplicate_of}. Check this isn't the same person enrolled twice.`);
    loaders.workers();
  } catch (e) { $("e-error2").textContent = e.message; }
}

$("e-finish").onclick = finishEnrollment;
$("e-cancel").onclick = () => dialog.close();
$("e-cancel2").onclick = async () => {
  stopCamera();
  // an abandoned enrollment shouldn't leave face data behind
  if (enrollId) await api(`/api/workers/${enrollId}`, { method: "DELETE" }).catch(() => {});
  dialog.close();
  loaders.workers();
};
dialog.addEventListener("close", stopCamera);

// open the tab named in the URL (e.g. /dashboard#muster), else "On site"
const startTab = document.querySelector(`#tabs button[data-tab="${location.hash.slice(1)}"]`);
if (startTab) { startTab.hidden = false; startTab.click(); } else loaders.onsite();
