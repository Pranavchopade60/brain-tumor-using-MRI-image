const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
let API = store.get("api", "http://127.0.0.1:8000"), file = null, last = null;
const preview = $("preview");

/* ---------- theme, toast, settings ---------- */
document.documentElement.dataset.theme = store.get("theme", "dark");
$("theme").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = t; store.set("theme", t);
};
function toast(msg) {
  const d = document.createElement("div"); d.className = "toast"; d.textContent = msg;
  $("toasts").appendChild(d); setTimeout(() => d.remove(), 3200);
}
$("gear").onclick = () => { $("apiUrl").value = API; $("setDlg").showModal(); };
$("saveApi").onclick = () => {
  API = $("apiUrl").value.trim().replace(/\/$/, ""); store.set("api", API);
  $("setDlg").close(); checkServer(); toast("API URL saved");
};

/* ---------- server status ---------- */
async function checkServer() {
  const s = $("status");
  try {
    const r = await fetch(`${API}/health`); if (!r.ok) throw 0;
    s.className = "status on"; s.querySelector("em").textContent = "Model online";
  } catch { s.className = "status off"; s.querySelector("em").textContent = "Server offline"; }
}
checkServer(); setInterval(checkServer, 15000);

/* ---------- file handling ---------- */
function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return toast("Please choose an image file");
  if (f.size > 10 * 1024 * 1024) return toast("Image is larger than 10 MB");
  file = f; preview.src = URL.createObjectURL(f);
  preview.hidden = false; $("ph").hidden = true;
  $("analyze").disabled = $("reset").disabled = false;
  $("cam").hidden = true;
  $("result").innerHTML = '<div class="empty">Ready – click “Analyze Scan”.</div>';
}
$("file").onchange = (e) => setFile(e.target.files[0]);
const drop = $("drop");
["dragenter", "dragover"].forEach((v) => drop.addEventListener(v, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((v) => drop.addEventListener(v, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => { const f = e.clipboardData && e.clipboardData.files[0]; if (f) setFile(f); });
$("reset").onclick = () => {
  file = null; $("file").value = ""; preview.hidden = true; $("ph").hidden = false;
  $("analyze").disabled = $("reset").disabled = true; $("cam").hidden = true;
  $("result").innerHTML = '<div class="empty">Results will appear here after analysis.</div>';
};

/* ---------- analyze ---------- */
$("analyze").onclick = async () => {
  if (!file) return;
  $("analyze").disabled = true; $("scanline").hidden = false; $("cam").hidden = true;
  $("result").innerHTML = '<div class="loading"><b></b>Analyzing scan…</div>';
  const fd = new FormData(); fd.append("file", file);
  const t0 = performance.now();
  try {
    const r = await fetch(`${API}/predict`, { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Prediction failed");
    d.ms = Math.round(performance.now() - t0); last = d;
    render(d); showCam(d); saveHist(d);
    toast(d.has_tumor ? "⚠ Tumor pattern detected" : "✓ No tumor detected");
  } catch (err) {
    $("result").innerHTML = `<div class="empty">❌ ${err.message}<br><small>Is the backend running at ${API}?</small></div>`;
  } finally { $("analyze").disabled = false; $("scanline").hidden = true; }
};

const ring = (c) => `<svg viewBox="0 0 120 120" class="ring"><circle cx="60" cy="60" r="52"/><circle class="v" cx="60" cy="60" r="52" style="--o:${326.7 * (1 - c / 100)}"/><text x="60" y="67">${c.toFixed(1)}%</text></svg>`;

function render(d, old) {
  const bars = Object.entries(d.probabilities).sort((a, b) => b[1] - a[1]).map(([n, p], i) =>
    `<div class="bar ${i ? "" : "top"}"><div class="row"><span>${n}</span><span>${p.toFixed(2)}%</span></div><div class="t"><div class="f" data-w="${p}"></div></div></div>`).join("");
  $("result").innerHTML = `<div class="res ${d.has_tumor ? "bad" : "good"}">
    <div class="rtop">${ring(d.confidence)}<div>
      <span class="pill">${d.has_tumor ? "⚠ Tumor detected" : "✓ No tumor"}</span>
      <h3>${d.label}</h3>
      <p>${d.has_tumor ? "Region: <b>" + (d.location || "see heatmap") + "</b>" : "No tumor pattern found."}</p>
      ${d.ms ? `<small>⏱ ${d.ms} ms</small>` : ""}
    </div></div>${bars}
    <div class="btns"><button class="btn" id="pdf">📄 Download Report</button>${old ? "" : '<button class="btn ghost" id="dlmap">🖼 Heatmap</button>'}</div></div>`;
  requestAnimationFrame(() => document.querySelectorAll(".bar .f").forEach((e) => (e.style.width = e.dataset.w + "%")));
  $("pdf").onclick = () => report(d);
  if (!old) $("dlmap").onclick = () => { const a = document.createElement("a"); a.href = d.gradcam; a.download = "gradcam.png"; a.click(); };
}

/* ---------- Grad-CAM ---------- */
function showCam(d) {
  if (!d.gradcam) return;
  $("cam").hidden = false;
  $("camBase").src = preview.src; $("camTop").src = d.gradcam;
  $("camInfo").innerHTML = d.has_tumor
    ? `Suspected region: <b>${d.location || "highlighted area"}</b>. Red/yellow = strongest influence on the prediction; the white box marks the approximate area.`
    : "No tumor found. The heatmap only shows where the model focused.";
}
$("op").oninput = (e) => ($("camTop").style.opacity = e.target.value / 100);
document.querySelectorAll(".seg button").forEach((b) => (b.onclick = () => {
  document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
  $("stack").classList.toggle("side", b.dataset.v === "side");
}));
const openZoom = () => { if (!last || !last.gradcam) return; $("zoomImg").src = last.gradcam; $("zoomDlg").showModal(); };
$("zoom").onclick = openZoom; $("stack").onclick = openZoom;

/* ---------- history ---------- */
function thumb() {
  const c = document.createElement("canvas"); c.width = c.height = 96;
  const g = c.getContext("2d"), w = preview.naturalWidth, h = preview.naturalHeight, s = Math.min(w, h);
  g.drawImage(preview, (w - s) / 2, (h - s) / 2, s, s, 0, 0, 96, 96);
  return c.toDataURL("image/jpeg", 0.7);
}
function saveHist(d) {
  const h = store.get("hist", []);
  h.unshift({ t: Date.now(), thumb: thumb(), label: d.label, confidence: d.confidence, has_tumor: d.has_tumor, probabilities: d.probabilities, location: d.location, ms: d.ms });
  store.set("hist", h.slice(0, 12)); drawHist();
}
function drawHist() {
  const h = store.get("hist", []);
  $("hist").innerHTML = h.length ? h.map((x, i) =>
    `<button class="hi" data-i="${i}"><img src="${x.thumb}" alt="" /><b>${x.label}</b><small>${x.confidence.toFixed(1)}% · ${new Date(x.t).toLocaleString([], { dateStyle: "short", timeStyle: "short" })}</small><i class="${x.has_tumor ? "bad" : "good"}"></i></button>`).join("")
    : '<p class="muted">No scans yet – your analyses will be saved here (in this browser only).</p>';
}
$("hist").onclick = (e) => {
  const b = e.target.closest(".hi"); if (!b) return;
  render(store.get("hist", [])[b.dataset.i], true); $("cam").hidden = true;
  document.getElementById("scan").scrollIntoView(); toast("Saved result (heatmap is not stored)");
};
$("clearHist").onclick = () => { store.set("hist", []); drawHist(); toast("History cleared"); };
drawHist();

/* ---------- PDF report ---------- */
function report(d) {
  const J = window.jspdf && window.jspdf.jsPDF;
  if (!J) return toast("PDF library not loaded (needs internet access)");
  const p = new J();
  p.setFontSize(20); p.text("NeuroScan AI - MRI Analysis Report", 14, 20);
  p.setFontSize(10); p.text(new Date().toLocaleString(), 14, 27);
  p.setFontSize(15); p.text(`Result: ${d.label} (${d.confidence.toFixed(2)}%)`, 14, 40);
  p.setFontSize(11); let y = 48;
  if (d.location) { p.text("Suspected region: " + d.location, 14, y); y += 8; }
  Object.entries(d.probabilities).forEach(([k, v]) => { p.text(`${k}: ${v.toFixed(2)}%`, 14, y); y += 6; });
  if (d.gradcam && preview.naturalWidth) {
    const c = document.createElement("canvas"), s = Math.min(1, 500 / preview.naturalWidth);
    c.width = preview.naturalWidth * s; c.height = preview.naturalHeight * s;
    c.getContext("2d").drawImage(preview, 0, 0, c.width, c.height);
    const w = 88, h = w * c.height / c.width; y += 6;
    p.addImage(c.toDataURL("image/jpeg", 0.85), "JPEG", 14, y, w, h);
    p.addImage(d.gradcam, "PNG", 108, y, w, h);
    p.setFontSize(9); p.text("Original MRI", 14, y + h + 5); p.text("Grad-CAM heatmap", 108, y + h + 5);
  }
  p.setFontSize(9);
  p.text(p.splitTextToSize("Disclaimer: educational project, not a medical device. This report must not be used for diagnosis. Consult a qualified doctor.", 180), 14, 280);
  p.save("neuroscan-report.pdf");
}

/* ---------- tumor info cards ---------- */
const TYPES = [
  ["🧠", "Glioma", "Arises from glial (supporting) cells of the brain. Ranges from slow-growing to aggressive grades.", "Intra-axial"],
  ["🛡️", "Meningioma", "Grows from the meninges, the membranes covering the brain. Often slow-growing and frequently benign.", "Extra-axial"],
  ["🎯", "Pituitary", "Develops in the pituitary gland at the base of the brain; can affect hormones and vision. Most are benign adenomas.", "Sellar region"],
  ["✅", "No Tumor", "No tumor-like pattern was found in the scan by the model. A doctor must still confirm.", "Normal"],
];
$("typeGrid").innerHTML = TYPES.map(([e, n, t, g]) => `<div class="tcard"><div class="e">${e}</div><h3>${n}</h3><p>${t}</p><span class="tag">${g}</span></div>`).join("");
