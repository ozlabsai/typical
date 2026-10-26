// Typical air-defence demo. Three altitudes: the decision for one threat (panel), its story over time (flight strip),
// tonight's outcome (Tonight block). One accent (brand blue) points at the focal threat and the call; all else greyscale.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (m) => (m < 0.001 ? "<$1k" : m < 1 ? `$${Math.round(m * 1000)}k` : m < 1000 ? `$${m.toFixed(1)}M` : `$${(m / 1000).toFixed(2)}B`);
const BLUE = [37, 72, 246], GREY = [138, 138, 142], INK = [16, 16, 16], HIT = [255, 59, 48];
const VIEWS = { theatre: [40.5, 31.2, 4.7, 48], north: [35.35, 32.95, 8.3, 50], south: [39.5, 24.5, 4.6, 45] };
const SCEN_VIEW = { northern_barrage: "north", april_mixed: "theatre", october_saturation: "theatre", june_attrition: "theatre", houthi_eilat: "south" };
const DONE = ["killed", "leaked", "fell_open"];

// ---------------------------------------------------------------- copy (EN / HE)
const I18N = {
  en: {
    brand: "Air defence", skip: "Skip ahead", settings: "Settings", lang: "עברית", play: "Play", pause: "Pause", deciding: "Deciding…",
    skipped: (x) => `Skipped ${x} of quiet`,
    cls: { B: "Ballistic missile", R: "Rocket", C: "Cruise missile", D: "Drone" },
    to: "to", openGround: "Open ground near", priority: "priority", impactIn: "Impact in", follow: "Follow the most urgent",
    following: "Most urgent threat", pinned: "Pinned",
    callBy: { typical: "Typical's call", planner: "The planner's call", naive: "Fire at everything" },
    conf: (p) => `${p}% confident`,
    act: { hold: "Let it fall", fire: (e) => `Fire ${e} now`, fire2: (e) => `Fire two ${e} now`, wait: (e) => `Wait for ${e}` },
    pk: ["is almost certain to stop it", "is very likely to stop it", "is likely to stop it", "may stop it", "is unlikely to stop it"],
    whyFire: (e, pk) => `${e} can engage now and ${pk}.`,
    whyFire2: (e, pk) => `Two ${e} together: the pair ${pk}.`,
    whyWait: (e, m, pk) => `${e} reaches it in ${m} and ${pk}.`,
    whyHoldOpen: "It will land in open ground. Nothing worth an interceptor is at risk.",
    whyHold: "The damage it can do is worth less than any interceptor that can reach it.",
    vs: { raid: (r) => `${r} would spend interceptors tonight's raid still needs.`, campaign: (r) => `${r} would spend stock the campaign has to keep.`,
      local: (r, d) => `${r} costs ${d} more for this threat, in interceptors and risk.` },
    status: { killed: "Intercepted", leaked: "Hit a defended site", fell_open: "Fell in open ground" },
    operator: "Sent to the operator, who took the planner's call",
    unsure: (x, p) => `Typical was unsure: its top pick, ${x}, had ${p}%`,
    agrees: "The planner agrees.", differs: (x) => `The planner would: ${x}.`,
    reads: "What Typical reads", notYet: "No interceptor can reach it yet. Typical decides once one can.",
    empty: "Press Play. The most urgent threat and Typical's call appear here.",
    tonight: "Tonight", attacker: (x) => `Attacker spent ${x}`, spent: "Spent", hits: "Hits on sites", a3: "Arrow 3 left",
    greyIs: (p) => `Grey: ${p}`, naive: "firing at everything", plannerLow: "the planner", foot: (a, ms) => `Agrees with the planner ${a}% · ${ms} ms per call`, cmp: "Compare planner",
    planner: "Planner",
    strip: (id, t, s) => `Flight of ${id} · ${t} to ${s}`, now: "now", impact: "impact", calls: "Calls", hit: "hit", miss: "miss",
    posture: "Magazine posture", postures: ["Healthy", "Conserve", "Critical"], prio: "Priority sites", assets: "Assets and conditions",
    allied: "US THAAD and Navy", weather: "Cloud cover (laser degraded)", knobs: "Model and sensors", pkScale: "Pk scale", escTop: "Escalate below",
    warn: "Early warning, min", dmg: "Damage weights, $M per hit", tier: { strategic: "Strategic site", city: "City", infra: "Infrastructure" },
    prioMult: "Priority ×", warhead: "Warhead weight (1 = 500 kg)", perSite: "Value per site (blank = by type)",
    note: "Posture, stock, damage weights and priorities feed every lens. Typical reads them as sentences; the planner reads numbers.",
  },
  he: {
    brand: "הגנה אווירית", skip: "דלג קדימה", settings: "הגדרות", lang: "English", play: "הפעל", pause: "השהה", deciding: "מחליט…",
    skipped: (x) => `דולגו ${x} שקטות`,
    cls: { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" },
    to: "לעבר", openGround: "שטח פתוח ליד", priority: "בעדיפות", impactIn: "פגיעה בעוד", follow: "עקוב אחרי הדחוף",
    following: "האיום הדחוף ביותר", pinned: "נעוץ",
    callBy: { typical: "ההחלטה של Typical", planner: "ההחלטה של המתכנן", naive: "יורים על הכול" },
    conf: (p) => `ביטחון ${p}%`,
    act: { hold: "לתת לו ליפול", fire: (e) => `לירות ${e} עכשיו`, fire2: (e) => `לירות שני ${e} עכשיו`, wait: (e) => `להמתין ל${e}` },
    pk: ["כמעט בטוח שיעצור אותו", "סביר מאוד שיעצור אותו", "סביר שיעצור אותו", "עשוי לעצור אותו", "לא סביר שיעצור אותו"],
    whyFire: (e, pk) => `${e} יכול ליירט עכשיו, ו${pk}.`,
    whyFire2: (e, pk) => `שני ${e} יחד: ${pk}.`,
    whyWait: (e, m, pk) => `${e} יגיע אליו בעוד ${m}, ו${pk}.`,
    whyHoldOpen: "הוא ייפול בשטח פתוח. אין שם דבר ששווה מיירט.",
    whyHold: "הנזק שהוא יכול לגרום שווה פחות מכל מיירט שמגיע אליו.",
    vs: { raid: (r) => `${r} ינצל מיירטים שהמתקפה הלילה עוד צריכה.`, campaign: (r) => `${r} ינצל מלאי שהמערכה חייבת לשמור.`,
      local: (r, d) => `${r} עולה ${d} יותר על האיום הזה, במיירטים ובסיכון.` },
    status: { killed: "יורט", leaked: "פגע באתר מוגן", fell_open: "נפל בשטח פתוח" },
    operator: "הועבר למפעיל, שביצע את החלטת המתכנן",
    unsure: (x, p) => `Typical לא היה בטוח: הבחירה המובילה שלו, ${x}, קיבלה ${p}%`,
    agrees: "המתכנן מסכים.", differs: (x) => `המתכנן היה בוחר: ${x}.`,
    reads: "מה Typical קורא (באנגלית)", notYet: "אף מיירט עדיין לא מגיע אליו. Typical יחליט כשיגיע.",
    empty: "לחצו הפעל. האיום הדחוף ביותר וההחלטה של Typical יופיעו כאן.",
    tonight: "הלילה", attacker: (x) => `התוקף הוציא ${x}`, spent: "הוצאה", hits: "פגיעות באתרים", a3: "חץ 3 שנותרו",
    greyIs: (p) => `באפור: ${p}`, naive: "יורים על הכול", plannerLow: "המתכנן", foot: (a, ms) => `הסכמה עם המתכנן ${a}% · ${ms} מ״ש להחלטה`, cmp: "השווה למתכנן",
    planner: "מתכנן",
    strip: (id, t, s) => `מסלול ${id} · ${t} לעבר ${s}`, now: "עכשיו", impact: "פגיעה", calls: "החלטות", hit: "פגע", miss: "החטיא",
    posture: "מצב מלאים", postures: ["תקין", "שימור", "קריטי"], prio: "אתרים בעדיפות", assets: "נכסים ותנאים",
    allied: "THAAD וצי ארה״ב", weather: "עננות (לייזר מוגבל)", knobs: "מודל וחיישנים", pkScale: "מקדם Pk", escTop: "למפעיל מתחת",
    warn: "התרעה מוקדמת, דק׳", dmg: "משקלי נזק, מיליון $ לפגיעה", tier: { strategic: "אתר אסטרטגי", city: "עיר", infra: "תשתית" },
    prioMult: "מכפיל עדיפות", warhead: "משקל ראש קרב (1 = 500 ק״ג)", perSite: "ערך לאתר (ריק = לפי סוג)",
    note: "מצב המלאים, המלאי, משקלי הנזק והעדיפויות נכנסים לכל עדשה. Typical קורא אותם כמשפטים; המתכנן כמספרים.",
  },
};
let LANG = localStorage.getItem("airdef-lang") || "en";
const T = () => I18N[LANG];
const nm = (o) => (LANG === "he" && o.name_he ? o.name_he : o.name);
const glyphs = (s) => (LANG === "he" ? [...s].reverse().join("") : s);   // deck.gl TextLayer has no bidi
const effName = (e) => nm(META.effectors[e]).replace("F-15/F-35 ", "").replace("F-16 laser-guided rocket (APKWS)", "APKWS").replace("GPS jamming (EW)", "EW jamming").replace(" laser", "");
const actLabel = (key) => { if (!key) return "—"; if (key === "hold") return T().act.hold; const [a, e] = key.split(":"); return T().act[a](effName(e)); };
const siteName = (sid) => { const s = META.sites.find((x) => x.id === sid); return s ? nm(s) : ""; };
const fmtT = (s) => { const neg = s < 0; s = Math.abs(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60); return (neg ? "−" : "") + (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0"); };
const mins = (s) => (LANG === "he" ? (s < 90 ? `${Math.round(s)} שניות` : `${Math.round(s / 60)} דקות`) : (s < 90 ? `${Math.round(s)} seconds` : `${Math.round(s / 60)} minutes`));
const pkIdx = (p) => (p >= 0.95 ? 0 : p >= 0.85 ? 1 : p >= 0.7 ? 2 : p >= 0.5 ? 3 : 4);

let META, PATHS = {}, PATH_SCEN = null, S = null, fetchedAt = 0, selected = null, pinned = false, detail = null, showPlanner = false;
const viewPol = "typical";
const flashSeen = new Map();

// ---------------------------------------------------------------- map
const map = new maplibregl.Map({ container: "map", style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  center: [35.0, 31.6], zoom: 6.6, pitch: 45, maxPitch: 75, attributionControl: { compact: true } });
const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
map.addControl(overlay);
const flyTo = (v) => { const [lon, lat, zoom, pitch] = VIEWS[v]; map.flyTo({ center: [lon, lat], zoom, pitch, duration: 1600 }); };
const api = (p, body) => fetch(p, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {}).then((r) => r.json());

async function init() {
  META = await api("/api/meta");
  $("#lang").onclick = () => { LANG = LANG === "en" ? "he" : "en"; localStorage.setItem("airdef-lang", LANG); applyLang(); };
  $("#scen").onclick = async (e) => { const s = e.target.closest("[data-s]")?.dataset.s; if (!s) return; selected = null; pinned = false; detail = null; await api("/api/control", { action: "reset", scenario: s }); flyTo(SCEN_VIEW[s]); };
  $("#speeds").innerHTML = [1, 5, 20, 60].map((x) => `<button class="dc-tab" data-x="${x}" role="tab">${x}×</button>`).join("");
  $("#speeds").onclick = (e) => { const x = e.target.closest("[data-x]")?.dataset.x; if (x) api("/api/control", { action: "speed", speed: +x }); };
  $("#play").onclick = () => api("/api/control", { action: S?.running ? "pause" : "play" });
  $("#skip").onclick = () => api("/api/control", { action: "skip" });
  $("#gear").onclick = () => { const o = $("#sheet").classList.toggle("open"); $("#gear").setAttribute("aria-expanded", o); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#sheet").classList.remove("open"); $("#gear").setAttribute("aria-expanded", false); } });
  $("#decision").onclick = (e) => {
    if (e.target.closest("#followBtn")) { pinned = false; pickFollow(true); }
    if (e.target.closest(".dc-accordion__trigger")) e.target.closest(".dc-accordion__item").classList.toggle("is-open");
  };
  $("#tonight").onclick = (e) => { if (e.target.closest("#cmpBtn")) { showPlanner = !showPlanner; renderTonight(); } };
  await poll();
  applyLang();
  flyTo(SCEN_VIEW[S.scenario]);
  setInterval(poll, 250);
  setInterval(loadDetail, 600);
  requestAnimationFrame(frame);
}

function applyLang() {
  document.documentElement.lang = LANG;
  document.documentElement.dir = LANG === "he" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = T()[el.dataset.i18n]));
  $("#lang").textContent = T().lang;
  $("#scen").innerHTML = Object.entries(META.scenarios).map(([k, v]) => {
    const [title, blurb] = LANG === "he" ? [v.title_he, v.blurb_he] : [v.title, v.blurb];
    return `<button class="dc-tab" role="tab" data-s="${k}" title="${esc(blurb)}">${esc(title.split(":")[0])}</button>`;
  }).join("");
  renderBar(); renderTonight(); renderDecision(); renderStrip(); buildSheet();
}

async function poll() {
  try { S = await api(`/api/state?policy=${viewPol}`); fetchedAt = performance.now(); } catch { return; }
  if (S.scenario !== PATH_SCEN) {
    const P = await api("/api/paths");
    PATHS = Object.fromEntries(P.threats.map((t) => [t.id, t]));
    PATH_SCEN = P.scenario; flashSeen.clear();
  }
  if (!pinned) pickFollow(false);
  renderBar(); renderTonight();
}

function pickFollow(force) {   // the most urgent threat that already has a call, else the most urgent live one
  const live = S.tracks.filter((x) => !DONE.includes(x.status));
  const cand = live.filter((x) => x.choice).sort((a, b) => a.tti - b.tti)[0] || [...live].sort((a, b) => a.tti - b.tti)[0];
  const cur = S.tracks.find((x) => x.id === selected);
  const stale = !cur || (DONE.includes(cur.status) && S.t - cur.impact > 4 * Math.max(1, S.speed));   // let an outcome show briefly
  if (cand && (force || stale)) { selected = cand.id; loadDetail(); }
}
function select(id) { selected = id; pinned = true; loadDetail(); }
async function loadDetail() {
  if (!selected) { detail = null; renderDecision(); renderStrip(); return; }
  try { detail = await api(`/api/track/${selected}?policy=${viewPol}`); } catch { return; }
  renderDecision(); renderStrip();
}

function simNow() { if (!S) return 0; if (!S.running || S.backlog) return S.t; return Math.min(S.end, S.t + ((performance.now() - fetchedAt) / 1000) * S.speed); }
function posAt(path, t) {
  const p = path.p;
  if (t <= p[0][3]) return p[0];
  if (t >= p[p.length - 1][3]) return p[p.length - 1];
  let lo = 0, hi = p.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (p[m][3] <= t) lo = m; else hi = m; }
  const a = p[lo], b = p[hi], f = (t - a[3]) / (b[3] - a[3]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, t];
}

// ---------------------------------------------------------------- map layers (greyscale; the focal threat in blue)
function frame() {
  requestAnimationFrame(frame);
  if (!S || !META) return;
  const t = simNow(), prio = new Set(S.cfg.priority_sites);
  const live = [], trails = [], futures = [];
  for (const tr of S.tracks) {
    const path = PATHS[tr.id]; if (!path) continue;
    if (t > Math.min(S.killed[tr.id] ?? Infinity, path.t0 + path.T)) continue;
    const cur = posAt(path, t), sel = tr.id === selected, col = sel ? BLUE : GREY;
    const past = path.p.filter((q) => q[3] < t).map((q) => q.slice(0, 3)); past.push(cur.slice(0, 3));
    trails.push({ path: past, color: [...col, sel ? 255 : 150], w: sel ? 3 : 1.25 });
    futures.push({ path: [cur.slice(0, 3), ...path.p.filter((q) => q[3] > t).map((q) => q.slice(0, 3))], color: [...col, sel ? 110 : 45], w: sel ? 2 : 1 });
    live.push({ pos: cur.slice(0, 3), tr, sel });
  }
  const batt = Object.fromEntries(META.batteries.map((b) => [b.id, b]));
  const shots = [], flashes = [], nowMs = performance.now();
  for (const f of S.flights) {
    if (t < f.t0) continue;
    const b = batt[f.b], g = Math.min(1, (t - f.t0) / Math.max(1, f.t1 - f.t0)), pts = [];
    for (let k = 0; k <= 24 * g; k++) { const s = k / 24; pts.push([b.lon + (f.lon - b.lon) * s, b.lat + (f.lat - b.lat) * s, f.alt * Math.pow(s, 0.6)]); }
    if (g < 1) shots.push({ path: pts, sel: f.th === selected });
    else {
      const key = `${f.b}${f.th}${f.t1}`;
      if (!flashSeen.has(key)) flashSeen.set(key, nowMs);
      const age = (nowMs - flashSeen.get(key)) / 1000;
      if (age < 1.6) flashes.push({ pos: [f.lon, f.lat, f.alt], age, hit: f.hit });
    }
  }
  const selTr = S.tracks.find((x) => x.id === selected);
  const leaks = S.tracks.filter((x) => x.status === "leaked").map((x) => PATHS[x.id]?.p.at(-1)).filter(Boolean);
  const batts = META.batteries.filter((b) => S.cfg.allied || !META.effectors[b.eff].allied);
  overlay.setProps({ layers: [
    new deck.ScatterplotLayer({ id: "sites", data: META.sites, getPosition: (d) => [d.lon, d.lat], getRadius: (d) => d.r * 1000,
      getFillColor: (d) => (prio.has(d.id) ? [16, 16, 16, 14] : [0, 0, 0, 0]), stroked: true, lineWidthUnits: "pixels",
      getLineWidth: (d) => (prio.has(d.id) ? 1.5 : 0.75), getLineColor: (d) => (prio.has(d.id) ? [...INK, 200] : [197, 197, 199, 255]),
      updateTriggers: { getFillColor: [...prio].join(), getLineColor: [...prio].join(), getLineWidth: [...prio].join() } }),
    new deck.TextLayer({ id: "siteNames", data: META.sites.filter((s) => prio.has(s.id) && map.getZoom() > 5.6), getPosition: (d) => [d.lon, d.lat],
      getText: (d) => glyphs(nm(d)), getSize: 12, getColor: [...INK, 220], getPixelOffset: [0, -16], fontFamily: "Figtree, Heebo, Arial, sans-serif",
      fontWeight: 500, characterSet: "auto", outlineWidth: 3, outlineColor: [255, 255, 255, 230], fontSettings: { sdf: true }, updateTriggers: { getText: LANG } }),
    new deck.ScatterplotLayer({ id: "batt", data: batts, getPosition: (d) => [d.lon, d.lat], radiusUnits: "pixels", getRadius: 2.25, getFillColor: [80, 80, 80, 200] }),
    selTr && !DONE.includes(selTr.status) && new deck.ScatterplotLayer({ id: "ipp", data: [selTr], getPosition: (d) => [d.ipp[0], d.ipp[1]],
      getRadius: (d) => Math.max(d.ipp[2], 0.6) * 1000, filled: true, stroked: true, getFillColor: [...BLUE, 22], getLineColor: [...BLUE, 220], lineWidthUnits: "pixels", getLineWidth: 1.5 }),
    new deck.PathLayer({ id: "future", data: futures, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w }),
    new deck.PathLayer({ id: "trail", data: trails, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w }),
    new deck.PathLayer({ id: "shots", data: shots, getPath: (d) => d.path, getColor: (d) => (d.sel ? [...BLUE, 255] : [...INK, 110]), widthUnits: "pixels", getWidth: (d) => (d.sel ? 2.5 : 1) }),
    new deck.ScatterplotLayer({ id: "heads", data: live, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => (d.sel ? 7 : 3.5),
      getFillColor: (d) => (d.sel ? BLUE : [110, 110, 114]), stroked: true, getLineColor: [255, 255, 255], lineWidthUnits: "pixels", getLineWidth: (d) => (d.sel ? 2.5 : 1),
      pickable: true, onClick: (i) => select(i.object.tr.id) }),
    new deck.ScatterplotLayer({ id: "flash", data: flashes, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => 5 + d.age * 22, filled: false,
      stroked: true, lineWidthUnits: "pixels", getLineWidth: 1.5, getLineColor: (d) => [...(d.hit ? INK : GREY), 255 * (1 - d.age / 1.6)] }),
    new deck.ScatterplotLayer({ id: "leaks", data: leaks, getPosition: (d) => [d[0], d[1]], radiusUnits: "pixels", getRadius: 6, getFillColor: [...HIT, 255],
      stroked: true, getLineColor: [...HIT, 70], lineWidthUnits: "pixels", getLineWidth: 7 }),
  ].filter(Boolean) });
  $("#clock").textContent = "T+" + fmtT(t);
  stripNow(t);
}

// ---------------------------------------------------------------- bar
let lastSkip = { v: 0, at: -1e9 };
function renderBar() {
  if (!S) return;
  $("#play").textContent = S.running ? T().pause : T().play;
  document.querySelectorAll("#speeds .dc-tab").forEach((b) => b.setAttribute("aria-selected", +b.dataset.x === S.speed));
  document.querySelectorAll("#scen .dc-tab").forEach((b) => b.setAttribute("aria-selected", b.dataset.s === S.scenario));
  if (S.skipped !== lastSkip.v) lastSkip = { v: S.skipped, at: performance.now() };
  const skip = S.running && S.skipped > 0 && performance.now() - lastSkip.at < 2500;
  $("#note").textContent = S.backlog ? T().deciding : skip ? T().skipped(fmtT(S.skipped)) : "";
}

// ---------------------------------------------------------------- decision panel
function reason(i, D) {
  const tt = T(), o = Object.fromEntries(i.opts.map(([k, , total, lenses, pk, launch]) => [k, { total, lenses, pk, launch }]));
  const c = o[D.choice]; if (!c) return "";
  const [act, e] = D.choice.split(":");
  let s = act === "fire" ? tt.whyFire(effName(e), tt.pk[pkIdx(c.pk)]) : act === "fire2" ? tt.whyFire2(effName(e), tt.pk[pkIdx(c.pk)])
    : act === "wait" ? tt.whyWait(effName(e), mins(Math.max(0, c.launch - i.t)), tt.pk[pkIdx(c.pk)]) : D.open_ground ? tt.whyHoldOpen : tt.whyHold;
  const runner = Object.keys(o).filter((k) => k !== D.choice).sort((a, b) => o[a].total - o[b].total)[0];
  if (runner) {   // why not the runner-up: the lens where it loses most
    const r = o[runner], d = Object.fromEntries(["local", "raid", "campaign"].map((k) => [k, (r.lenses[k] || 0) - (c.lenses[k] || 0)]));
    const k = Object.keys(d).sort((a, b) => d[b] - d[a])[0];
    if (d[k] > 0.0005) s += " " + (k === "local" ? tt.vs.local(actLabel(runner), money(d[k])) : tt.vs[k](actLabel(runner)));
  }
  return s;
}

function renderDecision() {
  const el = $("#decision"), tt = T();
  if (!detail || !META) { el.innerHTML = `<p class="empty">${tt.empty}</p>`; return; }
  const D = detail, ty = META.threat_types[D.kind], done = DONE.includes(D.status);
  const target = D.open_ground ? `${tt.openGround} <b>${esc(siteName(D.site))}</b>`
    : `${tt.to} <b>${esc(siteName(D.site))}</b>${D.priority ? `<span class="tag">${tt.priority}</span>` : ""}`;
  let h = `<div class="row"><p class="kicker">${pinned ? tt.pinned : tt.following} · <span class="num">${D.id}</span> · ${tt.cls[D.cls]}</p>
      ${pinned ? `<button id="followBtn" class="dc-pill dc-pill--ghost dc-pill--sm">${tt.follow}</button>` : ""}</div>
    <div class="row"><div><h2 class="threat">${esc(nm(ty).replace(/ (MRBM|SRBM)$/i, ""))}</h2><div class="target">${target}</div></div>
      <div class="eta"><p class="kicker">${done ? "" : tt.impactIn}</p><div class="v num">${done ? "" : fmtT(D.impact - D.t)}</div></div></div>`;
  const i = D.info;
  if (done) {
    el.innerHTML = h + `<div class="call"><p class="kicker">${tt.callBy[viewPol]}</p><div class="what" style="color:${D.status === "leaked" ? "var(--hit)" : "var(--dc-ink-2)"}">${tt.status[D.status]}</div></div>`;
    return;
  }
  if (!i) { el.innerHTML = h + `<p class="empty">${tt.notYet}</p>`; return; }
  const top = i.dist ? Object.keys(i.dist).sort((a, b) => i.dist[b] - i.dist[a])[0] : null;
  const conf = !i.dist ? null : D.escalated ? tt.unsure(actLabel(top), Math.round(i.dist[top] * 100)) : tt.conf(Math.round(i.dist[D.choice] * 100));
  h += `<div class="call"><p class="kicker">${tt.callBy[viewPol]}</p><div class="what">${esc(actLabel(D.choice))}</div>
      ${conf ? `<span class="conf">${esc(conf)}</span>` : ""}<p class="why">${esc(reason(i, D))}</p>
      ${D.escalated ? `<div class="op">${tt.operator}</div>` : ""}</div>`;
  const order = i.dist ? Object.keys(i.dist).sort((a, b) => i.dist[b] - i.dist[a]) : i.opts.map((o) => o[0]);
  const rest = order.filter((k) => k !== D.choice).slice(0, 3);
  if (rest.length) h += `<div class="alts">${rest.map((k) => `<div class="alt"><span>${esc(actLabel(k))}</span><span class="num">${i.dist ? Math.round(i.dist[k] * 100) + "%" : ""}</span></div>`).join("")}</div>`;
  if (i.planner) h += `<p class="agree">${i.planner === D.choice ? tt.agrees : esc(tt.differs(actLabel(i.planner)))}</p>`;
  const open = el.querySelector(".reads .dc-accordion__item")?.classList.contains("is-open");
  h += `<div class="reads dc-accordion"><div class="dc-accordion__item ${open ? "is-open" : ""}"><button class="dc-accordion__trigger">${tt.reads}<span>${open ? "−" : "+"}</span></button>
      <div class="dc-accordion__panel"><div><pre>${esc(i.state)}</pre></div></div></div></div>`;
  el.innerHTML = h;
}

// ---------------------------------------------------------------- tonight
function renderTonight() {
  if (!S) return;
  const L = S.ledger, ty = L.typical, nv = L.naive, pl = L.planner, tt = T();
  if (!ty) return;
  const a3 = (x) => (x.init.arrow3 ? `${Math.round(x.left.arrow3)}/${x.init.arrow3}` : "—");
  const other = showPlanner && pl ? pl : nv, cmp = (f) => f(other);
  const d = Math.max(ty.decisions, 1);
  $("#tonight").innerHTML = `<div class="row"><p class="kicker">${tt.tonight}</p><p class="kicker num">${tt.attacker(money(S.attacker_cost))}</p></div>
    <div class="stats">
      <div class="stat"><div class="l">${tt.spent}</div><div class="v num">${money(ty.spent)}</div><div class="vs num">${cmp((x) => money(x.spent))}</div></div>
      <div class="stat"><div class="l">${tt.hits}</div><div class="v num ${ty.leaks ? "hit" : ""}">${ty.leaks}</div><div class="vs num">${cmp((x) => x.leaks)}</div></div>
      <div class="stat"><div class="l">${tt.a3}</div><div class="v num">${a3(ty)}</div><div class="vs num">${cmp(a3)}</div></div>
    </div>
    <div class="legend"><i></i>${tt.greyIs(showPlanner && pl ? tt.plannerLow : tt.naive)}</div>
    <div class="tfoot"><span>${S.model !== "planner only" ? tt.foot(Math.round((ty.agree / d) * 100), Math.round(ty.ms / Math.max(ty.calls, 1))) : ""}</span>
      <button id="cmpBtn" class="chip" aria-pressed="${showPlanner}">${tt.cmp}</button></div>`;
}

// ---------------------------------------------------------------- flight strip (the threat's story over time)
let stripGeom = null;
function renderStrip() {
  const el = $("#strip"), tt = T();
  if (!detail || !META) { el.innerHTML = ""; stripGeom = null; return; }
  const D = detail, W = Math.max(el.clientWidth - 48, 400), H = Math.max(el.clientHeight - 44, 100), LW = 128, PW = W - LW - 64;
  const hist0 = D.history.length ? D.history[0][0] : D.t;
  const x0 = Math.max(D.t0, Math.min(D.t, hist0) - 0.1 * (D.impact - Math.min(D.t, hist0))), span = Math.max(D.impact - x0, 1);
  const X = (t) => LW + ((Math.min(Math.max(t, x0), D.impact) - x0) / span) * PW;
  const effs = [...new Set([...Object.keys(D.windows), ...D.shots.map((s) => s.eff)])].sort((a, b) => META.effectors[b].layer - META.effectors[a].layer).slice(0, 6);
  const top = 34, RH = Math.min(22, (H - top - 22) / Math.max(effs.length, 1)), yb = top + effs.length * RH;
  const callEff = D.choice && D.choice !== "hold" ? D.choice.split(":")[1] : null;
  let g = `<text x="0" y="14">${tt.calls}</text>`;
  D.history.forEach(([t, key], n) => {   // labelled call markers, alternating rows so neighbours don't collide
    const x = X(t), y = n % 2 ? 28 : 14, last = n === D.history.length - 1;
    g += `<circle cx="${x}" cy="${y - 4}" r="4" fill="${last ? "#2548f6" : "#8a8a8e"}"/><text class="call-l" x="${x + 8}" y="${y}" ${last ? 'style="fill:#2548f6"' : ""}>${esc(actLabel(key))}</text>`;
  });
  effs.forEach((e, k) => {
    const y = top + k * RH, w = D.windows[e], on = e === callEff;
    g += `<text class="lane-l ${on ? "on" : ""}" x="0" y="${y + RH * 0.65}">${esc(effName(e))}</text>`;
    g += `<rect x="${LW}" y="${y + RH * 0.2}" width="${PW}" height="${RH * 0.6}" rx="${RH * 0.3}" fill="#f5f5f5"/>`;
    if (w) g += `<rect x="${X(w[0])}" y="${y + RH * 0.2}" width="${Math.max(4, X(w[1]) - X(w[0]))}" height="${RH * 0.6}" rx="${RH * 0.3}" fill="${on ? "#2548f6" : "#c5c5c7"}"/>`;
    for (const s of D.shots.filter((x) => x.eff === e)) {
      const fin = D.t >= s.t1;
      g += `<line x1="${X(s.t0)}" x2="${X(fin ? s.t1 : D.t)}" y1="${y + RH / 2}" y2="${y + RH / 2}" stroke="#101010" stroke-width="2.5" stroke-linecap="round"/>`;
      if (fin) g += `<text x="${X(s.t1) + 6}" y="${y + RH * 0.7}" style="fill:${s.hit ? "#101010" : "#8a8a8e"};font-size:13px">${s.hit ? "✓ " + tt.hit : "✕ " + tt.miss}</text>`;
    }
  });
  const step = [10, 30, 60, 120, 300, 600, 1800, 3600, 7200].find((s) => span / s <= 8) || 7200;
  for (let tm = step; D.impact - tm >= x0; tm += step) g += `<text x="${X(D.impact - tm)}" y="${yb + 16}" text-anchor="middle">−${fmtT(tm)}</text>`;
  g += `<line x1="${X(D.impact)}" x2="${X(D.impact)}" y1="${top}" y2="${yb}" stroke="#101010" stroke-width="1.5"/>`;
  g += `<text x="${X(D.impact) + 6}" y="${top + 12}" style="fill:#101010">${tt.impact}</text>`;
  g += `<g id="nowline"><line x1="0" x2="0" y1="${top - 2}" y2="${yb + 2}" stroke="#2548f6" stroke-width="2"/><text x="0" y="${yb + 16}" text-anchor="middle" style="fill:#2548f6;font-weight:500">${tt.now}</text></g>`;
  const ty = META.threat_types[D.kind];
  const dest = D.open_ground ? `${tt.openGround} ${siteName(D.site)}` : siteName(D.site);
  el.innerHTML = `<p class="kicker" style="margin:0 0 .25rem">${esc(tt.strip(D.id, nm(ty), dest))}</p><svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" direction="ltr">${g}</svg>`;
  stripGeom = { X };
  stripNow(simNow());
}
function stripNow(t) {   // the playhead moves every frame, between strip re-renders
  const g = document.getElementById("nowline");
  if (g && stripGeom) g.setAttribute("transform", `translate(${stripGeom.X(t)},0)`);
}

// ---------------------------------------------------------------- settings sheet
function buildSheet() {
  if (!S) return;
  const tt = T(), c = S.cfg;
  const rng = (attrs, label, val, shown) => `<label class="rng"><span>${label}</span><input type="range" ${attrs} value="${val}"><output class="num">${shown}</output></label>`;
  const heCls = { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" }, enCls = { B: "Ballistic missile", R: "Rocket", C: "Cruise missile", D: "Drone" };
  const whLabel = (k) => (META.threat_types[k] ? nm(META.threat_types[k]) : (LANG === "he" ? heCls : enCls)[k]);
  $("#sheet").innerHTML = `
    <div class="field"><p class="kicker">${tt.posture}</p><div class="dc-tabs dc-tabs--sm" id="reserve">${tt.postures.map((p, k) => `<button class="dc-tab" data-r="${k}" aria-selected="${c.reserve === k}">${p}</button>`).join("")}</div></div>
    <div class="field"><p class="kicker">${tt.prio}</p><div class="chips" id="prio">${META.sites.map((s) => `<button class="chip" data-site="${s.id}" aria-pressed="${c.priority_sites.includes(s.id)}">${esc(nm(s))}</button>`).join("")}</div></div>
    <div class="field"><p class="kicker">${tt.assets}</p><div class="chips"><button class="chip" id="allied" aria-pressed="${c.allied}">${tt.allied}</button><button class="chip" id="weather" aria-pressed="${c.iron_beam_weather === "cloudy"}">${tt.weather}</button></div></div>
    <div class="field"><p class="kicker">${tt.knobs}</p>
      ${rng('id="pk" min="0.6" max="1.1" step="0.05"', tt.pkScale, c.pk_scale, c.pk_scale)}
      ${rng('id="thr" min="0.2" max="0.9" step="0.02"', tt.escTop, c.conf_threshold, c.conf_threshold)}
      ${rng('id="warn" min="0" max="30" step="1"', tt.warn, Math.round(c.forecast_s / 60), Math.round(c.forecast_s / 60))}</div>
    <div class="field"><p class="kicker">${tt.dmg}</p>
      ${Object.keys(c.tier_value).map((k) => rng(`data-tier="${k}" min="0" max="5000" step="50"`, tt.tier[k], Math.round(c.tier_value[k]), `$${Math.round(c.tier_value[k])}M`)).join("")}
      ${rng('id="pmul" min="1" max="10" step="0.5"', tt.prioMult, c.priority_multiplier, `×${c.priority_multiplier}`)}
      <details class="more"><summary>${tt.warhead}</summary>${Object.keys(c.warhead).map((k) => rng(`data-wh="${k}" min="0" max="2" step="0.01"`, esc(whLabel(k)), c.warhead[k], c.warhead[k])).join("")}</details>
      <details class="more"><summary>${tt.perSite}</summary><div class="sitegrid">${META.sites.map((s) => `<span>${esc(nm(s))}</span><input data-sv="${s.id}" type="number" min="0" step="50" value="${c.site_value[s.id] ?? ""}" placeholder="${Math.round(c.tier_value[s.tier])}" aria-label="${esc(nm(s))}">`).join("")}</div></details></div>
    <p class="note">${tt.note}</p>`;
  const set = (patch) => api("/api/config", patch).then((r) => { S.cfg = r.cfg; });
  $("#reserve").onclick = (e) => { const b = e.target.closest("[data-r]"); if (!b) return; document.querySelectorAll("#reserve .dc-tab").forEach((x) => x.setAttribute("aria-selected", x === b)); set({ reserve: +b.dataset.r }); };
  const press = (b) => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", on); return on; };
  $("#prio").onclick = (e) => { const b = e.target.closest("[data-site]"); if (!b) return; press(b); set({ priority_sites: [...document.querySelectorAll('#prio [aria-pressed="true"]')].map((x) => x.dataset.site) }); };
  $("#allied").onclick = (e) => set({ allied: press(e.currentTarget) });
  $("#weather").onclick = (e) => set({ iron_beam_weather: press(e.currentTarget) ? "cloudy" : "clear" });
  const slide = (el, fmt, patch) => (el.onchange = () => { el.nextElementSibling.textContent = fmt(el.value); set(patch()); });
  const all = (sel, key) => Object.fromEntries([...document.querySelectorAll(sel)].map((x) => [x.dataset[key], +x.value]));
  slide($("#pk"), (v) => v, () => ({ pk_scale: +$("#pk").value }));
  slide($("#thr"), (v) => v, () => ({ conf_threshold: +$("#thr").value }));
  slide($("#warn"), (v) => v, () => ({ forecast_s: 60 * +$("#warn").value }));
  slide($("#pmul"), (v) => `×${v}`, () => ({ priority_multiplier: +$("#pmul").value }));
  document.querySelectorAll("[data-tier]").forEach((r) => slide(r, (v) => `$${v}M`, () => ({ tier_value: all("[data-tier]", "tier") })));
  document.querySelectorAll("[data-wh]").forEach((r) => slide(r, (v) => v, () => ({ warhead: all("[data-wh]", "wh") })));
  document.querySelectorAll("[data-sv]").forEach((inp) => (inp.onchange = () => set({ site_value: Object.fromEntries([...document.querySelectorAll("[data-sv]")].filter((x) => x.value !== "").map((x) => [x.dataset.sv, +x.value])) })));
}

window.addEventListener("resize", () => renderStrip());
init();
