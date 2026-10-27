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
    speed: "Speed", speedDesc: ["Real time", "Calm walkthrough", "A raid in a few minutes", "Fast-forward"],
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
    vsNaive: (x) => `${x} firing at everything`, details: "Details", foot: (a, ms) => `Agrees with the planner ${a}% · ${ms} ms per call`, cmp: "Compare planner",
    planner: "Planner",
    strip: (id, t, s) => `Flight of ${id} · ${t} to ${s}`, now: "now", impact: "impact", calls: "Calls", hit: "hit", miss: "miss",
    posture: "Magazine posture", postures: ["Healthy", "Conserve", "Critical"], prio: "Priority sites", assets: "Assets and conditions",
    allied: "US THAAD and Navy", weather: "Cloud cover (laser degraded)", knobs: "Model and sensors", pkScale: "Pk scale", escTop: "Escalate below",
    warn: "Early warning, min", dmg: "Damage weights, $M per hit", tier: { strategic: "Strategic site", city: "City", infra: "Infrastructure" },
    prioMult: "Priority ×", warhead: "Warhead weight (1 = 500 kg)", perSite: "Value per site (blank = by type)",
    note: "Posture, stock, damage weights and priorities feed every lens. Typical reads them as sentences; the planner reads numbers.",
  },
  he: {
    speed: "מהירות", speedDesc: ["זמן אמת", "הדגמה רגועה", "מתקפה בכמה דקות", "הרצה מהירה"],
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
    vsNaive: (x) => `${x} ביריות על הכול`, details: "פרטים", foot: (a, ms) => `הסכמה עם המתכנן ${a}% · ${ms} מ״ש להחלטה`, cmp: "השווה למתכנן",
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
  $("#scenBtn").onclick = (e) => { e.stopPropagation(); $("#scenDrop").classList.toggle("is-open"); };
  $("#speedBtn").onclick = (e) => { e.stopPropagation(); $("#speedDrop").classList.toggle("is-open"); };
  document.addEventListener("click", () => document.querySelectorAll(".dc-dropdown").forEach((d) => d.classList.remove("is-open")));
  $("#speedMenu").onclick = (e) => { const x = e.target.closest("[data-x]")?.dataset.x; if (x) api("/api/control", { action: "speed", speed: +x }); };
  $("#scenMenu").onclick = async (e) => {
    const s = e.target.closest("[data-s]")?.dataset.s; if (!s) return;
    selected = null; pinned = false; detail = null; await api("/api/control", { action: "reset", scenario: s }); flyTo(SCEN_VIEW[s]);
  };
  $("#play").onclick = () => api("/api/control", { action: S?.running ? "pause" : "play" });
  $("#gear").onclick = () => { const o = $("#sheet").classList.toggle("open"); $("#gear").setAttribute("aria-expanded", o); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("#sheet").classList.remove("open"); $("#gear").setAttribute("aria-expanded", false); } });
  $("#decision").onclick = (e) => {
    if (e.target.closest("#followBtn")) { pinned = false; pickFollow(true); }
    if (e.target.closest(".dc-accordion__trigger")) e.target.closest(".dc-accordion__item").classList.toggle("is-open");
  };
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
  $("#scenMenu").innerHTML = Object.entries(META.scenarios).map(([k, v]) => {
    const [title, blurb] = LANG === "he" ? [v.title_he, v.blurb_he] : [v.title, v.blurb];
    return `<button class="dc-dropdown__item" data-s="${k}"><span class="dc-dropdown__text"><span class="dc-dropdown__label">${esc(title)}</span><span class="dc-dropdown__desc">${esc(blurb)}</span></span></button>`;
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

// One display clock per frame: eases toward the server's (extrapolated) time, never runs backwards, snaps only on
// real jumps (auto-skip, reset). Polls arrive every 250 ms with server time that wobbles; showing it raw jitters.
let shown = null, lastFrame = performance.now(), tNow = 0;
function simNow() {
  if (!S) return 0;
  const now = performance.now(), dtr = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  const live = S.running && !S.backlog;
  const target = live ? Math.min(S.end, S.t + ((now - fetchedAt) / 1000) * S.speed) : S.t;
  if (shown === null || !S.running || Math.abs(target - shown) > Math.max(30, 4 * S.speed)) return (shown = target);
  const step = live ? dtr * S.speed : 0;
  shown = Math.max(shown, shown + step + (target - shown - step) * Math.min(1, dtr * 3));
  return shown;
}
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
  const t = (tNow = simNow()), prio = new Set(S.cfg.priority_sites);
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
  stripNow(t);
}

// ---------------------------------------------------------------- bar
let lastSkip = { v: 0, at: -1e9 };
const SPEEDS = [[1, 0], [5, 1], [20, 2], [60, 3]];
function renderBar() {
  if (!S) return;
  $("#play").textContent = S.running ? T().pause : T().play;
  $("#speedBtn").firstChild.textContent = `${T().speed} ${S.speed}× `;
  if (renderBar.key === S.speed + LANG) return;
  renderBar.key = S.speed + LANG;
  $("#speedMenu").innerHTML = SPEEDS.map(([x, d]) => `<button class="dc-dropdown__item" role="menuitemradio" aria-checked="${x === S.speed}" data-x="${x}">
    <span class="dc-dropdown__text"><span class="dc-dropdown__label num">${x}×</span><span class="dc-dropdown__desc">${T().speedDesc[d]}</span></span></button>`).join("");
  const sc = META.scenarios[S.scenario];
  $("#scenBtn").firstChild.textContent = (LANG === "he" ? sc.title_he : sc.title).split(":")[0] + " ";
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
  const dest = D.open_ground ? `${tt.openGround} ${esc(siteName(D.site))}` : `${tt.to} ${esc(siteName(D.site))}${D.priority ? `<span class="tag">${tt.priority}</span>` : ""}`;
  let h = `<p class="kicker">${tt.cls[D.cls]}${pinned ? ` · <button id="followBtn" class="linkish">${tt.follow}</button>` : ""}</p>
    <div class="row"><div><h2 class="threat">${esc(nm(ty).replace(/ (MRBM|SRBM)$/i, ""))}</h2><div class="target">${dest}</div></div>
      <div class="eta num" id="eta" data-impact="${D.impact}">${done ? "" : fmtT(D.impact - tNow)}</div></div>`;
  const i = D.info;
  if (done) {
    el.innerHTML = h + `<div class="what" style="color:${D.status === "leaked" ? "var(--hit)" : "var(--dc-ink-2)"}">${tt.status[D.status]}</div>`;
    return;
  }
  if (!i) { el.innerHTML = h + `<p class="empty">${tt.notYet}</p>`; return; }
  h += `<div class="what">${esc(actLabel(D.choice))}</div><p class="why">${esc(reason(i, D))}</p>`;
  if (D.escalated) h += `<div class="op">${tt.operator}</div>`;
  // everything below the reason is detail: one closed disclosure
  const top = i.dist ? Object.keys(i.dist).sort((a, b) => i.dist[b] - i.dist[a]) : [];
  const conf = i.dist ? (D.escalated ? tt.unsure(actLabel(top[0]), Math.round(i.dist[top[0]] * 100)) : tt.conf(Math.round(i.dist[D.choice] * 100))) : "";
  const open = el.querySelector("details.detail")?.open;
  h += `<details class="detail" ${open ? "open" : ""}><summary>${tt.details}</summary>
    ${conf ? `<p class="dline">${esc(conf)}</p>` : ""}
    ${top.filter((k) => k !== D.choice).slice(0, 3).map((k) => `<div class="alt"><span>${esc(actLabel(k))}</span><span class="num">${Math.round(i.dist[k] * 100)}%</span></div>`).join("")}
    ${i.planner ? `<p class="dline">${i.planner === D.choice ? tt.agrees : esc(tt.differs(actLabel(i.planner)))}</p>` : ""}
    <p class="dline">${tt.reads}</p><pre>${esc(i.state)}</pre></details>`;
  el.innerHTML = h;
}

// ---------------------------------------------------------------- tonight
function renderTonight() {
  if (!S) return;
  const L = S.ledger, ty = L.typical, nv = L.naive, tt = T();
  if (!ty) return;
  $("#tonight").innerHTML = `<p class="kicker">${tt.tonight}</p><div class="stats">
      <div class="stat"><div class="v num">${money(ty.spent)}</div><div class="l">${tt.spent}</div><div class="vs">${tt.vsNaive(money(nv.spent))}</div></div>
      <div class="stat"><div class="v num ${ty.leaks ? "hit" : ""}">${ty.leaks}</div><div class="l">${tt.hits}</div><div class="vs">${tt.vsNaive(nv.leaks)}</div></div>
    </div>`;
}

// ---------------------------------------------------------------- flight strip (the threat's story over time)
let stripGeom = null;
function renderStrip() {   // one lifeline per threat: launch -> impact, the calls on it, shots as ticks, now as a blue dot
  const el = $("#strip"), tt = T();
  if (!detail || !META) { el.innerHTML = ""; stripGeom = null; return; }
  const D = detail, W = Math.max(el.clientWidth - 48, 400), Y = 46, PAD = 8;
  const X = (t) => PAD + ((Math.min(Math.max(t, D.t0), D.impact) - D.t0) / Math.max(D.impact - D.t0, 1)) * (W - 2 * PAD - 70);
  let g = `<line x1="${X(D.t0)}" x2="${X(D.impact)}" y1="${Y}" y2="${Y}" stroke="#e2e2e4" stroke-width="4" stroke-linecap="round"/>
    <line id="pastline" x1="${X(D.t0)}" x2="${X(D.t0)}" y1="${Y}" y2="${Y}" stroke="#101010" stroke-width="4" stroke-linecap="round"/>`;
  let lastX = -1e9, row = 0;
  D.history.forEach(([t, key], n) => {   // calls above the line; alternate height only when labels would collide
    const x = X(t), last = n === D.history.length - 1;
    row = x - lastX < 150 ? 1 - row : 0; lastX = x;
    const y = Y - 14 - row * 16, col = last ? "#2548f6" : "#5d5d5d";
    g += `<line x1="${x}" x2="${x}" y1="${y + 3}" y2="${Y}" stroke="${col}" stroke-width="1"/><text x="${x}" y="${y}" style="fill:${col};font-size:13px${last ? ";font-weight:500" : ""}">${esc(actLabel(key))}</text>`;
  });
  for (const s of D.shots) {   // shots below the line
    const fin = D.t >= s.t1;
    g += `<text x="${X(fin ? s.t1 : s.t0)}" y="${Y + 22}" text-anchor="middle" style="fill:${fin ? (s.hit ? "#101010" : "#8a8a8e") : "#2548f6"};font-size:13px">${fin ? (s.hit ? "✓" : "✕") : "↑"} ${esc(effName(s.eff))}</text>`;
  }
  g += `<text x="${X(D.impact) + 10}" y="${Y + 4}" style="fill:#101010;font-size:13px">${tt.impact}</text>`;
  g += `<circle id="nowdot" cx="0" cy="${Y}" r="6" fill="#2548f6" stroke="#fff" stroke-width="2"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} 80" width="${W}" height="80" direction="ltr">${g}</svg>`;
  stripGeom = { X, t0: D.t0 };
  stripNow(tNow);
}
function stripNow(t) {   // the now dot, the flown part and the countdown move every frame, between re-renders
  const d = document.getElementById("nowdot"), p = document.getElementById("pastline"), eta = document.getElementById("eta");
  if (d && stripGeom) { d.setAttribute("cx", stripGeom.X(t)); p.setAttribute("x2", stripGeom.X(t)); }
  if (eta && eta.textContent) eta.textContent = fmtT(Math.max(0, +eta.dataset.impact - t));
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
