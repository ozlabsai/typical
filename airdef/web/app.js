// Typical air-defence decision desk: polls /api/state, interpolates trajectories by sim time, draws with deck.gl. EN / HE.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (m) => (m < 0.001 ? "<$1k" : m < 1 ? `$${Math.round(m * 1000)}k` : m < 1000 ? `$${m.toFixed(1)}M` : `$${(m / 1000).toFixed(2)}B`);
const CLS_COLOR = { B: [255, 90, 54], R: [255, 140, 70], C: [255, 200, 90], D: [235, 120, 200] };
const VIEWS = {
  theatre: { longitude: 40.5, latitude: 31.2, zoom: 4.7, pitch: 50, bearing: 0 },
  north: { longitude: 35.35, latitude: 32.95, zoom: 8.3, pitch: 50, bearing: 0 },
  south: { longitude: 39.5, latitude: 24.5, zoom: 4.6, pitch: 45, bearing: 0 },
};
const SCEN_VIEW = { northern_barrage: "north", april_mixed: "theatre", october_saturation: "theatre", june_attrition: "theatre", houthi_eilat: "south" };

// ---------------------------------------------------------------- i18n
const I18N = {
  en: {
    brand: "Air Defence Decision Desk", busy: "re-deciding…", play: "Play", pause: "Pause", skip: "Skip transit", reset: "Reset", model: "model",
    liveTracks: "Live tracks", tabDecision: "Decision", tabConsole: "Commander's console", raid: "Raid", attSpend: "attacker spend",
    launched: "launched", lgThreat: "threat, flown / predicted", lgInt: "interceptor", lgIpp: "predicted impact zone", lang: "עברית",
    summary: (n, e, h, x) => `${n} on screen · <b>${e}</b> engaging/planned · ${h} held · ${x} escalated`,
    status: { pending: "tracking", planned: "planned", held: "hold", engaging: "engaging", killed: "killed", leaked: "impact", fell_open: "fell open", operator: "operator" },
    pol: { naive: ["Fire at everything", "best Pk, no doctrine"], optimizer: ["Cost-aware optimizer", "numeric objective"], typical: ["Typical", "doctrine model, live"] },
    kills: "kills", hits: "hits on sites", letFall: "let fall", toOp: "to operator", exchange: "exchange", follows: "follows doctrine", agrees: "agrees w/ optimizer",
    hold: "Hold fire", ifMissed: "if missed", noInt: "no interceptor",
    act: { hold: "Hold fire", protection: "Fire the sure-kill plan", value: "Fire the cheapest reliable plan" },
    pk: ["almost certain", "very likely", "likely", "may stop it", "unlikely"],
    empty: "Select a track on the map or the board to see what Typical reads and how it decides.",
    notDecided: "Not decided yet: no interceptor can reach it inside the decision horizon.",
    viewing: "viewing", reads: (p) => `What ${p} reads`, readsNote: "", typicalCall: "Typical's call: a probability over the doctrine's actions; the solver turns the action into a plan",
    nullBar: "∅ none of these: escalate to operator", doctrine: "doctrine", optimizer: "optimizer", executing: "executing",
    escBanner: "Typical's confidence was below the escalation bar, so the track went to the operator, who executed the optimizer's plan.",
    latency: "model latency", naive: "fire-at-everything", allPlans: "Every feasible plan · expected cost incl. leak risk + magazine value",
    ordersHead: "Commander's orders: Typical reads these above the standing rules", issue: "Issue order", clear: "Clear all orders",
    noOrders: "No orders. The three standing rules apply.", freeText: "free text: outside the rule templates",
    placeholder: "When the threat … , hold fire / fire the sure-kill plan / fire the cheapest reliable plan.",
    prio: "Priority sites", posture: "Magazine posture", postures: ["Healthy stocks", "Conserve upper tier", "Critical"],
    assets: "Assets & conditions", allied: "US THAAD + Navy", weather: "Cloud cover (laser degraded)",
    knobs: "Knobs (notional values)", pkScale: "Pk scale", escTop: "Escalate if top below", escNull: "Escalate if ∅ above",
    dmg: "Damage weights: what a hit costs ($M-equivalent; feeds the optimizer and the 'cheapest reliable' plan)",
    tier: { strategic: "strategic site", city: "city", infra: "infrastructure" }, prioMult: "priority ×",
    warhead: "Warhead weight (1 = 500 kg MRBM)", perSite: "Per-site value, $M (blank = tier default)",
    note: "Typical reads the rules, the orders and the priority sites. Posture, weather, allied assets, Pk and damage weights change which concrete plan each of Typical's actions resolves to.",
    presets: ["When the threat is a drone, hold fire.", "When the threat will hit Haifa Bay refineries, fire the sure-kill plan.",
      "When the threat will hit Kiryat Shmona, hold fire.", "When the threat is a ballistic missile, fire the sure-kill plan.", "Treat Haifa Bay refineries as a priority site."],
  },
  he: {
    brand: "שולחן החלטות הגנה אווירית", busy: "מחשב מחדש…", play: "הפעל", pause: "השהה", skip: "דלג על מעבר", reset: "איפוס", model: "מודל",
    liveTracks: "מטרות פעילות", tabDecision: "החלטה", tabConsole: "מסוף המפקד", raid: "המתקפה", attSpend: "עלות התוקף",
    launched: "שוגרו", lgThreat: "איום, מסלול שעבר / חזוי", lgInt: "מיירט", lgIpp: "אזור פגיעה חזוי", lang: "English",
    summary: (n, e, h, x) => `${n} על המסך · <b>${e}</b> ביירוט/מתוכננים · ${h} באחיזת אש · ${x} הועברו למפעיל`,
    status: { pending: "במעקב", planned: "מתוכנן", held: "אחיזת אש", engaging: "ביירוט", killed: "יורט", leaked: "פגיעה", fell_open: "נפל בשטח פתוח", operator: "למפעיל" },
    pol: { naive: ["יורים על הכול", "Pk מרבי, בלי תורה"], optimizer: ["מייעל עלות", "פונקציית מטרה מספרית"], typical: ["Typical", "מודל תורת לחימה, חי"] },
    kills: "יירוטים", hits: "פגיעות באתרים", letFall: "הונחו ליפול", toOp: "למפעיל", exchange: "יחס עלות", follows: "נאמנות לתורה", agrees: "הסכמה עם המייעל",
    hold: "אחיזת אש", ifMissed: "אם החטיא", noInt: "ללא מיירט",
    act: { hold: "להחזיק אש", protection: "תוכנית השמדה ודאית", value: "התוכנית הזולה והאמינה" },
    pk: ["כמעט ודאי", "סביר מאוד", "סביר", "ייתכן", "לא סביר"],
    empty: "בחרו מטרה במפה או ברשימה כדי לראות מה Typical קורא ואיך הוא מחליט.",
    notDecided: "טרם הוחלט: אף מיירט אינו מגיע אליה בתוך אופק ההחלטה.",
    viewing: "תצוגה:", reads: (p) => `מה ${p} קורא`, readsNote: " (המודל קורא את הכללים באנגלית; פקודות בעברית נשארות בעברית)",
    typicalCall: "ההחלטה של Typical: הסתברות על פעולות התורה; הממטב הופך את הפעולה לתוכנית",
    nullBar: "∅ אף אחת: העבר למפעיל", doctrine: "תורה", optimizer: "מייעל", executing: "מבוצע",
    escBanner: "הביטחון של Typical היה מתחת לסף, ולכן המטרה הועברה למפעיל, שביצע את תוכנית המייעל.",
    latency: "זמן תגובת המודל", naive: "יורים על הכול", allPlans: "כל התוכניות האפשריות · עלות צפויה כולל סיכון פגיעה וערך המלאי",
    ordersHead: "פקודות המפקד: Typical קורא אותן לפני הכללים הקבועים", issue: "הוצא פקודה", clear: "נקה את כל הפקודות",
    noOrders: "אין פקודות. שלושת הכללים הקבועים חלים.", freeText: "טקסט חופשי: מחוץ לתבניות הכללים",
    placeholder: "כאשר האיום הוא … : להחזיק אש / לירות את תוכנית ההשמדה הוודאית / לירות את התוכנית הזולה והאמינה.",
    prio: "אתרים בעדיפות", posture: "מצב מלאים", postures: ["מלאים תקינים", "שימור שכבה עליונה", "קריטי"],
    assets: "נכסים ותנאים", allied: "THAAD וצי ארה״ב", weather: "עננות (לייזר מוגבל)",
    knobs: "כיוונונים (ערכים משוערים)", pkScale: "מקדם Pk", escTop: "למפעיל אם הביטחון מתחת", escNull: "למפעיל אם ∅ מעל",
    dmg: "משקלי נזק: מחיר פגיעה (מיליון $ שקולים; משפיע על המייעל ועל 'התוכנית הזולה והאמינה')",
    tier: { strategic: "אתר אסטרטגי", city: "עיר", infra: "תשתית" }, prioMult: "מכפיל עדיפות",
    warhead: "משקל ראש קרב (1 = טיל בליסטי 500 ק״ג)", perSite: "ערך לאתר, מיליון $ (ריק = לפי סוג האתר)",
    note: "Typical קורא את הכללים, הפקודות והאתרים בעדיפות. מצב המלאים, מזג האוויר, נכסי בעלות הברית, Pk ומשקלי הנזק קובעים לאיזו תוכנית קונקרטית כל פעולה מתורגמת.",
    presets: ["כאשר האיום הוא כטב״ם: להחזיק אש.", "איומים לעבר בתי הזיקוק במפרץ חיפה: לירות את תוכנית ההשמדה הוודאית.",
      "איומים לעבר קריית שמונה: להחזיק אש.", "כאשר האיום הוא טיל בליסטי: לירות את תוכנית ההשמדה הוודאית.", "בתי הזיקוק במפרץ חיפה הוא אתר בעדיפות הלילה."],
  },
};
let LANG = localStorage.getItem("airdef-lang") || "en";
const L_ = () => I18N[LANG];
const nm = (o) => (LANG === "he" && o.name_he ? o.name_he : o.name);
const glyphs = (s) => (LANG === "he" ? [...s].reverse().join("") : s);   // deck.gl TextLayer has no bidi: reverse Hebrew by hand

function applyLang() {
  document.documentElement.lang = LANG;
  document.documentElement.dir = LANG === "he" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = L_()[el.dataset.i18n]));
  $("#lang").textContent = L_().lang;
  if (!META) return;
  $("#scen").innerHTML = Object.entries(META.scenarios).map(([k, v]) => {
    const [title, blurb] = LANG === "he" ? [v.title_he, v.blurb_he] : [v.title, v.blurb];
    return `<button data-s="${k}" title="${esc(blurb)}">${esc(title.split(":")[0])}</button>`;
  }).join("");
  if (S) { renderTop(); renderBoard(); renderLedger(); buildConsole(); renderDetail(); }
}

let META, PATHS = {}, PATH_SCEN = null, S = null, fetchedAt = 0, selected = null, viewPol = "typical", detail = null;
const flashSeen = new Map();

// ---------------------------------------------------------------- map
const map = new maplibregl.Map({
  container: "map", style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  center: [35.0, 31.6], zoom: 6.6, pitch: 45, maxPitch: 75, attributionControl: { compact: true },
});
const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
map.addControl(overlay);
function flyTo(v) { const x = VIEWS[v]; map.flyTo({ center: [x.longitude, x.latitude], zoom: x.zoom, pitch: x.pitch, bearing: x.bearing, duration: 1600 }); }

// ---------------------------------------------------------------- api
const api = (p, body) => fetch(p, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {}).then((r) => r.json());
const effShort = (k) => nm(META.effectors[k]).replace("F-15/F-35 ", "").replace("F-16 laser-guided rocket (APKWS)", "APKWS").replace("GPS jamming (EW)", "EW jamming").replace(" laser", "");

async function init() {
  META = await api("/api/meta");
  $("#modelName").textContent = META.model;
  $("#lang").onclick = () => { LANG = LANG === "en" ? "he" : "en"; localStorage.setItem("airdef-lang", LANG); applyLang(); };
  $("#scen").onclick = async (e) => { const s = e.target.dataset.s; if (!s) return; selected = null; await api("/api/control", { action: "reset", scenario: s }); flyTo(SCEN_VIEW[s]); renderDetail(); };
  $("#speeds").innerHTML = [1, 5, 20, 60, 200].map((x) => `<button data-x="${x}">${x}×</button>`).join("");
  $("#speeds").onclick = (e) => e.target.dataset.x && api("/api/control", { action: "speed", speed: +e.target.dataset.x });
  $("#play").onclick = () => api("/api/control", { action: S?.running ? "pause" : "play" });
  $("#skip").onclick = () => api("/api/control", { action: "skip" });
  $("#reset").onclick = () => api("/api/control", { action: "reset" });
  document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => {
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("on", p.id === "p-" + b.dataset.tab));
  }));
  document.querySelectorAll(".pol").forEach((el) => (el.onclick = () => { viewPol = el.dataset.pol; document.querySelectorAll(".pol").forEach((x) => x.classList.toggle("view", x === el)); }));
  $("#rows").onclick = (e) => { const r = e.target.closest(".row"); if (r) select(r.dataset.id); };
  await poll();
  applyLang();
  flyTo(SCEN_VIEW[S.scenario]);
  setInterval(poll, 250);
  setInterval(() => selected && loadDetail(), 700);
  requestAnimationFrame(frame);
}

async function poll() {
  try { S = await api(`/api/state?policy=${viewPol}`); fetchedAt = performance.now(); } catch { return; }
  if (S.scenario !== PATH_SCEN) {
    const P = await api("/api/paths");
    PATHS = Object.fromEntries(P.threats.map((t) => [t.id, t]));
    PATH_SCEN = P.scenario; flashSeen.clear();
  }
  renderTop(); renderBoard(); renderLedger(); renderOrders();
}

function simNow() {   // extrapolate between polls so motion is smooth
  if (!S) return 0;
  if (!S.running || S.backlog) return S.t;
  return Math.min(S.end, S.t + ((performance.now() - fetchedAt) / 1000) * S.speed);
}

function posAt(path, t) {   // path.p = [[lon, lat, alt_m, t], ...]
  const p = path.p;
  if (t <= p[0][3]) return p[0];
  if (t >= p[p.length - 1][3]) return p[p.length - 1];
  let lo = 0, hi = p.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (p[m][3] <= t) lo = m; else hi = m; }
  const a = p[lo], b = p[hi], f = (t - a[3]) / (b[3] - a[3]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, t];
}

// ---------------------------------------------------------------- deck layers
function frame() {
  requestAnimationFrame(frame);
  if (!S || !META) return;
  const t = simNow();
  const trackById = Object.fromEntries(S.tracks.map((x) => [x.id, x]));
  const prio = new Set(S.cfg.priority_sites);
  const live = [], trails = [], futures = [];
  for (const tr of S.tracks) {
    const path = PATHS[tr.id]; if (!path) continue;
    const end = Math.min(S.killed[tr.id] ?? Infinity, path.t0 + path.T);
    if (t > end) continue;
    const cur = posAt(path, t);
    const past = path.p.filter((q) => q[3] < t).map((q) => q.slice(0, 3)); past.push(cur.slice(0, 3));
    const fut = [cur.slice(0, 3), ...path.p.filter((q) => q[3] > t).map((q) => q.slice(0, 3))];
    const col = CLS_COLOR[path.cls];
    trails.push({ path: past, color: [...col, 230], id: tr.id });
    futures.push({ path: fut, color: [...col, tr.status === "held" ? 35 : 80], id: tr.id });
    live.push({ pos: cur.slice(0, 3), color: col, tr });
  }
  const batt = Object.fromEntries(META.batteries.map((b) => [b.id, b]));
  const shots = [], flashes = [], misses = [];
  const nowMs = performance.now();
  for (const f of S.flights) {
    if (t < f.t0) continue;
    const b = batt[f.b], g = Math.min(1, (t - f.t0) / Math.max(1, f.t1 - f.t0));
    const n = 24, pts = [];
    for (let k = 0; k <= n * g; k++) { const s = k / n; pts.push([b.lon + (f.lon - b.lon) * s, b.lat + (f.lat - b.lat) * s, f.alt * Math.pow(s, 0.6)]); }
    if (g < 1) shots.push({ path: pts });
    else {
      const key = `${f.b}${f.th}${f.t1}`;
      if (!flashSeen.has(key)) flashSeen.set(key, nowMs);
      const age = (nowMs - flashSeen.get(key)) / 1000;
      if (age < 1.6) flashes.push({ pos: [f.lon, f.lat, f.alt], age, hit: f.hit });
      if (!f.hit && age < 6) misses.push({ pos: [f.lon, f.lat, f.alt] });
    }
  }
  const ipps = S.tracks.filter((x) => !["killed", "fell_open", "leaked"].includes(x.status) && t < (PATHS[x.id]?.t0 ?? 0) + (PATHS[x.id]?.T ?? 0));
  const leaks = S.tracks.filter((x) => x.status === "leaked").map((x) => { const p = PATHS[x.id]?.p; return p ? p[p.length - 1] : null; }).filter(Boolean);
  const sel = selected && trackById[selected] && PATHS[selected];
  const batts = META.batteries.filter((b) => S.cfg.allied || !META.effectors[b.eff].allied);

  overlay.setProps({ layers: [
    new deck.ScatterplotLayer({ id: "sites", data: META.sites, getPosition: (d) => [d.lon, d.lat], getRadius: (d) => d.r * 1000,
      getFillColor: (d) => (prio.has(d.id) ? [255, 181, 71, 40] : [240, 238, 235, 18]), stroked: true, lineWidthUnits: "pixels", getLineWidth: (d) => (prio.has(d.id) ? 1.5 : 0.6),
      getLineColor: (d) => (prio.has(d.id) ? [255, 181, 71, 200] : [240, 238, 235, 70]), updateTriggers: { getFillColor: [...prio].join(), getLineColor: [...prio].join(), getLineWidth: [...prio].join() } }),
    new deck.TextLayer({ id: "siteNames", data: META.sites, visible: map.getZoom() > 6.8, getPosition: (d) => [d.lon, d.lat], getText: (d) => glyphs(nm(d)), getSize: 11,
      getColor: [240, 238, 235, 150], getPixelOffset: [0, -14], fontFamily: "Helvetica Neue, Arial, sans-serif", fontWeight: 500, characterSet: "auto", updateTriggers: { getText: LANG } }),
    new deck.ScatterplotLayer({ id: "batt", data: batts, getPosition: (d) => [d.lon, d.lat], radiusUnits: "pixels", getRadius: 3.5, getFillColor: [90, 209, 255, 220] }),
    new deck.TextLayer({ id: "battNames", visible: map.getZoom() > 7.5, data: batts, getPosition: (d) => [d.lon, d.lat],
      getText: (d) => d.id, getSize: 9.5, getColor: [90, 209, 255, 170], getPixelOffset: [0, 11], fontFamily: "Menlo, monospace" }),
    new deck.ScatterplotLayer({ id: "ipp", data: ipps, getPosition: (d) => [d.ipp[0], d.ipp[1]], getRadius: (d) => Math.max(d.ipp[2], 0.6) * 1000,
      filled: true, stroked: true, getFillColor: (d) => (d.status === "held" ? [160, 157, 151, 18] : [255, 181, 71, 26]),
      getLineColor: (d) => (d.id === selected ? [255, 255, 255, 230] : d.status === "held" ? [160, 157, 151, 120] : [255, 181, 71, 170]), lineWidthUnits: "pixels", getLineWidth: (d) => (d.id === selected ? 2 : 1) }),
    new deck.PathLayer({ id: "future", data: futures, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => (d.id === selected ? 2 : 1) }),
    new deck.PathLayer({ id: "trail", data: trails, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => (d.id === selected ? 3 : 1.6) }),
    new deck.PathLayer({ id: "shots", data: shots, getPath: (d) => d.path, getColor: [90, 209, 255, 255], widthUnits: "pixels", getWidth: 2 }),
    new deck.ScatterplotLayer({ id: "heads", data: live, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => (d.tr.id === selected ? 7 : 4),
      getFillColor: (d) => d.color, stroked: true, getLineColor: (d) => (d.tr.escalated ? [255, 181, 71] : [20, 19, 18]), lineWidthUnits: "pixels", getLineWidth: (d) => (d.tr.escalated ? 2 : 1),
      pickable: true, onClick: (i) => select(i.object.tr.id) }),
    new deck.ScatterplotLayer({ id: "flash", data: flashes, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => 6 + d.age * 26,
      filled: false, stroked: true, lineWidthUnits: "pixels", getLineWidth: 2, getLineColor: (d) => (d.hit ? [255, 255, 255, 255 * (1 - d.age / 1.6)] : [160, 157, 151, 200 * (1 - d.age / 1.6)]) }),
    new deck.ScatterplotLayer({ id: "miss", data: misses, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: 3, getFillColor: [160, 157, 151, 160] }),
    new deck.ScatterplotLayer({ id: "leaks", data: leaks, getPosition: (d) => [d[0], d[1]], radiusUnits: "pixels", getRadius: 6, getFillColor: [255, 77, 77, 230],
      stroked: true, getLineColor: [255, 77, 77, 90], lineWidthUnits: "pixels", getLineWidth: 6 }),
    sel && new deck.PathLayer({ id: "selPath", data: [{ p: PATHS[selected].p.map((q) => q.slice(0, 3)) }], getPath: (d) => d.p, getColor: [255, 255, 255, 60], widthUnits: "pixels", getWidth: 1 }),
  ].filter(Boolean) });
  $("#clock").textContent = "T+" + fmtT(t);
}

const fmtT = (s) => { s = Math.max(0, s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60); return (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0"); };

// ---------------------------------------------------------------- panels
function planShort(key) {
  if (!key) return "—";
  if (key === "hold") return L_().hold;
  const m = key.match(/^(\w+?)x([12])$/);
  if (m) return (m[2] === "2" ? "2× " : "") + effShort(m[1]);
  const [a, b] = key.split(">");
  return `${effShort(a)} → ${effShort(b)} ${L_().ifMissed}`;
}
const pkWord = (p) => L_().pk[p >= 0.95 ? 0 : p >= 0.85 ? 1 : p >= 0.7 ? 2 : p >= 0.5 ? 3 : 4];
const siteName = (sid) => { const s = META.sites.find((x) => x.id === sid); return s ? nm(s) : ""; };

function renderTop() {
  $("#play").textContent = S.running ? L_().pause : L_().play;
  $("#play").classList.toggle("on", S.running);
  document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", +b.dataset.x === S.speed));
  document.querySelectorAll("#scen button").forEach((b) => b.classList.toggle("on", b.dataset.s === S.scenario));
  $("#busy").style.visibility = S.backlog ? "visible" : "hidden";
}

function renderBoard() {
  const order = { engaging: 0, planned: 1, pending: 2, held: 3, leaked: 4, killed: 5, fell_open: 6 };
  const rows = [...S.tracks].sort((a, b) => (order[a.status] - order[b.status]) || a.tti - b.tti);
  const n = (st) => S.tracks.filter((x) => x.status === st).length;
  $("#boardSummary").innerHTML = L_().summary(S.tracks.length, n("engaging") + n("planned"), n("held"), S.tracks.filter((x) => x.escalated).length);
  const max = Math.floor(($("#rows").clientHeight || 400) / 52);
  $("#rows").innerHTML = rows.slice(0, max).map((r) => {
    const ty = META.threat_types[r.kind], col = CLS_COLOR[ty.cls];
    const conf = r.conf != null ? `<span class="mono">${Math.round(r.conf * 100)}%</span>` : "";
    const op = r.escalated && ["planned", "held", "pending"].includes(r.status);
    return `<div class="row ${r.id === selected ? "sel" : ""}" data-id="${r.id}">
      <span class="dot" style="background:rgb(${col})"></span><span class="id mono">${r.id}</span>
      <span class="what">${esc(nm(ty))}</span>
      <span class="chip c-${op ? "esc" : r.status}">${L_().status[op ? "operator" : r.status]}</span>
      <span></span><span class="rec"><b>${esc(planShort(r.plan))}</b> ${conf} ${r.site ? "· " + esc(siteName(r.site)) : ""}</span>
      <span class="tti mono">${r.tti > 0 && !["killed", "leaked", "fell_open"].includes(r.status) ? fmtT(r.tti) : ""}</span></div>`;
  }).join("");
}

function magRow(L) {
  return ["arrow3", "arrow2", "thaad", "dsling", "ironDome", "aim9x"].filter((e) => L.init[e]).map((e) => {
    const f = L.left[e] / L.init[e];
    return `<div class="mag ${f < 0.2 ? "crit" : f < 0.5 ? "low" : ""}">${esc(effShort(e))} <span class="mono">${Math.round(L.left[e])}/${L.init[e]}</span><i><u style="width:${Math.max(0, f) * 100}%"></u></i></div>`;
  }).join("");
}

function renderLedger() {
  const T = L_();
  $("#attCost").textContent = money(S.attacker_cost);
  $("#launched").textContent = S.launched; $("#total").textContent = S.total;
  for (const [k, L] of Object.entries(S.ledger)) {
    const d = Math.max(L.decisions, 1);
    const extra = k === "typical" && S.model !== "optimizer only"
      ? `<div class="kv" style="margin-top:4px"><span>${T.follows} <b>${Math.round((L.follow / d) * 100)}%</b></span><span>${T.agrees} <b>${Math.round((L.agree / d) * 100)}%</b></span><span><b class="mono">${Math.round(L.ms / Math.max(L.calls, 1))}</b> ms</span></div>` : "";
    $(`.pol.${k}`).innerHTML = `<h4>${T.pol[k][0]} <small>${T.pol[k][1]}</small></h4>
      <div class="big mono">${money(L.spent)}</div>
      <div class="kv"><span>${T.kills} <b>${L.kills}</b></span><span>${T.hits} <b style="color:${L.leaks ? "var(--bad)" : "inherit"}">${L.leaks}</b></span><span>${T.letFall} <b>${L.fell_open}</b></span>${k === "typical" ? `<span>${T.toOp} <b>${L.escalated}</b></span>` : ""}<span>${T.exchange} <b class="mono">${(L.spent / Math.max(S.attacker_cost, 1e-6)).toFixed(1)}:1</b></span></div>
      ${extra}<div class="mags">${magRow(L)}</div>`;
  }
}

// ---------------------------------------------------------------- decision detail
function select(id) {
  selected = id; renderBoard();
  document.querySelector('#tabs button[data-tab="decision"]').click();
  loadDetail();
}

async function loadDetail() {
  if (!selected) return;
  detail = await api(`/api/track/${selected}?policy=${viewPol}`);
  renderDetail();
}

function renderDetail() {
  const T = L_(), el = $("#p-decision");
  if (!selected || !detail) { el.innerHTML = `<div class="empty">${T.empty}</div>`; return; }
  const i = detail.info;
  const ty = Object.values(META.threat_types).find((x) => x.name === detail.name);
  const head = `<div id="dTitle">${detail.id} · ${esc(ty ? nm(ty) : detail.name)}</div><div id="dSub">${T.status[detail.status] ?? detail.status} · ${T.viewing} <b>${T.pol[viewPol][0]}</b></div>`;
  if (!i) { el.innerHTML = head + `<div class="empty">${T.notDecided}</div>`; return; }
  const state = esc(i.state).replace(/(Situation\.[\s\S]*)$/, '<span class="hl">$1</span>');
  const plan = Object.fromEntries(i.plans.map(([k, label, pk, cost]) => [k, { pk, cost }]));
  const planLine = (k) => (k === "hold" ? T.noInt : `${planShort(k)} · ${money(plan[k].cost)} · ${pkWord(plan[k].pk)}`);
  const mark = (cls, txt) => `<span class="mark ${cls}">${txt}</span>`;
  let bars = "";
  if (i.dist) {
    bars = Object.keys(T.act).sort((a, b) => i.dist[b] - i.dist[a]).map((a) => {
      const p = i.dist[a], k = i.resolved[a];
      const marks = (a === i.rule_action ? mark("m-rule", T.doctrine) : "") + (k === i.opt ? mark("m-opt", T.optimizer) : "");
      return `<div class="bar ${a === i.action ? "top" : ""}"><div class="lab"><div class="fill" style="width:${p * 100}%"></div><span class="txt">${T.act[a]}${marks}<br><span style="color:var(--mute);font-size:11.5px">→ ${esc(planLine(k))}</span></span></div><span class="p mono">${(p * 100).toFixed(0)}%</span></div>`;
    }).join("") + `<div class="bar"><div class="lab"><div class="fill" style="width:${i.p_null * 100}%;background:rgba(160,157,151,.2);border-color:var(--mute)"></div><span class="txt">${T.nullBar}</span></div><span class="p mono">${(i.p_null * 100).toFixed(0)}%</span></div>`;
  }
  const all = i.plans.map(([k]) => `<div class="bar ${k === detail.plan ? "top" : ""}"><div class="lab" style="padding:2px 8px"><span class="txt" style="font-size:11.5px">${esc(planLine(k))}${k === i.opt ? mark("m-opt", T.optimizer) : ""}${k === detail.plan ? mark("m-rule", T.executing) : ""}</span></div><span class="p mono" style="font-size:11px;color:var(--mute)">${money(i.scores[k])}</span></div>`).join("");
  const escl = detail.escalated ? `<div id="esc" style="display:block">${T.escBanner}</div>` : "";
  const foot = `<div id="dFoot">
      <span>${T.executing} <b>${esc(planShort(detail.plan))}</b></span><span>${T.optimizer} <b>${esc(planShort(detail.others.optimizer))}</b></span>
      ${i.ms != null ? `<span>${T.latency} <b class="mono">${i.ms.toFixed(0)} ms</b></span>` : "<span></span>"}
      <span>${T.naive} <b>${esc(planShort(detail.others.naive))}</b></span></div>`;
  el.innerHTML = head + escl + `<div class="eyebrow">${T.reads(viewPol === "typical" ? "Typical" : T.pol[viewPol][0])}${T.readsNote}</div><div id="dState">${state}</div>
    ${i.dist ? `<div class="eyebrow">${T.typicalCall}</div>${bars}` : ""}
    ${foot}<div class="eyebrow" style="margin-top:14px">${T.allPlans}</div>${all}`;
}

// ---------------------------------------------------------------- commander's console
function renderOrders() {
  const box = $("#orders"); if (!box) return;
  const T = L_(), orders = S.cfg.orders || [], ok = S.orders_parsed || [];
  const html = orders.map((o, k) => `<div class="bar"><div class="lab" style="padding:4px 8px"><span class="txt">${k + 1}. ${esc(o)}${ok[k] === false ? `<span class="mark m-opt">${T.freeText}</span>` : ""}</span></div><button data-del="${k}" style="padding:2px 8px">×</button></div>`).join("") || `<div style="color:var(--mute)">${T.noOrders}</div>`;
  if (box.dataset.html !== html) { box.innerHTML = html; box.dataset.html = html; }
}

function buildConsole() {
  const T = L_(), c = S.cfg;
  const sites = META.sites.map((s) => `<button data-site="${s.id}" class="${c.priority_sites.includes(s.id) ? "on" : ""}">${esc(nm(s))}</button>`).join("");
  const rng = (attrs, label, val, shown) => `<div class="rng"><span>${label}</span><input type="range" ${attrs} value="${val}"><span class="mono">${shown}</span></div>`;
  const whLabel = (k) => (META.threat_types[k] ? nm(META.threat_types[k]) : { B: "ballistic missile", R: "rocket", C: "cruise missile", D: "drone" }[k]);
  const heCls = { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" };
  $("#p-console").innerHTML = `
    <div class="field"><span class="eyebrow">${T.ordersHead}</span>
      <div id="orders"></div>
      <textarea id="intent" style="height:44px;margin-top:6px" placeholder="${esc(T.placeholder)}"></textarea>
      <div class="presets">${T.presets.map((p) => `<button data-intent="${esc(p)}">${esc(p)}</button>`).join("")}</div>
      <div class="seg" style="margin-top:6px"><button id="issue" class="on">${T.issue}</button><button id="clearIntent">${T.clear}</button></div></div>
    <div class="field"><span class="eyebrow">${T.prio}</span><div class="seg" id="prio">${sites}</div></div>
    <div class="field"><span class="eyebrow">${T.posture}</span><div class="seg" id="reserve">
      ${T.postures.map((p, k) => `<button data-r="${k}" class="${c.reserve === k ? "on" : ""}">${p}</button>`).join("")}</div></div>
    <div class="field"><span class="eyebrow">${T.assets}</span><div class="seg">
      <button id="allied" class="${c.allied ? "on" : ""}">${T.allied}</button>
      <button id="weather" class="${c.iron_beam_weather === "cloudy" ? "on" : ""}">${T.weather}</button></div></div>
    <div class="field"><span class="eyebrow">${T.knobs}</span>
      ${rng('id="pk" min="0.6" max="1.1" step="0.05"', T.pkScale, c.pk_scale, c.pk_scale)}
      ${rng('id="thr" min="0.34" max="0.9" step="0.02"', T.escTop, c.conf_threshold, c.conf_threshold)}
      ${rng('id="abs" min="0.3" max="1" step="0.05"', T.escNull, c.abstain_threshold, c.abstain_threshold)}</div>
    <div class="field"><span class="eyebrow">${T.dmg}</span>
      ${Object.keys(c.tier_value).map((t) => rng(`data-tier="${t}" min="0" max="5000" step="50"`, T.tier[t], c.tier_value[t], `$${c.tier_value[t]}M`)).join("")}
      ${rng('id="pmul" min="1" max="10" step="0.5"', T.prioMult, c.priority_multiplier, `×${c.priority_multiplier}`)}
      <details class="dmg"><summary>${T.warhead}</summary>
        ${Object.keys(c.warhead).map((k) => rng(`data-wh="${k}" min="0" max="2" step="0.01"`, esc(LANG === "he" && heCls[k] ? heCls[k] : whLabel(k)), c.warhead[k], c.warhead[k])).join("")}</details>
      <details class="dmg"><summary>${T.perSite}</summary><div class="sitegrid">
        ${META.sites.map((s) => `<span>${esc(nm(s))}</span><input data-sv="${s.id}" type="number" min="0" step="50" value="${c.site_value[s.id] ?? ""}" placeholder="${c.tier_value[s.tier]}">`).join("")}</div></details>
      <div class="eyebrow" style="margin-top:8px;text-transform:none;letter-spacing:0">${T.note}</div></div>`;
  const set = (patch) => api("/api/config", patch).then((r) => { S.cfg = r.cfg; S.orders_parsed = r.parsed; renderOrders(); });
  renderOrders();
  $("#issue").onclick = () => { const o = $("#intent").value.trim(); if (o) { $("#intent").value = ""; set({ orders: [...(S.cfg.orders || []), o] }); } };
  $("#clearIntent").onclick = () => set({ orders: [] });
  $("#orders").onclick = (e) => { const k = e.target.dataset.del; if (k != null) set({ orders: S.cfg.orders.filter((_, j) => j !== +k) }); };
  document.querySelectorAll("[data-intent]").forEach((b) => (b.onclick = () => { $("#intent").value = b.dataset.intent; }));
  $("#prio").onclick = (e) => {
    const s = e.target.dataset.site; if (!s) return;
    e.target.classList.toggle("on");
    set({ priority_sites: [...document.querySelectorAll("#prio .on")].map((b) => b.dataset.site) });
  };
  $("#reserve").onclick = (e) => {
    if (e.target.dataset.r == null) return;
    document.querySelectorAll("#reserve button").forEach((b) => b.classList.toggle("on", b === e.target));
    set({ reserve: +e.target.dataset.r });
  };
  $("#allied").onclick = (e) => { e.target.classList.toggle("on"); set({ allied: e.target.classList.contains("on") }); };
  $("#weather").onclick = (e) => { e.target.classList.toggle("on"); set({ iron_beam_weather: e.target.classList.contains("on") ? "cloudy" : "clear" }); };
  const slide = (el, fmt, patch) => (el.onchange = () => { el.nextElementSibling.textContent = fmt(el.value); set(patch()); });
  slide($("#pk"), (v) => v, () => ({ pk_scale: +$("#pk").value }));
  slide($("#thr"), (v) => v, () => ({ conf_threshold: +$("#thr").value }));
  slide($("#abs"), (v) => v, () => ({ abstain_threshold: +$("#abs").value }));
  slide($("#pmul"), (v) => `×${v}`, () => ({ priority_multiplier: +$("#pmul").value }));
  const all = (sel, key) => Object.fromEntries([...document.querySelectorAll(sel)].map((x) => [x.dataset[key], +x.value]));
  document.querySelectorAll("[data-tier]").forEach((r) => slide(r, (v) => `$${v}M`, () => ({ tier_value: all("[data-tier]", "tier") })));
  document.querySelectorAll("[data-wh]").forEach((r) => slide(r, (v) => v, () => ({ warhead: all("[data-wh]", "wh") })));
  document.querySelectorAll("[data-sv]").forEach((inp) => (inp.onchange = () => set({
    site_value: Object.fromEntries([...document.querySelectorAll("[data-sv]")].filter((x) => x.value !== "").map((x) => [x.dataset.sv, +x.value])),
  })));
}

init();
