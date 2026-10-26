// Typical air-defence demo: map + one threat card (timeline of its flight, the call now, the lenses) + tonight's score.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (m) => (m < 0.001 ? "<$1k" : m < 1 ? `$${Math.round(m * 1000)}k` : m < 1000 ? `$${m.toFixed(1)}M` : `$${(m / 1000).toFixed(2)}B`);
const CLS_COLOR = { B: [255, 90, 54], R: [255, 140, 70], C: [255, 200, 90], D: [235, 120, 200] };
const VIEWS = { theatre: [40.5, 31.2, 4.7, 50], north: [35.35, 32.95, 8.3, 50], south: [39.5, 24.5, 4.6, 45] };
const SCEN_VIEW = { northern_barrage: "north", april_mixed: "theatre", october_saturation: "theatre", june_attrition: "theatre", houthi_eilat: "south" };
const DONE = ["killed", "leaked", "fell_open"];

// ---------------------------------------------------------------- i18n
const I18N = {
  en: {
    brand: "Air Defence", busy: "deciding…", skipped: (x) => `skipped ${x}`, play: "Play", pause: "Pause", skip: "Skip transit", lang: "עברית",
    cls: { B: "ballistic missile", R: "rocket", C: "cruise missile", D: "drone" },
    to: "to", openGround: "open ground near", prio: "priority", secondary: "secondary", impactIn: "impact in",
    following: "Following the most urgent threat", pinned: "Pinned", followBtn: "follow most urgent",
    calls: "calls", now: "now", impact: "impact",
    act: { hold: "Let it fall", fire: (e) => `Fire ${e} now`, fire2: (e) => `Fire two ${e} now`, wait: (e) => `Wait for ${e}` },
    status: { pending: "tracking", waiting: "waiting", held: "letting it fall", inflight: "interceptor in flight", killed: "intercepted", leaked: "hit", fell_open: "fell in open ground" },
    callNow: "Typical's call now", plannerCall: "Planner's call now", naiveCall: "Fire-at-everything's call now",
    lensLocal: "this threat", lensRaid: "tonight's raid", lensCampaign: "the campaign",
    why: (t, l, r, c) => `Expected cost ${t}: this threat ${l}` + (r ? `, tonight's raid +${r}` : "") + (c ? `, the campaign +${c}` : ""),
    esc: "Below the confidence bar: the operator took the planner's call.",
    agrees: "The planner agrees.", differs: (x) => `The planner would: ${x}.`,
    reads: "What Typical reads", missed: "missed", hit: "hit", empty: "Press Play. The most urgent threat appears here.",
    notYet: "No interceptor can reach it yet.",
    score: "Tonight", pol: { typical: "Typical", naive: "Fire at everything", planner: "Planner" },
    rows: { spent: "Spent", hits: "Hits on defended sites", kills: "Intercepted", a3: "Arrow 3 left" },
    foot: (a, ms) => `agrees with planner ${a}% · ${ms} ms per call`, comparePlanner: "compare planner",
    attacker: (x) => `attacker spent ${x}`,
    posture: "Magazine posture", postures: ["Healthy", "Conserve", "Critical"], prioSites: "Priority sites",
    assets: "Assets & conditions", allied: "US THAAD + Navy", weather: "Cloud cover (laser degraded)",
    knobs: "Escalation & Pk", pkScale: "Pk scale", escTop: "Escalate below", warn: "Early warning (min)",
    dmg: "Damage weights ($M-equivalent of a hit)", tier: { strategic: "strategic", city: "city", infra: "infrastructure" },
    prioMult: "priority ×", warhead: "Warhead weight (1 = 500 kg)", perSite: "Per-site value, $M (blank = tier)",
    note: "Posture, stock, damage weights and priorities feed all three lenses; Typical reads them as sentences, the planner as numbers.",
  },
  he: {
    brand: "הגנה אווירית", busy: "מחליט…", skipped: (x) => `דולגו ${x}`, play: "הפעל", pause: "השהה", skip: "דלג על מעבר", lang: "English",
    cls: { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" },
    to: "לעבר", openGround: "שטח פתוח ליד", prio: "בעדיפות", secondary: "משני", impactIn: "פגיעה בעוד",
    following: "עוקב אחרי האיום הדחוף ביותר", pinned: "נעוץ", followBtn: "עקוב אחרי הדחוף",
    calls: "החלטות", now: "עכשיו", impact: "פגיעה",
    act: { hold: "לתת לו ליפול", fire: (e) => `לירות ${e} עכשיו`, fire2: (e) => `לירות שני ${e} עכשיו`, wait: (e) => `להמתין ל${e}` },
    status: { pending: "במעקב", waiting: "בהמתנה", held: "נותנים ליפול", inflight: "מיירט באוויר", killed: "יורט", leaked: "פגיעה", fell_open: "נפל בשטח פתוח" },
    callNow: "ההחלטה של Typical עכשיו", plannerCall: "ההחלטה של המתכנן עכשיו", naiveCall: "ההחלטה של 'יורים על הכול'",
    lensLocal: "האיום הזה", lensRaid: "המתקפה הלילה", lensCampaign: "המערכה",
    why: (t, l, r, c) => `עלות צפויה ${t}: האיום הזה ${l}` + (r ? `, המתקפה הלילה +${r}` : "") + (c ? `, המערכה +${c}` : ""),
    esc: "מתחת לסף הביטחון: המפעיל ביצע את החלטת המתכנן.",
    agrees: "המתכנן מסכים.", differs: (x) => `המתכנן היה בוחר: ${x}.`,
    reads: "מה Typical קורא (באנגלית)", missed: "החטיא", hit: "פגע", empty: "לחצו הפעל. האיום הדחוף ביותר יופיע כאן.",
    notYet: "אף מיירט עדיין לא מגיע אליו.",
    score: "הלילה", pol: { typical: "Typical", naive: "יורים על הכול", planner: "מתכנן" },
    rows: { spent: "הוצאה", hits: "פגיעות באתרים מוגנים", kills: "יורטו", a3: "חץ 3 שנותרו" },
    foot: (a, ms) => `הסכמה עם המתכנן ${a}% · ${ms} מ״ש להחלטה`, comparePlanner: "השווה למתכנן",
    attacker: (x) => `התוקף הוציא ${x}`,
    posture: "מצב מלאים", postures: ["תקין", "שימור", "קריטי"], prioSites: "אתרים בעדיפות",
    assets: "נכסים ותנאים", allied: "THAAD וצי ארה״ב", weather: "עננות (לייזר מוגבל)",
    knobs: "סף העברה ו-Pk", pkScale: "מקדם Pk", escTop: "העבר למפעיל מתחת", warn: "התרעה מוקדמת (דק׳)",
    dmg: "משקלי נזק (מיליון $ שקולים לפגיעה)", tier: { strategic: "אסטרטגי", city: "עיר", infra: "תשתית" },
    prioMult: "מכפיל עדיפות", warhead: "משקל ראש קרב (1 = 500 ק״ג)", perSite: "ערך לאתר, מיליון $ (ריק = לפי סוג)",
    note: "מצב המלאים, המלאי, משקלי הנזק והעדיפויות נכנסים לשלוש העדשות; Typical קורא אותם כמשפטים, המתכנן כמספרים.",
  },
};
let LANG = localStorage.getItem("airdef-lang") || "en";
const T = () => I18N[LANG];
const nm = (o) => (LANG === "he" && o.name_he ? o.name_he : o.name);
const glyphs = (s) => (LANG === "he" ? [...s].reverse().join("") : s);   // deck.gl TextLayer has no bidi
const effName = (e) => nm(META.effectors[e]).replace("F-15/F-35 ", "").replace("F-16 laser-guided rocket (APKWS)", "APKWS").replace("GPS jamming (EW)", "EW").replace(" laser", "");
const actLabel = (key) => { if (!key) return "—"; if (key === "hold") return T().act.hold; const [a, e] = key.split(":"); return T().act[a](effName(e)); };
const siteName = (sid) => { const s = META.sites.find((x) => x.id === sid); return s ? nm(s) : ""; };
const fmtT = (s) => { const neg = s < 0; s = Math.abs(s); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60); return (neg ? "−" : "") + (h ? h + ":" : "") + String(m).padStart(2, "0") + ":" + String(x).padStart(2, "0"); };

let META, PATHS = {}, PATH_SCEN = null, S = null, fetchedAt = 0, selected = null, pinned = false, viewPol = "typical", detail = null, showPlanner = false;
const flashSeen = new Map();

// ---------------------------------------------------------------- map
const map = new maplibregl.Map({ container: "map", style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  center: [35.0, 31.6], zoom: 6.6, pitch: 45, maxPitch: 75, attributionControl: { compact: true } });
const overlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
map.addControl(overlay);
const flyTo = (v) => { const [lon, lat, zoom, pitch] = VIEWS[v]; map.flyTo({ center: [lon, lat], zoom, pitch, duration: 1600 }); };
const api = (p, body) => fetch(p, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : {}).then((r) => r.json());

async function init() {
  META = await api("/api/meta");
  $("#lang").onclick = () => { LANG = LANG === "en" ? "he" : "en"; localStorage.setItem("airdef-lang", LANG); applyLang(); };
  $("#scen").onclick = async (e) => { const s = e.target.dataset.s; if (!s) return; selected = null; pinned = false; detail = null; await api("/api/control", { action: "reset", scenario: s }); flyTo(SCEN_VIEW[s]); };
  $("#speeds").innerHTML = [1, 5, 20, 60, 200].map((x) => `<button data-x="${x}">${x}×</button>`).join("");
  $("#speeds").onclick = (e) => e.target.dataset.x && api("/api/control", { action: "speed", speed: +e.target.dataset.x });
  $("#play").onclick = () => api("/api/control", { action: S?.running ? "pause" : "play" });
  $("#skip").onclick = () => api("/api/control", { action: "skip" });
  $("#gear").onclick = () => { $("#drawer").classList.toggle("open"); $("#gear").classList.toggle("on"); };
  $("#card").onclick = (e) => { if (e.target.id === "followBtn") { pinned = false; pickFollow(); } };
  $("#score").onclick = (e) => { if (e.target.id === "cmpBtn") { showPlanner = !showPlanner; renderScore(); } };
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
    return `<button data-s="${k}" title="${esc(blurb)}">${esc(title.split(":")[0])}</button>`;
  }).join("");
  renderTop(); renderScore(); renderCard(); buildDrawer();
}

async function poll() {
  try { S = await api(`/api/state?policy=${viewPol}`); fetchedAt = performance.now(); } catch { return; }
  if (S.scenario !== PATH_SCEN) {
    const P = await api("/api/paths");
    PATHS = Object.fromEntries(P.threats.map((t) => [t.id, t]));
    PATH_SCEN = P.scenario; flashSeen.clear();
  }
  if (!pinned) pickFollow();
  renderTop(); renderScore();
}

function pickFollow() {   // the most urgent threat that has a call, else the most urgent live one
  const live = S.tracks.filter((x) => !DONE.includes(x.status));
  const cand = live.filter((x) => x.choice).sort((a, b) => a.tti - b.tti)[0] || live.sort((a, b) => a.tti - b.tti)[0];
  const cur = S.tracks.find((x) => x.id === selected);
  if (cand && (!cur || DONE.includes(cur.status))) { selected = cand.id; loadDetail(); }
}

function select(id) { selected = id; pinned = true; loadDetail(); }
async function loadDetail() {
  if (!selected) { detail = null; renderCard(); return; }
  try { detail = await api(`/api/track/${selected}?policy=${viewPol}`); } catch { return; }
  renderCard();
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

// ---------------------------------------------------------------- map layers
function frame() {
  requestAnimationFrame(frame);
  if (!S || !META) return;
  const t = simNow(), prio = new Set(S.cfg.priority_sites), dim = selected ? 0.45 : 1;
  const live = [], trails = [], futures = [];
  for (const tr of S.tracks) {
    const path = PATHS[tr.id]; if (!path) continue;
    if (t > Math.min(S.killed[tr.id] ?? Infinity, path.t0 + path.T)) continue;
    const cur = posAt(path, t), sel = tr.id === selected, a = sel ? 1 : dim;
    const past = path.p.filter((q) => q[3] < t).map((q) => q.slice(0, 3)); past.push(cur.slice(0, 3));
    const col = CLS_COLOR[path.cls];
    trails.push({ path: past, color: [...col, 220 * a], w: sel ? 3 : 1.4 });
    futures.push({ path: [cur.slice(0, 3), ...path.p.filter((q) => q[3] > t).map((q) => q.slice(0, 3))], color: [...col, (sel ? 140 : 50) * a], w: sel ? 2 : 1 });
    live.push({ pos: cur.slice(0, 3), color: col, tr, sel });
  }
  const batt = Object.fromEntries(META.batteries.map((b) => [b.id, b]));
  const shots = [], flashes = [], nowMs = performance.now();
  for (const f of S.flights) {
    if (t < f.t0) continue;
    const b = batt[f.b], g = Math.min(1, (t - f.t0) / Math.max(1, f.t1 - f.t0)), pts = [];
    for (let k = 0; k <= 24 * g; k++) { const s = k / 24; pts.push([b.lon + (f.lon - b.lon) * s, b.lat + (f.lat - b.lat) * s, f.alt * Math.pow(s, 0.6)]); }
    if (g < 1) shots.push({ path: pts });
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
      getFillColor: (d) => (prio.has(d.id) ? [255, 181, 71, 34] : [240, 238, 235, 14]), stroked: true, lineWidthUnits: "pixels",
      getLineWidth: (d) => (prio.has(d.id) ? 1.4 : 0.5), getLineColor: (d) => (prio.has(d.id) ? [255, 181, 71, 190] : [240, 238, 235, 60]),
      updateTriggers: { getFillColor: [...prio].join(), getLineColor: [...prio].join(), getLineWidth: [...prio].join() } }),
    new deck.TextLayer({ id: "siteNames", data: META.sites.filter((s) => (prio.has(s.id) && map.getZoom() > 6) || map.getZoom() > 7.2), getPosition: (d) => [d.lon, d.lat],
      getText: (d) => glyphs(nm(d)), getSize: 11, getColor: (d) => (prio.has(d.id) ? [255, 181, 71, 210] : [240, 238, 235, 120]), getPixelOffset: [0, -14],
      fontFamily: "Helvetica Neue, Arial, sans-serif", fontWeight: 500, characterSet: "auto", updateTriggers: { getText: LANG, getColor: [...prio].join() } }),
    new deck.ScatterplotLayer({ id: "batt", data: batts, getPosition: (d) => [d.lon, d.lat], radiusUnits: "pixels", getRadius: 2.5, getFillColor: [90, 209, 255, 170] }),
    selTr && !DONE.includes(selTr.status) && new deck.ScatterplotLayer({ id: "ipp", data: [selTr], getPosition: (d) => [d.ipp[0], d.ipp[1]],
      getRadius: (d) => Math.max(d.ipp[2], 0.6) * 1000, filled: true, stroked: true, getFillColor: [255, 181, 71, 26], getLineColor: [255, 181, 71, 200], lineWidthUnits: "pixels", getLineWidth: 1.5 }),
    new deck.PathLayer({ id: "future", data: futures, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w }),
    new deck.PathLayer({ id: "trail", data: trails, getPath: (d) => d.path, getColor: (d) => d.color, widthUnits: "pixels", getWidth: (d) => d.w }),
    new deck.PathLayer({ id: "shots", data: shots, getPath: (d) => d.path, getColor: [90, 209, 255, 255], widthUnits: "pixels", getWidth: 2 }),
    new deck.ScatterplotLayer({ id: "heads", data: live, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => (d.sel ? 7 : 3.5),
      getFillColor: (d) => [...d.color, d.sel ? 255 : 255 * dim], stroked: true, getLineColor: (d) => (d.sel ? [255, 255, 255] : [20, 19, 18]),
      lineWidthUnits: "pixels", getLineWidth: (d) => (d.sel ? 2 : 1), pickable: true, onClick: (i) => select(i.object.tr.id) }),
    new deck.ScatterplotLayer({ id: "flash", data: flashes, getPosition: (d) => d.pos, radiusUnits: "pixels", getRadius: (d) => 6 + d.age * 26, filled: false,
      stroked: true, lineWidthUnits: "pixels", getLineWidth: 2, getLineColor: (d) => (d.hit ? [255, 255, 255, 255 * (1 - d.age / 1.6)] : [160, 157, 151, 200 * (1 - d.age / 1.6)]) }),
    new deck.ScatterplotLayer({ id: "leaks", data: leaks, getPosition: (d) => [d[0], d[1]], radiusUnits: "pixels", getRadius: 6, getFillColor: [255, 77, 77, 230],
      stroked: true, getLineColor: [255, 77, 77, 80], lineWidthUnits: "pixels", getLineWidth: 6 }),
  ].filter(Boolean) });
  $("#clock").textContent = "T+" + fmtT(t);
}

// ---------------------------------------------------------------- top + score
function renderTop() {
  if (!S) return;
  $("#play").textContent = S.running ? T().pause : T().play;
  $("#play").classList.toggle("on", S.running);
  document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", +b.dataset.x === S.speed));
  document.querySelectorAll("#scen button").forEach((b) => b.classList.toggle("on", b.dataset.s === S.scenario));
  const showSkip = S.skipped > 0 && S.running && performance.now() - (renderTop.lastSkip?.at ?? -1e9) < 2500;
  if (S.skipped !== renderTop.lastSkip?.v) renderTop.lastSkip = { v: S.skipped, at: performance.now() };
  $("#busy").textContent = S.backlog ? T().busy : showSkip ? T().skipped(fmtT(S.skipped)) : "";
  $("#busy").style.visibility = S.backlog || showSkip ? "visible" : "hidden";
}

function renderScore() {
  if (!S) return;
  const L = S.ledger, cols = ["typical", "naive", ...(showPlanner ? ["planner"] : [])].filter((k) => L[k]);
  const a3 = (x) => (x.init.arrow3 ? `${Math.round(x.left.arrow3)}/${x.init.arrow3}` : "—");
  const row = (name, f, bad) => `<tr><td>${name}</td>${cols.map((k) => `<td class="${k === "typical" ? "typ" : ""} ${bad && bad(L[k]) ? "bad" : ""}">${f(L[k])}</td>`).join("")}</tr>`;
  const ty = L.typical, d = Math.max(ty?.decisions || 0, 1);
  $("#score").innerHTML = `<div class="eyebrow">${T().score} · ${T().attacker(money(S.attacker_cost))}</div>
    <table><tr><th></th>${cols.map((k) => `<th>${T().pol[k]}</th>`).join("")}</tr>
    ${row(T().rows.spent, (x) => money(x.spent))}
    ${row(T().rows.hits, (x) => x.leaks, (x) => x.leaks > 0)}
    ${row(T().rows.kills, (x) => x.kills)}
    ${row(T().rows.a3, a3)}</table>
    <div class="scorefoot"><span>${ty && S.model !== "planner only" ? T().foot(Math.round((ty.agree / d) * 100), Math.round(ty.ms / Math.max(ty.calls, 1))) : ""}</span>
    <button id="cmpBtn" class="${showPlanner ? "on" : ""}">${T().comparePlanner}</button></div>`;
}

// ---------------------------------------------------------------- the threat card
function timeline(D) {
  const W = 384, LW = 84, PW = W - LW - 8, now = D.t, end = D.impact;
  const hist0 = D.history.length ? D.history[0][0] : now;
  const x0 = Math.max(D.t0, Math.min(now, hist0) - 0.12 * (end - Math.min(now, hist0))), span = Math.max(end - x0, 1);
  const X = (t) => LW + ((Math.min(Math.max(t, x0), end) - x0) / span) * PW;
  const effs = [...new Set([...Object.keys(D.windows), ...D.shots.map((s) => s.eff)])].sort((a, b) => META.effectors[b].layer - META.effectors[a].layer);
  const RH = 17, top = 22, H = top + effs.length * RH + 22;
  let g = "";
  g += `<text x="0" y="12">${T().calls}</text>`;
  for (const [t, key] of D.history) {
    const c = key === "hold" ? "#a19d97" : key.startsWith("wait") ? "#ffb547" : "#5ad1ff";
    g += `<g><title>${esc(actLabel(key))} · T+${fmtT(t)}</title><rect x="${X(t) - 4}" y="4" width="8" height="8" transform="rotate(45 ${X(t)} 8)" fill="${c}"/></g>`;
  }
  effs.forEach((e, i) => {
    const y = top + i * RH, w = D.windows[e];
    g += `<text x="0" y="${y + 11}">${esc(effName(e))}</text><line x1="${LW}" x2="${LW + PW}" y1="${y + 7}" y2="${y + 7}" stroke="rgba(240,238,235,.07)"/>`;
    if (w) g += `<rect x="${X(w[0])}" y="${y + 2}" width="${Math.max(2, X(w[1]) - X(w[0]))}" height="10" rx="3" fill="rgba(90,209,255,.22)" stroke="rgba(90,209,255,.55)"/>`;
    for (const s of D.shots.filter((x) => x.eff === e)) {
      const done = now >= s.t1, col = !done ? "#5ad1ff" : s.hit ? "#7fd18b" : "#ff4d4d";
      g += `<line x1="${X(s.t0)}" x2="${X(Math.min(now, s.t1))}" y1="${y + 7}" y2="${y + 7}" stroke="${col}" stroke-width="2.5"/>`;
      if (done) g += `<text x="${X(s.t1) + 4}" y="${y + 11}" style="fill:${col}">${s.hit ? "✓" : "✕"}</text>`;
    }
  });
  const yb = top + effs.length * RH;
  g += `<line x1="${X(now)}" x2="${X(now)}" y1="0" y2="${yb + 4}" stroke="#ffb547" stroke-width="1.5"/>`;
  g += `<text x="${X(now)}" y="${yb + 16}" text-anchor="middle" style="fill:#ffb547">${T().now}</text>`;
  g += `<line x1="${LW + PW}" x2="${LW + PW}" y1="0" y2="${yb + 4}" stroke="#ff5a36" stroke-width="1.5"/>`;
  g += `<text x="${LW + PW}" y="${yb + 16}" text-anchor="end" style="fill:#ff5a36">${T().impact}</text>`;
  return `<svg class="tl" viewBox="0 0 ${W} ${H}" height="${H}" direction="ltr">${g}</svg>`;
}

function lensBar(o, scale) {
  const [, , total, L] = o, tot = Math.max(total, 1e-9), w = (v) => `${(100 * v) / tot * (scale ?? 1)}%`;
  return `<div class="lens"><i class="l-local" style="width:${w(L.local)}"></i><i class="l-raid" style="width:${w(L.raid)}"></i><i class="l-campaign" style="width:${w(L.campaign)}"></i></div>`;
}

function renderCard() {
  const el = $("#card");
  if (!detail || !META) { el.innerHTML = `<div class="empty">${T().empty}</div>`; return; }
  const D = detail, ty = META.threat_types[D.kind], tt = T();
  const target = D.open_ground ? `${tt.openGround} ${esc(siteName(D.site))}` :
    `${tt.to} <b>${esc(siteName(D.site))}</b> · <span class="${D.priority ? "prio" : ""}">${D.priority ? tt.prio : tt.secondary}</span>`;
  const tti = D.impact - D.t;
  let html = `<div class="chead"><div><div class="ctitle">${esc(nm(ty))}</div><div class="csub">${tt.cls[D.cls]} · ${target}</div></div>
    <div class="tti"><div class="eyebrow">${DONE.includes(D.status) ? tt.status[D.status] : tt.impactIn}</div><div class="big">${DONE.includes(D.status) ? "" : fmtT(tti)}</div></div></div>
    <div class="follow"><span>${pinned ? tt.pinned : tt.following} · ${D.id}</span>${pinned ? `<button id="followBtn">${tt.followBtn}</button>` : ""}</div>`;
  html += timeline(D);
  if (D.history.length) {
    html += `<div class="log">${D.history.slice(-4).map(([t, k]) => `<span class="mono">T+${fmtT(t)}</span> <b>${esc(actLabel(k))}</b>`).join(" → ")}</div>`;
  }
  const i = D.info;
  if (!i) { el.innerHTML = html + `<div class="why">${tt.notYet}</div>`; return; }
  const opts = Object.fromEntries(i.opts.map((o) => [o[0], o]));
  const chosen = D.choice && opts[D.choice] ? opts[D.choice] : null;
  const order = i.dist ? Object.keys(i.dist).sort((a, b) => i.dist[b] - i.dist[a]) : i.opts.map((o) => o[0]).sort((a, b) => opts[a][2] - opts[b][2]);
  const p = (k) => (i.dist ? `${Math.round(i.dist[k] * 100)}%` : money(opts[k][2]));
  html += `<div class="call"><div class="eyebrow">${viewPol === "typical" ? tt.callNow : viewPol === "planner" ? tt.plannerCall : tt.naiveCall}</div>`;
  if (chosen) {
    const [, , total, L] = chosen;
    html += `<div class="callmain"><span class="what">${esc(actLabel(chosen[0]))}</span><span class="p">${p(chosen[0])}</span></div>
      ${lensBar(chosen)}<div class="why">${esc(tt.why(money(total), money(L.local), L.raid > 0.0005 ? money(L.raid) : "", L.campaign > 0.0005 ? money(L.campaign) : ""))}</div>
      <div class="lenskey" style="margin-top:6px"><span class="k-local">${tt.lensLocal}</span><span class="k-raid">${tt.lensRaid}</span><span class="k-campaign">${tt.lensCampaign}</span></div>`;
  }
  if (D.escalated) html += `<div class="esc">${tt.esc}</div>`;
  const maxT = Math.max(...order.slice(0, 4).map((k) => Math.min(opts[k][2], 50)));
  html += `<div class="alts">${order.filter((k) => k !== D.choice).slice(0, 3).map((k) =>
    `<div class="alt"><span>${esc(actLabel(k))} <span class="mono" style="color:var(--mute)">· ${money(opts[k][2])}</span></span><span class="p">${p(k)}</span>${lensBar(opts[k], Math.min(opts[k][2], 50) / maxT)}</div>`).join("")}</div>`;
  if (viewPol === "typical" && i.planner) html += `<div class="agree">${i.planner === D.choice ? tt.agrees : esc(tt.differs(actLabel(i.planner)))}</div>`;
  html += `<details class="reads"><summary>${tt.reads}</summary><pre>${esc(i.state)}</pre></details></div>`;
  const open = el.querySelector("details.reads")?.open;
  el.innerHTML = html;
  if (open) el.querySelector("details.reads").open = true;
}

// ---------------------------------------------------------------- settings drawer
function buildDrawer() {
  if (!S) return;
  const tt = T(), c = S.cfg;
  const rng = (attrs, label, val, shown) => `<div class="rng"><span>${label}</span><input type="range" ${attrs} value="${val}"><span>${shown}</span></div>`;
  const heCls = { B: "טיל בליסטי", R: "רקטה", C: "טיל שיוט", D: "כטב״ם" }, enCls = { B: "ballistic missile", R: "rocket", C: "cruise missile", D: "drone" };
  const whLabel = (k) => (META.threat_types[k] ? nm(META.threat_types[k]) : (LANG === "he" ? heCls : enCls)[k]);
  $("#drawer").innerHTML = `
    <div class="field"><span class="eyebrow">${tt.posture}</span><div class="seg" id="reserve">${tt.postures.map((p, k) => `<button data-r="${k}" class="${c.reserve === k ? "on" : ""}">${p}</button>`).join("")}</div></div>
    <div class="field"><span class="eyebrow">${tt.prioSites}</span><div class="seg" id="prio">${META.sites.map((s) => `<button data-site="${s.id}" class="${c.priority_sites.includes(s.id) ? "on" : ""}">${esc(nm(s))}</button>`).join("")}</div></div>
    <div class="field"><span class="eyebrow">${tt.assets}</span><div class="seg"><button id="allied" class="${c.allied ? "on" : ""}">${tt.allied}</button><button id="weather" class="${c.iron_beam_weather === "cloudy" ? "on" : ""}">${tt.weather}</button></div></div>
    <div class="field"><span class="eyebrow">${tt.knobs}</span>
      ${rng('id="pk" min="0.6" max="1.1" step="0.05"', tt.pkScale, c.pk_scale, c.pk_scale)}
      ${rng('id="thr" min="0.2" max="0.9" step="0.02"', tt.escTop, c.conf_threshold, c.conf_threshold)}
      ${rng('id="warn" min="0" max="30" step="1"', tt.warn, Math.round(c.forecast_s / 60), Math.round(c.forecast_s / 60))}</div>
    <div class="field"><span class="eyebrow">${tt.dmg}</span>
      ${Object.keys(c.tier_value).map((k) => rng(`data-tier="${k}" min="0" max="5000" step="50"`, tt.tier[k], Math.round(c.tier_value[k]), `$${Math.round(c.tier_value[k])}M`)).join("")}
      ${rng('id="pmul" min="1" max="10" step="0.5"', tt.prioMult, c.priority_multiplier, `×${c.priority_multiplier}`)}
      <details class="dmg"><summary>${tt.warhead}</summary>${Object.keys(c.warhead).map((k) => rng(`data-wh="${k}" min="0" max="2" step="0.01"`, esc(whLabel(k)), c.warhead[k], c.warhead[k])).join("")}</details>
      <details class="dmg"><summary>${tt.perSite}</summary><div class="sitegrid">${META.sites.map((s) => `<span>${esc(nm(s))}</span><input data-sv="${s.id}" type="number" min="0" step="50" value="${c.site_value[s.id] ?? ""}" placeholder="${Math.round(c.tier_value[s.tier])}">`).join("")}</div></details></div>
    <div class="note">${tt.note}</div>`;
  const set = (patch) => api("/api/config", patch).then((r) => { S.cfg = r.cfg; });
  $("#reserve").onclick = (e) => { if (e.target.dataset.r == null) return; document.querySelectorAll("#reserve button").forEach((b) => b.classList.toggle("on", b === e.target)); set({ reserve: +e.target.dataset.r }); };
  $("#prio").onclick = (e) => { if (!e.target.dataset.site) return; e.target.classList.toggle("on"); set({ priority_sites: [...document.querySelectorAll("#prio .on")].map((b) => b.dataset.site) }); };
  $("#allied").onclick = (e) => { e.target.classList.toggle("on"); set({ allied: e.target.classList.contains("on") }); };
  $("#weather").onclick = (e) => { e.target.classList.toggle("on"); set({ iron_beam_weather: e.target.classList.contains("on") ? "cloudy" : "clear" }); };
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

init();
