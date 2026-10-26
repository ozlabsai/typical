// Typical air-defence decision desk: polls /api/state, interpolates trajectories by sim time, draws with deck.gl.
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (m) => (m < 0.001 ? "<$1k" : m < 1 ? `$${Math.round(m * 1000)}k` : m < 1000 ? `$${m.toFixed(1)}M` : `$${(m / 1000).toFixed(2)}B`);
const CLS_COLOR = { B: [255, 90, 54], R: [255, 140, 70], C: [255, 200, 90], D: [235, 120, 200] };
const VIEWS = {
  theatre: { longitude: 40.5, latitude: 31.2, zoom: 4.7, pitch: 50, bearing: 0 },
  israel: { longitude: 35.0, latitude: 31.6, zoom: 6.6, pitch: 45, bearing: 0 },
  north: { longitude: 35.35, latitude: 32.95, zoom: 8.3, pitch: 50, bearing: 0 },
  south: { longitude: 39.5, latitude: 24.5, zoom: 4.6, pitch: 45, bearing: 0 },
};
const SCEN_VIEW = { northern_barrage: "north", april_mixed: "theatre", october_saturation: "theatre", june_attrition: "theatre", houthi_eilat: "south" };
const SHORT_EFF = {};

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

async function init() {
  META = await api("/api/meta");
  for (const [k, e] of Object.entries(META.effectors)) SHORT_EFF[k] = e.name.replace("F-15/F-35 ", "").replace("F-16 laser-guided rocket (APKWS)", "APKWS").replace("GPS jamming (EW)", "EW jamming").replace(" laser", "");
  $("#modelName").textContent = META.model;
  $("#scen").innerHTML = Object.entries(META.scenarios).map(([k, v]) => `<button data-s="${k}" title="${esc(v.blurb)}">${esc(v.title.split(":")[0])}</button>`).join("");
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
  flyTo(SCEN_VIEW[S.scenario]);
  buildConsole();
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
  renderTop(); renderBoard(); renderLedger();
}

// sim time between polls: extrapolate so motion is smooth
function simNow() {
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
    const killedAt = S.killed[tr.id];
    const end = Math.min(killedAt ?? Infinity, path.t0 + path.T);
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

  overlay.setProps({ layers: [
    new deck.ScatterplotLayer({ id: "sites", data: META.sites, getPosition: (d) => [d.lon, d.lat], getRadius: (d) => d.r * 1000,
      getFillColor: (d) => (prio.has(d.id) ? [255, 181, 71, 40] : [240, 238, 235, 18]), stroked: true, lineWidthUnits: "pixels", getLineWidth: (d) => (prio.has(d.id) ? 1.5 : 0.6),
      getLineColor: (d) => (prio.has(d.id) ? [255, 181, 71, 200] : [240, 238, 235, 70]), updateTriggers: { getFillColor: [...prio].join(), getLineColor: [...prio].join(), getLineWidth: [...prio].join() } }),
    new deck.TextLayer({ id: "siteNames", data: META.sites, visible: map.getZoom() > 6.8, getPosition: (d) => [d.lon, d.lat], getText: (d) => d.name, getSize: 11, getColor: [240, 238, 235, 150],
      getPixelOffset: [0, -14], fontFamily: "Helvetica Neue, Arial, sans-serif", fontWeight: 500, characterSet: "auto" }),
    new deck.ScatterplotLayer({ id: "batt", data: META.batteries.filter((b) => S.cfg.allied || !META.effectors[b.eff].allied), getPosition: (d) => [d.lon, d.lat],
      radiusUnits: "pixels", getRadius: 3.5, getFillColor: [90, 209, 255, 220], updateTriggers: { data: S.cfg.allied } }),
    new deck.TextLayer({ id: "battNames", visible: map.getZoom() > 7.5, data: META.batteries.filter((b) => S.cfg.allied || !META.effectors[b.eff].allied), getPosition: (d) => [d.lon, d.lat],
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
  if (key === "hold") return "Hold fire";
  const m = key.match(/^(\w+?)x([12])$/);
  if (m) return (m[2] === "2" ? "2× " : "") + SHORT_EFF[m[1]];
  const [a, b] = key.split(">");
  return `${SHORT_EFF[a]} → ${SHORT_EFF[b]} if missed`;
}

function renderTop() {
  $("#play").textContent = S.running ? "Pause" : "Play";
  $("#play").classList.toggle("on", S.running);
  document.querySelectorAll("#speeds button").forEach((b) => b.classList.toggle("on", +b.dataset.x === S.speed));
  document.querySelectorAll("#scen button").forEach((b) => b.classList.toggle("on", b.dataset.s === S.scenario));
  $("#busy").style.visibility = S.backlog ? "visible" : "hidden";
}

const STATUS = { pending: "tracking", planned: "planned", held: "hold", engaging: "engaging", killed: "killed", leaked: "impact", fell_open: "fell open" };
function renderBoard() {
  const order = { engaging: 0, planned: 1, pending: 2, held: 3, leaked: 4, killed: 5, fell_open: 6 };
  const rows = [...S.tracks].sort((a, b) => (order[a.status] - order[b.status]) || a.tti - b.tti);
  const n = (st) => S.tracks.filter((x) => x.status === st).length;
  $("#boardSummary").innerHTML = `${S.tracks.length} on screen · <b>${n("engaging") + n("planned")}</b> engaging/planned · ${n("held")} held · ${S.tracks.filter((x) => x.escalated).length} escalated`;
  const max = Math.floor(($("#rows").clientHeight || 400) / 52);
  $("#rows").innerHTML = rows.slice(0, max).map((r) => {
    const ty = META.threat_types[r.kind], col = CLS_COLOR[ty.cls];
    const conf = r.conf != null ? `<span class="mono">${Math.round(r.conf * 100)}%</span>` : "";
    const where = r.site ? r.site.replace(/_/g, " ") : "";
    return `<div class="row ${r.id === selected ? "sel" : ""}" data-id="${r.id}">
      <span class="dot" style="background:rgb(${col})"></span><span class="id mono">${r.id}</span>
      <span class="what">${esc(ty.name)}</span>
      <span class="chip c-${r.escalated && ["planned", "held", "pending"].includes(r.status) ? "esc" : r.status}">${r.escalated && ["planned", "held", "pending"].includes(r.status) ? "operator" : STATUS[r.status]}</span>
      <span></span><span class="rec"><b>${esc(planShort(r.plan))}</b> ${conf} ${where ? "· " + esc(where) : ""}</span>
      <span class="tti mono">${r.tti > 0 && !["killed", "leaked", "fell_open"].includes(r.status) ? fmtT(r.tti) : ""}</span></div>`;
  }).join("");
}

function magRow(L) {
  const show = ["arrow3", "arrow2", "thaad", "dsling", "ironDome", "aim9x"];
  return show.filter((e) => L.init[e]).map((e) => {
    const f = L.left[e] / L.init[e];
    return `<div class="mag ${f < 0.2 ? "crit" : f < 0.5 ? "low" : ""}">${SHORT_EFF[e]} <span class="mono">${Math.round(L.left[e])}/${L.init[e]}</span><i><u style="width:${Math.max(0, f) * 100}%"></u></i></div>`;
  }).join("");
}

const POL_NAME = { naive: ["Fire at everything", "best Pk, no doctrine"], optimizer: ["Cost-aware optimizer", "numeric objective"], typical: ["Typical", "doctrine model, live"] };
function renderLedger() {
  $("#attCost").textContent = money(S.attacker_cost);
  $("#launched").textContent = S.launched; $("#total").textContent = S.total;
  for (const [k, L] of Object.entries(S.ledger)) {
    const d = Math.max(L.decisions, 1);
    const extra = k === "typical" && S.model !== "optimizer only"
      ? `<div class="kv" style="margin-top:4px"><span>follows doctrine <b>${Math.round((L.follow / d) * 100)}%</b></span><span>agrees w/ optimizer <b>${Math.round((L.agree / d) * 100)}%</b></span><span><b>${Math.round(L.ms / d)}</b> ms</span></div>` : "";
    $(`.pol.${k}`).innerHTML = `<h4>${POL_NAME[k][0]} <small>${POL_NAME[k][1]}</small></h4>
      <div class="big mono">${money(L.spent)}</div>
      <div class="kv"><span>kills <b>${L.kills}</b></span><span>hits on sites <b style="color:${L.leaks ? "var(--bad)" : "inherit"}">${L.leaks}</b></span><span>let fall <b>${L.fell_open}</b></span>${k === "typical" ? `<span>to operator <b>${L.escalated}</b></span>` : ""}<span>exchange <b>${(L.spent / Math.max(S.attacker_cost, 1e-6)).toFixed(1)}:1</b></span></div>
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
  const el = $("#p-decision");
  if (!selected || !detail) { el.innerHTML = `<div class="empty">Select a track on the map or the board to see what Typical reads and how it decides.</div>`; return; }
  const i = detail.info;
  const head = `<div id="dTitle">${detail.id} · ${esc(detail.name)}</div><div id="dSub">${STATUS[detail.status] ?? detail.status} · viewing <b>${POL_NAME[viewPol][0]}</b></div>`;
  if (!i) { el.innerHTML = head + `<div class="empty">Not decided yet: no interceptor can reach it inside the decision horizon.</div>`; return; }
  const state = esc(i.state).replace(/(Situation\.[\s\S]*)$/, '<span class="hl">$1</span>');
  const plan = Object.fromEntries(i.plans.map(([k, label, pk, cost]) => [k, { label, pk, cost }]));
  const planLine = (k) => (k === "hold" ? "no interceptor" : `${planShort(k)} · ${money(plan[k].cost)} · ${pkWord(plan[k].pk)}`);
  let bars = "";
  if (i.dist) {
    const acts = Object.keys(ACTIONS).sort((a, b) => i.dist[b] - i.dist[a]);
    bars = acts.map((a) => {
      const p = i.dist[a], k = i.resolved[a];
      const marks = (a === i.rule_action ? `<span class="mark m-rule">doctrine</span>` : "") + (k === i.opt ? `<span class="mark m-opt">optimizer</span>` : "");
      return `<div class="bar ${a === i.action ? "top" : ""}"><div class="lab"><div class="fill" style="width:${p * 100}%"></div><span class="txt">${ACTIONS[a]}${marks}<br><span style="color:var(--mute);font-size:11.5px">→ ${esc(planLine(k))}</span></span></div><span class="p mono">${(p * 100).toFixed(0)}%</span></div>`;
    }).join("") + `<div class="bar"><div class="lab"><div class="fill" style="width:${i.p_null * 100}%;background:rgba(160,157,151,.2);border-color:var(--mute)"></div><span class="txt">∅ none of these: escalate to operator</span></div><span class="p mono">${(i.p_null * 100).toFixed(0)}%</span></div>`;
  }
  const all = i.plans.map(([k]) => `<div class="bar ${k === detail.plan ? "top" : ""}"><div class="lab" style="padding:2px 8px"><span class="txt" style="font-size:11.5px">${esc(planLine(k))}${k === i.opt ? '<span class="mark m-opt">optimizer</span>' : ""}${k === detail.plan ? '<span class="mark m-rule">executing</span>' : ""}</span></div><span class="p mono" style="font-size:11px;color:var(--mute)">${money(i.scores[k])}</span></div>`).join("");
  const escl = detail.escalated ? `<div id="esc" style="display:block">Typical's confidence was below the escalation bar, so the track went to the operator, who executed the optimizer's plan.</div>` : "";
  const foot = `<div id="dFoot">
      <span>executing <b>${esc(planShort(detail.plan))}</b></span><span>optimizer <b>${esc(planShort(detail.others.optimizer))}</b></span>
      ${i.ms != null ? `<span>model latency <b class="mono">${i.ms.toFixed(0)} ms</b></span>` : "<span></span>"}
      <span>fire-at-everything <b>${esc(planShort(detail.others.naive))}</b></span></div>`;
  el.innerHTML = head + escl + `<div class="eyebrow">What ${viewPol === "typical" ? "Typical" : "the policy"} reads</div><div id="dState">${state}</div>
    ${i.dist ? `<div class="eyebrow">Typical's call: a probability over the doctrine's actions; the solver turns the action into a plan</div>${bars}` : ""}
    ${foot}<div class="eyebrow" style="margin-top:14px">Every feasible plan · expected cost incl. leak risk + magazine value</div>${all}`;
}
const ACTIONS = { hold: "Hold fire", protection: "Fire the sure-kill plan", value: "Fire the cheapest reliable plan" };
const pkWord = (p) => (p >= 0.95 ? "almost certain" : p >= 0.85 ? "very likely" : p >= 0.7 ? "likely" : p >= 0.5 ? "may stop it" : "unlikely");

// ---------------------------------------------------------------- commander's console
const PRESETS = [
  "When the threat is a drone, hold fire.",
  "When the threat will hit Haifa Bay refineries, fire the sure-kill plan.",
  "When the threat will hit Kiryat Shmona, hold fire.",
  "When the threat is a ballistic missile, fire the sure-kill plan.",
  "When the threat is a rocket, fire the cheapest reliable plan.",
];
const RULE_RE = /^when the threat (is an? [a-z ]+?|will hit .+?|will land in open ground), (hold fire|fire the sure-kill plan|fire the cheapest reliable plan)\.?$/i;
function renderOrders() {
  $("#orders").innerHTML = (S.cfg.orders || []).map((o, k) => `<div class="bar"><div class="lab" style="padding:4px 8px"><span class="txt">${k + 1}. ${esc(o)}${RULE_RE.test(o.trim()) ? "" : '<span class="mark m-opt">free text: not in the rule grammar</span>'}</span></div><button data-del="${k}" style="padding:2px 8px">×</button></div>`).join("") || `<div style="color:var(--mute)">No orders. The three standing rules apply.</div>`;
}
function buildConsole() {
  const c = S.cfg;
  const sites = META.sites.map((s) => `<button data-site="${s.id}" class="${c.priority_sites.includes(s.id) ? "on" : ""}">${esc(s.name)}</button>`).join("");
  $("#p-console").innerHTML = `
    <div class="field"><span class="eyebrow">Commander's orders: Typical reads these above the standing rules</span>
      <div id="orders"></div>
      <textarea id="intent" style="height:44px;margin-top:6px" placeholder="When the threat … , hold fire / fire the sure-kill plan / fire the cheapest reliable plan."></textarea>
      <div class="presets">${PRESETS.map((t) => `<button data-intent="${esc(t)}">${esc(t)}</button>`).join("")}</div>
      <div class="seg" style="margin-top:6px"><button id="issue" class="on">Issue order</button><button id="clearIntent">Clear all orders</button></div></div>
    <div class="field"><span class="eyebrow">Priority sites</span><div class="seg" id="prio">${sites}</div></div>
    <div class="field"><span class="eyebrow">Magazine posture</span><div class="seg" id="reserve">
      ${["Healthy stocks", "Conserve upper tier", "Critical"].map((t, k) => `<button data-r="${k}" class="${c.reserve === k ? "on" : ""}">${t}</button>`).join("")}</div></div>
    <div class="field"><span class="eyebrow">Assets & conditions</span><div class="seg">
      <button id="allied" class="${c.allied ? "on" : ""}">US THAAD + Navy</button>
      <button id="weather" class="${c.iron_beam_weather === "cloudy" ? "on" : ""}">Cloud cover (laser degraded)</button></div></div>
    <div class="field"><span class="eyebrow">Knobs (notional values)</span>
      <div class="rng"><span>Pk scale</span><input id="pk" type="range" min="0.6" max="1.1" step="0.05" value="${c.pk_scale}"><span class="mono" id="pkv">${c.pk_scale}</span></div>
      <div class="rng"><span>Escalate if top below</span><input id="thr" type="range" min="0.34" max="0.9" step="0.02" value="${c.conf_threshold}"><span class="mono" id="thrv">${c.conf_threshold}</span></div>
      <div class="rng"><span>Escalate if ∅ above</span><input id="abs" type="range" min="0.3" max="1" step="0.05" value="${c.abstain_threshold}"><span class="mono" id="absv">${c.abstain_threshold}</span></div>
      ${["strategic", "city", "infra"].map((t) => `<div class="rng"><span>${t} site hit</span><input data-tier="${t}" type="range" min="50" max="3000" step="50" value="${c.tier_value[t]}"><span class="mono">$${c.tier_value[t]}M</span></div>`).join("")}
      <div class="eyebrow" style="margin-top:6px;text-transform:none;letter-spacing:0">Consequence weights feed the optimizer only. Typical reads the rules, the orders and the priority sites. Posture, weather, allied assets and Pk change which concrete plan each of Typical's actions resolves to.</div></div>`;
  const set = (patch) => api("/api/config", patch).then((r) => { S.cfg = r.cfg; renderOrders(); });
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
  $("#pk").onchange = (e) => { $("#pkv").textContent = e.target.value; set({ pk_scale: +e.target.value }); };
  $("#thr").onchange = (e) => { $("#thrv").textContent = e.target.value; set({ conf_threshold: +e.target.value }); };
  $("#abs").onchange = (e) => { $("#absv").textContent = e.target.value; set({ abstain_threshold: +e.target.value }); };
  document.querySelectorAll("[data-tier]").forEach((r) => (r.onchange = () => {
    r.nextElementSibling.textContent = `$${r.value}M`;
    set({ tier_value: Object.fromEntries([...document.querySelectorAll("[data-tier]")].map((x) => [x.dataset.tier, +x.value])) });
  }));
}

init();
