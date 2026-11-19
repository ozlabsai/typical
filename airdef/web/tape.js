// The demo without a server: plays a baked tape (bake.py) through the same API app.js speaks to the live server.
// State at any time is rebuilt from the tape's frames (each holds only what changed); the clock, the quiet-time skips
// and Focus's story pacing (director.py) run here.
(() => {
  const DT = 4, QUIET_S = 15, STORY_S = 20, SEEK = 60, BEAT_S = 7, OUTCOME_S = 3.5, FOLLOW_MAX = 60;
  const tapes = {};
  let meta = null;
  const V = { name: null, tape: null, t: 0, maxT: 0, running: false, speed: 10, rate: 10, mode: "focus", last: performance.now(),
    beat: null, next: 0, skipped: 0, cur: null, fi: -1 };

  async function getJSON(url) {   // .json.gz: decompress here unless the CDN already did (gzip magic bytes decide)
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    if (buf[0] !== 0x1f || buf[1] !== 0x8b) return JSON.parse(new TextDecoder().decode(buf));
    return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).json();
  }
  const tapeName = (s) => {   // drill:<family>:<case>, wrapping past the last case like the live server
    if (!s.startsWith("drill:")) return s;
    const [, fam, n] = s.split(":"), k = V.tape?.drill?.family === fam ? V.tape.drill.cases : Infinity;
    return `drill_${fam}_${+n % k}`;
  };

  async function load(name) {
    if (!tapes[name]) tapes[name] = await getJSON(`/data/${name}.json.gz`);
    return tapes[name];
  }
  async function reset(scenario) {
    const name = tapeName(scenario);
    await load(name);
    Object.assign(V, { name, tape: tapes[name], running: false, beat: null, next: 0, skipped: 0, cur: null, fi: -1 });
    V.t = V.maxT = V.tape.frames[0].t;
    seekTo(V.t);
  }

  // ---------------------------------------------------------------- state at time t
  function applyTo(i) {   // frames are deltas: apply them in order; going back rebuilds from the first
    if (i < V.fi) { V.cur = null; V.fi = -1; }
    if (!V.cur) V.cur = { tracks: new Map(), bat: new Map(), ledger: null, stats: null, units: [] };
    for (let k = V.fi + 1; k <= i; k++) {
      const f = V.tape.frames[k], c = V.cur;
      for (const [id, x] of Object.entries(f.upd || {})) c.tracks.set(id, x);
      for (const id of f.gone || []) c.tracks.delete(id);
      for (const b of f.bat || []) c.bat.set(b.id, b);
      if (f.ledger) c.ledger = f.ledger;
      if (f.stats) c.stats = f.stats;
      c.units = f.units;
    }
    V.fi = i;
  }
  const frameAt = (t) => {   // last frame at or before t
    const F = V.tape.frames; let lo = 0, hi = F.length - 1;
    if (t < F[0].t) return 0;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (F[m].t <= t) lo = m; else hi = m - 1; }
    return lo;
  };
  function seekTo(t) { V.t = Math.min(t, V.tape.end); V.maxT = Math.max(V.maxT, V.t); applyTo(frameAt(V.t)); }

  function units(t) {   // aircraft between two frames: interpolated, so they fly instead of hopping every step
    const F = V.tape.frames, a = F[V.fi], b = F[V.fi + 1];
    if (!b) return a.units;
    const f = Math.min(1, Math.max(0, (t - a.t) / Math.max(b.t - a.t, 1e-6)));
    return a.units.map((u) => { const v = b.units.find((x) => x.id === u.id) || u; return { ...u, lat: u.lat + (v.lat - u.lat) * f, lon: u.lon + (v.lon - u.lon) * f }; });
  }

  // ---------------------------------------------------------------- the clock and Focus's pacing
  function advance() {
    const now = performance.now(), real = Math.min((now - V.last) / 1000, 1);
    V.last = now;
    if (!V.tape) return;
    if (!V.running) { if (V.beat) V.beat.until += real * 1000; return; }   // paused: the story keeps its time
    if (V.t >= V.tape.end) return;
    const focus = V.mode === "focus" && !V.tape.drill, ms = V.tape.moments;
    let rate = V.speed;
    if (focus) {
      const b = V.beat;
      if (b) {
        if (b.follow && !b.done && V.t >= b.m.t_done) { b.done = true; b.until = Math.min(b.until, now + OUTCOME_S * 1000); }
        rate = b.follow && !b.done ? b.rate : 1;
        if (now > b.until) { V.beat = null; V.next = b.id + 1; }
      } else rate = SEEK;   // between stories: fast-forward to the next one
    }
    V.rate = rate;
    let t2 = V.t + real * rate;
    const F = V.tape.frames, a = F[V.fi], nf = F[V.fi + 1];   // the live server jumps quiet stretches: so does the tape
    if (nf && nf.t - a.t > QUIET_S + DT && t2 > a.t) { V.skipped = nf.t - a.t; t2 = Math.max(t2, nf.t - DT); }
    if (focus && !V.beat && V.next < ms.length && t2 >= ms[V.next].t) {   // a story starts
      const m = ms[V.next]; t2 = m.t;
      startBeat(m.id, now);
    }
    seekTo(t2);
  }
  function marks(tid) {   // when something happens to a threat: an event, a new call, a battery pick, a shot, its result
    const out = []; let key = null;
    for (const e of V.tape.threats[tid].log) { if (!e.key) out.push(e.t); else if (e.key !== key) { out.push(e.t); key = e.key; } }
    for (const f of V.tape.flights) if (f.th === tid) out.push(f.t0, f.t1);
    return out;
  }
  function jumpNext() {   // Next: the next thing that happens in the story on screen; after its outcome, the next story
    const b = V.beat, ms = V.tape.moments, t = V.t + 0.5;
    if (b && b.follow && !b.done) {
      const ids = b.m.type === "group" ? Object.keys(V.tape.threats) : [b.m.tid];
      const nx = Math.min(...ids.flatMap(marks).filter((x) => x > t && x <= b.m.t_done + DT));
      if (Number.isFinite(nx)) {   // land where the change reaches the screen: the story's next recorded state from then on
        const shown = Math.min(...ids.flatMap((id) => V.tape.threats[id].timeline.map((x) => x.t)).filter((x) => x >= nx - 0.01));
        seekTo(Math.min(Number.isFinite(shown) ? shown + 0.01 : nx + DT + 0.5, b.m.t_done + 0.01)); return;
      }
    }
    V.beat = null;
    const i = ms.findIndex((m) => m.t > V.t + 1e-6 && (!b || m.id > b.id));
    if (i < 0) { V.next = ms.length; seekTo(V.tape.end); return; }   // nothing left: the end
    seekTo(ms[i].t + 0.01); V.next = i + 1; startBeat(i);
  }
  function startBeat(id, now = performance.now()) {
    const m = V.tape.moments[id], follow = ["air", "cheaper", "payoff", "engage", "detect", "battery", "group"].includes(m.type);
    // the tape knows when the story ends: every story takes about STORY_S from its first call to its outcome
    const st = V.tape.story || {}, rate = Math.min(Math.max((m.t_done - m.t) / (st.story_s || STORY_S), 1), st.rate_max || 300);
    V.beat = { id, m, follow, done: false, rate, until: now + (follow ? st.follow_max || FOLLOW_MAX : BEAT_S) * 1000 };
  }

  // ---------------------------------------------------------------- the API app.js calls
  function state() {
    advance();
    const T = V.tape, c = V.cur, t = V.t;
    return {
      t, end: T.end, running: V.running, speed: V.speed, rate: V.running ? V.rate : V.speed, backlog: 0, skipped: V.skipped,
      scenario: T.scenario, model: meta?.model, cfg: T.cfg, mode: V.mode, drill: T.drill,
      tracks: [...c.tracks.values()].map((x) => ({ ...x, tti: x.impact - t })),
      flights: T.flights.filter((f) => f.t0 <= t), units: units(t), batteries: [...c.bat.values()],
      killed: Object.fromEntries(Object.entries(T.killed).filter(([, k]) => k <= t)),
      ledger: c.ledger || {}, stats: c.stats, attacker_cost: T.attacker_cost,
      launched: Object.values(T.threats).filter((x) => x.t0 <= t).length, total: T.total,
      moments: T.drill ? undefined : T.moments.filter((m) => m.t <= V.maxT),
      beat: V.beat && { id: V.beat.id, tid: V.beat.m.tid, follow: V.beat.follow, done: V.beat.done, left: Math.max(0, (V.beat.until - performance.now()) / 1000) },
      seek: V.mode === "focus" && V.running && !V.beat,
    };
  }
  function track(tid) {
    const th = V.tape.threats[tid], t = V.t;
    if (!th) return null;
    const d = [...th.timeline].reverse().find((x) => x.t <= t) || { status: "pending", choice: null, info: null, local: null, assign: null };
    return { id: tid, kind: th.kind, cls: th.cls, t, t0: th.t0, impact: th.impact, status: d.status, choice: d.choice, info: d.info,
      local: d.local, assign: d.assign, site: d.site, open_ground: d.open_ground, priority: d.priority, inside: d.inside, escalated: false,
      history: th.history.filter((h) => h[0] <= t), log: th.log.filter((e) => e.t <= t),
      shots: V.tape.flights.filter((f) => f.th === tid && f.t0 <= t).map((f) => ({ b: f.b, eff: f.eff, t0: f.t0, t1: f.t1, hit: f.hit })) };
  }
  async function control(c) {
    if (c.action === "play") { V.running = true; V.last = performance.now(); }
    else if (c.action === "pause") V.running = false;
    else if (c.action === "seek") seekTo(+c.t);   // jump within the tape (the video recorder uses it)
    else if (c.action === "speed") V.speed = Math.max(0.25, Math.min(+c.speed, 600));
    else if (c.action === "mode") { V.mode = c.mode; if (c.mode === "general") V.beat = null; }
    else if (c.action === "next") jumpNext();
    else if (c.action === "moment") { const m = V.tape.moments[c.index]; if (m) { seekTo(m.t + 0.01); V.next = m.id + 1; startBeat(m.id); V.running = true; } }   // +0.01: tape times are rounded
    else if (c.action === "reset") await reset(c.scenario || V.tape.drill && `drill:${V.tape.drill.family}:${V.tape.drill.case}` || V.tape.scenario);
    return { ok: true };
  }

  // Settings on the hosted demo: every recorded policy is its own tape. A change switches to that tape at the same moment.
  const wOf = (c) => {   // the weight that differs from the default, as recorded: city2, strategic0.5, prio5 ...
    const d = meta.default_weights;
    for (const t of Object.keys(d.tier)) { const r = (c.tier_value?.[t] ?? d.tier[t]) / d.tier[t]; if (Math.abs(r - 1) > 0.01) return `w${t}${+r.toFixed(2)}`; }
    return c.priority_multiplier != null && c.priority_multiplier !== d.prio ? `wprio${c.priority_multiplier}` : "";
  };
  const keyOf = (c) => [`r${c.reserve}`, `f${c.ammo_floor?.ironDome ?? 0}`, `b${c.floor_price}`, `p${c.border_price}`,
    c.priority_sites?.length === 1 ? `s${c.priority_sites[0]}` : "", wOf(c)].filter(Boolean).join("-");
  async function setPolicy(patch) {
    const base = V.name.split("@")[0], vars = meta.variants?.[base] || [];
    const parts = (k) => { const x = k.split("-"); return [...x.slice(0, 4), x.slice(4).find((y) => y[0] === "s") || "", x.slice(4).find((y) => y[0] === "w") || ""]; };
    const want = parts(keyOf({ ...V.tape.cfg, ...patch })), now = parts(keyOf(V.tape.cfg));
    const changed = want.map((x, i) => x !== now[i]);   // not recorded together: the closest recorded policy with the change asked for
    const score = (k) => { const p = parts(k); return p.every((x, i) => !changed[i] || x === want[i]) ? p.filter((x, i) => x === want[i]).length : -1; };
    const key = vars.reduce((b, k) => (score(k) > score(b) ? k : b), vars[0] || "");
    if (!key || score(key) < 0) return { ok: false, cfg: V.tape.cfg };
    const t = V.t, running = V.running;
    const name = key === meta.default_variant ? base : `${base}@${key}`;
    await load(name);
    Object.assign(V, { name, tape: tapes[name], beat: null, cur: null, fi: -1, maxT: t });
    seekTo(t);
    V.next = V.tape.moments.findIndex((m) => m.t > t); if (V.next < 0) V.next = V.tape.moments.length;
    V.running = running;
    return { ok: true, cfg: V.tape.cfg };
  }
  window.tapeVariants = () => (meta?.variants?.[V.name?.split("@")[0]] || []);   // the hosted Settings shows what was recorded

  window.tapeApi = async (path, body) => {
    if (path === "/api/meta") { meta = meta || await getJSON("/data/meta.json"); return meta; }
    if (!V.tape) await reset("level1");   // like the live server: start at the first step
    if (path === "/api/paths") return V.tape.paths;
    if (path.startsWith("/api/state")) return state();
    if (path.startsWith("/api/track/")) return track(decodeURIComponent(path.split("/")[3].split("?")[0]));
    if (path === "/api/control") return control(body);
    if (path === "/api/config") return setPolicy(body);
    throw new Error(`no tape route for ${path}`);
  };
})();
