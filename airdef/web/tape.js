// The demo without a server: plays a baked tape (bake.py) through the same API app.js speaks to the live server.
// State at any time is rebuilt from the tape's frames (each holds only what changed); the clock, the quiet-time skips
// and Focus's story pacing (director.py) run here.
(() => {
  const DT = 4, QUIET_S = 15, STORY_S = 25, SEEK = 60, BEAT_S = 7, OUTCOME_S = 3.5, FOLLOW_MAX = 60;
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

  async function reset(scenario) {
    const name = tapeName(scenario);
    if (!tapes[name]) tapes[name] = await getJSON(`/data/${name}.json.gz`);
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
        const th = V.tape.threats[b.m.tid];
        if (b.follow && !b.done && V.t >= b.m.t_done) { b.done = true; b.until = Math.min(b.until, now + OUTCOME_S * 1000); }
        rate = b.follow && !b.done ? Math.min(Math.max((th.impact - V.t) / STORY_S, 1), 300) : 1;
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
  function startBeat(id, now = performance.now()) {
    const m = V.tape.moments[id], follow = ["air", "cheaper", "payoff", "engage"].includes(m.type);
    V.beat = { id, m, follow, done: false, until: now + (follow ? FOLLOW_MAX : BEAT_S) * 1000 };
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
      local: d.local, assign: d.assign, site: d.site, open_ground: d.open_ground, priority: d.priority, escalated: false,
      history: th.history.filter((h) => h[0] <= t), log: th.log.filter((e) => e.t <= t),
      shots: V.tape.flights.filter((f) => f.th === tid && f.t0 <= t).map((f) => ({ b: f.b, eff: f.eff, t0: f.t0, t1: f.t1, hit: f.hit })) };
  }
  async function control(c) {
    if (c.action === "play") { V.running = true; V.last = performance.now(); }
    else if (c.action === "pause") V.running = false;
    else if (c.action === "speed") V.speed = Math.max(0.25, Math.min(+c.speed, 600));
    else if (c.action === "mode") { V.mode = c.mode; if (c.mode === "general") V.beat = null; }
    else if (c.action === "next") { V.beat = null; V.running = true; V.next = V.tape.moments.findIndex((m) => m.t > V.t + 1e-6); if (V.next < 0) V.next = V.tape.moments.length; }
    else if (c.action === "moment") { const m = V.tape.moments[c.index]; if (m) { seekTo(m.t); V.next = m.id + 1; startBeat(m.id); V.running = true; } }
    else if (c.action === "reset") await reset(c.scenario || V.tape.drill && `drill:${V.tape.drill.family}:${V.tape.drill.case}` || V.tape.scenario);
    return { ok: true };
  }

  window.tapeApi = async (path, body) => {
    if (path === "/api/meta") { meta = meta || await getJSON("/data/meta.json"); return meta; }
    if (!V.tape) await reset("houthi_eilat");   // the live server also starts on the Red Sea
    if (path === "/api/paths") return V.tape.paths;
    if (path.startsWith("/api/state")) return state();
    if (path.startsWith("/api/track/")) return track(decodeURIComponent(path.split("/")[3].split("?")[0]));
    if (path === "/api/control") return control(body);
    if (path === "/api/config") return { ok: true, cfg: V.tape.cfg };
    throw new Error(`no tape route for ${path}`);
  };
})();
