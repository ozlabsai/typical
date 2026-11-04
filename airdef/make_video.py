"""Record the demo video: watch a decision get made, then why it changes, what it saves, and that the policy is yours.

    cd airdef/static && python3 -m http.server 8796 &                 # the baked site (acts 1-4)
    uv run --no-sync uvicorn --app-dir airdef server:app --port 8797 &   # the live model (act 5: policy)
    uv run --no-project --with playwright python airdef/make_video.py [--out .context/airdef/video]

Acts 1-4 play the recorded tapes (deterministic: every take is the same); act 5 changes Settings on the live server,
so each new call is Typical's real answer. A scripted camera zooms to the element being explained.
"""
import argparse
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

W, H = 1920, 1080

KIT = """
(() => {
  const st = document.createElement('style');
  st.textContent = `
  body { transform-origin: 0 0; transition: transform 1s cubic-bezier(.2,.7,.2,1); }
  .vring { outline: 3px solid #2548f6 !important; outline-offset: 6px; border-radius: 10px; }
  #vcap { position: fixed; left: 32px; bottom: 40px; max-width: 760px; z-index: 50; padding: 16px 22px; border-radius: 14px;
    background: rgba(16,16,16,.88); color: #fff; font: 500 30px/1.3 Figtree, Heebo, Arial, sans-serif; letter-spacing: -.01em;
    opacity: 0; transform: translateY(8px); transition: opacity .3s, transform .3s; pointer-events: none; }
  #vcap.on { opacity: 1; transform: none; }
  #vcap b { color: #9db0ff; font-weight: 500; }
  #vcard { position: fixed; inset: 0; z-index: 60; background: #0d0d0f; color: #fff; display: grid; place-items: center; opacity: 0;
    pointer-events: none; transition: opacity .45s; font: 400 54px/1.2 Figtree, Heebo, Arial, sans-serif; letter-spacing: -.03em; text-align: center; padding: 0 12%; }
  #vcard.on { opacity: 1; } #vcard small { display: block; margin-top: 18px; font-size: 26px; color: #9a9aa0; letter-spacing: -.01em; }
  #vreads { position: fixed; left: 32px; top: 96px; width: 820px; z-index: 50; padding: 22px 26px; border-radius: 16px; background: rgba(255,255,255,.97);
    box-shadow: 0 12px 40px rgba(0,0,0,.18), 0 0 0 1px #e2e2e4; font: 400 22px/1.45 Figtree, Arial, sans-serif; color: #2a2a2a;
    opacity: 0; transition: opacity .35s; pointer-events: none; }
  #vreads.on { opacity: 1; }
  #vreads h4 { margin: 0 0 10px; font: 500 16px/1 Figtree, Arial; letter-spacing: .06em; text-transform: uppercase; color: #2548f6; }
  #vreads p { margin: 6px 0; opacity: 0; transform: translateX(-8px); transition: opacity .35s, transform .35s; }
  #vreads p.on { opacity: 1; transform: none; }`;
  document.head.appendChild(st);
  for (const id of ['vcap', 'vcard', 'vreads']) { const d = document.createElement('div'); d.id = id; document.documentElement.appendChild(d); }
  let cam = { tx: 0, ty: 0, s: 1 };
  const set = (c) => { cam = c; document.body.style.transform = c.s === 1 ? 'none' : `translate(${c.tx}px, ${c.ty}px) scale(${c.s})`; };
  window.VID = {
    zoom(sel, max = 2.4, fill = 0.8) {   // centre the element and enlarge it to `fill` of the screen (at most `max`)
      const el = document.querySelector(sel); if (!el) return false;
      const r = el.getBoundingClientRect(), x = (r.left - cam.tx) / cam.s, y = (r.top - cam.ty) / cam.s, w = r.width / cam.s, h = r.height / cam.s;
      const s = Math.min(max, (innerWidth * fill) / w, (innerHeight * fill) / h);
      let tx = innerWidth / 2 - s * (x + w / 2), ty = innerHeight / 2 - s * (y + h / 2);
      tx = Math.min(0, Math.max(innerWidth - s * innerWidth, tx)); ty = Math.min(0, Math.max(innerHeight - s * innerHeight, ty));
      set({ tx, ty, s }); return true;
    },
    reset() { set({ tx: 0, ty: 0, s: 1 }); },
    ring(sel) { document.querySelectorAll('.vring').forEach((e) => e.classList.remove('vring')); const el = document.querySelector(sel); if (el) el.classList.add('vring'); return !!el; },
    unring() { document.querySelectorAll('.vring').forEach((e) => e.classList.remove('vring')); },
    cap(html) { const c = document.getElementById('vcap'); c.classList.remove('on'); setTimeout(() => { c.innerHTML = html; c.classList.add('on'); }, 180); },
    uncap() { document.getElementById('vcap').classList.remove('on'); },
    card(html) { const c = document.getElementById('vcard'); c.innerHTML = html; c.classList.add('on'); },
    uncard() { document.getElementById('vcard').classList.remove('on'); },
    reads(title, lines) { const c = document.getElementById('vreads'); c.innerHTML = `<h4>${title}</h4>` + lines.map((l) => `<p>${l}</p>`).join(''); c.classList.add('on');
      [...c.querySelectorAll('p')].forEach((p, i) => setTimeout(() => p.classList.add('on'), 450 + i * 750)); },
    unreads() { document.getElementById('vreads').classList.remove('on'); },
  };
})();
"""


class Film:
    def __init__(self, page):
        self.pg = page

    def js(self, code, arg=None):
        return self.pg.evaluate(code, arg)

    def card(self, html, secs, during=None):   # `during`: done under the card (e.g. switching scenario), off camera
        self.js("h => VID.card(h)", html); t = time.time()
        if during:
            during()
        time.sleep(max(0.0, secs - (time.time() - t))); self.js("() => VID.uncard()"); time.sleep(0.5)

    def cap(self, html, secs=0.0):
        self.js("h => VID.cap(h)", html); time.sleep(secs)

    def zoom(self, sel, max=2.4, fill=0.8):
        self.js("a => VID.zoom(a[0], a[1], a[2])", [sel, max, fill])

    def ring(self, sel):
        self.js("s => VID.ring(s)", sel)

    def wide(self, secs=0.0):
        self.js("() => { VID.reset(); VID.unring(); }"); time.sleep(secs)

    def until(self, cond, timeout=60.0):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(f"() => {{ try {{ return !!({cond}); }} catch (e) {{ return false; }} }}"):
                return True
            time.sleep(0.15)
        print(f"  (gave up waiting for: {cond})", flush=True)
        return False

    def control(self, body):
        self.js("b => { pinned = false; detail = null; return api('/api/control', b); }", body)

    def scenario(self, name):
        self.pg.click("#scenBtn"); time.sleep(0.3); self.pg.click(f'#scenMenu [data-s="{name}"]'); time.sleep(2.5)


def record(url, out, act):
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"])
        ctx = b.new_context(viewport={"width": W, "height": H}, record_video_dir=str(out / f"raw_{act.__name__}"),
                            record_video_size={"width": W, "height": H})
        pg = ctx.new_page()
        t0 = time.time()
        pg.goto(url, wait_until="load")
        f = Film(pg)
        f.js(KIT)
        f.js("() => VID.card('')")   # cover the page while the map loads
        time.sleep(4)
        start = time.time() - t0
        act(f)
        path = pg.video.path()
        ctx.close(); b.close()
    clip = out / f"{act.__name__}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", path, "-c:v", "libx264", "-crf", "18", "-r", "30",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    return clip


# ---------------------------------------------------------------- acts 1-4: the recorded raids (static site)
def raids(f):
    js = f.js
    # 1. the hook (the scenario loads under the cover card)
    f.scenario("october_saturation")
    f.card("120 ballistic missiles.<br>12 minutes.<small>Every one needs a call: which interceptor, when, or none at all.</small>", 3.6)
    f.control({"action": "speed", "speed": 20}); f.control({"action": "play"})
    f.cap("A simulated raid on Israel. Typical makes every call.", 2.2)

    # 2. anatomy of one decision: the Ghadr heading for Dimona
    f.until("S.beat && S.beat.id === 0", 30); f.control({"action": "pause"}); time.sleep(0.9)
    f.until("detail && detail.info && detail.info.state", 10)
    lines = js("() => detail.info.state.split('\\n').slice(0, 5)")
    f.cap("One missile. Watch Typical decide.", 1.6)
    f.zoom("#list .row", max=2.0, fill=0.55); f.ring("#list .row"); time.sleep(2.0)
    f.wide(0.5)
    js("l => VID.reads('What Typical reads', l)", lines)
    f.cap("The threat, the interceptors that can reach it, their odds and cost, tonight's stock.", 5.6)
    js("() => VID.unreads()")
    f.zoom("#list .opts.live", max=2.3, fill=0.6); f.ring("#list .opts.live")
    f.cap("It weighs every option. Arrow 3 <b>50%</b>, wait <b>26%</b>, SM-3 <b>24%</b>.", 3.4)
    f.zoom("#list .story .what", max=2.2, fill=0.55); f.ring("#list .story .what")
    f.cap("The call: <b>Arrow 3</b>, now.", 2.0)
    f.zoom("#list .local", max=2.4, fill=0.7); f.ring("#list .local")
    f.cap("And the battery's own call: fire <b>one</b>, keep time to look before a second.", 3.2)
    f.wide(0.3)
    t_done = js("() => S.moments[0].t_done")
    f.control({"action": "play"}); f.control({"action": "seek", "t": t_done - 6})
    f.until("detail && detail.status === 'killed'", 12); time.sleep(0.4)
    f.zoom("#list .prog", max=2.0, fill=0.6); f.ring("#list .prog .step.ok")
    f.cap("<b>Intercepted.</b> One round, the cheapest one sure enough.", 3.0)
    f.wide(0.3); js("() => VID.uncap()")

    # 3. the call changes as the picture changes: the drone to Eilat
    def to_drone():
        f.scenario("houthi_eilat")
        f.control({"action": "speed", "speed": 20}); f.control({"action": "moment", "index": 1}); f.control({"action": "pause"})
    f.card("As the picture changes,<br>so does the call.", 3.4, to_drone)
    time.sleep(1.2); f.until("detail && detail.log && detail.log.length > 2", 10)
    f.cap("A drone heads for Eilat. No battery can reach it.", 2.4)
    f.zoom("#list .prog", max=2.0, fill=0.62); f.ring("#list .prog")
    f.cap("Typical asks again as it flies: <b>wait</b>, the track is not firm…", 3.2)
    f.ring("#list .prog .step.call:last-of-type")
    f.cap("…until the call turns: <b>send fighter jets.</b>", 2.8)
    f.wide(0.3)
    t_done = js("() => S.moments[1].t_done")
    f.control({"action": "play"}); f.control({"action": "seek", "t": t_done - 30})
    f.until("detail && detail.status === 'killed'", 15); time.sleep(0.6)
    f.cap("Intercepted by the jets, far from Eilat.", 2.2)
    js("() => VID.uncap()")

    # 4. money: wait for the cheaper interceptor, let rockets fall where they do no harm
    def to_north():
        f.scenario("northern_barrage")
        f.control({"action": "speed", "speed": 20}); f.control({"action": "moment", "index": 1}); f.control({"action": "pause"})
    f.card("A barrage from Lebanon.<br>174 rockets and drones.", 3.4, to_north)
    time.sleep(1.2); f.until("detail && detail.info", 10)
    f.zoom("#list .story .why", max=2.2, fill=0.6); f.ring("#list .story .why")
    f.cap("David's Sling could fire now. Iron Dome reaches it in seconds, at a <b>twentieth of the price</b>.", 3.8)
    f.zoom("#list .opts.live", max=2.3, fill=0.6); f.ring("#list .opts.live")
    f.cap("Typical waits: <b>100%</b> sure.", 2.0)
    f.wide(0.3)
    f.control({"action": "moment", "index": 2}); f.control({"action": "pause"}); time.sleep(1.4)
    f.cap("A whole salvo heading for open ground.", 2.0)
    f.zoom("#list .story", max=2.0, fill=0.5); f.ring("#list .story")
    f.cap("<b>117 rockets</b> left to fall. Not one interceptor spent on them.", 3.0)
    f.wide(0.3)
    f.pg.click('#mode [data-mode="general"]'); time.sleep(0.6)
    f.control({"action": "seek", "t": js("() => S.end") - 2}); f.control({"action": "play"}); time.sleep(1.5)
    f.control({"action": "pause"})
    f.zoom(".tonight", max=2.0, fill=0.62); f.ring(".tonight .held")
    f.cap("The night's bill: every interceptor counted, <b>$37M kept</b> by waiting for the cheaper one.", 3.8)
    f.wide(0.3); js("() => VID.uncap()")
    f.pg.click('#mode [data-mode="focus"]')


# ---------------------------------------------------------------- act 5: the policy is yours (live server, live model)
def policy(f):
    f.js("() => VID.uncard()")
    f.card("You set the policy.<small>Typical follows it.</small>", 4.0, lambda: (f.scenario("drill:last_chance:0"), time.sleep(1.5)))
    f.until("detail && detail.info && detail.info.dist", 20); time.sleep(0.6)
    f.cap("A rocket to Tiberias, seconds from impact. One David's Sling battery reaches it.", 3.0)
    pct = lambda k: round(100 * f.js(f"() => detail.info.dist['{k}']"))   # the live model's numbers, read off the screen
    f.zoom("#list .opts.live", max=2.3, fill=0.6); f.ring("#list .opts.live")
    f.cap(f"A close call: David's Sling <b>{pct('now:dsling')}%</b>, let it fall <b>{pct('hold')}%</b>.", 3.0)
    f.wide(0.3)

    def setting(click, cap):
        f.pg.click("#gear"); time.sleep(0.6)
        f.zoom("#sheet", max=1.4, fill=0.7); f.cap(cap, 0.4)
        f.pg.click(click); time.sleep(1.4)
        f.wide(0.2); f.pg.click("#gear"); time.sleep(0.4)

    setting('#reserve [data-r="2"]', "Magazines are critical: interceptors must last more nights.")
    f.until("detail && detail.choice === 'hold'", 15); time.sleep(0.8)
    f.zoom("#list .opts.live", max=2.3, fill=0.6); f.ring("#list .opts.live")
    f.cap(f"Typical lets it fall: <b>{pct('hold')}%</b>.", 2.6)
    f.wide(0.3)
    f.pg.click("#gear"); time.sleep(0.4); f.pg.click('#reserve [data-r="1"]'); time.sleep(0.6); f.pg.click("#gear")
    setting('#prio [data-site="tiberias"]', "Now make Tiberias a priority site.")
    f.until("detail && detail.choice === 'now:dsling' && detail.info.dist['now:dsling'] > 0.9", 15); time.sleep(0.8)
    f.zoom("#list .opts.live", max=2.3, fill=0.6); f.ring("#list .opts.live")
    f.cap(f"Typical fires David's Sling: <b>{pct('now:dsling')}%</b>.", 2.6)
    f.wide(0.3); f.js("() => VID.uncap()")
    f.pg.click("#gear"); time.sleep(0.3); f.pg.click('#prio [data-site="tiberias"]'); time.sleep(0.3); f.pg.click("#gear")   # leave the server as found
    f.card("Typical<small>Every call explained. Every call yours to set.</small>", 3.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", default="http://127.0.0.1:8796")
    ap.add_argument("--live", default="http://127.0.0.1:8797")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / ".context/airdef/video"))
    ap.add_argument("--only", choices=["raids", "policy"])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    acts = {"raids": (raids, a.static), "policy": (policy, a.live)}
    clips = [record(url, out, act) for name, (act, url) in acts.items() if not a.only or a.only == name]
    if a.only:
        print(clips[0]); return
    lst = out / "clips.txt"
    lst.write_text("".join(f"file '{c}'\n" for c in clips))
    final = out / "typical-airdefence.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    "-movflags", "+faststart", str(final)], check=True)
    print(f"{final}  ({final.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
