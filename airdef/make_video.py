"""Record the demo video from the static build, following VIDEO_SCRIPT.md point by point.

    cd airdef/static && python3 -m http.server 8796 &          # serve the baked site
    uv run --no-project --with playwright python airdef/make_video.py [--lang en|he] [--out .context/airdef/video]

The demo is deterministic, so every take is the same: each point appears as a caption when its moment is on screen.
"""
import argparse
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

TEXT = {
    "en": dict(
        c1="A missile raid is thousands of decisions.", c2="Which interceptor. When. Or none at all.",
        c3="Typical makes every call, in milliseconds.",
        r1="A ballistic missile heads for Tel Aviv.", r2="Typical waits seconds for the right interceptor.",
        r3="Arrow 3 fires. Intercepted.",
        d1="A drone heads for Eilat. No battery reaches it.", d2="Typical keeps asking as it flies.",
        d3="The call turns: send fighter jets.",
        h1="A battery holds fire: our own jets are near.",
        o1="Red Sea raid: 11 threats, 11 intercepted, 0 hits.",
        c4="Now a barrage: 174 rockets and drones from Lebanon.",
        n1="David's Sling could fire now.", n2="Typical waits seconds for Iron Dome, at a twentieth of the price.",
        n3="117 rockets will land in open ground.", n4="Not one interceptor spent on them.",
        g1="Every threat gets a call, as it happens.", g2="Tonight: $37M kept by waiting for the cheaper interceptor.",
        p1="Every option, played out to the end 40 times.", p2="Typical picks the best.",
        end="Typical · a decision model, fine-tuned to your doctrine."),
    "he": dict(
        c1="מתקפת טילים היא אלפי החלטות.", c2="איזה מיירט. מתי. או בכלל לא.", c3="Typical מקבל כל החלטה, באלפיות שנייה.",
        r1="טיל בליסטי בדרך לתל אביב.", r2="Typical ממתין שניות למיירט הנכון.", r3="חץ 3 יורה. יורט.",
        d1="כטב״ם בדרך לאילת. אף סוללה לא מגיעה אליו.", d2="Typical ממשיך לשאול לאורך המעוף.", d3="ההחלטה משתנה: לשלוח מטוסי קרב.",
        h1="סוללה עוצרת אש: המטוסים שלנו קרובים.",
        o1="מתקפה מים סוף: 11 איומים, 11 יירוטים, 0 פגיעות.",
        c4="ועכשיו מטח: 174 רקטות וכטב״מים מלבנון.",
        n1="קלע דוד יכול לירות עכשיו.", n2="Typical ממתין שניות לכיפת ברזל, בעשרים מהמחיר.",
        n3="117 רקטות ייפלו בשטח פתוח.", n4="אף מיירט לא נורה עליהן.",
        g1="כל איום מקבל החלטה, בזמן אמת.", g2="הלילה: 37 מיליון דולר נחסכו בהמתנה למיירט זול יותר.",
        p1="כל אפשרות הורצה עד הסוף, 40 פעמים.", p2="Typical בוחר את הטובה ביותר.",
        end="Typical · מודל החלטות, מותאם לתורת הלחימה שלכם."),
}

OVERLAY = """
const s = document.createElement('style');
s.textContent = `#vcap{position:fixed;top:92px;inset-inline-start:24px;max-width:640px;z-index:50;padding:14px 20px;border-radius:14px;
  background:rgba(16,16,16,.86);color:#fff;font:500 26px/1.3 Figtree,Heebo,Arial,sans-serif;letter-spacing:-.01em;opacity:0;transition:opacity .35s;pointer-events:none}
  #vcap.on{opacity:1}
  #vcard{position:fixed;inset:0;z-index:60;background:#0d0d0f;color:#fff;display:grid;place-items:center;opacity:0;pointer-events:none;
  transition:opacity .5s;font:400 46px/1.25 Figtree,Heebo,Arial,sans-serif;letter-spacing:-.03em;text-align:center;padding:0 12%}
  #vcard.on{opacity:1}`;
document.head.appendChild(s);
for (const id of ['vcap', 'vcard']) { const d = document.createElement('div'); d.id = id; document.body.appendChild(d); }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8796")
    ap.add_argument("--lang", default="en", choices=["en", "he"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / ".context/airdef/video"))
    a = ap.parse_args()
    T, out = TEXT[a.lang], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"])
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(out), record_video_size={"width": 1920, "height": 1080})
        pg = ctx.new_page()
        t0 = time.time()
        js = pg.evaluate

        def card(text, secs):
            js("t => { const c = document.getElementById('vcard'); c.textContent = t; c.classList.add('on'); }", text)
            time.sleep(secs)
            js("() => document.getElementById('vcard').classList.remove('on')")
            time.sleep(0.6)

        def cap(text, secs=0.0):
            js("t => { const c = document.getElementById('vcap'); c.classList.remove('on'); setTimeout(() => { c.textContent = t; c.classList.add('on'); }, 200); }", text)
            time.sleep(secs)

        def nocap():
            js("() => document.getElementById('vcap').classList.remove('on')")

        def until(cond, timeout=90.0):
            end = time.time() + timeout
            while time.time() < end:
                if js(f"() => {{ try {{ return !!({cond}); }} catch (e) {{ return false; }} }}"):
                    return True
                time.sleep(0.2)
            print(f"  (gave up waiting for: {cond})", flush=True)
            return False

        def control(body):
            js("b => { pinned = false; detail = null; return api('/api/control', b); }", body)

        def scenario(name):
            pg.click("#scenBtn"); time.sleep(0.4); pg.click(f'#scenMenu [data-s="{name}"]'); time.sleep(3)

        pg.goto(a.url, wait_until="load")
        js(OVERLAY)
        if a.lang == "he":
            pg.click("#lang")
        js("() => document.getElementById('vcard').classList.add('on')")   # cover the page while the map loads
        time.sleep(4)
        start = time.time() - t0
        card(T["c1"], 3.5); card(T["c2"], 3.5); card(T["c3"], 3.5)

        def beat(*texts, secs=3.5):   # freeze the frame on the moment while its captions are read, then play on
            control({"action": "pause"})
            for t in texts:
                cap(t, secs)
            control({"action": "play"})

        # Red Sea: three stories, then the result
        control({"action": "speed", "speed": 20}); control({"action": "play"})
        until("S.beat && S.beat.id === 0"); time.sleep(0.6)
        beat(T["r1"], T["r2"])                                    # the log shows the wait, then Arrow 3
        until("S.beat && S.beat.id === 0 && S.beat.done", 40); time.sleep(0.5)
        beat(T["r3"], secs=3); nocap()
        until("S.beat && S.beat.id === 1"); time.sleep(0.6)
        beat(T["d1"], T["d2"], T["d3"])                           # the log: re-asks as it flew, then the call turned to jets
        until("S.beat && S.beat.id === 1 && S.beat.done", 40); time.sleep(1.5); nocap()   # the intercept, uncaptioned
        until("S.beat && S.beat.id === 2"); time.sleep(0.8)
        beat(T["h1"], secs=4.5); nocap()
        until("raidOver()", 60); time.sleep(1); cap(T["o1"], 5); nocap()
        card(T["c4"], 4)

        # Northern front: the wait for Iron Dome, then the salvo let fall
        scenario("northern_barrage")
        control({"action": "speed", "speed": 20}); control({"action": "moment", "index": 1}); time.sleep(0.8)
        beat(T["n1"], T["n2"], secs=4); nocap()                   # the call: Wait for cheaper, David's Sling could fire
        until("detail && detail.choice === 'now:ironDome'", 30); time.sleep(2.5)
        control({"action": "moment", "index": 2})                 # skip the long final approach: on to the salvo
        until("S.beat && S.beat.id === 2", 30); time.sleep(1)
        beat(T["n3"], T["n4"]); nocap()
        time.sleep(2)

        # the whole raid, to its end: General
        pg.click('#mode [data-mode="general"]'); time.sleep(1)
        control({"action": "speed", "speed": 60}); control({"action": "play"})
        cap(T["g1"]); until("raidOver()", 90); time.sleep(1); cap(T["g2"], 6); nocap()

        # the proof
        pg.click('#mode [data-mode="focus"]'); time.sleep(0.5)
        scenario("drill:wait_for_cheaper:0"); time.sleep(2)
        cap(T["p1"], 4); cap(T["p2"], 4); nocap()
        card(T["end"], 4)
        video = pg.video.path()
        ctx.close(); b.close()
    mp4 = out / f"typical-airdefence-{a.lang}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", video, "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    print(f"{mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
