"""Opt-in public result cards. The owner creates / reads / revokes a link with POST / GET / DELETE /api/typically/models/{id}/share;
anyone with it sees GET /s/{share_id} (a small server-rendered page with og:* tags) and /s/{share_id}/og.png (1200x630).
Only aggregate numbers and the questions are rendered: never cases, rows or files.
Invite links /i/{invite_id}#{code} get the same kind of page (the invite's ticket as og:image) that forwards to sign-in.
Store: <data>/shares.json {share_id: {model_id, owner, created_at}}. Included at the end of server.py (reads server.TYPICALLY at call time)."""
import base64
import hashlib
import html
import io
import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from PIL import Image, ImageDraw, ImageFont

import server as S
import typically_auth as auth
import typically_deploy as D
import typically_models as M

router = APIRouter()
_lock = threading.Lock()
SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,32}$")
FONTS = Path(__file__).resolve().parent / "media" / "fonts"
# the site palette (site/style.css): warm paper, off-black ink, Typical green; standard = the app's --standard
INK, PAPER, MID, GREEN, STANDARD = "#292827", "#f0eeeb", "#938f89", "#2f5d50", "#b0612f"
DARK, GREEN_ON_DARK, STANDARD_ON_DARK = "#191817", "#7fcfae", "#e09a64"


def _file() -> Path:
    return S.TYPICALLY / "shares.json"


def _read() -> dict:
    return json.loads(f.read_text()) if (f := _file()).exists() else {}


def _write(shares: dict) -> None:
    (tmp := _file().with_suffix(".tmp")).write_text(json.dumps(shares, indent=1))
    os.replace(tmp, _file())


def card(model_id: str) -> dict | None:
    """The public numbers of a model: name, overall standard / yours on n held-out cases and per-decision scores; None until scored."""
    if model_id != "northwind" and not (job := S.TYPICALLY / "jobs" / model_id).is_dir():
        return None
    run = D.run_of(model_id)
    if not (metrics := M._metrics(run)):
        return None
    name = M.NORTHWIND["name"] if model_id == "northwind" else (M._json(job / "job.json") or {}).get("name") or model_id
    rev = M._json(S.TYPICALLY / "results" / run / "reveal.json") or {}
    per, pooled = D.baselines_of(model_id, rev or None)
    return {"name": name, "standard": metrics["standard"], "yours": metrics["yours"], "n_cases": metrics["n_cases"], "baseline": pooled,
            "decisions": [{"question": d["question"], "standard": d["standard"], "yours": d["yours"],
                           "baseline": per.get(d.get("key"), {}).get("baseline")} for d in rev.get("decisions", [])]}


# ---------------------------------------------------------------- owner endpoints

def _origin(request: Request) -> str:
    return (os.environ.get('TYPICALLY_PUBLIC_URL') or str(request.base_url)).rstrip('/')


def _link(request: Request, sid: str) -> str:
    return f"{_origin(request)}/s/{sid}"


def _mine(shares: dict, model_id: str) -> list[str]:
    return [sid for sid, s in shares.items() if s["model_id"] == model_id and s.get("owner") == auth.stamp().get("owner")]


def _out(request: Request, sid: str, rec: dict) -> dict:
    url = _link(request, sid)
    return {"id": sid, "url": url, "image": f"{url}/og.png", "created_at": rec["created_at"]}


def _owned(model_id: str) -> None:
    D.run_of(model_id)   # validates the id
    auth.check(model_id)


@router.get("/api/typically/models/{model_id}/share")
def get_share(model_id: str, request: Request):
    _owned(model_id)
    shares = _read()
    return {"share": next((_out(request, sid, shares[sid]) for sid in _mine(shares, model_id)), None)}


@router.post("/api/typically/models/{model_id}/share")
def create_share(model_id: str, request: Request):
    _owned(model_id)
    if card(model_id) is None:
        raise HTTPException(400, "This model has no results to share yet.")
    with _lock:
        shares = _read()
        if mine := _mine(shares, model_id):   # one link per model and owner: sharing again returns it
            return {"share": _out(request, mine[0], shares[mine[0]])}
        sid = secrets.token_urlsafe(9)
        shares[sid] = {"model_id": model_id, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **auth.stamp()}
        _write(shares)
    return {"share": _out(request, sid, shares[sid])}


@router.delete("/api/typically/models/{model_id}/share")
def revoke_share(model_id: str):
    _owned(model_id)
    with _lock:
        shares = _read()
        mine = _mine(shares, model_id)
        for sid in mine:
            del shares[sid]
        _write(shares)
    return {"revoked": len(mine)}


def revoke_all(model_id: str) -> int:
    """Every link of a model, any owner (archive)."""
    with _lock:
        shares = _read()
        keep = {sid: s for sid, s in shares.items() if s["model_id"] != model_id}
        _write(keep)
    return len(shares) - len(keep)


# ---------------------------------------------------------------- public: /s/{share_id}

def _public(share_id: str) -> dict:
    rec = _read().get(share_id) if SID_RE.fullmatch(share_id) else None
    if rec is None or (c := card(rec["model_id"])) is None:
        raise HTTPException(404, "This link was turned off or never existed.")
    return c


def _pct(x: float) -> str:
    return f"{int(x * 100 + 0.5)}%"   # half up, like the app's Math.round (round() is half-even: .605 -> 60 here, 61 there)


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="robots" content="noindex">
<meta name="description" content="{desc}">
<meta property="og:type" content="website"><meta property="og:site_name" content="Decision Factory">
<meta property="og:title" content="{title}"><meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}"><meta property="og:image" content="{url}/og.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{title}">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/favicon.png">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist+Mono:wght@400;500&family=Schibsted+Grotesk:wght@400;500;600;700&family=Heebo:wght@400;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:{paper};color:{ink};font:16px/1.5 'Schibsted Grotesk','Heebo',system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
main{{max-width:640px;margin:0 auto;padding:56px 20px 48px}}
.eyebrow{{font-size:13px;letter-spacing:.04em;color:{mid};margin:0 0 20px}}
.eyebrow b{{color:{ink};font-weight:600}}
h1{{font-size:32px;line-height:1.15;letter-spacing:-.01em;margin:0 0 28px;font-weight:700}}
.score{{display:flex;align-items:flex-end;gap:20px;flex-wrap:wrap}}
.score div{{display:grid;gap:2px}}
.score small{{font-size:13px;color:{mid}}}
.num{{font-family:'Geist Mono',ui-monospace,monospace;font-size:56px;line-height:1;font-weight:500;font-variant-numeric:tabular-nums}}
.arrow{{font-size:28px;color:{mid};padding-bottom:10px}}
.std{{color:{standard_color}}} .yours{{color:{green}}}
.note{{color:{mid};font-size:14px;margin:14px 0 36px}}
.panel{{background:#fff;border:1px solid #e3e0db;border-radius:14px;padding:8px 20px}}
.row{{padding:14px 0;border-top:1px solid #ece9e4}} .row:first-child{{border-top:0}}
.q{{font-weight:500;margin:0 0 8px}}
.bar{{display:grid;grid-template-columns:132px 1fr 44px;align-items:center;gap:10px;font-size:13px;color:{mid}}}
.track{{height:6px;border-radius:99px;background:#ece9e4;overflow:hidden}}
.fill{{height:100%;border-radius:99px}}
.bar span:last-child{{font-family:'Geist Mono',ui-monospace,monospace;text-align:end;color:{ink}}}
.cta{{display:inline-block;margin-top:32px;background:{green};color:#fff;text-decoration:none;font-weight:600;padding:12px 20px;border-radius:10px}}
.cta:focus-visible{{outline:3px solid {ink};outline-offset:2px}}
footer{{margin-top:40px;font-size:13px;color:{mid}}} footer a{{color:inherit}}
</style></head>
<body><main>
<p class="eyebrow"><b>Decision Factory</b> &middot; a model trained on one team's past decisions</p>
<h1 dir="auto">{name}</h1>
<div class="score" aria-label="{aria}">
<div><small>Standard model</small><span class="num std">{standard}</span></div>
<span class="arrow" aria-hidden="true">&rarr;</span>
<div><small>Trained model</small><span class="num yours">{yours}</span></div>
</div>
<p class="note">Agreement: how often each model gave the same answer as the team, on {n} test cases neither model trained on.{always}</p>
{bars}
<a class="cta" href="/app/">Train a model on your own decisions</a>
<footer>Made with Decision Factory, built on <a href="/">Typical</a> open decision models.</footer>
</main></body></html>"""


def _bar(label: str, x: float, color: str) -> str:
    return (f'<div class="bar"><span>{label}</span><span class="track"><span class="fill" style="display:block;width:{x * 100:.1f}%;'
            f'background:{color}"></span></span><span>{_pct(x)}</span></div>')


@router.get("/s/{share_id}", response_class=HTMLResponse)
def share_page(share_id: str, request: Request):
    try:
        c = _public(share_id)
    except HTTPException:
        return HTMLResponse("<!doctype html><title>Not found</title><meta name=robots content=noindex>"
                            "<p style='font-family:system-ui;padding:40px'>This link was turned off or never existed.</p>", status_code=404)
    e = html.escape
    rows = "".join(f'<div class="row"><p class="q" dir="auto">{e(d["question"])}</p>'
                   + (_bar("Most common answer", d["baseline"], MID) if d.get("baseline") is not None else "")
                   + f'{_bar("Standard model", d["standard"], STANDARD)}{_bar("Trained model", d["yours"], GREEN)}</div>' for d in c["decisions"])
    title = f'{c["name"]}: {_pct(c["yours"])} agreement (standard model: {_pct(c["standard"])})'
    always = "" if c.get("baseline") is None else f' Always giving the most common answer would agree {_pct(c["baseline"])} of the time.'
    return PAGE.format(title=e(title), name=e(c["name"]), url=e(_link(request, share_id)), n=f"{c['n_cases']:,}", always=e(always),
                       desc=e(f'Trained with Decision Factory. Agreement measured on {c["n_cases"]:,} test cases.'),
                       aria=e(f'Agreement: standard model {_pct(c["standard"])}, trained model {_pct(c["yours"])}'),
                       standard=_pct(c["standard"]), yours=_pct(c["yours"]), bars=f'<div class="panel">{rows}</div>' if rows else "",
                       paper=PAPER, ink=INK, mid=MID, green=GREEN, standard_color=STANDARD)


# ---------------------------------------------------------------- og.png

def _font(size: int, weight: int, text: str = "", mono: bool = False) -> ImageFont.FreeTypeFont:
    hebrew = any("֐" <= ch <= "׿" for ch in text)   # Geist Mono has no Hebrew either: Heebo for both
    f = ImageFont.truetype(str(FONTS / ("Heebo.ttf" if hebrew else "GeistMono.ttf" if mono else "SchibstedGrotesk.ttf")), size)
    f.set_variation_by_axes([weight])
    return f


def _visual(text: str) -> str:
    # ponytail: Pillow here has no libraqm (no bidi); a Hebrew string is reversed whole, so mixed Hebrew + digits/latin reads wrong.
    # Upgrade: pip install python-bidi (or Pillow with raqm) if Hebrew names with numbers matter.
    return text[::-1] if any("֐" <= ch <= "׿" for ch in text) else text


def _fit(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> str:
    while text and draw.textlength(text, font=font) > width:
        text = text[:-2] + "…"
    return text


def og_png(c: dict) -> bytes:
    W, H, pad = 1200, 630, 72
    img = Image.new("RGB", (W, H), DARK)
    d = ImageDraw.Draw(img)
    d.text((pad, pad - 8), "Decision Factory", font=_font(30, 700), fill=PAPER)
    d.text((pad + d.textlength("Decision Factory ", font=_font(30, 700)), pad - 8), "· a model trained on one team's past decisions", font=_font(30, 400), fill=MID)
    name_font = _font(60, 700, c["name"])
    d.text((pad, 150), _visual(_fit(d, c["name"], name_font, W - 2 * pad)), font=name_font, fill=PAPER)

    big, label = _font(150, 600), _font(30, 500)
    d.text((pad, 268), "Standard model", font=label, fill=MID)
    d.text((pad, 300), _pct(c["standard"]), font=big, fill=STANDARD_ON_DARK)
    x2 = pad + max(d.textlength("100%", font=big), 300) + 40
    d.text((x2 - 10, 360), "→", font=_font(70, 400), fill=MID)
    x2 += 100
    d.text((x2, 268), "Trained model", font=label, fill=MID)
    d.text((x2, 300), _pct(c["yours"]), font=big, fill=GREEN_ON_DARK)
    always = "" if c.get("baseline") is None else f" · most common answer: {_pct(c['baseline'])}"
    d.text((pad, H - pad - 34), f"Agreement on {c['n_cases']:,} test cases{always}", font=_font(30, 400), fill=MID)
    d.rectangle((0, H - 10, W, H), fill=GREEN_ON_DARK)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


@router.get("/s/{share_id}/og.png")
def share_image(share_id: str):
    return Response(og_png(_public(share_id)), media_type="image/png", headers={"Cache-Control": "public, max-age=300"})


# ---------------------------------------------------------------- invite ticket (typically_auth: GET /api/auth/invites/{code}/ticket.png)
# the app's DOM ticket (typically-app/src/components/invite-dialog.tsx, .ticket in index.css), drawn at 1 cqw = 12 px: keep the numbers in step
TICKET_INK, TICKET_BACKDROP = (59, 36, 20), "#2c1f17"
TICKET_BASE = ((0, "#f7b48a"), (.45, "#f6914f"), (.76, "#f97316"), (1, "#ff6a13"))   # left to right: peach to the stub's orange
TICKET_GLOWS = (("#f9731d", .46, .44, 19, 27),   # top first, like CSS layers: (color, x, y of the box, solid to cqw, gone at cqw):
                ("#fff3e6", .53, .38, 15, 29))   # an orange sphere over the light behind it, so only a rim of light shows


def name_lines(name: str, widest: float = 62, largest: float = 6) -> tuple[list[str], float]:
    """The name cut into 1-3 lines at spaces, and its size in cqw: the cut that sets it largest, fewer lines on a tie (the app's nameLines)."""
    words = name.split()
    n = len(words)
    best = ([name], 0.0)
    for cut in [()] + [(i,) for i in range(1, n)] + [(i, j) for i in range(1, n) for j in range(i + 1, n)]:
        b = (0, *cut, n)
        lines = [" ".join(words[b[k]:b[k + 1]]) for k in range(len(b) - 1)]
        size = min(largest, widest / (0.62 * max(map(len, lines))))
        if size > best[1] + 1e-9:
            best = (lines, size)
    return best


def ticket_png(name: str | None, inviter: str, no: str, host: str, created: str) -> bytes:
    import numpy as np
    W, H, u = 1200, 686, 12   # 7:4; u = px per cqw
    pad, cut, notch, perf, stub_x = 7 * u, 3.4 * u, 2.9 * u, 76 * u, 88 * u
    rgb = lambda c: np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], float)
    x, y = np.arange(W)[None, :], np.arange(H)[:, None]
    img = np.stack([np.interp(x / W, [s for s, _ in TICKET_BASE], [rgb(c)[k] for _, c in TICKET_BASE]) for k in range(3)], -1) * np.ones((H, 1, 1))
    for color, gx, gy, solid, gone in reversed(TICKET_GLOWS):
        a = np.clip((gone * u - np.hypot(x - gx * W, y - gy * H)) / ((gone - solid) * u), 0, 1)[..., None]
        img = img * (1 - a) + rgb(color) * a
    grain = np.random.default_rng(int(no)).random((H // 2 + 1, W // 2 + 1)).repeat(2, 0).repeat(2, 1)[:H, :W, None]   # coarse: 2 px
    img += (grain - .5) * 56
    base = Image.fromarray(img.clip(0, 255).astype("uint8")).convert("RGBA")

    S = 4   # the cut-outs drawn at 4x and scaled down: smooth edges
    mask = Image.new("L", (W * S, H * S), 255)
    m = ImageDraw.Draw(mask)
    for cx, cy, r in ((0, 0, cut), (W, 0, cut), (0, H, cut), (W, H, cut), (perf, 0, notch), (perf, H, notch)):   # concave corners + notches
        m.ellipse(((cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S), fill=0)
    mask = mask.resize((W, H), Image.LANCZOS)

    ink = TICKET_INK + (255,)
    layer = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(layer)
    yy = notch + u
    while yy < H - notch - u:   # the perforation: faint dashes
        d.rectangle((perf - 1.2, yy, perf + 1.2, min(yy + .7 * u, H - notch - u)), fill=TICKET_INK + (70,))
        yy += 1.3 * u

    def width(text: str, f, ls: float) -> float:
        return sum(d.textlength(ch, font=f) + ls * f.size for ch in text)

    def spaced(x0: float, y0: float, text: str, f, ls: float, fill=ink, draw=d) -> float:   # letter-spaced, left-middle anchored
        for ch in _visual(text):
            draw.text((x0, y0), ch, font=f, fill=fill, anchor="lm")
            x0 += d.textlength(ch, font=f) + ls * f.size
        return x0

    small, line = round(2.1 * u), 3.25 * u   # eyebrow + bottom line: 2.1cqw, line-height 1.55, letter-spacing .1em
    spaced(pad, pad + line / 2, "DECISION FACTORY PRESENTS", _font(small, 400, mono=True), .1)
    after = spaced(pad, pad + line * 1.5, "AN INVITATION FROM ", _font(small, 400, mono=True), .1)
    who, fw = inviter.upper(), _font(small, 400, inviter, mono=True)
    while who and width(who, fw, .1) > perf - pad - after:
        who = who[:-2] + "…"
    spaced(after, pad + line * 1.5, who, fw, .1)
    when = datetime.fromisoformat(created).strftime("%d %b %Y").upper()
    spaced(pad, H - pad - line / 2, f"{host.upper()} · {when}", _font(small, 400, mono=True), .1)

    lines, size = name_lines(" ".join((name or "").split()) or "You're invited")   # the fallback: invites from before names
    px, mid = size * u, (pad + 2 * line + H - pad - line) / 2
    for k, ln in enumerate(lines):
        spaced(pad, mid + (k + .5 - len(lines) / 2) * px, ln.upper(), _font(round(px), 400, ln, mono=True), .02)

    big = _font(17 * u, 600, mono=True)   # the stub: the ticket number's digits stacked, giant and faded, under a vertical ADMIT ONE
    for k, digit in enumerate(no):
        d.text((stub_x, H / 2 + (k - 1.5) * .76 * 17 * u), digit, font=big, fill=(255, 243, 230, 64), anchor="mm")
    f = _font(round(6.6 * u), 400, mono=True)
    strip = Image.new("RGBA", (int(width("ADMIT ONE", f, .04)) + 4, int(f.size * 1.3)))
    spaced(0, strip.height / 2, "ADMIT ONE", f, .04, TICKET_INK + (220,), ImageDraw.Draw(strip))
    strip = strip.rotate(-90, expand=True)   # reads top to bottom, like writing-mode: vertical-rl
    layer.alpha_composite(strip, (round(stub_x - strip.width / 2), round(H / 2 - strip.height / 2)))

    out = Image.alpha_composite(base, layer)
    out.putalpha(mask)
    # posted like the YC tickets: the ticket at 960 wide on a dark brown 16:9 card
    card = Image.new("RGBA", (1200, 675), TICKET_BACKDROP)
    t = out.resize((960, 549), Image.LANCZOS)
    card.alpha_composite(t, ((1200 - 960) // 2, (675 - 549) // 2))
    buf = io.BytesIO()
    card.convert("RGB").save(buf, "PNG", optimize=True)
    return buf.getvalue()


# ---------------------------------------------------------------- public: /i/{invite_id} (invite link previews)
# The link is /i/{id}#{code}: the code rides in the fragment, so it never reaches this server, its logs or a crawler. The page is
# og tags for the preview (the ticket, which never prints the code or email) and a script that forwards to sign-in with the code.

INVITE_STYLE = "body{margin:0;padding:48px 20px;background:#f0eeeb;color:#292827;font:16px/1.5 system-ui,sans-serif}a{color:#2f5d50}"
INVITE_SCRIPT = ("var c=location.hash.slice(1),n=document.querySelector('meta[name=invite-for]').content;"
                 "location.replace(c?'/#/?invite='+encodeURIComponent(c)+(n?'&for='+encodeURIComponent(n):''):'/');")
_csp_hash = lambda src: "'sha256-" + base64.b64encode(hashlib.sha256(src.encode()).digest()).decode() + "'"
INVITE_CSP = (f"default-src 'none'; script-src {_csp_hash(INVITE_SCRIPT)}; style-src {_csp_hash(INVITE_STYLE)}; img-src 'self'; "
              "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
INVITE_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="robots" content="noindex"><meta name="referrer" content="no-referrer">
<meta name="description" content="{desc}"><meta name="invite-for" content="{name}">
<meta property="og:type" content="website"><meta property="og:site_name" content="Decision Factory">
<meta property="og:title" content="{title}"><meta property="og:description" content="{desc}">
<meta property="og:url" content="{url}"><meta property="og:image" content="{url}/ticket.png">
<meta property="og:image:width" content="1200"><meta property="og:image:height" content="675">
<meta property="og:image:alt" content="{alt}">
<meta name="twitter:card" content="summary_large_image">
<style>{style}</style>
</head><body>
<p>Opening your invite to Decision Factory&hellip; <a href="/">Continue</a></p>
<noscript><p>This invite needs JavaScript. Turn it on and open the link again.</p></noscript>
<script>{script}</script>
</body></html>"""
INVITE_DESC = "Decision Factory trains a model on your team's past decisions and shows how sure it is of every answer."


def _invite(invite_id: str) -> dict:
    if not SID_RE.fullmatch(invite_id) or (i := auth.by_id(invite_id)) is None:
        raise HTTPException(404, "This invite was revoked or never existed.")
    return i


@router.get("/i/{invite_id}", response_class=HTMLResponse)
def invite_page(invite_id: str, request: Request):
    try:
        i = _invite(invite_id)
    except HTTPException:
        return HTMLResponse("<!doctype html><title>Not found</title><meta name=robots content=noindex>"
                            "<p style='font-family:system-ui;padding:40px'>This invite was revoked or never existed.</p>", status_code=404)
    e, name, inviter = html.escape, i.get("name") or i.get("for") or "", i.get("from")
    title = (f"{inviter} invited {name or 'you'} to Decision Factory" if inviter
             else f"{name}, you're invited to Decision Factory" if name else "You're invited to Decision Factory")
    page = INVITE_PAGE.format(title=e(title), desc=e(INVITE_DESC), name=e(name), url=e(f"{_origin(request)}/i/{invite_id}"),
                              alt=e(f"An ADMIT ONE ticket for {name or 'you'}"), style=INVITE_STYLE, script=INVITE_SCRIPT)
    return HTMLResponse(page, headers={"Content-Security-Policy": INVITE_CSP, "Cache-Control": "no-store"})


@router.get("/i/{invite_id}/ticket.png")
def invite_image(invite_id: str, request: Request):
    png = auth.ticket(_invite(invite_id), request.url.netloc)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
