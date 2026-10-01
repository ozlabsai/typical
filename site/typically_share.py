"""Opt-in public result cards. The owner creates / reads / revokes a link with POST / GET / DELETE /api/typically/models/{id}/share;
anyone with it sees GET /s/{share_id} (a small server-rendered page with og:* tags) and /s/{share_id}/og.png (1200x630).
Only aggregate numbers and the questions are rendered: never cases, rows or files.
Store: <data>/shares.json {share_id: {model_id, owner, created_at}}. Included at the end of server.py (reads server.TYPICALLY at call time)."""
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
    return {"name": name, "standard": metrics["standard"], "yours": metrics["yours"], "n_cases": metrics["n_cases"],
            "decisions": [{"question": d["question"], "standard": d["standard"], "yours": d["yours"]} for d in rev.get("decisions", [])]}


# ---------------------------------------------------------------- owner endpoints

def _link(request: Request, sid: str) -> str:
    return f"{(os.environ.get('TYPICALLY_PUBLIC_URL') or str(request.base_url)).rstrip('/')}/s/{sid}"


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
        raise HTTPException(400, "this model has no results to share yet")
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


# ---------------------------------------------------------------- public: /s/{share_id}

def _public(share_id: str) -> dict:
    rec = _read().get(share_id) if SID_RE.fullmatch(share_id) else None
    if rec is None or (c := card(rec["model_id"])) is None:
        raise HTTPException(404, "this link was turned off or never existed")
    return c


def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="robots" content="noindex">
<meta name="description" content="{desc}">
<meta property="og:type" content="website"><meta property="og:site_name" content="typically">
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
.bar{{display:grid;grid-template-columns:72px 1fr 44px;align-items:center;gap:10px;font-size:13px;color:{mid}}}
.track{{height:6px;border-radius:99px;background:#ece9e4;overflow:hidden}}
.fill{{height:100%;border-radius:99px}}
.bar span:last-child{{font-family:'Geist Mono',ui-monospace,monospace;text-align:end;color:{ink}}}
.cta{{display:inline-block;margin-top:32px;background:{green};color:#fff;text-decoration:none;font-weight:600;padding:12px 20px;border-radius:10px}}
.cta:focus-visible{{outline:3px solid {ink};outline-offset:2px}}
footer{{margin-top:40px;font-size:13px;color:{mid}}} footer a{{color:inherit}}
</style></head>
<body><main>
<p class="eyebrow"><b>typically</b> &middot; a decision model fine-tuned on its own past decisions</p>
<h1 dir="auto">{name}</h1>
<div class="score" aria-label="{aria}">
<div><small>Standard</small><span class="num std">{standard}</span></div>
<span class="arrow" aria-hidden="true">&rarr;</span>
<div><small>Fine-tuned</small><span class="num yours">{yours}</span></div>
</div>
<p class="note">Agreement with the team's own decisions on {n} held-out cases neither model saw while learning.</p>
{bars}
<a class="cta" href="/app/">Train your own on your decisions</a>
<footer>Made with typically, built on <a href="/">Typical</a> open decision models.</footer>
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
    rows = "".join(f'<div class="row"><p class="q" dir="auto">{e(d["question"])}</p>{_bar("Standard", d["standard"], STANDARD)}'
                   f'{_bar("Fine-tuned", d["yours"], GREEN)}</div>' for d in c["decisions"])
    title = f'{c["name"]}: {_pct(c["yours"])} vs {_pct(c["standard"])} for the standard model'
    return PAGE.format(title=e(title), name=e(c["name"]), url=e(_link(request, share_id)), n=c["n_cases"],
                       desc=e(f'Fine-tuned with typically and measured on {c["n_cases"]} held-out cases.'),
                       aria=e(f'Standard {_pct(c["standard"])}, fine-tuned {_pct(c["yours"])}'),
                       standard=_pct(c["standard"]), yours=_pct(c["yours"]), bars=f'<div class="panel">{rows}</div>' if rows else "",
                       paper=PAPER, ink=INK, mid=MID, green=GREEN, standard_color=STANDARD)


# ---------------------------------------------------------------- og.png

def _font(size: int, weight: int, text: str = "") -> ImageFont.FreeTypeFont:
    hebrew = any("֐" <= ch <= "׿" for ch in text)
    f = ImageFont.truetype(str(FONTS / ("Heebo.ttf" if hebrew else "SchibstedGrotesk.ttf")), size)
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
    d.text((pad, pad - 8), "typically", font=_font(30, 700), fill=PAPER)
    d.text((pad + d.textlength("typically ", font=_font(30, 700)), pad - 8), "· fine-tuned decision model", font=_font(30, 400), fill=MID)
    name_font = _font(60, 700, c["name"])
    d.text((pad, 150), _visual(_fit(d, c["name"], name_font, W - 2 * pad)), font=name_font, fill=PAPER)

    big, label = _font(150, 600), _font(30, 500)
    d.text((pad, 268), "Standard", font=label, fill=MID)
    d.text((pad, 300), _pct(c["standard"]), font=big, fill=STANDARD_ON_DARK)
    x2 = pad + max(d.textlength("100%", font=big), 300) + 40
    d.text((x2 - 10, 360), "→", font=_font(70, 400), fill=MID)
    x2 += 100
    d.text((x2, 268), "Fine-tuned", font=label, fill=MID)
    d.text((x2, 300), _pct(c["yours"]), font=big, fill=GREEN_ON_DARK)
    d.text((pad, H - pad - 34), f"Agreement with the team's own decisions on {c['n_cases']} held-out cases", font=_font(30, 400), fill=MID)
    d.rectangle((0, H - 10, W, H), fill=GREEN_ON_DARK)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


@router.get("/s/{share_id}/og.png")
def share_image(share_id: str):
    return Response(og_png(_public(share_id)), media_type="image/png", headers={"Cache-Control": "public, max-age=300"})
