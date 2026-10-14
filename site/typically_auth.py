"""Hosting a small invited audience: invite-code sign-in, per-user ownership and training quotas (deploy/README.md).

Off unless TYPICALLY_AUTH=1, so local dev is unchanged (no user, everything visible, no quotas). When on:
- POST /api/auth/login {code, name?} sets a signed session cookie (HMAC-SHA256, TYPICALLY_SECRET); GET /api/auth/me; POST /api/auth/logout.
  Invite codes: TYPICALLY_INVITES (comma-separated) + <data>/invites.json (a JSON list, the operator's) + <data>/issued_invites.json
  (made by users: GET [?all=1, admins] / POST {name, email} / DELETE /api/auth/invites, GET /api/auth/invites/{code}/ticket.png;
  INVITE_LIMIT active each, admins unlimited; joined / last_seen are stamped on sign-in); TYPICALLY_ADMINS (codes) are admins.
  Each issued invite also has a public, non-secret id: its link is /i/{id}#{code} (typically_share: the og preview page + ticket).
  Removing a code from the invites signs its sessions out (checked on every request).
- every /api/* request needs a session, except /api/auth/* and /api/health. /v1/models/{id}/decide (its own bearer keys), the
  public share cards (/s/*), the invite previews (/i/*), the static site and the /app bundle (it renders the sign-in screen itself) stay open.
- ownership: job.json / upload meta carry "owner"; can_see() is the one rule (own + shared + admin sees all).
- quotas: admit() is called by server.start_job for every training run (/teach and /retrain); throttle() rate-limits the endpoints that
  call the operator's LLM key on every request (/analyze, /parse_question).
"""
import base64
import contextvars
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

router = APIRouter()
_issued_lock = threading.RLock()   # issued_invites.json (re-entrant: _issued() backfills ids). ponytail: one process; a file lock if several
COOKIE, MAX_AGE = "typically_session", 30 * 86400
SHARED = "*"   # owner of what every user sees: the base models and the Northwind sample
SHARED_MODELS = {"typical-small", "typical-medium", "northwind"}
OPEN_API = ("/api/auth/", "/api/health")
INVITE_LIMIT = 5   # active invites a non-admin may have made


@dataclass(frozen=True)
class User:
    uid: str
    name: str
    admin: bool


_user: contextvars.ContextVar[User | None] = contextvars.ContextVar("typically_user", default=None)


def enabled() -> bool:
    return os.environ.get("TYPICALLY_AUTH") == "1"


def current() -> User | None:
    return _user.get()


def _root():
    import server   # read at call time (the tests patch server.TYPICALLY)
    return server.TYPICALLY


# ---------------------------------------------------------------- invites + session cookie

def _secret() -> bytes:
    if not (s := os.environ.get("TYPICALLY_SECRET")):
        raise RuntimeError("TYPICALLY_AUTH=1 needs TYPICALLY_SECRET (python -c 'import secrets; print(secrets.token_urlsafe(32))')")
    return s.encode()


def _codes(var: str) -> set[str]:
    return {c.strip() for c in os.environ.get(var, "").split(",") if c.strip()}


def _issued() -> list[dict]:
    """[{id, code, by: uid, from: inviter's name, name, email, created, joined?, last_seen?}] of the invites users made (the operator's
    own stay in invites.json). Entries from before name + email were required may carry "for" (the name) or neither; entries from
    before the public id get one here, on first read."""
    with _issued_lock:
        issued = json.loads(f.read_text()) if (f := _root() / "issued_invites.json").exists() else []
        if missing := [i for i in issued if "id" not in i]:
            for i in missing:
                i["id"] = _new_id(issued)
            _write_issued(issued)
        return issued


def _new_id(issued: list[dict]) -> str:
    """The invite's public id (/i/{id}): random, unrelated to the code, safe in logs and link previews."""
    while (iid := secrets.token_urlsafe(8)) in {i.get("id") for i in issued}:
        pass
    return iid


def invites() -> set[str]:
    f = _root() / "invites.json"
    return (_codes("TYPICALLY_INVITES") | _codes("TYPICALLY_ADMINS") | (set(json.loads(f.read_text())) if f.exists() else set())
            | {i["code"] for i in _issued()})


def uid_of(code: str) -> str:
    """Stable across secret rotation (a new secret signs everyone out but keeps their models)."""
    return hashlib.sha256(f"typically:{code}".encode()).hexdigest()[:16]


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _sign(payload: str) -> str:
    return _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())


def make_cookie(uid: str, name: str, now: float | None = None) -> str:
    payload = _b64(json.dumps({"u": uid, "n": name, "e": int((now or time.time()) + MAX_AGE)}).encode())
    return f"{payload}.{_sign(payload)}"


def read_cookie(value: str | None) -> User | None:
    """The signed user, or None: bad signature, expired, malformed, or its invite was revoked."""
    payload, _, sig = (value or "").partition(".")
    if not payload or not hmac.compare_digest(sig, _sign(payload)):
        return None
    try:
        d = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        uid, name, exp = d["u"], d["n"], d["e"]
    except (ValueError, KeyError, TypeError):
        return None
    if exp < time.time() or uid not in {uid_of(c) for c in invites()}:
        return None
    return User(uid, name, uid in {uid_of(c) for c in _codes("TYPICALLY_ADMINS")})


async def middleware(request: Request, call_next):
    if not enabled():
        return await call_next(request)
    user = read_cookie(request.cookies.get(COOKIE))
    path = request.url.path
    if user is None and path.startswith("/api/") and not path.startswith(OPEN_API):
        return JSONResponse({"detail": "Sign in first."}, status_code=401)
    token = _user.set(user)   # the endpoint's threadpool copies this context
    try:
        return await call_next(request)
    finally:
        _user.reset(token)


def _public(u: User) -> dict:
    return {"id": u.uid, "name": u.name, "admin": u.admin}


class Login(BaseModel):
    code: str = Field(min_length=1, max_length=200)
    name: str | None = Field(None, max_length=60)


@router.post("/api/auth/login")
def login(req: Login, request: Request, response: Response):
    if not enabled():
        raise HTTPException(400, "Sign-in is off on this server.")
    code = req.code.strip()
    # ponytail: no attempt limiter; invite codes must be long random strings (deploy/README.md), which makes guessing hopeless
    if not any(hmac.compare_digest(code.encode(), c.encode()) for c in invites()):
        raise HTTPException(401, "That invite code is not valid.")
    uid = uid_of(code)
    issued = _seen(uid)   # an invite a user made: its first sign-in stamps "joined"; no name typed -> the one the inviter gave
    name = " ".join((req.name or "").split()) or (issued and _view(issued)["name"]) or f"Guest {uid[:4]}"
    local = request.url.hostname in ("localhost", "127.0.0.1") and request.headers.get("x-forwarded-proto", request.url.scheme) != "https"
    response.set_cookie(COOKIE, make_cookie(uid, name), max_age=MAX_AGE, httponly=True, secure=not local, samesite="lax", path="/")
    return {"auth": True, "user": _public(User(uid, name, uid in {uid_of(c) for c in _codes("TYPICALLY_ADMINS")}))}


@router.get("/api/auth/me")
def me():
    if not enabled():
        return {"auth": False, "user": None}
    if (u := current()) is None:
        raise HTTPException(401, "Sign in first.")
    _seen(u.uid)   # the app calls /me once per load: that is "last seen"
    return {"auth": True, "user": _public(u)}


@router.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


# ---------------------------------------------------------------- invites users make

def _write_issued(issued: list[dict]) -> None:   # caller holds _issued_lock
    f = _root() / "issued_invites.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)   # they are credentials: 0600 from the first byte
    with os.fdopen(fd, "w") as out:
        out.write(json.dumps(issued, indent=1))
    os.replace(tmp, f)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seen(uid: str) -> dict | None:
    """Stamp the issued invite behind this user: joined (first sign-in, kept) and last_seen. None for the operator's codes."""
    with _issued_lock:
        issued = _issued()
        if (i := next((i for i in issued if uid_of(i["code"]) == uid), None)) is None:
            return None
        now = _now()
        i["joined"] = i.get("joined") or now
        i["last_seen"] = now
        _write_issued(issued)
    return i


def _inviter() -> User:
    if not enabled():
        raise HTTPException(404, "Not found")
    if (u := current()) is None:
        raise HTTPException(401, "Sign in first.")
    return u


def ticket_no(code: str) -> str:
    """The ticket's printed number: stable, derived from the code, and says nothing about it."""
    return f"{int(hashlib.sha256(f'ticket:{code}'.encode()).hexdigest()[:8], 16) % 10000:04d}"


def _view(i: dict) -> dict:
    return {"id": i["id"], "code": i["code"], "no": ticket_no(i["code"]), "name": i.get("name") or i.get("for"), "email": i.get("email"),
            "inviter": i.get("from"), "created": i["created"], "joined": i.get("joined"), "last_seen": i.get("last_seen")}


def _mine(u: User, everyone: bool = False) -> dict:
    issued = _issued()
    mine = [i for i in issued if i["by"] == u.uid]
    return {"invites": [_view(i) for i in (issued if everyone else mine)],
            "limit": None if u.admin else INVITE_LIMIT, "left": None if u.admin else max(0, INVITE_LIMIT - len(mine))}


@router.get("/api/auth/invites")
def my_invites(everyone: bool = Query(False, alias="all")):
    """Yours; ?all=1 (admins only): everyone's, each with its inviter's name."""
    u = _inviter()
    if everyone and not u.admin:
        raise HTTPException(403, "Only admins can see every invite.")
    return _mine(u, everyone)


EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")   # ponytail: a shape check; the invite link is what reaches them, not mail


class NewInvite(BaseModel):
    name: str = Field(max_length=60)    # printed on the ticket, prefills their name
    email: str = Field(max_length=254)  # kept for the inviter and admins; never printed on the ticket nor put in the link

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not (v := " ".join(v.split())):
            raise ValueError("Enter their name.")
        return v

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        if not EMAIL.fullmatch(v := v.strip()):
            raise ValueError("Enter a valid email address.")
        return v


@router.post("/api/auth/invites")
def create_invite(req: NewInvite):
    u = _inviter()
    with _issued_lock:
        issued = _issued()
        if not u.admin and sum(i["by"] == u.uid for i in issued) >= INVITE_LIMIT:
            raise HTTPException(429, f"You can have {INVITE_LIMIT} invites at a time. Revoke one to make a new one.")
        code = secrets.token_urlsafe(24)
        _write_issued(issued + [{"id": _new_id(issued), "code": code, "by": u.uid, "from": u.name, "name": req.name, "email": req.email, "created": _now()}])
    return {"code": code, **_mine(u)}


@router.delete("/api/auth/invites/{code}")
def revoke_invite(code: str):
    """Owner or admin. The code leaves invites(), so its holder is signed out and their API keys stop on the next request;
    every invite that holder made (and so on down the chain) is revoked with it."""
    u = _inviter()
    with _issued_lock:
        issued = _issued()
        keep = [i for i in issued if not (hmac.compare_digest(i["code"].encode(), code.encode()) and (u.admin or i["by"] == u.uid))]
        if len(keep) == len(issued):
            raise HTTPException(404, "We can't find that invite.")
        gone = {uid_of(code)}   # the revoked holder's own invites go too, and theirs in turn: a removed person leaves no way back in
        while cut := [i for i in keep if i["by"] in gone]:
            gone |= {uid_of(i["code"]) for i in cut}
            keep = [i for i in keep if i["by"] not in gone]
        _write_issued(keep)
    return _mine(u)


@router.get("/api/auth/invites/{code}/ticket.png")
def invite_ticket(code: str, request: Request):
    """The invite as a printable ticket (owner or admin; 404 otherwise). Never prints the code or the email: tickets get posted."""
    u = _inviter()
    i = next((i for i in _issued() if hmac.compare_digest(i["code"].encode(), code.encode()) and (u.admin or i["by"] == u.uid)), None)
    if i is None:
        raise HTTPException(404, "We can't find that invite.")
    return Response(ticket(i, request.url.netloc, u.name), media_type="image/png", headers={"Cache-Control": "private, no-store"})


def ticket(i: dict, host: str, inviter: str = "Decision Factory") -> bytes:
    import typically_share   # Pillow + fonts live there (the og card)
    return typically_share.ticket_png(_view(i)["name"], i.get("from") or inviter, ticket_no(i["code"]), host, i["created"])


def by_id(iid: str) -> dict | None:
    """The issued invite behind a public id (the /i/{id} preview), or None: sign-in off, unknown id, or revoked."""
    return next((i for i in _issued() if i["id"] == iid), None) if enabled() else None


# ---------------------------------------------------------------- ownership: the one rule

def stamp() -> dict:
    """Merge into job.json / upload meta: {"owner": uid} for a signed-in user, {} with auth off."""
    return {"owner": u.uid} if (u := current()) else {}


def can_see(owner: str | None) -> bool:
    if not enabled():
        return True
    u = current()
    return u is not None and (u.admin or owner in (SHARED, u.uid))


def owner_of(model_id: str) -> str | None:
    if model_id in SHARED_MODELS:
        return SHARED
    f = _root() / "jobs" / model_id / "job.json"
    try:
        return json.loads(f.read_text()).get("owner") if f.exists() else None
    except ValueError:
        return None


def visible(model_id: str) -> bool:
    return can_see(owner_of(model_id))


def check(model_id: str) -> None:
    """404 (not 403: someone else's model does not exist for you) unless the caller may use this model."""
    if not visible(model_id):
        raise HTTPException(404, "We can't find that model. It may have been archived.")


def model_of(name: str) -> str:
    """Model id from any of the names the API takes: "typical-small", "local:co_<slug>", "co_<slug>", "<slug>"."""
    import typically_deploy as D
    name = name.removeprefix("local:")
    return name if name in SHARED_MODELS else D.model_id_of(name) if name.startswith("co_") else name


# ---------------------------------------------------------------- quotas

def _int(var: str, default: int) -> int:
    return int(os.environ.get(var) or default)


def admit(slug: str, running: dict[str, str | None]) -> None:
    """Raise unless one more training run may start; record it. `running`: slug -> owner of every job in this process
    (server._jobs, held under its lock). ponytail: in-process counts, like the job runner itself; one server process only."""
    if slug in running:
        raise HTTPException(409, "This model is training right now. Wait for it to finish.")
    if len(running) >= _int("TYPICALLY_MAX_JOBS", 2):
        raise HTTPException(429, "Every training GPU on this server is busy. Try again in a few minutes.")
    if (u := current()) is None:
        return
    if u.uid in running.values():
        raise HTTPException(429, "You already have a model training. Wait for it to finish before starting another.")
    log, now, limit = _root() / "usage.jsonl", time.time(), _int("TYPICALLY_DAILY_RUNS", 3)
    runs = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if not u.admin and sum(r["uid"] == u.uid and now - r["at"] < 86400 for r in runs) >= limit:
        raise HTTPException(429, f"You've used today's {limit} training runs. You can train again tomorrow.")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"uid": u.uid, "slug": slug, "at": now}) + "\n")


# per user: (calls, window seconds) of the endpoints that call the operator's LLM key on every request
LIMITS = {"analyze": (30, 3600), "parse_question": (120, 3600)}
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def throttle(what: str) -> None:
    """429 once the signed-in user made LIMITS[what] calls within its window; no-op with auth off (local dev).
    ponytail: in-process sliding window, reset on restart, one server process only; a shared store (Redis) if it ever runs several."""
    if (u := current()) is None:
        return
    n, window = LIMITS[what]
    now = time.monotonic()
    with _hits_lock:
        hits = _hits[(u.uid, what)]
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= n:
            raise HTTPException(429, f"Too many requests. You can do this {n} times an hour. Try again in a few minutes.")
        hits.append(now)
