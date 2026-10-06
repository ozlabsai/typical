"""Hosting a small invited audience: invite-code sign-in, per-user ownership and training quotas (deploy/README.md).

Off unless TYPICALLY_AUTH=1, so local dev is unchanged (no user, everything visible, no quotas). When on:
- POST /api/auth/login {code, name?} sets a signed session cookie (HMAC-SHA256, TYPICALLY_SECRET); GET /api/auth/me; POST /api/auth/logout.
  Invite codes: TYPICALLY_INVITES (comma-separated) + <data>/invites.json (a JSON list); TYPICALLY_ADMINS (codes) are admins.
  Removing a code from the invites signs its sessions out (checked on every request).
- every /api/* request needs a session, except /api/auth/* and /api/health. /v1/models/{id}/decide (its own bearer keys), the
  public share cards (/s/*), the static site and the /app bundle (it renders the sign-in screen itself) stay open.
- ownership: job.json / upload meta carry "owner"; can_see() is the one rule (own + shared + admin sees all).
- quotas: admit() is called by server.start_job for every training run (/train and /retrain).
"""
import base64
import contextvars
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter()
COOKIE, MAX_AGE = "typically_session", 30 * 86400
SHARED = "*"   # owner of what every user sees: the base models and the Northwind sample
SHARED_MODELS = {"typical-small", "typical-medium", "northwind"}
OPEN_API = ("/api/auth/", "/api/health")


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


def invites() -> set[str]:
    f = _root() / "invites.json"
    return _codes("TYPICALLY_INVITES") | _codes("TYPICALLY_ADMINS") | (set(json.loads(f.read_text())) if f.exists() else set())


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
        return JSONResponse({"detail": "sign in first"}, status_code=401)
    token = _user.set(user)   # the endpoint's threadpool copies this context
    try:
        return await call_next(request)
    finally:
        _user.reset(token)


def _public(u: User) -> dict:
    return {"id": u.uid, "name": u.name, "admin": u.admin}


class Login(BaseModel):
    code: str = Field(min_length=1, max_length=200)
    name: str | None = Field(None, max_length=40)


@router.post("/api/auth/login")
def login(req: Login, request: Request, response: Response):
    if not enabled():
        raise HTTPException(400, "sign-in is off on this server")
    code = req.code.strip()
    # ponytail: no attempt limiter; invite codes must be long random strings (deploy/README.md), which makes guessing hopeless
    if not any(hmac.compare_digest(code.encode(), c.encode()) for c in invites()):
        raise HTTPException(401, "that invite code is not valid")
    uid = uid_of(code)
    name = " ".join((req.name or "").split()) or f"Guest {uid[:4]}"
    local = request.url.hostname in ("localhost", "127.0.0.1") and request.headers.get("x-forwarded-proto", request.url.scheme) != "https"
    response.set_cookie(COOKIE, make_cookie(uid, name), max_age=MAX_AGE, httponly=True, secure=not local, samesite="lax", path="/")
    return {"auth": True, "user": _public(User(uid, name, uid in {uid_of(c) for c in _codes("TYPICALLY_ADMINS")}))}


@router.get("/api/auth/me")
def me():
    if not enabled():
        return {"auth": False, "user": None}
    if (u := current()) is None:
        raise HTTPException(401, "sign in first")
    return {"auth": True, "user": _public(u)}


@router.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


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
        raise HTTPException(404, "no such model")


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
        raise HTTPException(409, "that model is being taught right now; wait for it to finish")
    if len(running) >= _int("TYPICALLY_MAX_JOBS", 2):
        raise HTTPException(429, "all training machines are busy right now; try again in a few minutes")
    if (u := current()) is None:
        return
    if u.uid in running.values():
        raise HTTPException(429, "you already have a model training; wait for it to finish before starting another")
    log, now, limit = _root() / "usage.jsonl", time.time(), _int("TYPICALLY_DAILY_RUNS", 3)
    runs = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if not u.admin and sum(r["uid"] == u.uid and now - r["at"] < 86400 for r in runs) >= limit:
        raise HTTPException(429, f"you have used today's {limit} training runs; you can train again tomorrow")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps({"uid": u.uid, "slug": slug, "at": now}) + "\n")
