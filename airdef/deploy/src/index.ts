// typical-ad.ozlabs.ai: a shared-password login in front of the demo; each signed-in visitor gets their own
// container (its own raid, clock and settings), which sleeps when they leave.
import { Container, getContainer } from "@cloudflare/containers";

export class AirdefDemo extends Container {
  defaultPort = 8790;
  sleepAfter = "20m";
}

interface Env {
  AIRDEF: DurableObjectNamespace<AirdefDemo>;
  DEMO_PASSWORD: string; // wrangler secret put DEMO_PASSWORD
}

const COOKIE = "airdef_session";
const enc = new TextEncoder();

async function hmac(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(value));
  return [...new Uint8Array(sig)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function sameSecret(a: string, b: string): Promise<boolean> {
  // compare digests, so the timing-safe comparison sees equal lengths and leaks nothing about the password
  const [x, y] = await Promise.all([a, b].map((s) => crypto.subtle.digest("SHA-256", enc.encode(s))));
  return crypto.subtle.timingSafeEqual(x, y);
}

async function session(req: Request, env: Env): Promise<string | null> {
  const raw = (req.headers.get("Cookie") || "").split(/;\s*/).find((c) => c.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1);
  if (!raw) return null;
  const [sid, sig] = raw.split(".");
  return sid && sig && (await sameSecret(sig, await hmac(env.DEMO_PASSWORD, sid))) ? sid : null;
}

function loginPage(error = "", status = 200): Response {
  return new Response(
    `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Typical · Air defence</title>
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500&display=swap" rel="stylesheet">
<style>
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #fafafa; font-family: Figtree, Arial, sans-serif; color: #101010; }
  form { width: 20rem; display: grid; gap: .75rem; }
  h1 { font-weight: 400; font-size: 1.5rem; letter-spacing: -.03em; margin: 0 0 .5rem; } h1 span { color: #8a8a8e; }
  input { font: inherit; padding: .75rem 1rem; border: 1px solid #e2e2e4; border-radius: 999px; background: #fff; }
  button { font: inherit; font-weight: 500; padding: .75rem 1rem; border: 0; border-radius: 999px; background: #2548f6; color: #fff; cursor: pointer; }
  p { margin: 0; color: #ff3b30; font-size: .9375rem; min-height: 1.25rem; }
</style></head><body>
<form method="post" action="/login"><h1>Typical <span>Air defence</span></h1>
<input type="password" name="password" placeholder="Password" autofocus required autocomplete="current-password">
<button type="submit">Enter</button><p>${error}</p></form></body></html>`,
    { status, headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" } },
  );
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (!env.DEMO_PASSWORD) return new Response("Not configured.", { status: 503 });   // never open without a password
    const url = new URL(req.url);
    if (url.pathname === "/login" && req.method === "POST") {
      const password = String((await req.formData()).get("password") || "");
      if (!(await sameSecret(password, env.DEMO_PASSWORD))) return loginPage("Wrong password.", 401);
      const sid = crypto.randomUUID();
      return new Response(null, {
        status: 303,
        headers: { Location: "/", "Set-Cookie": `${COOKIE}=${sid}.${await hmac(env.DEMO_PASSWORD, sid)}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=43200` },
      });
    }
    const sid = await session(req, env);
    if (!sid) return url.pathname.startsWith("/api/") ? new Response("unauthorized", { status: 401 }) : loginPage();
    return getContainer(env.AIRDEF, sid).fetch(req);
  },
};
