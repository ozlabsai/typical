// Typed client for site/server.py (/api/typically/*). Plain fetch; non-2xx throws Error(server `detail`).

export type DecisionType = "choice" | "noul" | "score"

export interface Result {
  probs: Record<string, number>
  p_null: number
  argmax: string
  expected?: number // score only
}

export const PHASES = ["queued", "starting_gpu", "uploading", "training", "evaluating", "downloading", "done", "failed"] as const
export type Phase = (typeof PHASES)[number]

/** Training curves aligned on `step` (null where that step logged no value); step_time = recent seconds per step.
 *  best_on: error on the company's own validation examples (best.pt is picked on it); acc: agreement with them (0-1). */
export interface Series { step: number[]; loss: (number | null)[]; val: (number | null)[]; best_on: (number | null)[]; acc?: (number | null)[]; step_time: number | null }

/** A milestone of a job (scripts/typically_job.event): `key` is localised by the UI; detail is data (a GPU name, a size); n / of count progress. */
export interface JobEvent { t: string; key: string; detail?: string; n?: number; of?: number }

/** The pod's last nvidia-smi sample (MiB) + training throughput. */
export interface GpuStats { util: number; mem_used: number; mem_total: number; tok_s: number | null }

export interface TrainStatus {
  phase: Phase
  started_at: string
  updated_at: string
  message: string
  pod_id: string | null
  job_id?: string
  steps?: number
  series?: Series // present from training on: parsed from the pod log
  events?: JobEvent[]
  gpu?: GpuStats
  [extra: string]: unknown
}

/** A 401 anywhere means the session is gone (expired, signed out elsewhere, invite revoked): the auth gate shows sign-in. */
export const SIGNED_OUT = "typically:signed-out"
const checkAuth = (res: Response) => { if (res.status === 401) window.dispatchEvent(new Event(SIGNED_OUT)) }

async function call<T>(path: string, body?: unknown, method?: "DELETE", root = "/api/typically/"): Promise<T> {
  const res = await fetch(`${root}${path}`, method ? { method } : body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  checkAuth(res)
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({ detail: undefined }))
    if (typeof detail?.message === "string") throw Object.assign(new Error(detail.message), { detail }) // a structured error: callers read e.detail
    // FastAPI 422 detail is an array of validation errors
    throw new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export type Source =
  | { kind: "csv"; text: string; name?: string }
  | { kind: "xlsx"; data_base64: string; name?: string; sheet?: string }
  | { kind: "upload"; token: string }
  | { kind: "hf"; dataset: string; config?: string; split?: string; limit?: number }
  | { kind: "sheets"; url: string }
  | { kind: "mail"; data_base64: string; name?: string; owner?: string }
  | { kind: "sample"; name?: "enron" | "enron-decisions" } // no name: Northwind

export interface BuildV2 {
  records_token: string
  plan: import("@/lib/project").DatasetPlan
  name: string
  base: "small" | "medium"
  enrich: import("@/lib/project").Enrich
  settings: { steps: number; holdout: number; seed: number }
}

export interface Snippets { curl: string; python: string; javascript: string; sdk?: string }
export interface Capabilities { ai: boolean; hf_namespace: string | null; languages: string[]; can_push?: boolean }
let caps: Promise<Capabilities> | undefined

export const api = {
  trainStatus: (slug: string) => call<TrainStatus>(`train/${encodeURIComponent(slug)}`),
  stopTraining: (slug: string) => call<{ stopping: string }>(`train/${encodeURIComponent(slug)}/stop`, {}),
  analyze: (source: Source, lang: "en" | "he", intent?: import("@/lib/project").Intent) => call<import("@/lib/project").Analysis>("analyze", { source, lang, intent }),
  /** Fetched once per page load (a failure is retried on the next call). can_push: may this user push to Hugging Face. */
  capabilities: () => (caps ??= call<Capabilities>("capabilities").catch((e) => { caps = undefined; throw e })),
  teach: (req: BuildV2) => call<TrainStatus & { job: string }>("teach", req), // build + train in one job; returns at once
  createKey: (model_id: string) => call<{ key: string; prefix: string }>("keys", { model_id }),
  snippets: (model_id: string) => call<Snippets>(`snippets/${encodeURIComponent(model_id)}`),
  push: (req: { run: string; repo?: string; private: boolean }) => call<{ url: string; files: string[] }>("push", req),
  datasets: () => call<{ datasets: Dataset[] }>("datasets").then((r) => r.datasets),
  deleteDataset: (token: string) => call<unknown>(`datasets/${encodeURIComponent(token)}`, undefined, "DELETE"),
}

export interface Dataset {
  token: string // "sample" / "sample-enron" / "sample-enron-decisions" for the samples
  name: string
  kind: string | null
  rows: number
  columns: string[]
  created: string | null
  sample: boolean
  models: { id: string; name: string }[] // trained from it
}

export interface Call {
  answer: string
  p: number
}

export interface Reveal {
  company: string
  tuned: string
  n_cases: number
  n_answers: number
  score: { standard: number; yours: number }
  decisions: {
    key: string; question: string; type: DecisionType; labels: string[]; standard: number; yours: number
    majority?: string; baseline?: number // always giving the most common answer; absent in reveals scored before it existed
    catch?: { standard: number | null; yours: number | null } // yes/no: share of the real "yes" cases in each model's top catch_top
  }[]
  catch_top?: number
  fixed: number
  broken: number
  abstained: { standard: number; yours: number }
  disagreements: { case: string; key: string; question: string; standard: Call; yours: Call; decided: string }[]
  measured: string
}

export type Computing = { status: "computing"; done: number; total: number }

/** GET /api/typically/results/{project}: 202 {status:"computing"} while scoring, then 200 Reveal. */
export async function results(project: string): Promise<Reveal | Computing> {
  const res = await fetch(`/api/typically/results/${encodeURIComponent(project)}`)
  checkAuth(res)
  if (res.status === 202) return res.json()
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({ detail: undefined }))
    throw new Error(typeof detail === "string" ? detail : `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export const downloadUrl = (run: string) => `/api/typically/download/${encodeURIComponent(run)}`

// ---- models library + chat (site/typically_models.py)

export interface LibraryDecision { column: string; question: string; type: DecisionType; labels: string[] }
export interface LibraryModel {
  id: string
  name: string
  kind: "base" | "custom"
  base: "small" | "medium"
  params?: string
  status: "ready" | "training" | "failed"
  progress?: number
  message?: string
  code?: string // the job's status code (code.* copy); "failed" = shutdown unconfirmed, show `message`
  description?: string
  sample?: boolean
  run?: string
  steps?: number
  created_at?: string
  decisions?: LibraryDecision[]
  metrics?: { standard: number; yours: number; n_cases: number } | null
  has_key?: boolean
  hf_repo?: string | null
  examples?: string[]
  version?: number // 1 = trained from the data; 2+ = retrained with chat corrections ("<root>_v<N>")
  parent?: string | null
  corrections?: number
}
export interface Library { base: LibraryModel[]; custom: LibraryModel[] }
export interface AskQuestion { question: string; type: DecisionType; labels?: string[] }
export interface AskResult { model: string; results: Result[]; base?: { model: string; results: Result[] }; ms: number }

export const library = {
  list: () => call<Library>("models/library"),
  get: (id: string) => call<LibraryModel>(`models/library/${encodeURIComponent(id)}`),
  archive: (id: string) => call<{ archived: string }>(`models/library/${encodeURIComponent(id)}`, undefined, "DELETE"),
  ask: (req: { model: string; case: string; questions: AskQuestion[]; compare_with_base?: boolean }) => call<AskResult>("ask", req),
  parseQuestion: (text: string, lang: string) => call<AskQuestion & { source: string }>("parse_question", { text, lang }),
}

// ---- corrections loop (site/typically_corrections.py)

export interface Correction { case: string; question: string; type: DecisionType; labels: string[]; answer: string; model_answer?: string; p?: number }
export interface CorrectionItem extends Correction { n: number; at: string }
const corr = (id: string, rest = "") => `models/${encodeURIComponent(id)}/corrections${rest}`

export const corrections = {
  list: (id: string) => call<{ count: number; items: CorrectionItem[] }>(corr(id)),
  add: (id: string, c: Correction) => call<{ count: number; n: number }>(corr(id), c),
  remove: (id: string, n: number) => call<{ count: number }>(corr(id, `/${n}`), undefined, "DELETE"),
  retrain: (id: string) => call<{ id: string; name: string; version: number }>(`models/${encodeURIComponent(id)}/retrain`, {}),
}

// ---- sign-in (site/typically_auth.py) and share cards (site/typically_share.py)

export interface User { id: string; name: string; admin: boolean }
/** auth=false: the server runs without sign-in (local dev). /me is 401 when sign-in is on and there is no session. */
export interface Me { auth: boolean; user: User | null }

export const session = {
  me: async (): Promise<Me | null> => {
    const res = await fetch("/api/auth/me")
    if (res.status === 401) return null
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
    return res.json()
  },
  login: (code: string, name: string) => call<Me>("login", { code, name: name || null }, undefined, "/api/auth/"),
  logout: () => call<{ ok: boolean }>("logout", {}, undefined, "/api/auth/"),
}

export interface Share { id: string; url: string; image: string; created_at: string }
const share = (id: string) => `models/${encodeURIComponent(id)}/share`

export const shares = {
  get: (id: string) => call<{ share: Share | null }>(share(id)).then((r) => r.share),
  create: (id: string) => call<{ share: Share }>(share(id), {}).then((r) => r.share),
  revoke: (id: string) => call<{ revoked: number }>(share(id), undefined, "DELETE"),
}

/** Invite codes the signed-in user made (sign-in on only). limit/left: null for admins (unlimited).
 *  name / email: null on invites from before they were required; joined: their first sign-in (null = pending). */
export interface Invite {
  code: string; no: string; name: string | null; email: string | null; inviter: string | null
  created: string; joined: string | null; last_seen: string | null
}
export interface Invites { invites: Invite[]; limit: number | null; left: number | null }
// the app's own path: / hosted, /app/ local; &for= prefills the invitee's name on the sign-in screen (never the email: links get forwarded)
export const inviteLink = (i: Invite) => `${new URL(".", window.location.href).href}#/?invite=${encodeURIComponent(i.code)}${i.name ? `&for=${encodeURIComponent(i.name)}` : ""}`
export const ticketUrl = (code: string) => `/api/auth/invites/${encodeURIComponent(code)}/ticket.png`

export const invites = {
  list: (all = false) => call<Invites>(all ? "invites?all=1" : "invites", undefined, undefined, "/api/auth/"),  // all: admins, everyone's
  create: (name: string, email: string) => call<Invites & { code: string }>("invites", { name, email }, undefined, "/api/auth/"),
  revoke: (code: string) => call<Invites>(`invites/${encodeURIComponent(code)}`, undefined, "DELETE", "/api/auth/"),
}
