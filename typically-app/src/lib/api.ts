// Typed client for site/server.py (/api/typically/*). Plain fetch; non-2xx throws Error(server `detail`).

export type DecisionType = "choice" | "noul" | "score"

export interface Query {
  type: DecisionType
  question: string
  labels?: string[] // choice/score need >= 2; noul is fixed no/yes
}

export interface Decision {
  column: string
  question: string
  type: DecisionType
}

export interface Models {
  base: string[]
  tuned: string[] // "local:<slug>"
}

export interface Preview {
  columns: string[]
  rows: Record<string, string>[] // first 5
  n: number
  uniques: Record<string, number>
  values: Record<string, string[]> // only columns with <= 20 unique values
}

export type Dist = Record<string, Record<string, number>> // decision -> label -> count

export interface Build {
  job: string
  data_dir: string
  splits: Record<string, number>
  balance: { before: Dist; after: Dist }
  command: string
}

export interface Result {
  probs: Record<string, number>
  p_null: number
  argmax: string
  expected?: number // score only
}

export interface Run {
  results: Result[]
  ms: number
  device: string
}

export interface Compare {
  models: Record<string, Run & { model: string }>
  base: string | null   // the released model "base" resolved to
}

export const PHASES = ["queued", "starting_gpu", "uploading", "training", "evaluating", "downloading", "done", "failed"] as const
export type Phase = (typeof PHASES)[number]

export interface Agreement {
  n: number
  overall: { base: number; yours: number }
  decisions: Record<string, { base: number; yours: number; n: number }>
}

/** Training curves aligned on `step` (null where that step logged no value); step_time = recent seconds per step. */
export interface Series { step: number[]; loss: (number | null)[]; val: (number | null)[]; best_on: (number | null)[]; step_time: number | null }

export interface TrainStatus {
  phase: Phase
  started_at: string
  updated_at: string
  message: string
  pod_id: string | null
  job_id?: string
  steps?: number
  agreement?: Agreement // present when phase === "done"
  series?: Series // present while training / evaluating: parsed from the pod log
  [extra: string]: unknown
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api/typically/${path}`, body === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const { detail } = await res.json().catch(() => ({ detail: undefined }))
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
  | { kind: "sample" }

export interface BuildV2 {
  records_token: string
  plan: import("@/lib/project").DatasetPlan
  name: string
  base: "small" | "medium"
  enrich: import("@/lib/project").Enrich
  settings: { steps: number; holdout: number; seed: number }
}

export interface Snippets { curl: string; python: string; javascript: string; sdk: string }

export const api = {
  models: () => call<Models>("models"),
  preview: (csv_text: string) => call<Preview>("preview", { csv_text }),
  build: (req: { csv_text: string; text_col: string; decisions: Decision[]; name: string }) => call<Build>("build", req),
  compare: (req: { state: string; decisions: Query[]; models: string[] }) => call<Compare>("compare", req),
  train: (name: string) => call<TrainStatus>("train", { name }),
  trainStatus: (slug: string) => call<TrainStatus>(`train/${encodeURIComponent(slug)}`),
  analyze: (source: Source, lang: "en" | "he") => call<import("@/lib/project").Analysis>("analyze", { source, lang }),
  capabilities: () => call<{ ai: boolean; hf_namespace: string | null; languages: string[] }>("capabilities"),
  buildPlan: (req: BuildV2) => call<Build>("build", req),
  trainV2: (req: { name: string; base: "small" | "medium"; steps: number }) => call<TrainStatus>("train", req),
  createKey: (model_id: string) => call<{ key: string; prefix: string }>("keys", { model_id }),
  snippets: (model_id: string) => call<Snippets>(`snippets/${encodeURIComponent(model_id)}`),
  push: (req: { run: string; repo?: string; private: boolean }) => call<{ url: string; files: string[] }>("push", req),
  datasets: () => call<{ datasets: Dataset[] }>("datasets").then((r) => r.datasets),
  deleteDataset: async (token: string) => {
    const r = await fetch(`/api/typically/datasets/${encodeURIComponent(token)}`, { method: "DELETE" })
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `${r.status} ${r.statusText}`)
  },
}

export interface Dataset {
  token: string // "sample" for the Northwind sample
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
  decisions: { key: string; question: string; type: DecisionType; labels: string[]; standard: number; yours: number }[]
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
  status: "ready" | "training" | "failed" | "queued"
  progress?: number
  message?: string
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
}
export interface Library { base: LibraryModel[]; custom: LibraryModel[] }
export interface AskQuestion { question: string; type: DecisionType; labels?: string[] }
export interface AskResult { model: string; results: Result[]; base?: { model: string; results: Result[] }; ms: number }

export const library = {
  list: () => call<Library>("models/library"),
  get: (id: string) => call<LibraryModel>(`models/library/${encodeURIComponent(id)}`),
  archive: (id: string) =>
    fetch(`/api/typically/models/library/${encodeURIComponent(id)}`, { method: "DELETE" }).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`)
    }),
  ask: (req: { model: string; case: string; questions: AskQuestion[]; compare_with_base?: boolean }) => call<AskResult>("ask", req),
  parseQuestion: (text: string, lang: string) => call<AskQuestion & { source: string }>("parse_question", { text, lang }),
}
