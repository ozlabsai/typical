import type { DecisionType, Query } from "@/lib/api"
import type { T } from "@/lib/i18n"

// ---- DatasetPlan: the agent's reading of the data (.context/typically/FLOW.md), confirmed by the user

export interface CasePart { column: string; role: "fact" | "body"; sentence?: string | null; confidence: number; reason: string }
export interface PlanDecision {
  column: string
  include: boolean
  question: string
  type: DecisionType
  labels: string[] // ordered; score = low -> high
  mapping: Record<string, string | null> // observed (normalised) value -> label; null = drop row
  confidence: number
  reasons: string[]
  needs_review: boolean
}
export interface DatasetPlan {
  case: { parts: CasePart[]; max_tokens: number; overflow: string }
  decisions: PlanDecision[]
  excluded: { column: string; why: string; confidence: number; reason: string }[]
  issues: { kind: string; column?: string | null; count: number; detail: string; action: string }[]
  languages: { code: string; share: number }[]
}
export interface ObservedValue { value: string; count: number; raws: string[] } // value = normalised spelling

export interface Analysis {
  name_hint: string
  n_rows: number
  columns: string[]
  profile: { columns?: Record<string, { values?: ObservedValue[] } & Record<string, unknown>> } & Record<string, unknown>
  plan: DatasetPlan
  plan_source: "llm" | "heuristic"
  preview_cases: { case: string; answers: Record<string, string> }[]
  records_token: string
  warnings?: string[]
}

export type Base = "small" | "medium"
export type Preset = "quick" | "balanced" | "thorough"
export const PRESETS: Record<Preset, { steps: number }> = { quick: { steps: 200 }, balanced: { steps: 400 }, thorough: { steps: 800 } }
export const BASES: Record<Base, { label: string; size: string; minutes: number; beta?: boolean }> = {
  small: { label: "typical-small", size: "1.7B", minutes: 10 },
  medium: { label: "typical-medium", size: "4B", minutes: 25, beta: true },
}

export interface Enrich { balance: boolean; dedupe_soft: boolean; policy: Record<string, string>; synthetic: boolean; languages: string[] }
export interface Settings { preset: Preset; holdout: number; seed: number }

export interface Project {
  name: string
  base: Base
  source: string // "northwind_tickets.csv", "hf: bitext/...", ...
  sample: boolean
  analysis: Analysis
  plan: DatasetPlan
  enrich: Enrich
  settings: Settings
  slug?: string
  run?: string // "co_f" -> model "local:co_f"
  resultsKey?: string // "northwind" or the slug
}

export const slugify = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "").slice(0, 40)
export const modelId = (p: Project) => (p.sample ? "northwind" : p.slug ?? slugify(p.name))

export const included = (p: Project) => p.plan.decisions.filter((d) => d.include)
export const queries = (p: Project): Query[] => included(p).map(({ type, question, labels }) => ({ type, question, labels }))

const observed = (p: Project, d: PlanDecision) => p.analysis.profile.columns?.[d.column]?.values ?? []
const target = (d: PlanDecision, v: ObservedValue) => d.mapping[v.value] ?? (d.labels.includes(v.value) ? v.value : null)

/** Answer counts per label for a decision, folded through its mapping. */
export function labelCounts(p: Project, d: PlanDecision) {
  const out: Record<string, number> = Object.fromEntries(d.labels.map((l) => [l, 0]))
  for (const v of observed(p, d)) {
    const to = target(d, v)
    if (to) out[to] = (out[to] ?? 0) + v.count
  }
  return out
}

/** "yes <- Yes, Y, TRUE": raw spellings in the file that differ from the label they were merged into. */
export function merges(p: Project, d: PlanDecision) {
  const by: Record<string, string[]> = {}
  for (const v of observed(p, d)) {
    const to = target(d, v)
    for (const raw of v.raws) if (to && raw !== to) (by[to] ??= []).push(raw)
  }
  return by
}

export const say = (t: T, type: DecisionType, label: string) => (type === "noul" ? t(label === "yes" ? "ans.yes" : "ans.no") : label)
export const pct = (x: number) => `${Math.round(x * 100)}%`

export const TYPES: DecisionType[] = ["choice", "noul", "score"]

// The Northwind sample: the questions its model was taught, in the candidate order it was trained with
// (scripts/typically_spike.py DECISIONS), and the arm-f model from spike 3 (.context/typically/RESULTS.md).
export const SAMPLE = {
  name: "Northwind triage",
  run: "co_f",
  resultsKey: "northwind",
  questions: {
    team: { question: "Which team should handle this ticket?", type: "choice", labels: ["Claims", "Billing", "Dispatch", "Sales"] },
    escalate: { question: "Should we escalate this to a manager?", type: "noul", labels: ["no", "yes"] },
    urgency: { question: "How urgent is this ticket?", type: "score", labels: ["0", "1", "2", "3"] },
    refund: { question: "Can the agent refund this without approval?", type: "noul", labels: ["no", "yes"] },
  } as Record<string, { question: string; type: DecisionType; labels: string[] }>,
  tryCase:
    "Customer: Cedar Foods (enterprise plan, US). Tickets from this customer in the last 30 days: 1. Shipment value: $450, shipped 4 days ago.\n\n" +
    "Hi, two cases of glassware arrived cracked this morning. Photos attached. Can you sort out a replacement?",
}

export const LANGUAGES = ["es", "fr", "de", "pt", "he", "ar"] as const
