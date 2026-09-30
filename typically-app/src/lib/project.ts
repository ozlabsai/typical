import type { DecisionType, Preview, Query } from "@/lib/api"

export interface DecisionDraft {
  column: string
  include: boolean
  question: string
  type: DecisionType
  labels: string[] // candidates in the order the model sees them
  counts: Record<string, number>
}

export interface Project {
  name: string
  fileName: string
  csvText: string
  preview: Preview
  textCol: string
  decisions: DecisionDraft[]
  sample: boolean
  slug?: string // set once a dataset is built
  run?: string // trained model run, e.g. "co_f" -> model "local:co_f"
  resultsKey?: string // "northwind" or the slug
}

const YES_NO = new Set(["yes", "no", "true", "false", "1", "0", "y", "n"])
export const isYesNo = (values: string[]) => values.length === 2 && values.every((v) => YES_NO.has(v.toLowerCase()))
const isScale = (values: string[]) => values.length >= 3 && values.every((v) => /^-?\d+$/.test(v))

const human = (col: string) => col.replace(/[_-]+/g, " ").trim().toLowerCase()

export function defaultQuestion(col: string, type: DecisionType) {
  const c = human(col)
  return type === "noul" ? `Should we ${c}?` : type === "score" ? `How much ${c}?` : `Which ${c}?`
}

export function detectType(values: string[]): DecisionType {
  return isYesNo(values) ? "noul" : isScale(values) ? "score" : "choice"
}

export function labelsFor(type: DecisionType, values: string[]) {
  if (type === "noul") return ["no", "yes"]
  return type === "score" ? [...values].sort((a, b) => Number(a) - Number(b)) : [...values].sort()
}

/** Text column = the one with the most distinct values; every other column with 2-20 values is a decision. */
export function draftFromPreview(preview: Preview, counts: Record<string, Record<string, number>>) {
  const textCol = [...preview.columns].sort((a, b) => preview.uniques[b] - preview.uniques[a])[0]
  const decisions = preview.columns
    .filter((c) => c !== textCol && preview.values[c] && preview.uniques[c] >= 2)
    .map<DecisionDraft>((column) => {
      const values = preview.values[column]
      const type = detectType(values)
      return { column, include: true, type, question: defaultQuestion(column, type), labels: labelsFor(type, values), counts: counts[column] ?? {} }
    })
  return { textCol, decisions }
}

/** Per-column answer counts, parsed client-side (the preview only returns 5 rows). */
export function countAnswers(csvText: string, columns: string[]) {
  const rows = parseCsv(csvText)
  const header = rows[0] ?? []
  const out: Record<string, Record<string, number>> = {}
  for (const c of columns) {
    const i = header.indexOf(c)
    if (i < 0) continue
    out[c] = {}
    for (const r of rows.slice(1)) if (r[i] !== undefined && r[i] !== "") out[c][r[i]] = (out[c][r[i]] ?? 0) + 1
  }
  return out
}

/** Minimal RFC 4180 parser (quotes, escaped quotes, CRLF). */
function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = [], cell = "", q = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (q) {
      if (ch === '"' && text[i + 1] === '"') (cell += '"'), i++
      else if (ch === '"') q = false
      else cell += ch
    } else if (ch === '"') q = true
    else if (ch === ",") row.push(cell), (cell = "")
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++
      row.push(cell), rows.push(row), (row = []), (cell = "")
    } else cell += ch
  }
  if (cell || row.length) row.push(cell), rows.push(row)
  return rows.filter((r) => r.some((c) => c !== ""))
}

export const queries = (p: Project): Query[] =>
  p.decisions.filter((d) => d.include).map(({ type, question, labels }) => ({ type, question, labels }))

export const say = (type: DecisionType, label: string) =>
  type === "noul" ? (label === "yes" ? "Yes" : "No") : label

export const pct = (x: number) => `${Math.round(x * 100)}%`

// The Northwind sample: the questions its model was taught, in the candidate order it was trained with
// (scripts/typically_spike.py DECISIONS), and the arm-f model from spike 3.
export const SAMPLE = {
  name: "Northwind Freight",
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
