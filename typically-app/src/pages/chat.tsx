import { useEffect, useMemo, useRef, useState } from "react"
import { ArrowUp, ChevronDown, ListChecks, Loader2, Plus, Sparkles, Trash2, X } from "lucide-react"

import { StatusDot } from "@/components/app-sidebar"
import { msg } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { library, type AskQuestion, type AskResult, type LibraryModel, type Result } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import { pct, say, TYPES } from "@/lib/project"
import { go } from "@/lib/router"
import { cn } from "@/lib/utils"

interface Turn {
  id: string
  model: string
  modelName: string
  case: string
  questions: AskQuestion[]
  compare: boolean
  result?: AskResult
  error?: string
}

const STORE = "typically.chat"
const load = (): Turn[] => {
  try { return JSON.parse(localStorage.getItem(STORE) ?? "[]") } catch { return [] }
}

/* ------------------------------------------------------------------ answer rendering */

function Answer({ q, r, tone, label }: { q: AskQuestion; r: Result; tone: "standard" | "yours"; label?: string }) {
  const { t } = useI18n()
  const ranked = Object.entries(r.probs).sort((a, b) => b[1] - a[1])
  const unsure = r.p_null >= 0.5
  return (
    <div className="grid gap-2">
      {label && <span className="flex items-center gap-1.5 text-xs text-muted-foreground"><span className={cn("size-1.5 rounded-full", tone === "standard" ? "bg-standard" : "bg-yours")} />{label}</span>}
      <div className="flex items-baseline gap-2">
        <span className="text-lg font-semibold tracking-tight">{say(t, q.type, r.argmax)}</span>
        <span className="font-mono text-sm text-muted-foreground tabular">{pct(r.probs[r.argmax])}</span>
        {unsure && <Badge variant="outline" className="font-normal">{t("chat.notSure")}</Badge>}
      </div>
      <div className="grid gap-1">
        {ranked.slice(0, 6).map(([l, p]) => (
          <div key={l} className="grid grid-cols-[minmax(0,7rem)_1fr_2.5rem] items-center gap-2 text-xs">
            <span dir="auto" className="truncate text-muted-foreground">{say(t, q.type, l)}</span>
            <div className="h-1 overflow-hidden rounded-full bg-muted">
              <div className={cn("h-full rounded-full transition-[width] duration-200", l === r.argmax ? (tone === "standard" ? "bg-standard" : "bg-yours") : "bg-muted-foreground/30")} style={{ width: `${p * 100}%` }} />
            </div>
            <span className="text-end font-mono tabular text-muted-foreground">{pct(p)}</span>
          </div>
        ))}
        {r.p_null >= 0.15 && <p className="text-xs text-muted-foreground">{t("chat.noneFit")}: {pct(r.p_null)}</p>}
      </div>
    </div>
  )
}

function TurnView({ turn }: { turn: Turn }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const long = turn.case.length > 280
  return (
    <div className="grid gap-4">
      {/* the user's case + questions */}
      <div className="ms-auto grid max-w-[85%] gap-2 rounded-2xl rounded-se-sm bg-muted px-4 py-3">
        <p dir="auto" className={cn("whitespace-pre-wrap text-sm", long && !open && "line-clamp-4")}>{turn.case}</p>
        {long && <button type="button" onClick={() => setOpen(!open)} className="justify-self-start text-xs text-muted-foreground hover:text-foreground">{open ? "−" : "…"}</button>}
        <div className="flex flex-wrap gap-1.5">
          {turn.questions.map((q, i) => <Badge key={i} variant="outline" className="bg-background font-normal" dir="auto">{q.question}</Badge>)}
        </div>
      </div>
      {/* the model's answers */}
      <div className="grid gap-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="grid size-5 place-items-center rounded-full bg-primary/10"><span className="size-1.5 rounded-full bg-primary" /></span>
          <span className="font-medium text-foreground">{turn.modelName}</span>
          {turn.result && <span className="font-mono tabular">{t("chat.ms", { n: Math.round(turn.result.ms) })}</span>}
        </div>
        {turn.error ? (
          <p role="alert" className="text-sm text-destructive">{turn.error}</p>
        ) : !turn.result ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> {t("chat.thinking")}</p>
        ) : (
          <div className="grid gap-3">
            {turn.questions.map((q, i) => {
              const mine = turn.result!.results[i], base = turn.result!.base?.results[i]
              return (
                <div key={i} className="rounded-xl border bg-card p-4">
                  <div className="mb-3 flex items-start justify-between gap-3">
                    <p dir="auto" className="text-sm font-medium">{q.question}</p>
                    {base && base.argmax !== mine.argmax && <Badge variant="secondary">{t("pg.disagree")}</Badge>}
                  </div>
                  {base ? (
                    <div className="grid gap-5 sm:grid-cols-2">
                      <Answer q={q} r={base} tone="standard" label={t("ev.standardCol")} />
                      <Answer q={q} r={mine} tone="yours" label={turn.modelName} />
                    </div>
                  ) : (
                    <Answer q={q} r={mine} tone="yours" />
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ composer */

function QuestionChip({ q, onChange, onRemove }: { q: AskQuestion; onChange: (q: AskQuestion) => void; onRemove: () => void }) {
  const { t } = useI18n()
  const [edit, setEdit] = useState(false)
  return (
    <div className={cn("max-w-full min-w-0 rounded-lg border bg-background text-sm", edit && "w-full")}>
      <div className="flex items-center gap-1 ps-2.5">
        <button type="button" onClick={() => setEdit(!edit)} className="flex min-w-0 flex-1 items-center gap-1.5 py-1 text-start" aria-label={t("chat.edit")}>
          <span dir="auto" className="truncate">{q.question}</span>
          <span className="hidden shrink-0 text-xs text-muted-foreground sm:inline">· {t(`type.${q.type}`)}</span>
          <ChevronDown className={cn("size-3 shrink-0 text-muted-foreground transition-transform", edit && "rotate-180")} />
        </button>
        <Button variant="ghost" size="icon-xs" onClick={onRemove} aria-label={t("chat.remove")}><X /></Button>
      </div>
      {edit && (
        <div className="grid gap-2 border-t p-2.5 sm:grid-cols-[1fr_140px]">
          <Input dir="auto" value={q.question} onChange={(e) => onChange({ ...q, question: e.target.value })} aria-label={t("chat.edit")} />
          <Select value={q.type} onValueChange={(v) => onChange({ ...q, type: v as AskQuestion["type"], labels: v === "noul" ? ["no", "yes"] : v === "score" && q.type !== "score" ? ["1", "2", "3", "4", "5"] : q.labels })}>
            <SelectTrigger className="w-full" aria-label={t("chat.type")}><SelectValue /></SelectTrigger>
            <SelectContent>{TYPES.map((ty) => <SelectItem key={ty} value={ty}>{t(`type.${ty}`)}</SelectItem>)}</SelectContent>
          </Select>
          {q.type !== "noul" && (
            <div className="grid gap-1 sm:col-span-2">
              <Label className="text-xs text-muted-foreground">{t("chat.options")} <span className="font-normal">({t("chat.optionsHint")})</span></Label>
              <Input dir="auto" defaultValue={(q.labels ?? []).join(", ")} onBlur={(e) => onChange({ ...q, labels: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ page */

export function ChatPage({ modelId }: { modelId?: string }) {
  const { t, lang } = useI18n()
  const { lib, find } = useLibrary()
  const [turns, setTurns] = useState<Turn[]>(load)
  const [text, setText] = useState("")
  const [questions, setQuestions] = useState<AskQuestion[]>([])
  const [draft, setDraft] = useState("")
  const [parsing, setParsing] = useState(false)
  const [compare, setCompare] = useState(true)
  const [examples, setExamples] = useState<string[]>([])
  const end = useRef<HTMLDivElement>(null)

  const ready = useMemo(() => [...(lib?.custom ?? []).filter((m) => m.status === "ready"), ...(lib?.base ?? [])], [lib])
  const model: LibraryModel | undefined = find(modelId) ?? ready[0]
  const custom = model?.kind === "custom"

  useEffect(() => { localStorage.setItem(STORE, JSON.stringify(turns.slice(-50))) }, [turns])
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }) }, [turns])  // braces: scrollIntoView returns a Promise in newer Chromium, which React would call as cleanup
  useEffect(() => {
    if (!model) return
    if (custom && model.decisions?.length && !questions.length) setQuestions(model.decisions.map(({ question, type, labels }) => ({ question, type, labels })))
    library.get(model.id).then((m) => setExamples(m.examples ?? [])).catch(() => setExamples([]))
  }, [model?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  async function addQuestion() {
    const q = draft.trim()
    if (!q) return
    setParsing(true)
    try {
      const p = await library.parseQuestion(q, lang)
      setQuestions((qs) => [...qs, { question: p.question || q, type: p.type, labels: p.labels }])
      setDraft("")
    } catch {
      setQuestions((qs) => [...qs, { question: q, type: "noul", labels: ["no", "yes"] }])
      setDraft("")
    } finally {
      setParsing(false)
    }
  }

  async function send() {
    if (!model || !text.trim() || !questions.length) return
    const turn: Turn = { id: crypto.randomUUID(), model: model.id, modelName: model.name, case: text.trim(), questions, compare: custom && compare }
    setTurns((ts) => [...ts, turn])
    setText("")
    try {
      const result = await library.ask({ model: model.id, case: turn.case, questions, compare_with_base: turn.compare })
      setTurns((ts) => ts.map((x) => (x.id === turn.id ? { ...x, result } : x)))
    } catch (e) {
      setTurns((ts) => ts.map((x) => (x.id === turn.id ? { ...x, error: msg(e) } : x)))
    }
  }

  const shown = turns.filter((x) => x.model === model?.id)

  return (
    <div className="flex h-[calc(100svh-3.5rem)] flex-col">
      {/* toolbar */}
      <div className="flex flex-wrap items-center gap-3 border-b px-4 py-2.5 md:px-6">
        <Select value={model?.id ?? ""} onValueChange={(id) => go({ name: "chat", model: id })}>
          <SelectTrigger className="w-full max-w-64" aria-label={t("chat.model")}>
            <SelectValue placeholder={t("chat.model")} />
          </SelectTrigger>
          <SelectContent>
            {(lib?.custom ?? []).filter((m) => m.status === "ready").length > 0 && (
              <SelectGroup>
                <SelectLabel>{t("app.yourModels")}</SelectLabel>
                {lib!.custom.filter((m) => m.status === "ready").map((m) => (
                  <SelectItem key={m.id} value={m.id}><StatusDot status={m.status} /> <span dir="auto">{m.name}</span></SelectItem>
                ))}
              </SelectGroup>
            )}
            <SelectGroup>
              <SelectLabel>{t("app.base")}</SelectLabel>
              {(lib?.base ?? []).map((m) => <SelectItem key={m.id} value={m.id}>{m.name} <span className="text-muted-foreground">{m.params}</span></SelectItem>)}
            </SelectGroup>
          </SelectContent>
        </Select>
        {custom && (
          <Label className="flex items-center gap-2 text-sm font-normal">
            <Switch checked={compare} onCheckedChange={setCompare} /> {t("chat.compare")}
          </Label>
        )}
        <div className="ms-auto">
          {shown.length > 0 && (
            <Button variant="ghost" size="sm" onClick={() => setTurns((ts) => ts.filter((x) => x.model !== model?.id))}>
              <Trash2 data-icon="inline-start" /> {t("chat.clear")}
            </Button>
          )}
        </div>
      </div>

      {/* transcript */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto grid max-w-3xl gap-8 px-4 py-8 md:px-6">
          {!shown.length && model && (
            <div className="grid gap-6 pt-6 text-center">
              <div className="mx-auto grid size-12 place-items-center rounded-2xl bg-primary/10"><Sparkles className="size-5 text-primary" /></div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight" dir="auto">{t("chat.emptyTitle", { name: model.name })}</h1>
                <p className="mx-auto mt-2 max-w-lg text-sm text-muted-foreground">{t("chat.emptyBody")}</p>
              </div>
              {examples.length > 0 && (
                <div className="grid gap-2 text-start sm:grid-cols-3">
                  {examples.slice(0, 3).map((ex, i) => (
                    <button key={i} type="button" onClick={() => setText(ex)}
                      className="rounded-xl border bg-card p-3 text-start text-xs text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground">
                      <span className="mb-1 block font-medium text-foreground">{t("chat.try")}</span>
                      <span dir="auto" className="line-clamp-4">{ex.split("\n\n").at(-1)}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
          {shown.map((turn) => <TurnView key={turn.id} turn={turn} />)}
          <div ref={end} />
        </div>
      </div>

      {/* composer */}
      <div className="border-t bg-background px-4 pt-3 pb-4 md:px-6">
        <div className="mx-auto grid max-w-3xl min-w-0 gap-2 rounded-2xl border bg-card p-2 shadow-xs focus-within:ring-3 focus-within:ring-ring/30">
          <Textarea
            dir="auto" value={text} rows={3} placeholder={t("chat.casePlaceholder")}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => (e.metaKey || e.ctrlKey) && e.key === "Enter" && (e.preventDefault(), send())}
            className="max-h-60 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
          <div className="flex flex-wrap items-start gap-1.5 px-1">
            {questions.map((q, i) => (
              <QuestionChip key={i} q={q}
                onChange={(nq) => setQuestions((qs) => qs.map((x, j) => (j === i ? nq : x)))}
                onRemove={() => setQuestions((qs) => qs.filter((_, j) => j !== i))} />
            ))}
          </div>
          <div className="flex min-w-0 items-center gap-2 px-1">
            <div className="relative min-w-0 flex-1">
              <Input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={t("chat.questionPlaceholder")}
                onKeyDown={(e) => e.key === "Enter" && !e.metaKey && !e.ctrlKey && (e.preventDefault(), addQuestion())}
                className="h-8 border-0 bg-muted/50 pe-16 shadow-none" />
              <Button variant="ghost" size="xs" onClick={addQuestion} disabled={!draft.trim() || parsing} className="absolute end-1 top-1">
                {parsing ? <Loader2 className="animate-spin" /> : <Plus />} {t("chat.add")}
              </Button>
            </div>
            {custom && model?.decisions?.length ? (
              <Button variant="ghost" size="sm" onClick={() => setQuestions(model.decisions!.map(({ question, type, labels }) => ({ question, type, labels })))}>
                <ListChecks data-icon="inline-start" /> <span className="hidden sm:inline">{t("chat.trained")}</span>
              </Button>
            ) : null}
            <Button size="icon" className="rounded-full" onClick={send} disabled={!text.trim() || !questions.length || !model}
              aria-label={`${t("chat.send")} (${t("chat.sendHint")})`} title={!questions.length ? t("chat.noQuestions") : t("chat.sendHint")}>
              <ArrowUp />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
