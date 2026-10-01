import { useEffect, useMemo, useRef, useState } from "react"
import { ArrowUp, Check, FileText, ListChecks, Loader2, Pencil, Sparkles, Trash2, Undo2, X } from "lucide-react"
import { toast } from "sonner"

import { StatusDot } from "@/components/app-sidebar"
import { HeaderActions } from "@/components/layout"
import { msg } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { corrections, library, type AskQuestion, type AskResult, type LibraryModel, type Result } from "@/lib/api"
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
  fixes?: Record<number, Fix> // question index -> the correction saved for it
}
interface Fix { answer: string; n: number } // n: its line in the model's corrections file (DELETE .../corrections/{n})

const STORE = "typically.chat"
const load = (): Turn[] => {
  try { return JSON.parse(localStorage.getItem(STORE) ?? "[]") } catch { return [] }
}
const COLLAPSE_AT = 160 // a paste this long (or multi-line) becomes a Case card, like Claude's pasted-text chips
const words = (s: string) => s.trim().split(/\s+/).filter(Boolean).length

/* ------------------------------------------------------------------ case card (composer + transcript) */

function CaseCard({ text, onEdit, onRemove, className }: { text: string; onEdit?: () => void; onRemove?: () => void; className?: string }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const body = text.trim()
  return (
    <div className={cn("grid gap-1.5 rounded-lg border bg-muted/40 p-3", className)}>
      <div className="flex items-center gap-2">
        <FileText className="size-4 shrink-0 text-muted-foreground" />
        <span className="text-xs font-medium">{t("chat.caseLabel")}</span>
        <span className="text-xs text-muted-foreground tabular">{t("chat.words", { n: words(body) })}</span>
        <span className="ms-auto flex items-center gap-0.5">
          {onEdit && <Button variant="ghost" size="icon-xs" onClick={onEdit} aria-label={t("chat.editCase")}><Pencil /></Button>}
          {onRemove && <Button variant="ghost" size="icon-xs" onClick={onRemove} aria-label={t("chat.removeCase")}><X /></Button>}
        </span>
      </div>
      <p dir="auto" className={cn("text-sm whitespace-pre-wrap text-muted-foreground", !open && "line-clamp-2")}>{body}</p>
      {body.length > COLLAPSE_AT && (
        <button type="button" onClick={() => setOpen(!open)} className="justify-self-start text-xs text-muted-foreground hover:text-foreground">
          {t(open ? "chat.collapse" : "chat.expand")}
        </button>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ answers */

function Answer({ q, r, tone, label }: { q: AskQuestion; r: Result; tone: "standard" | "yours"; label?: string }) {
  const { t } = useI18n()
  const ranked = Object.entries(r.probs).sort((a, b) => b[1] - a[1])
  return (
    <div className="grid gap-2">
      {label && <span className="flex items-center gap-1.5 text-xs text-muted-foreground"><span className={cn("size-1.5 rounded-full", tone === "standard" ? "bg-standard" : "bg-yours")} /><span dir="auto">{label}</span></span>}
      <div className="flex items-baseline gap-2">
        <span dir="auto" className="text-base font-semibold">{say(t, q.type, r.argmax)}</span>
        <span className="font-mono text-sm text-muted-foreground tabular">{pct(r.probs[r.argmax])}</span>
        {r.p_null >= 0.5 && <Badge variant="outline" className="font-normal">{t("chat.notSure")}</Badge>}
      </div>
      <div className="grid gap-1">
        {ranked.slice(0, 6).map(([l, p]) => (
          <div key={l} className="grid grid-cols-[minmax(0,7rem)_1fr_2.5rem] items-center gap-2 text-xs">
            <span dir="auto" className="truncate text-muted-foreground">{say(t, q.type, l)}</span>
            <div className="h-1 overflow-hidden rounded-full bg-muted">
              <div className={cn("h-full rounded-full transition-[width] duration-200", l === r.argmax ? (tone === "standard" ? "bg-standard" : "bg-yours") : "bg-muted-foreground/30")} style={{ width: `${p * 100}%` }} />
            </div>
            <span className="text-end font-mono text-muted-foreground tabular">{pct(p)}</span>
          </div>
        ))}
        {r.p_null >= 0.15 && <p className="text-xs text-muted-foreground">{t("chat.noneFit")}: {pct(r.p_null)}</p>}
      </div>
    </div>
  )
}

const optionsOf = (q: AskQuestion, r: Result) => (q.type === "noul" ? ["no", "yes"] : q.labels?.length ? q.labels : Object.keys(r.probs))

/** "Wrong?" on an answer card: pick the right option; once saved it becomes a "Correct: X" chip (click to change) with an undo. */
function WrongAction({ q, r, fix, onPick, onUndo }: { q: AskQuestion; r: Result; fix?: Fix; onPick: (answer: string) => void; onUndo: () => void }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const options = optionsOf(q, r)
  return (
    <span className="flex items-center gap-0.5">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          {fix ? (
            <button type="button" className="inline-flex items-center gap-1 rounded-full border border-yours/40 bg-yours/10 py-0.5 ps-1.5 pe-2 text-xs text-yours hover:bg-yours/15">
              <Check className="size-3" /> <span dir="auto">{t("chat.correct", { answer: say(t, q.type, fix.answer) })}</span>
            </button>
          ) : (
            <Button variant="ghost" size="xs" className="text-muted-foreground">{t("chat.wrong")}</Button>
          )}
        </PopoverTrigger>
        <PopoverContent align="end" className="grid w-60 gap-0.5 p-1.5">
          <p className="px-2 pt-1 pb-1.5 text-xs font-medium">{t("chat.wrongTitle")}</p>
          {options.map((l) => (
            <button key={l} type="button" disabled={l === r.argmax || l === fix?.answer} onClick={() => { setOpen(false); onPick(l) }}
              className="flex items-center justify-between gap-2 rounded-md px-2 py-1.5 text-start text-sm hover:bg-muted disabled:pointer-events-none disabled:text-muted-foreground">
              <span dir="auto" className="truncate">{say(t, q.type, l)}</span>
              {l === r.argmax ? <span className="shrink-0 text-xs">{t("chat.modelSaid")}</span> : l === fix?.answer && <Check className="size-3.5 shrink-0" />}
            </button>
          ))}
          <p className="px-2 pt-1.5 pb-1 text-xs text-muted-foreground">{t("chat.wrongHint")}</p>
        </PopoverContent>
      </Popover>
      {fix && (
        <Button variant="ghost" size="icon-xs" className="text-muted-foreground" onClick={onUndo} aria-label={t("chat.undo")} title={t("chat.undo")}>
          <Undo2 />
        </Button>
      )}
    </span>
  )
}

function TurnView({ turn, onFix, onUndo }: { turn: Turn; onFix?: (i: number, answer: string) => void; onUndo?: (i: number) => void }) {
  const { t } = useI18n()
  return (
    <div className="grid gap-4">
      <div className="ms-auto grid w-full max-w-[85%] gap-2">
        <CaseCard text={turn.case} />
        <div className="flex flex-wrap justify-end gap-1.5">
          {turn.questions.map((q, i) => <Badge key={i} variant="secondary" className="font-normal" dir="auto">{q.question}</Badge>)}
        </div>
      </div>
      <div className="grid gap-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="grid size-5 place-items-center rounded-full bg-primary/10"><span className="size-1.5 rounded-full bg-primary" /></span>
          <span dir="auto" className="font-medium text-foreground">{turn.modelName}</span>
          {turn.result && <span className="font-mono tabular">{t("chat.ms", { n: Math.round(turn.result.ms) })}</span>}
        </div>
        {turn.error ? (
          <p role="alert" className="text-sm text-destructive">{turn.error}</p>
        ) : !turn.result ? (
          <p className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" /> {t("chat.thinking")}</p>
        ) : (
          turn.questions.map((q, i) => {
            const mine = turn.result!.results[i], base = turn.result!.base?.results[i]
            return (
              <div key={i} className="grid gap-3 rounded-xl border bg-card p-4">
                <div className="flex items-start justify-between gap-3">
                  <p dir="auto" className="text-sm font-medium">{q.question}</p>
                  <span className="flex shrink-0 items-center gap-1.5">
                    {base && base.argmax !== mine.argmax && <Badge variant="secondary">{t("pg.disagree")}</Badge>}
                    {onFix && onUndo && <WrongAction q={q} r={mine} fix={turn.fixes?.[i]} onPick={(a) => onFix(i, a)} onUndo={() => onUndo(i)} />}
                  </span>
                </div>
                {base ? (
                  <div className="grid gap-5 sm:grid-cols-2">
                    <Answer q={q} r={base} tone="standard" label={t("ev.standardCol")} />
                    <Answer q={q} r={mine} tone="yours" label={turn.modelName} />
                  </div>
                ) : <Answer q={q} r={mine} tone="yours" />}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ composer */

function QuestionChip({ q, onChange, onRemove }: { q: AskQuestion; onChange: (q: AskQuestion) => void; onRemove: () => void }) {
  const { t } = useI18n()
  return (
    <span className="inline-flex max-w-full min-w-0 items-center rounded-full border bg-background text-xs">
      <Popover>
        <PopoverTrigger asChild>
          <button type="button" className="flex min-w-0 items-center rounded-s-full py-1 ps-2.5 pe-1 text-start hover:bg-muted/60" aria-label={t("chat.edit")} title={t(`type.${q.type}`)}>
            <span dir="auto" className="truncate">{q.question}</span>
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" className="grid w-80 gap-3">
          <div className="grid gap-1.5"><Label className="text-xs">{t("chat.edit")}</Label>
            <Input dir="auto" value={q.question} onChange={(e) => onChange({ ...q, question: e.target.value })} /></div>
          <div className="grid gap-1.5"><Label className="text-xs">{t("chat.type")}</Label>
            <Select value={q.type} onValueChange={(v) => onChange({ ...q, type: v as AskQuestion["type"], labels: v === "noul" ? ["no", "yes"] : v === "score" && q.type !== "score" ? ["1", "2", "3", "4", "5"] : q.labels })}>
              <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
              <SelectContent>{TYPES.map((ty) => <SelectItem key={ty} value={ty}>{t(`type.${ty}`)}</SelectItem>)}</SelectContent>
            </Select></div>
          {q.type !== "noul" && (
            <div className="grid gap-1.5"><Label className="text-xs">{t("chat.options")} <span className="font-normal text-muted-foreground">({t("chat.optionsHint")})</span></Label>
              <Input dir="auto" defaultValue={(q.labels ?? []).join(", ")} onBlur={(e) => onChange({ ...q, labels: e.target.value.split(",").map((s) => s.trim()).filter(Boolean) })} /></div>
          )}
        </PopoverContent>
      </Popover>
      <button type="button" onClick={onRemove} aria-label={t("chat.remove")} className="me-0.5 grid size-5 shrink-0 place-items-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground"><X className="size-3" /></button>
    </span>
  )
}

function ModelPicker({ model }: { model?: LibraryModel }) {
  const { t } = useI18n()
  const { lib } = useLibrary()
  const mine = (lib?.custom ?? []).filter((m) => m.status === "ready")
  return (
    <Select value={model?.id ?? ""} onValueChange={(id) => go({ name: "chat", model: id })}>
      <SelectTrigger size="sm" className="h-8 max-w-52 border-0 bg-transparent ps-1 shadow-none hover:bg-muted dark:bg-transparent" aria-label={t("chat.model")}>
        <SelectValue placeholder={t("chat.model")} />
      </SelectTrigger>
      <SelectContent align="start">
        {mine.length > 0 && (
          <SelectGroup>
            <SelectLabel>{t("app.yourModels")}</SelectLabel>
            {mine.map((m) => <SelectItem key={m.id} value={m.id}><StatusDot status={m.status} /> <span dir="auto">{m.name}</span></SelectItem>)}
          </SelectGroup>
        )}
        <SelectGroup>
          <SelectLabel>{t("app.base")}</SelectLabel>
          {(lib?.base ?? []).map((m) => <SelectItem key={m.id} value={m.id}>{m.name} <span className="text-muted-foreground">{m.params}</span></SelectItem>)}
        </SelectGroup>
      </SelectContent>
    </Select>
  )
}

/* ------------------------------------------------------------------ page */

export function ChatPage({ modelId }: { modelId?: string }) {
  const { t, lang } = useI18n()
  const { lib, find, refresh } = useLibrary()
  const [turns, setTurns] = useState<Turn[]>(load)
  const [caseText, setCaseText] = useState("")
  const [caseCard, setCaseCard] = useState(false) // collapsed into a Case card
  const [questions, setQuestions] = useState<AskQuestion[]>([])
  const [draft, setDraft] = useState("")
  const [parsing, setParsing] = useState(false)
  const [compare, setCompare] = useState(true)
  const [examples, setExamples] = useState<string[]>([])
  const end = useRef<HTMLDivElement>(null)
  const caseInput = useRef<HTMLTextAreaElement>(null)

  const ready = useMemo(() => [...(lib?.custom ?? []).filter((m) => m.status === "ready"), ...(lib?.base ?? [])], [lib])
  const model: LibraryModel | undefined = find(modelId) ?? ready[0]
  const custom = model?.kind === "custom"
  const trained = (model?.decisions ?? []).map(({ question, type, labels }) => ({ question, type, labels }))
  const shown = turns.filter((x) => x.model === model?.id)
  const canSend = Boolean(model && caseText.trim() && questions.length)

  useEffect(() => { localStorage.setItem(STORE, JSON.stringify(turns.slice(-50))) }, [turns])
  // braces: scrollIntoView returns a Promise in newer Chromium, which React would call as an effect cleanup
  // on a new turn or a new answer only: a correction on an older card must not jump to the bottom
  const settled = turns.filter((x) => x.result || x.error).length
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }) }, [turns.length, settled])
  useEffect(() => {
    if (!model) return
    setQuestions(model.kind === "custom" && trained.length ? trained : [])
    library.get(model.id).then((m) => setExamples(m.examples ?? [])).catch(() => setExamples([]))
  }, [model?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const setCase = (text: string) => {
    setCaseText(text)
    setCaseCard(text.length > COLLAPSE_AT || text.trim().includes("\n"))
  }

  async function addQuestion() {
    const q = draft.trim()
    if (!q) return
    setParsing(true)
    try {
      const p = await library.parseQuestion(q, lang)
      setQuestions((qs) => [...qs, { question: p.question || q, type: p.type, labels: p.labels }])
    } catch {
      setQuestions((qs) => [...qs, { question: q, type: "noul", labels: ["no", "yes"] }])
    } finally {
      setDraft("")
      setParsing(false)
    }
  }

  async function send() {
    if (!canSend || !model) return
    const turn: Turn = { id: crypto.randomUUID(), model: model.id, modelName: model.name, case: caseText.trim(), questions, compare: custom && compare }
    setTurns((ts) => [...ts, turn])
    setCaseText("")
    setCaseCard(false)
    try {
      const result = await library.ask({ model: model.id, case: turn.case, questions, compare_with_base: turn.compare })
      setTurns((ts) => ts.map((x) => (x.id === turn.id ? { ...x, result } : x)))
    } catch (e) {
      setTurns((ts) => ts.map((x) => (x.id === turn.id ? { ...x, error: msg(e) } : x)))
    }
  }

  const setFix = (id: string, i: number, fix?: Fix) =>
    setTurns((ts) => ts.map((x) => {
      if (x.id !== id) return x
      const { [i]: _, ...rest } = x.fixes ?? {}
      return { ...x, fixes: fix ? { ...rest, [i]: fix } : rest }
    }))

  async function saveFix(turn: Turn, i: number, answer: string) {
    const q = turn.questions[i], r = turn.result!.results[i]
    try {
      const { count, n } = await corrections.add(turn.model, {
        case: turn.case, question: q.question, type: q.type, labels: optionsOf(q, r), answer, model_answer: r.argmax, p: r.probs[r.argmax],
      })
      setFix(turn.id, i, { answer, n })
      refresh()
      toast.success(count === 1 ? t("chat.correctedOne", { name: turn.modelName }) : t("chat.corrected", { n: count, name: turn.modelName }),
        { action: { label: t("chat.undo"), onClick: () => undoFix(turn, i, n) } })
    } catch (e) {
      toast.error(msg(e))
    }
  }

  // ponytail: undo deletes by line index; a correction removed elsewhere in between shifts it. Delete by (case, question) if that bites.
  async function undoFix(turn: Turn, i: number, n = turn.fixes?.[i]?.n) {
    if (n === undefined) return
    try {
      await corrections.remove(turn.model, n)
      setFix(turn.id, i)
      refresh()
      toast(t("chat.undone"))
    } catch (e) {
      toast.error(msg(e))
    }
  }

  const composer = (
    <div className="grid min-w-0 overflow-hidden rounded-xl border bg-card shadow-sm transition-shadow focus-within:shadow-md focus-within:ring-3 focus-within:ring-ring/20">
      {/* one inset for everything: 16px, the same as the answer cards (p-4); the controls divider runs full width */}
      {/* the case: a card once pasted, a textarea while typing */}
      {caseCard ? (
        <div className="px-4 pt-4"><CaseCard text={caseText} onEdit={() => { setCaseCard(false); setTimeout(() => caseInput.current?.focus()) }} onRemove={() => setCase("")} /></div>
      ) : (
        <Textarea
          ref={caseInput} dir="auto" value={caseText} rows={2} placeholder={t("chat.casePlaceholder")}
          onChange={(e) => setCaseText(e.target.value)}
          onPaste={(e) => {
            const pasted = e.clipboardData.getData("text")
            if (!caseText && (pasted.length > COLLAPSE_AT || pasted.includes("\n"))) { e.preventDefault(); setCase(pasted) }
          }}
          onBlur={() => { if (caseText) setCase(caseText) }}
          onKeyDown={(e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); send() } }}
          className="max-h-48 min-h-0 resize-none rounded-none border-0 bg-transparent px-4 pt-4 pb-1 shadow-none focus-visible:ring-0 dark:bg-transparent"
        />
      )}

      {/* the questions: chips plus a tag-style input on one line */}
      <div className="flex flex-wrap items-center gap-1 px-4 pt-2 pb-4">
        {questions.map((q, i) => (
          <QuestionChip key={i} q={q}
            onChange={(nq) => setQuestions((qs) => qs.map((x, j) => (j === i ? nq : x)))}
            onRemove={() => setQuestions((qs) => qs.filter((_, j) => j !== i))} />
        ))}
        <span className="relative flex min-w-40 flex-1 items-center">
          <input
            value={draft} onChange={(e) => setDraft(e.target.value)} placeholder={t("chat.addQuestion")} aria-label={t("chat.addQuestion")}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.metaKey && !e.ctrlKey) { e.preventDefault(); addQuestion() }
              if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); send() }
              if (e.key === "Backspace" && !draft && questions.length) setQuestions((qs) => qs.slice(0, -1))
            }}
            className={cn("h-6 w-full min-w-0 bg-transparent text-xs outline-none placeholder:text-muted-foreground", questions.length ? "px-1.5" : "px-0")}
          />
          {parsing && <Loader2 className="absolute end-1 size-3.5 animate-spin text-muted-foreground" />}
        </span>
      </div>

      {/* controls row */}
      <div className="flex items-center gap-1 border-t ps-3 pe-4 py-2">
        <ModelPicker model={model} />
        {custom && (
          <Label className="flex h-8 items-center gap-2 rounded-md px-2 text-sm font-normal text-muted-foreground hover:bg-muted">
            <Switch size="sm" checked={compare} onCheckedChange={setCompare} /> {t("chat.compareShort")}
          </Label>
        )}
        {custom && trained.length > 0 && JSON.stringify(questions) !== JSON.stringify(trained) && (
          <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={() => setQuestions(trained)} title={t("chat.trained")}>
            <ListChecks data-icon="inline-start" /> <span className="hidden md:inline">{t("chat.trained")}</span>
          </Button>
        )}
        <Button size="icon-sm" className="ms-auto rounded-full" onClick={send} disabled={!canSend}
          aria-label={`${t("chat.send")} (${t("chat.sendHint")})`} title={!questions.length ? t("chat.noQuestions") : t("chat.sendHint")}>
          <ArrowUp />
        </Button>
      </div>
    </div>
  )

  return (
    <div className="relative flex h-full flex-col">
      {shown.length > 0 && (
        <HeaderActions>
          <Button variant="ghost" size="icon-sm" onClick={() => setTurns((ts) => ts.filter((x) => x.model !== model?.id))} aria-label={t("chat.clear")} title={t("chat.clear")}>
            <Trash2 />
          </Button>
        </HeaderActions>
      )}

      {!shown.length ? (
        /* empty: everything centred, composer in the middle (ChatGPT / Gemini / Perplexity) */
        <div className="flex flex-1 items-center overflow-y-auto">
          <div className="mx-auto grid w-full max-w-3xl gap-6 px-4 py-10 md:px-6">
            <div className="grid justify-items-center gap-3 text-center">
              <div className="grid size-10 place-items-center rounded-xl bg-primary/10"><Sparkles className="size-5 text-primary" /></div>
              <h1 dir="auto" className="text-xl font-semibold tracking-tight">{model ? t("chat.emptyTitle", { name: model.name }) : t("chat.title")}</h1>
              <p className="max-w-lg text-sm text-muted-foreground">{t("chat.emptyBody")}</p>
            </div>
            {composer}
            {examples.length > 0 && (
              <div className="grid gap-2 sm:grid-cols-3">
                {examples.slice(0, 3).map((ex, i) => (
                  <button key={i} type="button" onClick={() => setCase(ex)}
                    className="grid content-start gap-1 rounded-xl border bg-card p-3 text-start text-xs text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground">
                    <span className="font-medium text-foreground">{t("chat.try")}</span>
                    <span dir="auto" className="line-clamp-3">{ex.split("\n\n").at(-1)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <>
          {/* conversation: one column; the composer floats at its foot with a fade, same width (Conductor / ChatGPT) */}
          <div className="flex-1 overflow-y-auto">
            <div className="mx-auto grid max-w-3xl gap-8 px-4 pt-8 pb-6 md:px-6">
              {shown.map((turn) => (
                <TurnView key={turn.id} turn={turn} onFix={custom ? (i, a) => saveFix(turn, i, a) : undefined} onUndo={(i) => undoFix(turn, i)} />
              ))}
              <div ref={end} />
            </div>
          </div>
          <div className="relative shrink-0">
            <div aria-hidden className="pointer-events-none absolute inset-x-0 -top-8 h-8 bg-gradient-to-t from-background to-transparent" />
            <div className="mx-auto max-w-3xl px-4 pb-4 md:px-6">{composer}</div>
          </div>
        </>
      )}
    </div>
  )
}
