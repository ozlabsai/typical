import { useState } from "react"
import { ArrowRight, ListChecks, Loader2, Plus, Sparkles, X } from "lucide-react"

import { OptionCard, PageHead, StepFooter } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { library, type AskQuestion } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { MAX_QUESTIONS, say, type Intent } from "@/lib/project"

/** A question's answers as chips; choice / scale answers can be retyped as a comma list (yes/no stays yes/no). */
function Answers({ q, onChange }: { q: AskQuestion; onChange: (q: AskQuestion) => void }) {
  const { t } = useI18n()
  const [editing, setEditing] = useState(false)
  const labels = q.labels ?? []
  const commit = (text: string) => {
    const next = [...new Set(text.split(",").map((s) => s.trim().slice(0, 80)).filter(Boolean))].slice(0, 20)
    if (next.length >= 2) onChange({ ...q, labels: next })
    setEditing(false)
  }
  if (editing)
    return (
      <Input autoFocus dir="auto" defaultValue={labels.join(", ")} aria-label={t("what.answersEdit")} className="h-8 text-sm"
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") commit(e.currentTarget.value); if (e.key === "Escape") setEditing(false) }} />
    )
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs text-muted-foreground">
      <span>{t("what.answers")}</span>
      {labels.map((l) => <Badge key={l} variant="outline" className="font-normal"><bdi>{say(t, q.type, l)}</bdi></Badge>)}
      {q.type !== "noul" && <Button variant="link" size="xs" className="h-auto px-1 text-xs" onClick={() => setEditing(true)}>{t("what.edit")}</Button>}
    </div>
  )
}

/** Step 1: learn every decision in the data, or the user's own questions (each read by /parse_question into a type + answers). */
export function WhatStep({ intent, setIntent, next }: { intent: Intent; setIntent: React.Dispatch<React.SetStateAction<Intent>>; next: () => void }) {
  const { t, lang } = useI18n()
  const [draft, setDraft] = useState("")
  const [parsing, setParsing] = useState(false)
  const questions = intent.mode === "questions"
  const full = intent.questions.length >= MAX_QUESTIONS
  const setQuestions = (f: (qs: AskQuestion[]) => AskQuestion[]) => setIntent((i) => ({ ...i, questions: f(i.questions) }))

  async function add() {
    const text = draft.trim()
    if (!text || parsing || full) return
    setParsing(true)
    const q = await library.parseQuestion(text, lang).then((p) => ({ question: p.question || text, type: p.type, labels: p.labels }),
      () => ({ question: text, type: "noul" as const, labels: ["no", "yes"] }))   // offline: a yes/no question, its answers editable later
    setQuestions((qs) => [...qs, q])
    setDraft("")
    setParsing(false)
  }

  const ready = !questions || intent.questions.length > 0 || Boolean(draft.trim())
  async function go() {
    if (questions && draft.trim()) await add()   // typed but not yet added: Continue adds it
    next()
  }

  return (
    <>
      <PageHead title={t("what.title")}>{t("what.lede")}</PageHead>
      <RadioGroup value={intent.mode} onValueChange={(v) => setIntent((i) => ({ ...i, mode: v as Intent["mode"] }))} className="grid gap-4 sm:grid-cols-2">
        {([["auto", Sparkles], ["questions", ListChecks]] as const).map(([m, Icon]) => (
          <Label key={m} htmlFor={`mode-${m}`} className="cursor-pointer font-normal">
            <OptionCard selected={intent.mode === m} className="h-full w-full p-5">
              <div className="flex items-start gap-3">
                <RadioGroupItem value={m} id={`mode-${m}`} className="mt-1" />
                <div className="grid gap-1">
                  <span className="flex items-center gap-2 text-base font-medium"><Icon className="size-4 text-primary" /> {t(`what.${m}`)}</span>
                  <span className="text-sm text-muted-foreground">{t(`what.${m}Body`)}</span>
                </div>
              </div>
            </OptionCard>
          </Label>
        ))}
      </RadioGroup>

      {questions && (
        <Card>
          <CardContent className="grid gap-4">
            {intent.questions.length > 0 && (
              <ol className="grid gap-3">
                {intent.questions.map((q, i) => (
                  <li key={i} className="grid gap-1.5 rounded-lg border p-3">
                    <div className="flex items-start gap-2">
                      <span dir="auto" className="min-w-0 flex-1 text-sm font-medium">{q.question}</span>
                      <Button variant="ghost" size="icon-xs" aria-label={t("chat.remove")} onClick={() => setQuestions((qs) => qs.filter((_, j) => j !== i))}><X /></Button>
                    </div>
                    <Answers q={q} onChange={(nq) => setQuestions((qs) => qs.map((x, j) => (j === i ? nq : x)))} />
                  </li>
                ))}
              </ol>
            )}
            <div className="flex gap-2">
              <Input dir="auto" value={draft} maxLength={300} disabled={full} autoFocus={!intent.questions.length}
                onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); add() } }}
                placeholder={t(full ? "what.full" : "what.placeholder", { n: MAX_QUESTIONS })} aria-label={t("what.add")} />
              <Button variant="outline" onClick={add} disabled={!draft.trim() || parsing || full}>
                {parsing ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Plus data-icon="inline-start" />} {t("what.add")}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid max-w-xl gap-1.5">
        <Label htmlFor="topic">{t("what.topic")} <span className="font-normal text-muted-foreground">({t("enr.optional")})</span></Label>
        <Input id="topic" dir="auto" maxLength={120} value={intent.topic} placeholder={t("what.topicPlaceholder")}
          onChange={(e) => setIntent((i) => ({ ...i, topic: e.target.value }))} />
      </div>

      <StepFooter>
        <Button onClick={go} disabled={!ready || parsing}>{t("common.continue")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>
      </StepFooter>
    </>
  )
}
