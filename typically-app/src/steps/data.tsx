import { useEffect, useRef, useState } from "react"
import { AlertCircle, AlertTriangle, ArrowRight, Check, ChevronDown, CircleHelp, Loader2, Upload } from "lucide-react"

import { msg, PageHead, StepFooter } from "@/components/shared"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, type Source } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { SAMPLE, say, type Analysis, type Intent, type PlanDecision, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"
import { ReviewDetails } from "@/steps/understand"

const TABLE = /\.(csv|xlsx)$/i
const MAIL = /\.(mbox|eml|zip)$/i
const MAIL_MB = 50 // site/typically_analyze.py MAX_MAIL_MB
const ACCEPT = ".csv,text/csv,.xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,.mbox,.eml,.zip,message/rfc822,application/mbox,application/zip"
const SAMPLES = [["northwind", "sample.title"], ["enron", "sample.enronTitle"]] as const
// the worst first: what to show of the data-health notes when only two fit
const SERIOUS = ["leakage", "conflicts", "pii", "empty", "rare_labels", "duplicates", "long_cases"]

const base64 = (f: File) => new Promise<string>((ok, fail) => {
  const r = new FileReader()
  r.onload = () => ok(String(r.result).split(",")[1] ?? "")
  r.onerror = () => fail(r.error)
  r.readAsDataURL(f)
})

/** "sample" / "sample-enron" / a records_token, as the Data page links here */
const fromToken = (token: string): Source =>
  token === "sample" ? { kind: "sample" } : token.startsWith("sample-") ? { kind: "sample", name: token.slice(7) as "enron" } : { kind: "upload", token }

function DropZone({ onPick }: { onPick: (f?: File) => void }) {
  const { t } = useI18n()
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [ai, setAi] = useState(false) // with an AI key, a summary of the columns and a few masked rows go to the model that reads them
  useEffect(() => void api.capabilities().then((c) => setAi(c.ai)).catch(() => setAi(false)), [])
  return (
    <div
      role="button" tabIndex={0} aria-label={t("add.drop")}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), input.current?.click())}
      onDragOver={(e) => (e.preventDefault(), setOver(true))}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => (e.preventDefault(), setOver(false), onPick(e.dataTransfer.files[0]))}
      className={cn("flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
        over ? "border-primary bg-primary/5" : "hover:bg-muted/50")}
    >
      <Upload className="size-6 text-muted-foreground" />
      <p className="max-w-md text-base font-medium">{t("add.drop")}</p>
      <span aria-hidden className={buttonVariants({ variant: "outline" })}>{t("add.choose")}</span>
      <p className="max-w-lg text-xs text-muted-foreground">{t(ai ? "upload.privateAi" : "upload.private")}</p>
      <input ref={input} type="file" accept={ACCEPT} className="sr-only" tabIndex={-1} onChange={(e) => onPick(e.target.files?.[0])} />
    </div>
  )
}

/** One decision the model would learn: a checkbox, the question in plain words, its answers, and where it came from. */
function DecisionRow({ d, set, note }: { d: PlanDecision; set: (include: boolean) => void; note: React.ReactNode }) {
  const { t } = useI18n()
  const more = d.labels.length - 8
  return (
    <Label className="flex cursor-pointer items-start gap-3 rounded-lg border p-3 font-normal transition-colors hover:bg-muted/40 has-[[data-state=unchecked]]:opacity-70">
      <Checkbox checked={d.include} onCheckedChange={(v) => set(v === true)} className="mt-0.5" />
      <div className="grid min-w-0 gap-1.5">
        <span dir="auto" className="text-sm font-medium">{d.question}</span>
        <span className="flex flex-wrap gap-1">
          {d.labels.slice(0, 8).map((l) => <Badge key={l} variant="outline" className="font-normal"><bdi>{say(t, d.type, l)}</bdi></Badge>)}
          {more > 0 && <span className="text-xs text-muted-foreground">+{more}</span>}
        </span>
        <span className="text-xs text-muted-foreground">{note}</span>
      </div>
    </Label>
  )
}

/** Step 2: one drop zone (tables and mailboxes alike), read at once with step 1's intent, then a plain summary; the rest under Review details. */
export function DataStep({ project, setProject, intent, from, back, next }: {
  project: Project | null; setProject: React.Dispatch<React.SetStateAction<Project | null>>; intent: Intent; from?: string; back: () => void; next: () => void
}) {
  const { t, lang } = useI18n()
  const [busy, setBusy] = useState<string | null>(null) // what is being read
  const [error, setError] = useState<string | null>(null)
  const [other, setOther] = useState(false) // "Use other data": the drop zone again over a finished summary
  const [hf, setHf] = useState({ dataset: "", split: "train" })
  const [sheet, setSheet] = useState("")
  const [owner, setOwner] = useState(project?.analysis.mail?.owner ?? "")

  /** Read a source with the current intent; `keep`: re-reading the same data (new intent, another sheet / owner) keeps what it was. */
  async function read(src: Source, label: string, keep?: Pick<Project, "input" | "sample" | "source">) {
    setBusy(label)
    setError(null)
    try {
      const analysis: Analysis = await api.analyze(src, lang, intent)
      const sample = keep?.sample ?? (src.kind === "sample" && !src.name)   // Northwind: already trained
      if (sample) {
        const asked = new Set(analysis.intent_matches?.map((m) => m.column))
        for (const d of analysis.plan.decisions) {
          const s = SAMPLE.questions[d.column]
          if (s) Object.assign(d, { type: s.type, labels: s.labels }, asked.has(d.column) ? {} : { question: s.question })
        }
      }
      const sheetName = analysis.sheets && analysis.sheets.length > 1 ? ` (${analysis.sheet})` : ""
      setProject((prev) => ({
        base: prev?.base ?? "small", source: keep?.source ?? `${analysis.name_hint}${sheetName}`, input: keep?.input ?? src, intent, sample, analysis, plan: analysis.plan,
        enrich: prev ? { ...prev.enrich, policy: {} } : { balance: true, dedupe_soft: true, policy: {}, synthetic: false, languages: [] },
        settings: prev?.settings ?? { preset: "balanced", holdout: 20, seed: 0 },
        ...(sample ? { run: SAMPLE.run, resultsKey: SAMPLE.resultsKey } : {}),
      }))
      setOwner(analysis.mail?.owner ?? "")
      setOther(false)
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(null)
    }
  }

  // arriving: data picked on the Data page is read at once; data read for another intent (step 1 changed since) is read again
  useEffect(() => {
    if (!project && from) read(fromToken(from), from)
    else if (project && JSON.stringify(project.intent) !== JSON.stringify(intent))
      read({ kind: "upload", token: project.analysis.records_token }, project.source, project)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  async function pick(f?: File) {
    if (!f) return
    if (!TABLE.test(f.name) && !MAIL.test(f.name)) return setError(t("upload.fileTypes"))
    if (MAIL.test(f.name) && f.size > MAIL_MB * 1024 * 1024) return setError(t("mail.tooBig", { n: MAIL_MB }))
    try {   // a file that can't be read (moved, locked, permissions) is an error on screen, not an unhandled rejection
      const src: Source = MAIL.test(f.name) ? { kind: "mail", data_base64: await base64(f), name: f.name }
        : /\.xlsx$/i.test(f.name) ? { kind: "xlsx", data_base64: await base64(f), name: f.name } : { kind: "csv", text: await f.text(), name: f.name }
      read(src, f.name)
    } catch (e) {
      setError(msg(e))
    }
  }

  const a = project?.analysis
  const p = project
  const setDecision = (column: string, include: boolean) =>
    p && setProject({ ...p, plan: { ...p.plan, decisions: p.plan.decisions.map((d) => (d.column === column ? { ...d, include } : d)) } })
  const matches = a?.intent_matches ?? []
  const matched = new Set(matches.map((m) => m.column))
  const rest = p ? p.plan.decisions.filter((d) => !matched.has(d.column)) : []
  const chosen = p ? p.plan.decisions.filter((d) => d.include).length : 0
  const notes = p ? [...(a!.warnings ?? []), ...[...p.plan.issues].sort((x, y) => SERIOUS.indexOf(x.kind) - SERIOUS.indexOf(y.kind))
    .map((i) => (i.column ? `${i.column}: ${i.detail}` : i.detail))] : []
  const found = p?.plan.decisions.length ?? 0
  const from_ = (d: PlanDecision) => t("add.fromColumn", { c: d.column })

  return (
    <>
      <PageHead title={t("add.title")}>{t("add.lede")}</PageHead>

      {busy ? (
        <Card>
          <CardContent className="flex min-h-56 flex-col items-center justify-center gap-2 text-center" role="status">
            <Loader2 className="size-6 animate-spin text-primary" />
            <p className="text-base font-medium">{t("create.reading")}</p>
            <p dir="auto" className="text-sm text-muted-foreground">{busy}</p>
          </CardContent>
        </Card>
      ) : !p || other ? (
        <div className="grid gap-4">
          <DropZone onPick={pick} />
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <span>{t("add.trySample")}</span>
            {SAMPLES.map(([name, title]) => (
              <Button key={name} variant="secondary" size="sm" onClick={() => read(name === "northwind" ? { kind: "sample" } : { kind: "sample", name }, t(title))}>{t(title)}</Button>
            ))}
            {other && <Button variant="ghost" size="sm" className="ms-auto" onClick={() => setOther(false)}>{t("add.keep")}</Button>}
          </div>
          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="-ms-2 text-muted-foreground">{t("add.other")} <ChevronDown data-icon="inline-end" /></Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="grid gap-6 pt-2">
              <form className="grid gap-2" onSubmit={(e) => (e.preventDefault(), read({ kind: "sheets", url: sheet.trim() }, t("src.sheet")))}>
                <Label htmlFor="sheet">{t("tab.sheets")}</Label>
                <div className="flex gap-2">
                  <Input id="sheet" dir="ltr" placeholder="https://docs.google.com/spreadsheets/d/…" value={sheet} onChange={(e) => setSheet(e.target.value)} />
                  <Button type="submit" variant="outline" disabled={!/docs\.google\.com\/spreadsheets\/d\//.test(sheet)}>{t("add.read")}</Button>
                </div>
                <p className="text-xs text-muted-foreground">{t("sheets.hint")}</p>
              </form>
              <form className="grid gap-2" onSubmit={(e) => (e.preventDefault(), read({ kind: "hf", dataset: hf.dataset.trim(), split: hf.split || "train", limit: 5000 }, `Hugging Face: ${hf.dataset}`))}>
                <Label htmlFor="hf-ds">{t("tab.hf")}</Label>
                <div className="flex flex-wrap gap-2">
                  <Input id="hf-ds" dir="ltr" placeholder="owner/dataset" className="min-w-48 flex-1" value={hf.dataset} onChange={(e) => setHf({ ...hf, dataset: e.target.value })} />
                  <Input aria-label={t("hf.split")} dir="ltr" className="w-28" value={hf.split} onChange={(e) => setHf({ ...hf, split: e.target.value })} />
                  <Button type="submit" variant="outline" disabled={!hf.dataset.includes("/")}>{t("add.read")}</Button>
                </div>
                <p className="text-xs text-muted-foreground">{t("hf.hint")}</p>
              </form>
            </CollapsibleContent>
          </Collapsible>
        </div>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">
                {t(found === 1 ? "add.found1" : found ? "add.found" : "add.foundNone", { n: found, m: a!.n_rows.toLocaleString() })}
              </CardTitle>
              <CardDescription className="flex flex-wrap items-center gap-x-2">
                <span dir="auto" className="truncate">{p.source}</span>
                <Button variant="link" size="xs" className="h-auto px-0" onClick={() => setOther(true)}>{t("create.otherData")}</Button>
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-2">
              {matches.map((m) => {
                const d = m.column && p.plan.decisions.find((x) => x.column === m.column)
                return d ? (
                  <DecisionRow key={m.question} d={d} set={(v) => setDecision(d.column, v)}
                    note={<span className="flex items-center gap-1 text-success"><Check className="size-3.5" /> {t("add.answeredBy", { c: d.column })}</span>} />
                ) : (
                  <div key={m.question} className="flex items-start gap-3 rounded-lg border border-dashed p-3">
                    <CircleHelp className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                    <div className="grid min-w-0 gap-1">
                      <span dir="auto" className="text-sm font-medium">{m.question}</span>
                      <span className="text-sm">{t("add.notFound")}</span>
                      <span className="text-xs text-muted-foreground">{t("add.notFoundHint")}</span>
                    </div>
                  </div>
                )
              })}
              {matches.length > 0 && rest.length > 0 && <p className="eyebrow pt-2">{t("add.alsoFound")}</p>}
              {rest.map((d) => <DecisionRow key={d.column} d={d} set={(v) => setDecision(d.column, v)} note={from_(d)} />)}
            </CardContent>
          </Card>

          {notes.length > 0 && (
            <Alert>
              <AlertTriangle />
              <AlertDescription>
                {notes.slice(0, 2).map((n) => <p key={n} dir="auto">{n}</p>)}
                {notes.length > 2 && <p className="text-muted-foreground">{t("add.moreNotes", { n: notes.length - 2 })}</p>}
              </AlertDescription>
            </Alert>
          )}

          <Collapsible>
            <CollapsibleTrigger asChild>
              <Button variant="outline" className="group w-full justify-between sm:w-auto">
                {t("add.details")} <ChevronDown data-icon="inline-end" className="transition-transform group-data-[state=open]:rotate-180" />
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-4">
              <ReviewDetails project={p} update={(patch) => setProject({ ...p, ...patch })}>
                {p.input.kind === "xlsx" && (a!.sheets?.length ?? 0) > 1 && (
                  <div className="grid max-w-sm gap-1.5">
                    <Label htmlFor="xlsx-sheet">{t("create.sheet")}</Label>
                    <Select value={a!.sheet} onValueChange={(s) => read({ ...(p.input as Extract<Source, { kind: "xlsx" }>), sheet: s }, s)}>
                      <SelectTrigger id="xlsx-sheet" className="w-full"><SelectValue /></SelectTrigger>
                      <SelectContent>{a!.sheets!.map((n) => <SelectItem key={n} value={n}><span dir="auto">{n}</span></SelectItem>)}</SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">{t("create.sheetHint", { n: a!.sheets!.length, sheet: a!.sheet ?? "" })}</p>
                  </div>
                )}
                {a!.mail && p.input.kind === "mail" && (
                  <form className="grid max-w-md gap-1.5" onSubmit={(e) => (e.preventDefault(), read({ ...p.input, owner: owner.trim() } as Source, p.source))}>
                    <p className="text-sm tabular">
                      {t("mail.stats", { owner: `⁨${a!.mail.owner}⁩`, messages: a!.mail.messages.toLocaleString(), threads: a!.mail.threads.toLocaleString(), inbound: a!.mail.inbound.toLocaleString() })}
                    </p>
                    <Label htmlFor="mail-owner">{t("mail.owner")}</Label>
                    <div className="flex gap-2">
                      <Input id="mail-owner" dir="ltr" type="text" inputMode="email" value={owner} onChange={(e) => setOwner(e.target.value)} />
                      <Button type="submit" variant="outline" disabled={!owner.trim() || owner.trim() === a!.mail.owner}>{t("add.readAgain")}</Button>
                    </div>
                    <p className="text-xs text-muted-foreground">{t("mail.ownerHint")}</p>
                  </form>
                )}
              </ReviewDetails>
            </CollapsibleContent>
          </Collapsible>
        </>
      )}

      {error && <p role="alert" dir="auto" className="flex items-center gap-2 text-sm text-destructive"><AlertCircle className="size-4 shrink-0" /> {error}</p>}
      <StepFooter back={back}>
        {p && !other && !busy && (
          <Button onClick={next} disabled={!chosen || a!.n_rows < 100}>
            {a!.n_rows < 100 ? t("und.need100") : <>{t("common.continue")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></>}
          </Button>
        )}
      </StepFooter>
    </>
  )
}
