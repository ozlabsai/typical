import { useEffect, useRef, useState } from "react"
import { AlertCircle, ArrowRight, Check, Circle, Copy, Download, FileSpreadsheet, Loader2, Upload, X } from "lucide-react"
import { toast } from "sonner"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { api, downloadUrl, results, type Compare, type DecisionType, type Reveal, type TrainStatus } from "@/lib/api"
import {
  countAnswers, defaultQuestion, draftFromPreview, isYesNo, labelsFor, pct, queries, SAMPLE, say,
  type DecisionDraft, type Project,
} from "@/lib/project"
import { cn } from "@/lib/utils"

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))

function PageHead({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{children}</p>
    </div>
  )
}

function Meter({ value, tone }: { value: number; tone: "standard" | "yours" }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-10 text-right font-mono text-sm tabular">{pct(value)}</span>
      <div className="h-1.5 w-full min-w-16 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full transition-[width] duration-200", tone === "standard" ? "bg-standard" : "bg-yours")} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

/* ---------------------------------------------------------------- 1. Data */

export function DataStep({ project, setProject, next }: { project: Project | null; setProject: (p: Project) => void; next: () => void }) {
  const [busy, setBusy] = useState(false)
  const [over, setOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  async function load(fileName: string, csvText: string, sample = false) {
    setBusy(true)
    setError(null)
    try {
      const preview = await api.preview(csvText)
      const { textCol, decisions } = draftFromPreview(preview, countAnswers(csvText, preview.columns))
      if (sample) for (const d of decisions) Object.assign(d, SAMPLE.questions[d.column] ?? {})
      setProject({
        name: sample ? SAMPLE.name : fileName.replace(/\.csv$/i, ""), fileName, csvText, preview, textCol, decisions, sample,
        ...(sample ? { run: SAMPLE.run, resultsKey: SAMPLE.resultsKey } : {}),
      })
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
    }
  }

  async function pick(file?: File) {
    if (!file) return
    if (!/\.csv$/i.test(file.name)) return setError("Only CSV for now. In Excel or Sheets: File, then Download or Save As, then CSV.")
    await load(file.name, await file.text())
  }

  async function sample() {
    setBusy(true)
    const res = await fetch("/data/typically_sample.csv").catch(() => null)
    if (!res?.ok) return setBusy(false), setError("Couldn't load the sample. Is the local server running?")
    await load("northwind_tickets.csv", await res.text(), true)
  }

  function setTextCol(textCol: string) {
    if (!project) return
    const { decisions } = draftFromPreview({ ...project.preview, columns: project.preview.columns.filter((c) => c !== textCol).concat(textCol) }, countAnswers(project.csvText, project.preview.columns))
    const keep = new Map(project.decisions.map((d) => [d.column, d]))
    setProject({ ...project, textCol, decisions: decisions.filter((d) => d.column !== textCol).map((d) => keep.get(d.column) ?? d) })
  }

  return (
    <>
      <PageHead title="Bring your past decisions">
        A CSV with one row per case. One column describes the case (a ticket, a request, a report); every other column is something you decided about it.
      </PageHead>
      {!project ? (
        <Card>
          <CardContent className="grid gap-4 md:grid-cols-[1fr_280px]">
            <div
              role="button"
              tabIndex={0}
              onClick={() => input.current?.click()}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
              onDragOver={(e) => (e.preventDefault(), setOver(true))}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => (e.preventDefault(), setOver(false), pick(e.dataTransfer.files[0]))}
              className={cn(
                "flex min-h-52 cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-8 text-center transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                over ? "border-primary bg-primary/5" : "hover:bg-muted/50",
              )}
            >
              <div className="grid size-10 place-items-center rounded-full bg-muted">
                {busy ? <Loader2 className="size-5 animate-spin" /> : <Upload className="size-5" />}
              </div>
              <div>
                <p className="text-sm font-medium">Drop a CSV here, or click to choose a file</p>
                <p className="mt-1 text-xs text-muted-foreground">Stays on this machine until you start training.</p>
              </div>
              <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => pick(e.target.files?.[0])} />
            </div>
            <div className="flex flex-col justify-between gap-4 rounded-lg bg-muted/50 p-5">
              <div>
                <p className="text-sm font-medium">No file handy?</p>
                <p className="mt-1 text-sm text-muted-foreground">Try Northwind Freight: 277 support tickets and the four calls their team made on each.</p>
              </div>
              <Button variant="outline" onClick={sample} disabled={busy}>
                <FileSpreadsheet data-icon="inline-start" /> Use the sample
              </Button>
            </div>
          </CardContent>
          {error && (
            <CardFooter>
              <p role="alert" className="flex items-center gap-2 text-sm text-destructive"><AlertCircle className="size-4" /> {error}</p>
            </CardFooter>
          )}
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><FileSpreadsheet className="size-4 text-muted-foreground" /> {project.fileName}</CardTitle>
            <CardDescription className="tabular">{project.preview.n} rows · {project.preview.columns.length} columns</CardDescription>
            <CardAction>
              <Button variant="ghost" size="sm" onClick={() => input.current?.click()}>Replace file</Button>
              <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => pick(e.target.files?.[0])} />
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-5">
            <div className="grid max-w-sm gap-2">
              <Label htmlFor="textcol">Which column describes each case?</Label>
              <Select value={project.textCol} onValueChange={setTextCol}>
                <SelectTrigger id="textcol" className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {project.preview.columns.map((c) => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
            <div className="overflow-hidden rounded-lg border">
              <Table>
                <TableHeader>
                  <TableRow className="bg-muted/50">
                    {project.preview.columns.map((c) => (
                      <TableHead key={c} className={cn(c === project.textCol && "w-[55%]")}>
                        {c}{c === project.textCol && <Badge variant="secondary" className="ml-2">case</Badge>}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {project.preview.rows.map((r, i) => (
                    <TableRow key={i}>
                      {project.preview.columns.map((c) => (
                        <TableCell key={c} className={cn("align-top", c === project.textCol ? "max-w-0 whitespace-normal" : "whitespace-nowrap")}>
                          <span className={cn(c === project.textCol && "line-clamp-2 text-muted-foreground")}>{r[c]}</span>
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
            {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
          </CardContent>
          <CardFooter className="justify-end">
            <Button onClick={next} disabled={!project.decisions.length}>Continue <ArrowRight data-icon="inline-end" /></Button>
          </CardFooter>
        </Card>
      )}
    </>
  )
}

/* ---------------------------------------------------------------- 2. Decisions */

const TYPES: { value: DecisionType; label: string }[] = [
  { value: "choice", label: "Pick one" },
  { value: "noul", label: "Yes or no" },
  { value: "score", label: "Scale" },
]

export function DecisionsStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
  const set = (i: number, patch: Partial<DecisionDraft>) =>
    update({ decisions: project.decisions.map((d, j) => (j === i ? { ...d, ...patch } : d)) })

  function setType(i: number, d: DecisionDraft, type: DecisionType) {
    const values = project.preview.values[d.column]
    const question = d.question === defaultQuestion(d.column, d.type) ? defaultQuestion(d.column, type) : d.question
    set(i, { type, labels: labelsFor(type, values), question })
  }

  const rare = project.decisions.filter((d) => d.include).flatMap((d) =>
    Object.entries(d.counts).filter(([, n]) => n < 10).map(([label, n]) => `${d.column}: “${label}” (${n})`))
  const chosen = project.decisions.filter((d) => d.include).length

  return (
    <>
      <PageHead title="What did you decide?">
        Each column becomes a question your Typical learns to answer. Word it the way you would ask a colleague. The answers come from your file.
      </PageHead>
      <Card>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-10 pl-6"><span className="sr-only">Include</span></TableHead>
                <TableHead className="w-32">Column</TableHead>
                <TableHead className="w-[38%]">Question</TableHead>
                <TableHead className="w-36">Answer type</TableHead>
                <TableHead className="pr-6">Answers in your file</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {project.decisions.map((d, i) => (
                <TableRow key={d.column} className={cn(!d.include && "text-muted-foreground")}>
                  <TableCell className="pl-6">
                    <Checkbox checked={d.include} onCheckedChange={(v) => set(i, { include: v === true })} aria-label={`Learn ${d.column}`} />
                  </TableCell>
                  <TableCell className="font-medium">{d.column}</TableCell>
                  <TableCell>
                    <Input value={d.question} disabled={!d.include} onChange={(e) => set(i, { question: e.target.value })} aria-label={`Question for ${d.column}`} />
                  </TableCell>
                  <TableCell>
                    <Select value={d.type} disabled={!d.include} onValueChange={(v) => setType(i, d, v as DecisionType)}>
                      <SelectTrigger className="w-full" aria-label={`Answer type for ${d.column}`}><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {TYPES.map((t) => (
                          <SelectItem key={t.value} value={t.value}
                            disabled={(t.value === "noul" && !isYesNo(project.preview.values[d.column])) || (t.value === "score" && !project.preview.values[d.column].every((v) => /^-?\d+$/.test(v)))}>
                            {t.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell className="pr-6">
                    <div className="flex flex-wrap gap-1.5">
                      {Object.entries(d.counts).sort((a, b) => (d.type === "score" ? Number(a[0]) - Number(b[0]) : b[1] - a[1])).map(([label, n]) => (
                        <Badge key={label} variant="outline" className="font-normal">
                          {label}<span className="text-muted-foreground tabular">{n}</span>
                        </Badge>
                      ))}
                    </div>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
        <CardFooter className="justify-between">
          <Button variant="ghost" onClick={back}>Back</Button>
          <div className="flex items-center gap-4">
            <span className="text-sm text-muted-foreground tabular">{chosen} of {project.decisions.length} selected</span>
            <Button onClick={next} disabled={!chosen}>Continue <ArrowRight data-icon="inline-end" /></Button>
          </div>
        </CardFooter>
      </Card>
      {rare.length > 0 && (
        <Alert className="mt-4">
          <AlertCircle />
          <AlertTitle>Some answers are rare</AlertTitle>
          <AlertDescription>
            {rare.join(", ")}. They get extra weight while it learns, but with fewer than 10 examples expect those answers to be the least reliable.
          </AlertDescription>
        </Alert>
      )}
    </>
  )
}

/* ---------------------------------------------------------------- 3. Train */

const PHASE_STEPS: { phase: TrainStatus["phase"]; label: string }[] = [
  { phase: "starting_gpu", label: "Starting a GPU" },
  { phase: "uploading", label: "Uploading your data" },
  { phase: "training", label: "Learning from your decisions" },
  { phase: "evaluating", label: "Testing on cases it hasn't seen" },
  { phase: "downloading", label: "Bringing your model back" },
]
const ORDER = ["queued", "starting_gpu", "uploading", "training", "evaluating", "downloading", "done"]

function useElapsed(since?: string) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!since) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [since])
  if (!since) return ""
  const s = Math.max(0, Math.floor((now - Date.parse(since)) / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}

export function TrainStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
  const [status, setStatus] = useState<TrainStatus | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [name, setName] = useState(project.name)
  const elapsed = useElapsed(status && status.phase !== "done" && status.phase !== "failed" ? status.started_at : undefined)
  const live = status && status.phase !== "done" && status.phase !== "failed"

  useEffect(() => {
    if (!project.slug || !live) return
    const t = setInterval(async () => {
      try {
        const s = await api.trainStatus(project.slug!)
        setStatus(s)
        if (s.phase === "done") {
          update({ run: (s.run as string) ?? `co_${project.slug}`, resultsKey: project.slug })
          toast.success(`${project.name}'s Typical is ready`)
        }
      } catch (e) {
        setError(msg(e))
      }
    }, 4000)
    return () => clearInterval(t)
  }, [project.slug, live]) // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setStarting(true)
    setError(null)
    try {
      const decisions = project.decisions.filter((d) => d.include).map(({ column, question, type }) => ({ column, question, type }))
      const built = await api.build({ csv_text: project.csvText, text_col: project.textCol, decisions, name })
      update({ name, slug: built.job, run: undefined })
      setStatus(await api.train(name))
    } catch (e) {
      setError(msg(e))
    } finally {
      setStarting(false)
    }
  }

  if (project.sample)
    return (
      <>
        <PageHead title="Train your Typical">Northwind's model is already trained, so the sample skips the wait.</PageHead>
        <Card>
          <CardHeader>
            <CardTitle>Northwind's Typical is ready</CardTitle>
            <CardDescription>
              It learned from Northwind's tickets with the same recipe your own file gets: about 10 minutes on one rented GPU. A fifth of the tickets were kept back so the results are measured on cases it never saw.
            </CardDescription>
          </CardHeader>
          <CardFooter className="justify-between">
            <Button variant="ghost" onClick={back}>Back</Button>
            <Button onClick={next}>See the results <ArrowRight data-icon="inline-end" /></Button>
          </CardFooter>
        </Card>
      </>
    )

  const n = project.preview.n
  const at = status ? ORDER.indexOf(status.phase) : -1
  return (
    <>
      <PageHead title="Train your Typical">
        We rent a GPU, teach a copy of Typical your {project.decisions.filter((d) => d.include).length} decisions, test it on cases it never saw, and shut the GPU down.
      </PageHead>
      <div className="grid gap-4 lg:grid-cols-[1fr_380px]">
        <Card>
          <CardHeader>
            <CardTitle>{status ? "Training" : "Ready to train"}</CardTitle>
            <CardDescription>{status ? status.message : "Nothing is sent anywhere until you press Start."}</CardDescription>
            {elapsed && <CardAction><span className="font-mono text-sm text-muted-foreground tabular">{elapsed}</span></CardAction>}
          </CardHeader>
          <CardContent>
            {!status ? (
              <div className="grid max-w-sm gap-2">
                <Label htmlFor="name">Name</Label>
                <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
                <p className="text-xs text-muted-foreground">Used for your model's file name. Private to you.</p>
              </div>
            ) : (
              <ol className="space-y-4">
                {PHASE_STEPS.map((p) => {
                  const idx = ORDER.indexOf(p.phase)
                  const state = status.phase === "failed" && idx === Math.max(at, 1) ? "failed" : idx < at || status.phase === "done" ? "done" : idx === at ? "current" : "pending"
                  return (
                    <li key={p.phase} className="flex gap-3">
                      <span className="mt-0.5">
                        {state === "done" ? <Check className="size-4 text-primary" /> : state === "current" ? <Loader2 className="size-4 animate-spin" /> : state === "failed" ? <X className="size-4 text-destructive" /> : <Circle className="size-4 text-muted-foreground/50" />}
                      </span>
                      <div className="flex-1">
                        <p className={cn("text-sm", state === "pending" && "text-muted-foreground")}>{p.label}</p>
                        {p.phase === "training" && state === "current" && typeof status.progress === "number" && (
                          <Progress value={(status.progress as number) * 100} className="mt-2 h-1.5" aria-label="Training progress" />
                        )}
                      </div>
                    </li>
                  )
                })}
              </ol>
            )}
            {(error || status?.phase === "failed") && (
              <Alert variant="destructive" className="mt-5">
                <AlertCircle />
                <AlertTitle>Training stopped</AlertTitle>
                <AlertDescription>{error ?? status?.message} The GPU was shut down, so nothing is still running or billing.</AlertDescription>
              </Alert>
            )}
          </CardContent>
          <CardFooter className="justify-between">
            <Button variant="ghost" onClick={back} disabled={Boolean(live)}>Back</Button>
            {status?.phase === "done" ? (
              <Button onClick={next}>See the results <ArrowRight data-icon="inline-end" /></Button>
            ) : (
              <Button onClick={start} disabled={starting || Boolean(live) || !name.trim()}>
                {starting && <Loader2 data-icon="inline-start" className="animate-spin" />}
                {status?.phase === "failed" || error ? "Try again" : "Start training"}
              </Button>
            )}
          </CardFooter>
        </Card>
        <Card size="sm" className="self-start">
          <CardHeader><CardTitle className="text-sm">What happens</CardTitle></CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
              <dt className="text-muted-foreground">Learns from</dt><dd className="tabular">about {Math.round(n * 0.7)} cases</dd>
              <dt className="text-muted-foreground">Tested on</dt><dd className="tabular">{Math.round(n * 0.2)} cases it never sees</dd>
              <dt className="text-muted-foreground">Runs on</dt><dd>one rented H100 GPU</dd>
              <dt className="text-muted-foreground">Takes</dt><dd>about 10 minutes</dd>
              <dt className="text-muted-foreground">Costs</dt><dd>roughly $1 of GPU time</dd>
            </dl>
          </CardContent>
        </Card>
      </div>
    </>
  )
}

/* ---------------------------------------------------------------- 4. Results */

/** One row per distinct case text (sample bodies repeat), wins AND losses: 7 where yours was right, 3 where it wasn't. */
function mixed(rows: Reveal["disagreements"]) {
  const body = (c: Reveal["disagreements"][number]) => c.case.split("\n\n").at(-1)
  const unique = rows.filter((c, i) => rows.findIndex((o) => body(o) === body(c)) === i)
  const won = unique.filter((c) => c.yours.answer === c.decided), lost = unique.filter((c) => c.yours.answer !== c.decided)
  return [...won.slice(0, 10 - Math.min(3, lost.length)), ...lost.slice(0, 3)]
}

export function ResultsStep({ project, next }: { project: Project; next: () => void }) {
  const [data, setData] = useState<Reveal | null>(null)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const r = await results(project.resultsKey!)
        if (!alive) return
        if ("status" in r) return setProgress({ done: r.done, total: r.total }), (timer = setTimeout(poll, 2500))
        setData(r)
      } catch (e) {
        if (alive) setError(msg(e))
      }
    }
    poll()
    return () => ((alive = false), clearTimeout(timer))
  }, [project.resultsKey])

  const who = project.sample ? "Northwind's" : "your"
  if (error)
    return (
      <Alert variant="destructive"><AlertCircle /><AlertTitle>Couldn't load the results</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>
    )
  if (!data)
    return (
      <>
        <PageHead title="Results">Scoring both models on the cases we kept back.</PageHead>
        <Card>
          <CardContent className="space-y-3">
            <p className="text-sm tabular">{progress ? `Scored ${progress.done} of ${progress.total} cases` : "Starting…"}</p>
            <Progress value={progress ? (progress.done / Math.max(1, progress.total)) * 100 : 0} className="h-1.5" />
            <div className="grid gap-4 pt-3 sm:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-24" />)}</div>
          </CardContent>
        </Card>
      </>
    )

  const gain = Math.round((data.score.yours - data.score.standard) * 100)
  const summary = `Standard Typical agrees with ${who} past decisions ${pct(data.score.standard)} of the time. ${project.sample ? "Northwind's own" : "Our own"} Typical: ${pct(data.score.yours)}. Measured on ${data.n_cases} cases neither model saw while learning.`

  return (
    <>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Results</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            How often each model gives the answer {project.sample ? "Northwind" : "you"} actually gave, on {data.n_cases} cases neither saw while learning ({data.n_answers} answers).
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => navigator.clipboard.writeText(summary).then(() => toast.success("Summary copied"))}>
            <Copy data-icon="inline-start" /> Copy summary
          </Button>
          <Button variant="outline" asChild>
            <a href={downloadUrl(project.run!)} download><Download data-icon="inline-start" /> Download model</a>
          </Button>
          <Button onClick={next}>Try it <ArrowRight data-icon="inline-end" /></Button>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card size="sm">
          <CardHeader><CardDescription>Standard Typical</CardDescription><p className="font-mono text-3xl font-medium tabular text-standard">{pct(data.score.standard)}</p></CardHeader>
          <CardContent className="text-sm text-muted-foreground">agrees with {who} decisions</CardContent>
        </Card>
        <Card size="sm" className="ring-1 ring-primary/30">
          <CardHeader><CardDescription>{project.sample ? "Northwind's" : "Your"} Typical</CardDescription><p className="font-mono text-3xl font-medium tabular text-yours">{pct(data.score.yours)}</p></CardHeader>
          <CardContent className="text-sm text-muted-foreground">agrees with {who} decisions</CardContent>
        </Card>
        <Card size="sm">
          <CardHeader><CardDescription>Difference</CardDescription><p className="font-mono text-3xl font-medium tabular">{gain >= 0 ? "+" : ""}{gain} pts</p></CardHeader>
          <CardContent className="text-sm text-muted-foreground tabular">fixed {data.fixed} answers, broke {data.broken}</CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>By decision</CardTitle>
          {data.abstained.standard > 0 && (
            <CardDescription>
              Standard Typical said “none of these fit” {data.abstained.standard} times; we count its best guess anyway.
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Question</TableHead>
                <TableHead className="w-[26%]"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-standard" />Standard</span></TableHead>
                <TableHead className="w-[26%]"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-yours" />{project.sample ? "Northwind's" : "Yours"}</span></TableHead>
                <TableHead className="w-20 pr-6 text-right">Change</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.decisions.map((d) => {
                const delta = Math.round((d.yours - d.standard) * 100)
                return (
                  <TableRow key={d.key}>
                    <TableCell className="pl-6 whitespace-normal">{d.question}</TableCell>
                    <TableCell><Meter value={d.standard} tone="standard" /></TableCell>
                    <TableCell><Meter value={d.yours} tone="yours" /></TableCell>
                    <TableCell className={cn("pr-6 text-right font-mono tabular", delta > 0 ? "text-yours" : "text-muted-foreground")}>
                      {delta > 0 ? "+" : ""}{delta}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Where they disagree</CardTitle>
          <CardDescription>Cases from the held-back set where the two models gave different answers, with what {project.sample ? "Northwind" : "you"} actually decided.</CardDescription>
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Case</TableHead>
                <TableHead>Question</TableHead>
                <TableHead>Standard</TableHead>
                <TableHead>{project.sample ? "Northwind's" : "Yours"}</TableHead>
                <TableHead className="pr-6">{project.sample ? "They decided" : "You decided"}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mixed(data.disagreements).map((c, i) => {
                const type = data.decisions.find((d) => d.key === c.key)?.type ?? "choice"
                const cell = (x: { answer: string; p: number }) => (
                  <span className="inline-flex items-center gap-1.5">
                    {x.answer === c.decided ? <Check className="size-3.5 text-yours" /> : <X className="size-3.5 text-muted-foreground" />}
                    <span className="font-medium">{say(type, x.answer)}</span>
                    <span className="font-mono text-xs text-muted-foreground tabular">{pct(x.p)}</span>
                  </span>
                )
                const body = c.case.split("\n\n").at(-1)
                return (
                  <TableRow key={i}>
                    <TableCell className="max-w-0 w-[38%] pl-6 whitespace-normal">
                      <span className="line-clamp-2 text-muted-foreground" title={c.case}>{body}</span>
                    </TableCell>
                    <TableCell className="whitespace-normal">{c.question}</TableCell>
                    <TableCell>{cell(c.standard)}</TableCell>
                    <TableCell>{cell(c.yours)}</TableCell>
                    <TableCell className="pr-6 font-medium">{say(type, c.decided)}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </>
  )
}

/* ---------------------------------------------------------------- 5. Playground */

export function PlaygroundStep({ project }: { project: Project }) {
  const first = project.preview.rows[0]?.[project.textCol] ?? ""
  const [text, setText] = useState(project.sample ? SAMPLE.tryCase : first)
  const [out, setOut] = useState<Compare | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mine = `local:${project.run}`
  const qs = queries(project)
  const decisions = project.decisions.filter((d) => d.include)

  async function run() {
    if (!text.trim()) return
    setBusy(true)
    setError(null)
    try {
      setOut(await api.compare({ state: text.trim(), decisions: qs, models: ["typical-small", mine] }))
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
    }
  }

  const cell = (r: { argmax: string; probs: Record<string, number> } | undefined, type: DecisionType, tone: "standard" | "yours") =>
    r && (
      <div className="space-y-1.5">
        <div className="flex items-baseline gap-2"><span className="font-medium">{say(type, r.argmax)}</span><span className="font-mono text-xs text-muted-foreground tabular">{pct(r.probs[r.argmax])}</span></div>
        <div className="h-1 w-full max-w-40 overflow-hidden rounded-full bg-muted">
          <div className={cn("h-full rounded-full", tone === "standard" ? "bg-standard" : "bg-yours")} style={{ width: `${r.probs[r.argmax] * 100}%` }} />
        </div>
      </div>
    )

  return (
    <>
      <PageHead title="Playground">Write or paste a new case. Both models answer every question, so you can see where yours differs.</PageHead>
      <Card>
        <CardContent className="space-y-3">
          <Label htmlFor="case">Case</Label>
          <Textarea id="case" value={text} rows={6} onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => (e.metaKey || e.ctrlKey) && e.key === "Enter" && run()} className="font-normal" />
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">⌘ Enter to run. The first run loads both models and takes a few seconds.</span>
            <Button onClick={run} disabled={busy || !text.trim()}>
              {busy && <Loader2 data-icon="inline-start" className="animate-spin" />} Ask both
            </Button>
          </div>
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {(out || busy) && (
        <Card className="mt-4">
          <CardContent className="px-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="pl-6">Question</TableHead>
                  <TableHead className="w-[26%]"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-standard" />Standard</span></TableHead>
                  <TableHead className="w-[26%] pr-6"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-yours" />{project.sample ? "Northwind's" : "Yours"}</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {decisions.map((d, i) => {
                  const a = out?.models["typical-small"]?.results[i], b = out?.models[mine]?.results[i]
                  return (
                    <TableRow key={d.column}>
                      <TableCell className="pl-6 whitespace-normal">
                        {d.question}
                        {a && b && a.argmax !== b.argmax && <Badge variant="secondary" className="ml-2">Disagree</Badge>}
                      </TableCell>
                      <TableCell>{busy ? <Skeleton className="h-8 w-28" /> : cell(a, d.type, "standard")}</TableCell>
                      <TableCell className="pr-6">{busy ? <Skeleton className="h-8 w-28" /> : cell(b, d.type, "yours")}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </>
  )
}
