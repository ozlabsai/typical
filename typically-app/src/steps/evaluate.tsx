import { useEffect, useState } from "react"
import { AlertCircle, ArrowRight, Check, Copy, Download, Loader2, X } from "lucide-react"
import { toast } from "sonner"

import { Meter, msg, PageHead } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { api, downloadUrl, results, type Compare, type DecisionType, type Reveal } from "@/lib/api"
import { included, pct, queries, SAMPLE, say, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

export function EvaluateStep({ project, next }: { project: Project; next: () => void }) {
  const [tab, setTab] = useState("results")
  return (
    <Tabs value={tab} onValueChange={setTab}>
      <TabsList className="mb-6">
        <TabsTrigger value="results">Results</TabsTrigger>
        <TabsTrigger value="playground">Playground</TabsTrigger>
      </TabsList>
      <TabsContent value="results"><Results project={project} next={next} /></TabsContent>
      <TabsContent value="playground"><Playground project={project} /></TabsContent>
    </Tabs>
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

function Results({ project, next }: { project: Project; next: () => void }) {
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
          <Button onClick={next}>Deploy <ArrowRight data-icon="inline-end" /></Button>
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

function Playground({ project }: { project: Project }) {
  const first = project.analysis.preview_cases[0]?.case ?? ""
  const [text, setText] = useState(project.sample ? SAMPLE.tryCase : first)
  const [out, setOut] = useState<Compare | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const mine = `local:${project.run}`
  const qs = queries(project)
  const decisions = included(project)

  async function run() {
    if (!text.trim()) return
    setBusy(true)
    setError(null)
    try {
      setOut(await api.compare({ state: text.trim(), decisions: qs, models: ["base", mine] }))
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
                  const a = out?.models["base"]?.results[i], b = out?.models[mine]?.results[i]
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
