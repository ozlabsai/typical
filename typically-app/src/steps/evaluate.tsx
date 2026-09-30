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
import { useI18n } from "@/lib/i18n"
import { included, pct, queries, SAMPLE, say, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

export function EvaluateStep({ project, next }: { project: Project; next: () => void }) {
  const { t } = useI18n()
  const [tab, setTab] = useState("results")
  return (
    <Tabs value={tab} onValueChange={setTab}>
      <TabsList className="mb-6">
        <TabsTrigger value="results">{t("ev.results")}</TabsTrigger>
        <TabsTrigger value="playground">{t("ev.playground")}</TabsTrigger>
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
  const { t } = useI18n()
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

  const who = t(project.sample ? "ev.whoNorthwind" : "ev.whoYour")
  const yoursCol = t(project.sample ? "ev.northwindCol" : "ev.yoursCol")
  if (error)
    return (
      <Alert variant="destructive"><AlertCircle /><AlertTitle>{t("ev.loadFail")}</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>
    )
  if (!data)
    return (
      <>
        <PageHead title={t("ev.results")}>{t("ev.scoring")}</PageHead>
        <Card>
          <CardContent className="space-y-3">
            <p className="text-sm tabular">{progress ? t("ev.scored", { a: progress.done, b: progress.total }) : t("ev.starting")}</p>
            <Progress value={progress ? (progress.done / Math.max(1, progress.total)) * 100 : 0} className="h-1.5" />
            <div className="grid gap-4 pt-3 sm:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-24" />)}</div>
          </CardContent>
        </Card>
      </>
    )

  const gain = Math.round((data.score.yours - data.score.standard) * 100)
  const summary = t("ev.summary", { who, a: pct(data.score.standard), mine: t(project.sample ? "ev.mineNorthwind" : "ev.mineOur"), b: pct(data.score.yours), n: data.n_cases })

  return (
    <>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("ev.results")}</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            {t("ev.lede", { who: project.sample ? "Northwind" : t("ev.you"), n: data.n_cases, a: data.n_answers })}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => navigator.clipboard.writeText(summary).then(() => toast.success(t("ev.summaryCopied")))}>
            <Copy data-icon="inline-start" /> {t("ev.copySummary")}
          </Button>
          <Button variant="outline" asChild>
            <a href={downloadUrl(project.run!)} download><Download data-icon="inline-start" /> {t("ev.download")}</a>
          </Button>
          <Button onClick={next}>{t("ev.deploy")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card size="sm">
          <CardHeader><CardDescription>{t("ev.standard")}</CardDescription><p className="font-mono text-3xl font-medium tabular text-standard">{pct(data.score.standard)}</p></CardHeader>
          <CardContent className="text-sm text-muted-foreground">{t("ev.agrees", { who })}</CardContent>
        </Card>
        <Card size="sm" className="ring-1 ring-primary/30">
          <CardHeader><CardDescription>{t(project.sample ? "ev.northwindTitle" : "ev.yoursTitle")}</CardDescription><p className="font-mono text-3xl font-medium tabular text-yours">{pct(data.score.yours)}</p></CardHeader>
          <CardContent className="text-sm text-muted-foreground">{t("ev.agrees", { who })}</CardContent>
        </Card>
        <Card size="sm">
          <CardHeader><CardDescription>{t("ev.difference")}</CardDescription><p className="font-mono text-3xl font-medium tabular">{gain >= 0 ? "+" : ""}{gain} {t("ev.pts")}</p></CardHeader>
          <CardContent className="text-sm text-muted-foreground tabular">{t("ev.fixed", { a: data.fixed, b: data.broken })}</CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader>
          <CardTitle>{t("ev.byDecision")}</CardTitle>
          {data.abstained.standard > 0 && (
            <CardDescription>
              {t("ev.abstained", { n: data.abstained.standard })}
            </CardDescription>
          )}
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="ps-6">{t("ev.question")}</TableHead>
                <TableHead className="w-[26%]"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-standard" />{t("ev.standardCol")}</span></TableHead>
                <TableHead className="w-[26%]"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-yours" />{yoursCol}</span></TableHead>
                <TableHead className="w-20 pe-6 text-end">{t("ev.change")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.decisions.map((d) => {
                const delta = Math.round((d.yours - d.standard) * 100)
                return (
                  <TableRow key={d.key}>
                    <TableCell className="ps-6 whitespace-normal"><span dir="auto">{d.question}</span></TableCell>
                    <TableCell><Meter value={d.standard} tone="standard" /></TableCell>
                    <TableCell><Meter value={d.yours} tone="yours" /></TableCell>
                    <TableCell className={cn("pe-6 text-end font-mono tabular", delta > 0 ? "text-yours" : "text-muted-foreground")}>
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
          <CardTitle>{t("ev.disagree")}</CardTitle>
          <CardDescription>{t("ev.disagreeLede", { who: project.sample ? "Northwind" : t("ev.you") })}</CardDescription>
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="ps-6">{t("ev.case")}</TableHead>
                <TableHead>{t("ev.question")}</TableHead>
                <TableHead>{t("ev.standardCol")}</TableHead>
                <TableHead>{yoursCol}</TableHead>
                <TableHead className="pe-6">{t(project.sample ? "ev.theyDecided" : "ev.youDecided")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mixed(data.disagreements).map((c, i) => {
                const type = data.decisions.find((d) => d.key === c.key)?.type ?? "choice"
                const cell = (x: { answer: string; p: number }) => (
                  <span className="inline-flex items-center gap-1.5">
                    {x.answer === c.decided ? <Check className="size-3.5 text-yours" /> : <X className="size-3.5 text-muted-foreground" />}
                    <span className="font-medium">{say(t, type, x.answer)}</span>
                    <span className="font-mono text-xs text-muted-foreground tabular">{pct(x.p)}</span>
                  </span>
                )
                const body = c.case.split("\n\n").at(-1)
                return (
                  <TableRow key={i}>
                    <TableCell className="max-w-0 w-[38%] ps-6 whitespace-normal">
                      <span dir="auto" className="line-clamp-2 text-muted-foreground" title={c.case}>{body}</span>
                    </TableCell>
                    <TableCell className="whitespace-normal"><span dir="auto">{c.question}</span></TableCell>
                    <TableCell>{cell(c.standard)}</TableCell>
                    <TableCell>{cell(c.yours)}</TableCell>
                    <TableCell className="pe-6 font-medium">{say(t, type, c.decided)}</TableCell>
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
  const { t } = useI18n()
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
        <div className="flex items-baseline gap-2"><span className="font-medium">{say(t, type, r.argmax)}</span><span className="font-mono text-xs text-muted-foreground tabular">{pct(r.probs[r.argmax])}</span></div>
        <div className="h-1 w-full max-w-40 overflow-hidden rounded-full bg-muted">
          <div className={cn("h-full rounded-full", tone === "standard" ? "bg-standard" : "bg-yours")} style={{ width: `${r.probs[r.argmax] * 100}%` }} />
        </div>
      </div>
    )

  return (
    <>
      <PageHead title={t("pg.title")}>{t("pg.lede")}</PageHead>
      <Card>
        <CardContent className="space-y-3">
          <Label htmlFor="case">{t("pg.case")}</Label>
          <Textarea id="case" dir="auto" value={text} rows={6} onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => (e.metaKey || e.ctrlKey) && e.key === "Enter" && run()} className="font-normal" />
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">{t("pg.hint")}</span>
            <Button onClick={run} disabled={busy || !text.trim()}>
              {busy && <Loader2 data-icon="inline-start" className="animate-spin" />} {t("pg.ask")}
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
                  <TableHead className="ps-6">{t("ev.question")}</TableHead>
                  <TableHead className="w-[26%]"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-standard" />{t("ev.standardCol")}</span></TableHead>
                  <TableHead className="w-[26%] pe-6"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-yours" />{t(project.sample ? "ev.northwindCol" : "ev.yoursCol")}</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {decisions.map((d, i) => {
                  const a = out?.models["base"]?.results[i], b = out?.models[mine]?.results[i]
                  return (
                    <TableRow key={d.column}>
                      <TableCell className="ps-6 whitespace-normal">
                        <span dir="auto">{d.question}</span>
                        {a && b && a.argmax !== b.argmax && <Badge variant="secondary" className="ms-2">{t("pg.disagree")}</Badge>}
                      </TableCell>
                      <TableCell>{busy ? <Skeleton className="h-8 w-28" /> : cell(a, d.type, "standard")}</TableCell>
                      <TableCell className="pe-6">{busy ? <Skeleton className="h-8 w-28" /> : cell(b, d.type, "yours")}</TableCell>
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
