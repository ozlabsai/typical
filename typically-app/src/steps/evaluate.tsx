import { useEffect, useState } from "react"
import { AlertCircle, ArrowRight, Check, Copy, Download, X } from "lucide-react"
import { toast } from "sonner"

import { SectionHeader, Stat } from "@/components/layout"
import { Meter, msg, PageHead } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { downloadUrl, results, type Reveal } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import type { ModelRef } from "@/lib/library"
import { pct, say, signed, snippet } from "@/lib/project"
import { cn } from "@/lib/utils"

/* ---------------------------------------------------------------- 4. Results */

/** One row per distinct case text (sample bodies repeat), wins AND losses: 7 where yours was right, 3 where it wasn't. */
function mixed(rows: Reveal["disagreements"]) {
  const body = (c: Reveal["disagreements"][number]) => snippet(c.case)
  const unique = rows.filter((c, i) => rows.findIndex((o) => body(o) === body(c)) === i)
  const won = unique.filter((c) => c.yours.answer === c.decided), lost = unique.filter((c) => c.yours.answer !== c.decided)
  return [...won.slice(0, 10 - Math.min(3, lost.length)), ...lost.slice(0, 3)]
}

export function Results({ model, next }: { model: ModelRef; next?: () => void }) {
  const { t } = useI18n()
  const [data, setData] = useState<Reveal | null>(null)
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const r = await results(model.resultsKey)
        if (!alive) return
        if ("status" in r) return setProgress({ done: r.done, total: r.total }), (timer = setTimeout(poll, 2500))
        setData(r)
      } catch (e) {
        if (alive) setError(msg(e))
      }
    }
    poll()
    return () => ((alive = false), clearTimeout(timer))
  }, [model.resultsKey])

  const who = t(model.sample ? "ev.whoNorthwind" : "ev.whoYour")
  const yoursCol = t(model.sample ? "ev.northwindCol" : "ev.yoursCol")
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
  const summary = t("ev.summary", { who, a: pct(data.score.standard), mine: t(model.sample ? "ev.mineNorthwind" : "ev.mineOur"), b: pct(data.score.yours), n: data.n_cases })

  return (
    <>
      <SectionHeader
        title={t("ev.results")}
        description={t("ev.lede", { who: t(model.sample ? "ev.gaveNorthwind" : "ev.gaveYou"), n: data.n_cases.toLocaleString(), a: data.n_answers.toLocaleString() })}
        actions={<>
          <Button variant="outline" onClick={() => navigator.clipboard.writeText(summary).then(() => toast.success(t("ev.summaryCopied")))}>
            <Copy data-icon="inline-start" /> {t("ev.copySummary")}
          </Button>
          <Button variant="outline" asChild>
            <a href={downloadUrl(model.run)} download><Download data-icon="inline-start" /> {t("ev.download")}</a>
          </Button>
          {next && <Button onClick={next}>{t("ev.deploy")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>}
        </>}
      />

      <div className="grid grid-cols-3 gap-2 sm:gap-4">
        <Stat label={t("ev.standard")} value={pct(data.score.standard)} tone="standard" note={t("ev.agrees", { who })} />
        <Stat label={t(model.sample ? "ev.northwindTitle" : "ev.yoursTitle")} value={pct(data.score.yours)} tone="yours" note={t("ev.agrees", { who })} highlight />
        <Stat label={t("ev.difference")} value={`${signed(gain)} ${t("ev.pts")}`} note={t("ev.fixed", { a: data.fixed, b: data.broken })} />
      </div>

      <Card>
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
                <TableHead className="hidden md:table-cell w-20 pe-6 text-end">{t("ev.change")}</TableHead>
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
                    <TableCell className={cn("hidden md:table-cell pe-6 text-end font-mono tabular", delta > 0 ? "text-yours" : "text-muted-foreground")}>
                      {signed(delta)}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>{t("ev.disagree")}</CardTitle>
          <CardDescription>{t("ev.disagreeLede", { who: t(model.sample ? "ev.decidedNorthwind" : "ev.decidedYou") })}</CardDescription>
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="ps-6">{t("ev.case")}</TableHead>
                <TableHead className="hidden md:table-cell">{t("ev.question")}</TableHead>
                <TableHead>{t("ev.standardCol")}</TableHead>
                <TableHead>{yoursCol}</TableHead>
                <TableHead className="hidden md:table-cell pe-6">{t(model.sample ? "ev.theyDecided" : "ev.youDecided")}</TableHead>
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
                const body = snippet(c.case)
                return (
                  <TableRow key={i}>
                    <TableCell className="max-w-0 w-[38%] ps-6 whitespace-normal">
                      <span dir="auto" className="line-clamp-2 text-muted-foreground" title={c.case}>{body}</span>
                      <span dir="auto" className="mt-1 block text-xs font-medium md:hidden">{c.question}</span>
                    </TableCell>
                    <TableCell className="hidden md:table-cell whitespace-normal"><span dir="auto">{c.question}</span></TableCell>
                    <TableCell>{cell(c.standard)}</TableCell>
                    <TableCell>{cell(c.yours)}</TableCell>
                    <TableCell className="hidden md:table-cell pe-6 font-medium">{say(t, type, c.decided)}</TableCell>
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

