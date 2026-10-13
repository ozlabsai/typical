import { useEffect, useState } from "react"
import { AlertCircle, Check, Copy, X } from "lucide-react"

import { SectionHeader, Stat } from "@/components/layout"
import { copy, Meter, msg } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { results, type Call, type Reveal } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import type { ModelRef } from "@/lib/library"
import { gain, pct, say, signed, snippet } from "@/lib/project"
import { cn } from "@/lib/utils"

/** Standard vs fine-tuned vs the difference: the same three numbers on a model's Overview and its Results. */
export function ScoreStats({ standard, yours, notes }: { standard: number; yours: number; notes?: [React.ReactNode, React.ReactNode, React.ReactNode] }) {
  const { t } = useI18n()
  return (
    <div className="grid grid-cols-3 gap-2 sm:gap-4">
      <Stat label={t("ev.standard")} value={pct(standard)} tone="standard" note={notes?.[0]} />
      <Stat label={t("ev.tuned")} value={pct(yours)} tone="yours" note={notes?.[1]} highlight />
      <Stat label={t("ev.difference")} value={`${signed(gain(yours, standard))} ${t("ev.pts")}`} note={notes?.[2]} />
    </div>
  )
}

/** One row per distinct case text (sample bodies repeat), wins AND losses: 7 where yours was right, 3 where it wasn't. */
function mixed(rows: Reveal["disagreements"]) {
  const body = (c: Reveal["disagreements"][number]) => snippet(c.case)
  const unique = rows.filter((c, i) => rows.findIndex((o) => body(o) === body(c)) === i)
  const won = unique.filter((c) => c.yours.answer === c.decided), lost = unique.filter((c) => c.yours.answer !== c.decided)
  return [...won.slice(0, 10 - Math.min(3, lost.length)), ...lost.slice(0, 3)]
}

/** ✓ Dispatch 96%: a model's answer, ticked when it is what was decided. */
function Answer({ x, decided, label }: { x: Call; decided: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {x.answer === decided ? <Check className="size-4 shrink-0 text-yours" /> : <X className="size-4 shrink-0 text-muted-foreground" />}
      <span dir="auto" className="font-medium">{label}</span>
      <span className="font-mono text-xs text-muted-foreground tabular">{pct(x.p)}</span>
    </span>
  )
}

export function Results({ model }: { model: ModelRef }) {
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
  if (error)
    return (
      <Alert variant="destructive"><AlertCircle /><AlertTitle>{t("ev.loadFail")}</AlertTitle><AlertDescription>{error}</AlertDescription></Alert>
    )
  if (!data)
    return (
      <>
        <SectionHeader description={t("ev.scoring")} />
        <Card>
          <CardContent className="grid gap-4">
            <p className="text-sm tabular">{progress ? t("ev.scored", { a: progress.done, b: progress.total }) : t("ev.starting")}</p>
            <Progress value={progress ? (progress.done / Math.max(1, progress.total)) * 100 : 0} className="h-1.5" />
            <div className="grid gap-4 pt-2 sm:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-24" />)}</div>
          </CardContent>
        </Card>
      </>
    )

  const mail = data.disagreements.some((c) => /^Subject: /m.test(c.case))   // a mail import: its yes/no "reply" reads as replies
  const summary = t("ev.summary", { who, a: pct(data.score.standard), mine: t(model.sample ? "ev.mineNorthwind" : "ev.mineOur"), b: pct(data.score.yours), n: data.n_cases })
  const decidedCol = t(model.sample ? "ev.theyDecided" : "ev.youDecided")
  const typeOf = (key: string) => data.decisions.find((d) => d.key === key)?.type ?? "choice"
  const rows = mixed(data.disagreements)

  return (
    <>
      <SectionHeader
        description={t("ev.lede", { who: t(model.sample ? "ev.gaveNorthwind" : "ev.gaveYou"), n: data.n_cases.toLocaleString(), a: data.n_answers.toLocaleString() })}
        actions={<Button variant="outline" size="sm" onClick={() => copy(t, summary, t("ev.summaryCopied"))}><Copy data-icon="inline-start" /> {t("ev.copySummary")}</Button>}
      />

      <ScoreStats standard={data.score.standard} yours={data.score.yours}
        notes={[t("ev.agrees", { who }), t("ev.agrees", { who }), t("ev.fixed", { a: data.fixed, b: data.broken })]} />

      <Card>
        <CardHeader>
          <CardTitle>{t("ev.byDecision")}</CardTitle>
          {data.abstained.standard > 0 && <CardDescription>{t("ev.abstained", { n: data.abstained.standard })}</CardDescription>}
        </CardHeader>
        <CardContent className="px-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="ps-6">{t("ev.question")}</TableHead>
                <TableHead className="hidden w-[26%] md:table-cell"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-standard" />{t("ev.standardCol")}</span></TableHead>
                <TableHead className="hidden w-[26%] md:table-cell"><span className="inline-flex items-center gap-2"><span className="size-2 rounded-full bg-yours" />{t("ev.tunedCol")}</span></TableHead>
                <TableHead className="w-20 pe-6 text-end">{t("ev.change")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.decisions.map((d) => {
                const delta = gain(d.yours, d.standard)
                return (
                  <TableRow key={d.key}>
                    <TableCell className="ps-6 whitespace-normal">
                      <span dir="auto">{d.question}</span>
                      {/* phones: no bars, the two numbers in one line */}
                      <span className="mt-1 flex items-center gap-1.5 font-mono text-sm tabular md:hidden">
                        <span className="text-standard">{pct(d.standard)}</span>
                        <span aria-hidden className="inline-block text-muted-foreground rtl:rotate-180">→</span>
                        <span className="font-medium text-yours">{pct(d.yours)}</span>
                      </span>
                      {d.catch?.yours != null && d.catch.standard != null && (
                        <p className="mt-1 text-sm font-medium">
                          {t("ev.catch", { k: pct(data.catch_top ?? 0.2), y: pct(d.catch.yours), s: pct(d.catch.standard),
                            what: t(mail && d.key === "reply" ? "ev.catchReplied" : "ev.catchYes") })}
                        </p>
                      )}
                      {/* only where it changes the reading: a model that does not beat always giving the most common answer */}
                      {d.majority != null && d.baseline != null && d.baseline >= Math.min(d.standard, d.yours) && (
                        <p className="mt-1 text-xs text-muted-foreground">{t("ev.baseline", { a: `\u2068${say(t, d.type, d.majority)}\u2069`, p: pct(d.baseline) })}</p>
                      )}
                    </TableCell>
                    <TableCell className="hidden md:table-cell"><Meter value={d.standard} tone="standard" /></TableCell>
                    <TableCell className="hidden md:table-cell"><Meter value={d.yours} tone="yours" /></TableCell>
                    <TableCell className={cn("pe-6 text-end align-top font-mono tabular md:align-middle", delta > 0 ? "text-yours" : delta < 0 ? "text-destructive" : "text-muted-foreground")}>
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
          {/* phones: one stacked row per case, the decided answer always visible */}
          <ul className="divide-y border-t md:hidden">
            {rows.map((c, i) => {
              const type = typeOf(c.key)
              return (
                <li key={i} className="grid gap-1.5 px-4 py-3 text-sm">
                  <span dir="auto" className="line-clamp-2 text-muted-foreground" title={c.case}>{snippet(c.case)}</span>
                  <span dir="auto" className="text-xs font-medium">{c.question}</span>
                  <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-standard" /><Answer x={c.standard} decided={c.decided} label={say(t, type, c.standard.answer)} /></span>
                    <span className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full bg-yours" /><Answer x={c.yours} decided={c.decided} label={say(t, type, c.yours.answer)} /></span>
                  </span>
                  <span className="text-xs text-muted-foreground">{decidedCol}: <span dir="auto" className="font-medium text-foreground">{say(t, type, c.decided)}</span></span>
                </li>
              )
            })}
          </ul>
          <Table className="hidden md:table">
            <TableHeader>
              <TableRow>
                <TableHead className="ps-6">{t("ev.case")}</TableHead>
                <TableHead>{t("ev.question")}</TableHead>
                <TableHead>{t("ev.standardCol")}</TableHead>
                <TableHead>{t("ev.tunedCol")}</TableHead>
                <TableHead className="pe-6">{decidedCol}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((c, i) => {
                const type = typeOf(c.key)
                return (
                  <TableRow key={i}>
                    <TableCell className="w-[34%] max-w-0 ps-6 whitespace-normal">
                      <span dir="auto" className="line-clamp-2 text-muted-foreground" title={c.case}>{snippet(c.case)}</span>
                    </TableCell>
                    <TableCell className="whitespace-normal"><span dir="auto">{c.question}</span></TableCell>
                    <TableCell><Answer x={c.standard} decided={c.decided} label={say(t, type, c.standard.answer)} /></TableCell>
                    <TableCell><Answer x={c.yours} decided={c.decided} label={say(t, type, c.yours.answer)} /></TableCell>
                    <TableCell className="pe-6 font-medium"><span dir="auto">{say(t, type, c.decided)}</span></TableCell>
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
