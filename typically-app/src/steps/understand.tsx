import { AlertTriangle, ArrowDown, ArrowRight, ArrowUp, Bot, ChevronDown, ListChecks } from "lucide-react"

import { PageHead } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import type { DecisionType } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { labelCounts, merges, TYPES, type PlanDecision, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

const WHY_KEYS = ["id", "timestamp", "pii", "leakage", "after_decision", "near_unique", "constant"] as const


function DecisionCard({ p, d, set }: { p: Project; d: PlanDecision; set: (patch: Partial<PlanDecision>) => void }) {
  const { t } = useI18n()
  const counts = labelCounts(p, d)
  const merged = merges(p, d)
  const move = (i: number, by: number) => {
    const labels = [...d.labels]
    ;[labels[i], labels[i + by]] = [labels[i + by], labels[i]]
    set({ labels })
  }
  return (
    <Card size="sm" className={cn(!d.include && "opacity-60")}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="font-mono text-sm">{d.column}</span>
          {d.needs_review && <Badge variant="outline" className="border-amber-500/50 text-amber-700 dark:text-amber-400">{t("und.check")}</Badge>}
        </CardTitle>
        <CardDescription>{d.reasons[0] ?? ""}</CardDescription>
        <CardAction><Switch checked={d.include} onCheckedChange={(v) => set({ include: v })} aria-label={t("und.learn", { c: d.column })} /></CardAction>
      </CardHeader>
      <CardContent className="grid gap-3">
        <div className="grid gap-3 sm:grid-cols-[1fr_150px]">
          <Input dir="auto" value={d.question} disabled={!d.include} onChange={(e) => set({ question: e.target.value })} aria-label={t("und.question", { c: d.column })} />
          <Select value={d.type} disabled={!d.include} onValueChange={(v) => set({ type: v as DecisionType })}>
            <SelectTrigger className="w-full" aria-label={t("und.type", { c: d.column })}><SelectValue /></SelectTrigger>
            <SelectContent>{TYPES.map((ty) => <SelectItem key={ty} value={ty}>{t(`type.${ty}`)}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {d.labels.map((l, i) => (
            <span key={l} className="inline-flex items-center gap-1">
              {d.type === "score" && i > 0 && <span className="text-xs text-muted-foreground">&lt;</span>}
              <Badge variant="outline" className="gap-1.5 font-normal">
                {l}<span className="text-muted-foreground tabular">{counts[l] ?? 0}</span>
                {d.type === "score" && d.include && (
                  <span className="-me-1 inline-flex">
                    <button type="button" aria-label={t("und.lower", { l })} disabled={i === 0} onClick={() => move(i, -1)} className="rounded p-0.5 hover:bg-muted disabled:opacity-30"><ArrowUp className="size-3 -rotate-90 rtl:rotate-90" /></button>
                    <button type="button" aria-label={t("und.higher", { l })} disabled={i === d.labels.length - 1} onClick={() => move(i, 1)} className="rounded p-0.5 hover:bg-muted disabled:opacity-30"><ArrowDown className="size-3 -rotate-90 rtl:rotate-90" /></button>
                  </span>
                )}
              </Badge>
            </span>
          ))}
        </div>
        {Object.keys(merged).length > 0 && (
          <p className="text-xs text-muted-foreground">
            {t("und.merged")}: {Object.entries(merged).map(([to, from]) => `${to} ← ${from.join(", ")}`).join(" · ")}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

export function UnderstandStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
  const { t } = useI18n()
  const { plan, analysis } = project
  const setDecision = (i: number, patch: Partial<PlanDecision>) =>
    update({ plan: { ...plan, decisions: plan.decisions.map((d, j) => (j === i ? { ...d, ...patch, needs_review: false } : d)) } })
  const facts = plan.case.parts.filter((c) => c.role === "fact").map((c) => c.column)
  const body = plan.case.parts.filter((c) => c.role === "body").map((c) => c.column)
  const review = [
    ...plan.excluded.filter((e) => e.why === "leakage" || e.why === "after_decision").map((e) => `${e.column}: ${e.reason}`),
    ...plan.decisions.filter((d) => d.needs_review).map((d) => `${d.column}: ${d.reasons.join(" ") || "low confidence"}`),
  ].slice(0, 3)
  const usable = analysis.n_rows
  const chosen = plan.decisions.filter((d) => d.include).length

  return (
    <>
      <PageHead
        title={t("und.title")}
        action={<Badge variant="secondary" className="gap-1.5"><Bot className="size-3.5" /> {t(analysis.plan_source === "llm" ? "und.byClaude" : "und.byRules")}</Badge>}
      >
        {t("und.lede")}
      </PageHead>

      <Card >
        <CardHeader>
          <CardTitle>{t("und.caseIs")}</CardTitle>
          <CardDescription>
            {[...facts, ...body].length ? <>{facts.length > 0 && <>{t("und.factsFrom")}<b className="font-medium text-foreground">{facts.join(", ")}</b>{body.length ? t("und.then") : ""}</>}{body.length > 0 && <>{t("und.textIn")}<b className="font-medium text-foreground">{body.join(", ")}</b></>}</> : t("und.noText")}
            {" · "}<span className="tabular">{t("und.rows", { n: usable.toLocaleString() })}</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          {analysis.preview_cases.slice(0, 3).map((c, i) => (
            <div key={i} className="rounded-lg bg-muted/50 p-3 text-xs">
              <p dir="auto" className="line-clamp-6 whitespace-pre-wrap text-muted-foreground">{c.case}</p>
              <div className="mt-2 flex flex-wrap gap-1">{Object.entries(c.answers).filter(([, v]) => v != null).map(([k, v]) => <Badge key={k} variant="outline" className="font-normal">{k}: {v}</Badge>)}</div>
            </div>
          ))}
        </CardContent>
      </Card>

      {review.length > 0 && (
        <Alert >
          <AlertTriangle />
          <AlertTitle>{t("und.review")}</AlertTitle>
          <AlertDescription><ul className="list-disc ps-4">{review.map((r) => <li key={r}>{r}</li>)}</ul></AlertDescription>
        </Alert>
      )}

      <div className="grid gap-3">
      <div className="flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-medium"><ListChecks className="size-4" /> {t("und.decisions")}</h2>
        <span className="text-sm text-muted-foreground tabular">{t("und.selected", { a: chosen, b: plan.decisions.length })}</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {plan.decisions.map((d, i) => <DecisionCard key={d.column} p={project} d={d} set={(patch) => setDecision(i, patch)} />)}
      </div>
      </div>

      {plan.issues.length > 0 && (
        <Card  size="sm">
          <CardHeader><CardTitle>{t("und.health")}</CardTitle></CardHeader>
          <CardContent>
            <ul className="grid gap-1.5 text-sm">
              {plan.issues.map((s, i) => (
                <li key={i} className="flex gap-2"><span className="text-muted-foreground">•</span><span>{s.column && <><bdi className="font-mono text-xs">{s.column}</bdi>{" · "}</>}{s.detail}{s.action ? <span className="text-muted-foreground"> {s.action}</span> : null}</span></li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {plan.excluded.length > 0 && (
        <Collapsible >
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="-ms-2">{t("und.ignored", { n: plan.excluded.length })} <ChevronDown data-icon="inline-end" /></Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ul className="grid gap-1 pt-2 text-sm">
              {plan.excluded.map((e) => <li key={e.column}><span className="font-mono">{e.column}</span> <span className="text-muted-foreground">: {(WHY_KEYS as readonly string[]).includes(e.why) ? t(`why.${e.why as (typeof WHY_KEYS)[number]}`) : e.why}. {e.reason}</span></li>)}
            </ul>
          </CollapsibleContent>
        </Collapsible>
      )}

      <Card  size="sm">
        <CardFooter className="justify-between">
          <Button variant="ghost" onClick={back}>{t("common.back")}</Button>
          <Button onClick={next} disabled={!chosen || usable < 100}>{usable < 100 ? t("und.need100") : <>{t("common.continue")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></>}</Button>
        </CardFooter>
      </Card>
    </>
  )
}
