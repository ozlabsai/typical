import { AlertTriangle, ArrowDown, ArrowRight, ArrowUp, Bot, ChevronDown } from "lucide-react"

import { SectionHeader } from "@/components/layout"
import { PageHead, StepFooter } from "@/components/shared"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import type { DecisionType } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { labelCounts, merges, TYPES, type PlanDecision, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

const WHY_KEYS = ["id", "timestamp", "pii", "leakage", "after_decision", "near_unique", "constant"] as const

/** A scale's answers as a vertical list, lowest first, each with buttons to move it lower / higher. */
function ScaleOrder({ d, counts, move }: { d: PlanDecision; counts: Record<string, number>; move: (i: number, by: number) => void }) {
  const { t } = useI18n()
  return (
    <div className="grid gap-1">
      <span className="text-xs text-muted-foreground">{t("und.lowest")}</span>
      <ol className="grid gap-1">
        {d.labels.map((l, i) => (
          <li key={l} className="flex items-center gap-2 rounded-lg border px-2 py-1 text-sm">
            <span dir="auto" className="min-w-0 flex-1 truncate">{l}</span>
            <span className="text-xs text-muted-foreground tabular">{counts[l] ?? 0}</span>
            {d.include && (
              <span className="flex">
                <Button variant="ghost" size="icon-xs" aria-label={t("und.lower", { l })} disabled={i === 0} onClick={() => move(i, -1)}><ArrowUp /></Button>
                <Button variant="ghost" size="icon-xs" aria-label={t("und.higher", { l })} disabled={i === d.labels.length - 1} onClick={() => move(i, 1)}><ArrowDown /></Button>
              </span>
            )}
          </li>
        ))}
      </ol>
      <span className="text-xs text-muted-foreground">{t("und.highest")}</span>
    </div>
  )
}

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
        {/* the question is the title (it grows with its text); the column it came from is metadata */}
        <CardTitle>
          <Textarea dir="auto" rows={1} value={d.question} disabled={!d.include} onChange={(e) => set({ question: e.target.value })}
            aria-label={t("und.question", { c: d.column })} className="min-h-0 resize-none text-base font-semibold md:text-sm" />
        </CardTitle>
        <CardDescription className="flex flex-wrap items-center gap-2 text-xs">
          <bdi>{d.column}</bdi>
          {d.needs_review && <Badge variant="outline" className="border-warning/60 text-warning-foreground">{t("und.check")}</Badge>}
        </CardDescription>
        <CardAction><Switch checked={d.include} onCheckedChange={(v) => set({ include: v })} aria-label={t("und.learn", { c: d.column })} /></CardAction>
      </CardHeader>
      <CardContent className="grid gap-4">
        {d.reasons[0] && <p className="text-sm text-muted-foreground">{d.reasons[0]}</p>}
        <Select value={d.type} disabled={!d.include} onValueChange={(v) => set({ type: v as DecisionType })}>
          <SelectTrigger className="w-full sm:w-48" aria-label={t("und.type", { c: d.column })}><SelectValue /></SelectTrigger>
          <SelectContent>{TYPES.map((ty) => <SelectItem key={ty} value={ty}>{t(`type.${ty}`)}</SelectItem>)}</SelectContent>
        </Select>
        {d.type === "score" ? <ScaleOrder d={d} counts={counts} move={move} /> : (
          <div className="flex flex-wrap items-center gap-2">
            {d.labels.map((l) => (
              <Badge key={l} variant="outline" className="gap-1.5 font-normal">
                <bdi>{l}</bdi><span className="text-muted-foreground tabular">{counts[l] ?? 0}</span>
              </Badge>
            ))}
          </div>
        )}
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
  const usable = analysis.n_rows
  const chosen = plan.decisions.filter((d) => d.include).length

  return (
    <>
      <PageHead
        title={t("und.title")}
        action={<Badge variant="secondary"><Bot /> {t(analysis.plan_source === "llm" ? "und.byAi" : "und.byRules")}</Badge>}
      >
        {t("und.lede")}
      </PageHead>

      {/* any source: an AI analysis that fell back to the rules, a file too big to read whole, no decision found */}
      {(analysis.warnings?.length ?? 0) > 0 && (
        <Alert>
          <AlertTriangle />
          <AlertDescription>{analysis.warnings!.map((w) => <p key={w} dir="auto">{w}</p>)}</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>{t("und.caseIs")}</CardTitle>
          <CardDescription>
            {[...facts, ...body].length ? <>{facts.length > 0 && <>{t("und.factsFrom")}<b className="font-medium text-foreground">{facts.join(", ")}</b>{body.length ? t("und.then") : ""}</>}{body.length > 0 && <>{t("und.textIn")}<b className="font-medium text-foreground">{body.join(", ")}</b></>}</> : t("und.noText")}
            {" · "}<span className="tabular">{t("und.rows", { n: usable.toLocaleString() })}</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          {analysis.preview_cases.slice(0, 3).map((c, i) => (
            <div key={i} className="rounded-lg bg-muted/50 p-3 text-sm">
              <p dir="auto" className="line-clamp-6 whitespace-pre-wrap text-muted-foreground">{c.case}</p>
              <div className="mt-2 flex flex-wrap gap-1">{Object.entries(c.answers).filter(([, v]) => v != null).map(([k, v]) => <Badge key={k} variant="outline" className="font-normal">{k}: {v}</Badge>)}</div>
            </div>
          ))}
        </CardContent>
      </Card>

      <section className="grid gap-4">
        <SectionHeader title={t("und.decisions")} actions={<span className="text-sm text-muted-foreground tabular">{t("und.selected", { a: chosen, b: plan.decisions.length })}</span>} />
        <div className="grid gap-4 md:grid-cols-2">
          {plan.decisions.map((d, i) => <DecisionCard key={d.column} p={project} d={d} set={(patch) => setDecision(i, patch)} />)}
        </div>
      </section>

      {plan.issues.length > 0 && (
        <Card size="sm">
          <CardHeader><CardTitle>{t("und.health")}</CardTitle></CardHeader>
          <CardContent>
            <ul className="grid gap-2 text-sm">
              {plan.issues.map((s, i) => (
                <li key={i} className="flex gap-2"><span className="text-muted-foreground">•</span><span>{s.column && <><bdi className="font-medium">{s.column}</bdi>{" · "}</>}{s.detail}{s.action ? <span className="text-muted-foreground"> {s.action}</span> : null}</span></li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {plan.excluded.length > 0 && (
        <Collapsible>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="-ms-2">{t("und.ignored", { n: plan.excluded.length })} <ChevronDown data-icon="inline-end" /></Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ul className="grid gap-1 pt-2 text-sm">
              {plan.excluded.map((e) => <li key={e.column}><bdi className="font-medium">{e.column}</bdi> <span className="text-muted-foreground">: {(WHY_KEYS as readonly string[]).includes(e.why) ? t(`why.${e.why as (typeof WHY_KEYS)[number]}`) : e.why}. {e.reason}</span></li>)}
            </ul>
          </CollapsibleContent>
        </Collapsible>
      )}

      <StepFooter back={back}>
        <Button onClick={next} disabled={!chosen || usable < 100}>{usable < 100 ? t("und.need100") : <>{t("common.continue")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></>}</Button>
      </StepFooter>
    </>
  )
}
