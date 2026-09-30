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
import { labelCounts, merges, TYPE_LABEL, type PlanDecision, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

const WHY: Record<string, string> = {
  id: "an ID", timestamp: "a date or time", pii: "personal data", leakage: "it gives the answer away",
  after_decision: "only known after the decision", near_unique: "different on every row", constant: "the same on every row",
}

function DecisionCard({ p, d, set }: { p: Project; d: PlanDecision; set: (patch: Partial<PlanDecision>) => void }) {
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
          {d.needs_review && <Badge variant="outline" className="border-amber-500/50 text-amber-700 dark:text-amber-400">Check</Badge>}
        </CardTitle>
        <CardDescription>{d.reasons[0] ?? ""}</CardDescription>
        <CardAction><Switch checked={d.include} onCheckedChange={(v) => set({ include: v })} aria-label={`Learn ${d.column}`} /></CardAction>
      </CardHeader>
      <CardContent className="grid gap-3">
        <div className="grid gap-3 sm:grid-cols-[1fr_150px]">
          <Input value={d.question} disabled={!d.include} onChange={(e) => set({ question: e.target.value })} aria-label={`Question for ${d.column}`} />
          <Select value={d.type} disabled={!d.include} onValueChange={(v) => set({ type: v as DecisionType })}>
            <SelectTrigger className="w-full" aria-label={`Answer type for ${d.column}`}><SelectValue /></SelectTrigger>
            <SelectContent>{(Object.keys(TYPE_LABEL) as DecisionType[]).map((t) => <SelectItem key={t} value={t}>{TYPE_LABEL[t]}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {d.labels.map((l, i) => (
            <span key={l} className="inline-flex items-center gap-1">
              {d.type === "score" && i > 0 && <span className="text-xs text-muted-foreground">&lt;</span>}
              <Badge variant="outline" className="gap-1.5 font-normal">
                {l}<span className="text-muted-foreground tabular">{counts[l] ?? 0}</span>
                {d.type === "score" && d.include && (
                  <span className="-mr-1 inline-flex">
                    <button type="button" aria-label={`Move ${l} lower`} disabled={i === 0} onClick={() => move(i, -1)} className="rounded p-0.5 hover:bg-muted disabled:opacity-30"><ArrowUp className="size-3 -rotate-90" /></button>
                    <button type="button" aria-label={`Move ${l} higher`} disabled={i === d.labels.length - 1} onClick={() => move(i, 1)} className="rounded p-0.5 hover:bg-muted disabled:opacity-30"><ArrowDown className="size-3 -rotate-90" /></button>
                  </span>
                )}
              </Badge>
            </span>
          ))}
        </div>
        {Object.keys(merged).length > 0 && (
          <p className="text-xs text-muted-foreground">
            Merged: {Object.entries(merged).map(([to, from]) => `${to} ← ${from.join(", ")}`).join(" · ")}
          </p>
        )}
      </CardContent>
    </Card>
  )
}

export function UnderstandStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
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
        title="Here's what we understood"
        action={<Badge variant="secondary" className="gap-1.5"><Bot className="size-3.5" /> {analysis.plan_source === "llm" ? "Read by Claude" : "Read by built-in rules"}</Badge>}
      >
        Check how each case is described and what you decided. Edit anything that's off; nothing is sent for training until the Train step.
      </PageHead>

      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Each case is</CardTitle>
          <CardDescription>
            {[...facts, ...body].length ? <>{facts.length > 0 && <>facts from <b className="font-medium text-foreground">{facts.join(", ")}</b>{body.length ? ", then " : ""}</>}{body.length > 0 && <>the text in <b className="font-medium text-foreground">{body.join(", ")}</b></>}</> : "No text column found"}
            {" · "}<span className="tabular">{usable.toLocaleString()} rows</span>
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 md:grid-cols-3">
          {analysis.preview_cases.slice(0, 3).map((c, i) => (
            <div key={i} className="rounded-lg bg-muted/50 p-3 text-xs">
              <p className="line-clamp-6 whitespace-pre-wrap text-muted-foreground">{c.case}</p>
              <div className="mt-2 flex flex-wrap gap-1">{Object.entries(c.answers).map(([k, v]) => <Badge key={k} variant="outline" className="font-normal">{k}: {v}</Badge>)}</div>
            </div>
          ))}
        </CardContent>
      </Card>

      {review.length > 0 && (
        <Alert className="mb-4">
          <AlertTriangle />
          <AlertTitle>Needs your eye</AlertTitle>
          <AlertDescription><ul className="list-disc pl-4">{review.map((r) => <li key={r}>{r}</li>)}</ul></AlertDescription>
        </Alert>
      )}

      <div className="mb-2 flex items-center justify-between">
        <h2 className="flex items-center gap-2 text-sm font-medium"><ListChecks className="size-4" /> Decisions</h2>
        <span className="text-sm text-muted-foreground tabular">{chosen} of {plan.decisions.length} selected</span>
      </div>
      <div className="grid gap-3 md:grid-cols-2">
        {plan.decisions.map((d, i) => <DecisionCard key={d.column} p={project} d={d} set={(patch) => setDecision(i, patch)} />)}
      </div>

      {plan.issues.length > 0 && (
        <Card className="mt-4" size="sm">
          <CardHeader><CardTitle>Data health</CardTitle></CardHeader>
          <CardContent>
            <ul className="grid gap-1.5 text-sm">
              {plan.issues.map((s, i) => (
                <li key={i} className="flex gap-2"><span className="text-muted-foreground">•</span><span>{s.detail}{s.action ? <span className="text-muted-foreground"> {s.action}</span> : null}</span></li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      {plan.excluded.length > 0 && (
        <Collapsible className="mt-4">
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="-ml-2">Ignored columns ({plan.excluded.length}) <ChevronDown data-icon="inline-end" /></Button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <ul className="grid gap-1 pt-2 text-sm">
              {plan.excluded.map((e) => <li key={e.column}><span className="font-mono">{e.column}</span> <span className="text-muted-foreground">: {WHY[e.why] ?? e.why}. {e.reason}</span></li>)}
            </ul>
          </CollapsibleContent>
        </Collapsible>
      )}

      <Card className="mt-6" size="sm">
        <CardFooter className="justify-between">
          <Button variant="ghost" onClick={back}>Back</Button>
          <Button onClick={next} disabled={!chosen || usable < 100}>{usable < 100 ? "Need at least 100 rows" : <>Continue <ArrowRight data-icon="inline-end" /></>}</Button>
        </CardFooter>
      </Card>
    </>
  )
}
