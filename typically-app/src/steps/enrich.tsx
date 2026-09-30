import { ArrowRight, Info } from "lucide-react"

import { PageHead } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { included, LANGUAGES, type Enrich, type Project } from "@/lib/project"

function Row({ id, title, children, checked, onChange, tags, disabled }: {
  id: string; title: string; children: React.ReactNode; checked: boolean; onChange?: (v: boolean) => void; tags?: React.ReactNode; disabled?: boolean
}) {
  return (
    <div className="flex items-start justify-between gap-6 py-4">
      <div className="grid gap-1">
        <Label htmlFor={id} className="flex flex-wrap items-center gap-2">{title}{tags}</Label>
        <div className="max-w-2xl text-sm text-muted-foreground">{children}</div>
      </div>
      <Switch id={id} checked={checked} onCheckedChange={onChange} disabled={disabled || !onChange} />
    </div>
  )
}

export function EnrichStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
  const e = project.enrich
  const set = (patch: Partial<Enrich>) => update({ enrich: { ...e, ...patch } })
  const ai = Boolean(project.anthropicKey)
  const needsKey = !ai && <Badge variant="outline" className="font-normal">Needs an AI key (Create step)</Badge>
  const experimental = <Badge variant="secondary">Experimental</Badge>
  const policyOn = Object.keys(e.policy).length > 0
  const train = Math.round(project.analysis.n_rows * (1 - project.settings.holdout / 100))

  return (
    <>
      <PageHead title="Improve your data">Optional. The defaults are what we recommend; the cases we keep back for testing are never changed.</PageHead>
      <Card>
        <CardContent className="divide-y py-0">
          <Row id="rules" title="Teach it to follow new rules" checked tags={<Badge variant="outline" className="font-normal">Always on</Badge>}>
            About 1 in 10 examples states a different rule and its answer, so your model still follows instructions you give it later instead of only repeating the past.
          </Row>
          <Row id="balance" title="Balance rare answers" checked={e.balance} onChange={(v) => set({ balance: v })}>
            Answers you rarely gave are shown more often while it learns, so it doesn't learn to ignore them.
          </Row>
          <Row id="dedupe" title="Merge identical cases" checked={e.dedupe_soft} onChange={(v) => set({ dedupe_soft: v })}>
            When the same case got different answers, it learns the split (say 70/30) instead of a coin flip.
          </Row>
          <div className="py-4">
            <Row id="policy" title="Add your written policy" checked={policyOn} onChange={(v) => set({ policy: v ? Object.fromEntries(included(project).map((d) => [d.column, ""])) : {} })}>
              One line per decision in your own words, like "Refunds only under $300 and never for starter plans". Helps with rules that examples alone don't show.
            </Row>
            {policyOn && (
              <div className="grid gap-3 pb-1">
                {included(project).map((d) => (
                  <div key={d.column} className="grid gap-1.5">
                    <Label htmlFor={`pol-${d.column}`} className="text-xs text-muted-foreground">{d.question}</Label>
                    <Textarea id={`pol-${d.column}`} rows={2} value={e.policy[d.column] ?? ""} onChange={(ev) => set({ policy: { ...e.policy, [d.column]: ev.target.value } })} placeholder="Optional" />
                  </div>
                ))}
              </div>
            )}
          </div>
          <Row id="synthetic" title="Write extra examples of rare answers" checked={e.synthetic} onChange={(v) => set({ synthetic: v })} disabled={!ai} tags={<>{experimental}{needsKey}</>}>
            Claude writes new cases in the style of yours for answers with few examples. Capped at 15% of the training data; you'll spot-check 10 before training.
          </Row>
          <div className="py-4">
            <Row id="langs" title="Understand other languages" checked={e.languages.length > 0} onChange={(v) => set({ languages: v ? ["es"] : [] })} disabled={!ai} tags={<>{experimental}{needsKey}</>}>
              Translates a fifth of your cases into each language, keeping your answers, and tests on translated cases too.
              {project.plan.languages.length > 0 && <> Your data is {project.plan.languages.map((l) => `${l.code} ${Math.round(l.share * 100)}%`).join(", ")}.</>}
            </Row>
            {e.languages.length > 0 && (
              <div className="flex flex-wrap gap-4 pb-1">
                {LANGUAGES.map(([code, name]) => (
                  <Label key={code} className="flex items-center gap-2 font-normal">
                    <Checkbox checked={e.languages.includes(code)} onCheckedChange={(v) => set({ languages: v ? [...e.languages, code] : e.languages.filter((c) => c !== code) })} />
                    {name}
                  </Label>
                ))}
              </div>
            )}
          </div>
        </CardContent>
        <Separator />
        <CardFooter className="justify-between">
          <Button variant="ghost" onClick={back}>Back</Button>
          <div className="flex items-center gap-4">
            <span className="hidden items-center gap-1.5 text-sm text-muted-foreground sm:flex"><Info className="size-3.5" /> Learns from about {train.toLocaleString()} of your cases</span>
            <Button onClick={next}>Continue <ArrowRight data-icon="inline-end" /></Button>
          </div>
        </CardFooter>
      </Card>
    </>
  )
}
