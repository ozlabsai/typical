import { useEffect, useState } from "react"
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
import { api } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
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
  const { t } = useI18n()
  const [ai, setAi] = useState(false)
  useEffect(() => void api.capabilities().then((c) => setAi(c.ai)).catch(() => setAi(false)), [])
  const e = project.enrich
  const set = (patch: Partial<Enrich>) => update({ enrich: { ...e, ...patch } })
  const needsKey = !ai && <Badge variant="outline" className="font-normal">{t("enr.unavailable")}</Badge>
  const experimental = <Badge variant="secondary">{t("enr.experimental")}</Badge>
  const policyOn = Object.keys(e.policy).length > 0
  const train = Math.round(project.analysis.n_rows * (1 - project.settings.holdout / 100))

  return (
    <>
      <PageHead title={t("enr.title")}>{t("enr.lede")}</PageHead>
      <Card>
        <CardContent className="divide-y py-0">
          <Row id="rules" title={t("enr.rules")} checked tags={<Badge variant="outline" className="font-normal">{t("enr.always")}</Badge>}>
            {t("enr.rulesBody")}
          </Row>
          <Row id="balance" title={t("enr.balance")} checked={e.balance} onChange={(v) => set({ balance: v })}>
            {t("enr.balanceBody")}
          </Row>
          <Row id="dedupe" title={t("enr.dedupe")} checked={e.dedupe_soft} onChange={(v) => set({ dedupe_soft: v })}>
            {t("enr.dedupeBody")}
          </Row>
          <div className="py-4">
            <Row id="policy" title={t("enr.policy")} checked={policyOn} onChange={(v) => set({ policy: v ? Object.fromEntries(included(project).map((d) => [d.column, ""])) : {} })}>
              {t("enr.policyBody")}
            </Row>
            {policyOn && (
              <div className="grid gap-3 pb-1">
                {included(project).map((d) => (
                  <div key={d.column} className="grid gap-1.5">
                    <Label htmlFor={`pol-${d.column}`} dir="auto" className="text-xs text-muted-foreground">{d.question}</Label>
                    <Textarea id={`pol-${d.column}`} dir="auto" rows={2} value={e.policy[d.column] ?? ""} onChange={(ev) => set({ policy: { ...e.policy, [d.column]: ev.target.value } })} placeholder={t("enr.optional")} />
                  </div>
                ))}
              </div>
            )}
          </div>
          <Row id="synthetic" title={t("enr.synthetic")} checked={e.synthetic} onChange={(v) => set({ synthetic: v })} disabled={!ai} tags={<>{experimental}{needsKey}</>}>
            {t("enr.syntheticBody")}
          </Row>
          <div className="py-4">
            <Row id="langs" title={t("enr.langs")} checked={e.languages.length > 0} onChange={(v) => set({ languages: v ? ["es"] : [] })} disabled={!ai} tags={<>{experimental}{needsKey}</>}>
              {t("enr.langsBody")}
              {project.plan.languages.length > 0 && <> {t("enr.yourData", { l: project.plan.languages.filter((l) => l.share >= 0.005).map((l) => `${l.code} ${Math.round(l.share * 100)}%`).join(", ") })}</>}
            </Row>
            {e.languages.length > 0 && (
              <div className="flex flex-wrap gap-4 pb-1">
                {LANGUAGES.map((code) => (
                  <Label key={code} className="flex items-center gap-2 font-normal">
                    <Checkbox checked={e.languages.includes(code)} onCheckedChange={(v) => set({ languages: v ? [...e.languages, code] : e.languages.filter((c) => c !== code) })} />
                    {t(`lang.${code}`)}
                  </Label>
                ))}
              </div>
            )}
          </div>
        </CardContent>
        <Separator />
        <CardFooter className="justify-between">
          <Button variant="ghost" onClick={back}>{t("common.back")}</Button>
          <div className="flex items-center gap-4">
            <span className="hidden items-center gap-1.5 text-sm text-muted-foreground sm:flex"><Info className="size-3.5" /> {t("enr.learnsFrom", { n: train.toLocaleString() })}</span>
            <Button onClick={next}>{t("common.continue")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>
          </div>
        </CardFooter>
      </Card>
    </>
  )
}
