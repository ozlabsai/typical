import { useEffect, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { api } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { included, LANGUAGES, type Enrich, type Project } from "@/lib/project"

/** One option: title + explanation + switch; `more` (the policy fields, the language picks) sits under it in the same row. */
function Row({ id, title, children, checked, onChange, tags, disabled, more }: {
  id: string; title: string; children: React.ReactNode; checked: boolean; onChange?: (v: boolean) => void; tags?: React.ReactNode; disabled?: boolean; more?: React.ReactNode
}) {
  return (
    <div className="grid gap-4 py-4">
      <div className="flex items-start justify-between gap-6">
        <div className="grid gap-1">
          <Label htmlFor={id} className="flex flex-wrap items-center gap-2">{title}{tags}</Label>
          <div className="max-w-2xl text-sm text-muted-foreground">{children}</div>
        </div>
        <Switch id={id} checked={checked} onCheckedChange={onChange} disabled={disabled || !onChange} />
      </div>
      {more}
    </div>
  )
}

/** The optional extras of a training run, under step 3's "Advanced options". */
export function EnrichOptions({ project, update }: { project: Project; update: (p: Partial<Project>) => void }) {
  const { t, language } = useI18n()
  const [ai, setAi] = useState(false)
  useEffect(() => void api.capabilities().then((c) => setAi(c.ai)).catch(() => setAi(false)), [])
  const e = project.enrich
  const set = (patch: Partial<Enrich>) => update({ enrich: { ...e, ...patch } })
  const needsKey = !ai && <Badge variant="outline" className="font-normal">{t("enr.unavailable")}</Badge>
  const experimental = <Badge variant="secondary">{t("enr.experimental")}</Badge>
  const policyOn = Object.keys(e.policy).length > 0
  const yours = project.plan.languages.filter((l) => l.share >= 0.005).map((l) => `${language(l.code)} ${Math.round(l.share * 100)}%`).join(", ")

  return (
    <div className="divide-y">
      <Row id="rules" title={t("enr.rules")} checked tags={<Badge variant="outline" className="font-normal">{t("enr.always")}</Badge>}>
        {t("enr.rulesBody")}
      </Row>
      <Row id="balance" title={t("enr.balance")} checked={e.balance} onChange={(v) => set({ balance: v })}>
        {t("enr.balanceBody")}
      </Row>
      <Row id="dedupe" title={t("enr.dedupe")} checked={e.dedupe_soft} onChange={(v) => set({ dedupe_soft: v })}>
        {t("enr.dedupeBody")}
      </Row>
      <Row id="policy" title={t("enr.policy")} checked={policyOn} onChange={(v) => set({ policy: v ? Object.fromEntries(included(project).map((d) => [d.column, ""])) : {} })}
        more={policyOn && (
          <div className="grid gap-4">
            {included(project).map((d) => (
              <div key={d.column} className="grid gap-1.5">
                <Label htmlFor={`pol-${d.column}`} dir="auto" className="font-normal text-muted-foreground">{d.question}</Label>
                <Textarea id={`pol-${d.column}`} dir="auto" rows={2} value={e.policy[d.column] ?? ""} onChange={(ev) => set({ policy: { ...e.policy, [d.column]: ev.target.value } })} placeholder={t("enr.optional")} />
              </div>
            ))}
          </div>
        )}>
        {t("enr.policyBody")}
      </Row>
      <Row id="synthetic" title={t("enr.synthetic")} checked={e.synthetic} onChange={(v) => set({ synthetic: v })} disabled={!ai} tags={<>{experimental}{needsKey}</>}>
        {t("enr.syntheticBody")}
      </Row>
      <Row id="langs" title={t("enr.langs")} checked={e.languages.length > 0} onChange={(v) => set({ languages: v ? ["es"] : [] })} disabled={!ai} tags={<>{experimental}{needsKey}</>}
        more={e.languages.length > 0 && (
          <div className="flex flex-wrap gap-4">
            {LANGUAGES.map((code) => (
              <Label key={code} className="flex items-center gap-2 font-normal">
                <Checkbox checked={e.languages.includes(code)} onCheckedChange={(v) => set({ languages: v ? [...e.languages, code] : e.languages.filter((c) => c !== code) })} />
                {language(code)}
              </Label>
            ))}
          </div>
        )}>
        {t("enr.langsBody")}
        {yours && <> {t("enr.yourData", { l: yours })}</>}
      </Row>
    </div>
  )
}
