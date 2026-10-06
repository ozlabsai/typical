import { useEffect, useRef, useState } from "react"
import { AlertCircle, ArrowRight, ChevronDown, Loader2 } from "lucide-react"
import { toast } from "sonner"

import { msg, OptionCard, PageHead, StatusCopy } from "@/components/shared"
import { settled, TrainingProgress } from "@/components/training-progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { api, type TrainStatus } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { BASES, included, PRESETS, type Preset, type Project } from "@/lib/project"

function useElapsed(since?: string) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!since) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [since])
  if (!since) return ""
  const s = Math.max(0, Math.floor((now - Date.parse(since)) / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}

export function TrainStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
  const { t } = useI18n()
  const [status, setStatus] = useState<TrainStatus | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const s = project.settings
  const live = status && status.phase !== "done" && status.phase !== "failed"
  const polling = !!status && !settled(status)
  const announced = useRef(false)
  const elapsed = useElapsed(live ? status.started_at : undefined)
  const base = BASES[project.base]
  const minutes = Math.round(base.minutes * (PRESETS[s.preset].steps / 400))

  useEffect(() => {
    if (!project.slug || !polling) return
    const timer = setInterval(async () => {
      try {
        const st = await api.trainStatus(project.slug!)
        setStatus(st)
        if (st.phase === "done" && !announced.current) {
          announced.current = true
          update({ run: (st.run as string) ?? `co_${project.slug}`, resultsKey: project.slug })
          toast.success(t("tr.toast", { name: project.name }))
        }
      } catch (e) {
        setError(msg(e))
      }
    }, 4000)
    return () => clearInterval(timer)
  }, [project.slug, polling]) // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setStarting(true)
    setError(null)
    announced.current = false
    try {
      // one call that returns at once: preparing the examples (incl. AI enrichment) is the job's first milestone on the card
      const st = await api.teach({
        records_token: project.analysis.records_token, plan: project.plan, name: project.name, base: project.base, enrich: project.enrich,
        settings: { steps: PRESETS[s.preset].steps, holdout: s.holdout, seed: s.seed },
      })
      update({ slug: st.job, run: undefined })
      setStatus(st)
    } catch (e) {
      setError(msg(e))
    } finally {
      setStarting(false)
    }
  }

  if (project.sample)
    return (
      <>
        <PageHead title={t("tr.title")}>{t("tr.sampleLede")}</PageHead>
        <Card>
          <CardHeader>
            <CardTitle>{t("tr.sampleReady", { name: project.name })}</CardTitle>
            <CardDescription>{t("tr.sampleBody", { base: BASES.small.label })}</CardDescription>
          </CardHeader>
          <CardFooter className="justify-between">
            <Button variant="ghost" onClick={back}>{t("common.back")}</Button>
            <Button onClick={next}>{t("tr.see")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>
          </CardFooter>
        </Card>
      </>
    )

  return (
    <>
      <PageHead title={t("tr.title")}>{t("tr.lede", { base: base.label, n: included(project).length })}</PageHead>
      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        {!status ? (
          <Card>
            <CardHeader><CardTitle>{t("tr.howThorough")}</CardTitle><CardDescription>{t("tr.howHint")}</CardDescription></CardHeader>
            <CardContent className="grid gap-5">
              <RadioGroup value={s.preset} onValueChange={(v) => update({ settings: { ...s, preset: v as Preset } })} className="grid gap-3 sm:grid-cols-3">
                {(Object.keys(PRESETS) as Preset[]).map((k) => (
                  <Label key={k} htmlFor={`pre-${k}`} className="cursor-pointer font-normal">
                    <OptionCard selected={s.preset === k} className="w-full">
                      <div className="flex items-start gap-3">
                        <RadioGroupItem value={k} id={`pre-${k}`} className="mt-0.5" />
                        <div className="grid gap-0.5">
                          <span className="font-medium">{t(`preset.${k}`)}</span>
                          <span className="text-xs text-muted-foreground">{t(`preset.${k}Note`)}</span>
                        </div>
                      </div>
                    </OptionCard>
                  </Label>
                ))}
              </RadioGroup>
              <Collapsible>
                <CollapsibleTrigger asChild><Button variant="ghost" size="sm" className="-ms-2">{t("tr.advanced")} <ChevronDown data-icon="inline-end" /></Button></CollapsibleTrigger>
                <CollapsibleContent className="grid gap-4 pt-2 sm:grid-cols-3">
                  <div className="grid gap-1.5"><Label htmlFor="steps">{t("tr.steps")}</Label><Input id="steps" value={PRESETS[s.preset].steps} readOnly className="tabular" /></div>
                  <div className="grid gap-1.5"><Label htmlFor="holdout">{t("tr.holdout")}</Label>
                    <Input id="holdout" type="number" min={10} max={40} value={s.holdout} onChange={(e) => update({ settings: { ...s, holdout: Math.min(40, Math.max(10, Number(e.target.value) || 20)) } })} /></div>
                  <div className="grid gap-1.5"><Label htmlFor="seed">{t("tr.seed")}</Label>
                    <Input id="seed" type="number" value={s.seed} onChange={(e) => update({ settings: { ...s, seed: Number(e.target.value) || 0 } })} /></div>
                  <p className="text-xs text-muted-foreground sm:col-span-3">{t("tr.advNote")}</p>
                </CollapsibleContent>
              </Collapsible>
            </CardContent>
            <CardFooter className="justify-between">
              <Button variant="ghost" onClick={back}>{t("common.back")}</Button>
              <Button onClick={start} disabled={starting}>
                {starting && <Loader2 data-icon="inline-start" className="animate-spin" />} {t("tr.start")}
              </Button>
            </CardFooter>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>{t(status.phase === "done" ? "tr.trained" : status.phase === "failed" ? "tr.stopped" : "tr.training")}</CardTitle>
              <CardDescription><StatusCopy code={status.code} message={status.message} /></CardDescription>
              {elapsed && <CardAction><span className="font-mono text-sm text-muted-foreground tabular">{elapsed}</span></CardAction>}
            </CardHeader>
            <CardContent><TrainingProgress status={status} /></CardContent>
            <CardFooter className="justify-end">
              {status.phase === "done" && <Button onClick={next}>{t("tr.see")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>}
              {status.phase === "failed" && <Button onClick={start}>{t("tr.retry")}</Button>}
            </CardFooter>
          </Card>
        )}
        <Card size="sm" className="self-start">
          <CardHeader><CardTitle className="text-sm">{t("tr.summary")}</CardTitle></CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
              <dt className="text-muted-foreground">{t("tr.model")}</dt><dd className="truncate">{project.name}</dd>
              <dt className="text-muted-foreground">{t("tr.startsFrom")}</dt><dd>{base.label}</dd>
              <dt className="text-muted-foreground">{t("tr.decisions")}</dt><dd className="tabular">{included(project).length}</dd>
              <dt className="text-muted-foreground">{t("tr.testedOn")}</dt><dd className="tabular">{t("tr.cases", { n: Math.round(project.analysis.n_rows * s.holdout / 100).toLocaleString() })}</dd>
              <dt className="text-muted-foreground">{t("tr.takes")}</dt><dd className="tabular">{t("tr.minutes", { n: minutes })}</dd>
            </dl>
          </CardContent>
        </Card>
      </div>
      {(error || status?.phase === "failed") && (
        <Alert variant="destructive" >
          <AlertCircle />
          <AlertTitle>{t("tr.failed")}</AlertTitle>
          <AlertDescription>{error ?? status?.message} {t("tr.failedBody")}</AlertDescription>
        </Alert>
      )}
    </>
  )
}
