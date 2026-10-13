import { useEffect, useRef, useState } from "react"
import { AlertCircle, ArrowRight, ChevronDown, Loader2 } from "lucide-react"
import { toast } from "sonner"

import { msg, OptionCard, PageHead, StatusCopy, StepFooter } from "@/components/shared"
import { Elapsed, settled, StatusLine, StopButton, TrainingProgress, useTrainStatus } from "@/components/training-progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { api, type TrainStatus } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { BASES, included, minutes, PRESETS, type Preset, type Project } from "@/lib/project"

const See = ({ next }: { next: () => void }) => {
  const { t } = useI18n()
  return <Button onClick={next}>{t("tr.see")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></Button>
}

export function TrainStep({ project, update, back, next }: { project: Project; update: (p: Partial<Project>) => void; back: () => void; next: () => void }) {
  const { t } = useI18n()
  const [started, setStarted] = useState<TrainStatus | null>(null) // teach()'s answer, until the first poll lands
  const [run, setRun] = useState(0)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null) // teach() itself was refused (name taken, quota): no GPU was involved
  const s = project.settings
  const [holdout, setHoldout] = useState(String(s.holdout))
  // the job outlives this step: coming back to it (stepper, Back) re-reads the job from the server instead of showing the form again
  const { status: polled, lost } = useTrainStatus(project.slug, !project.sample, run)
  const status = polled ?? started
  const live = status && status.phase !== "done" && status.phase !== "failed"
  const base = BASES[project.base]
  const prev = useRef(status?.phase)

  useEffect(() => {
    const phase = status?.phase
    if (phase === "done" && project.slug) {
      if (!project.run) update({ run: (status!.run as string) ?? `co_${project.slug}`, resultsKey: project.slug })
      if (prev.current && prev.current !== "done") toast.success(t("tr.toast", { name: project.name }))
    }
    prev.current = phase
  }, [status?.phase]) // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setStarting(true)
    setError(null)
    try {
      // one call that returns at once: preparing the examples (incl. AI enrichment) is the job's first milestone on the card
      const st = await api.teach({
        records_token: project.analysis.records_token, plan: project.plan, name: project.name, base: project.base, enrich: project.enrich,
        settings: { steps: PRESETS[s.preset].steps, holdout: s.holdout, seed: s.seed },
      })
      prev.current = st.phase
      update({ slug: st.job, run: undefined })
      setStarted(st)
      setRun((r) => r + 1)
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
        </Card>
        <StepFooter back={back}><See next={next} /></StepFooter>
      </>
    )

  return (
    <>
      <PageHead title={t("tr.title")}>{t("tr.lede", { base: base.label, n: included(project).length })}</PageHead>
      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        {!status ? (
          <Card>
            <CardHeader><CardTitle>{t("tr.howThorough")}</CardTitle><CardDescription>{t("tr.howHint")}</CardDescription></CardHeader>
            <CardContent className="grid gap-4">
              <RadioGroup value={s.preset} onValueChange={(v) => update({ settings: { ...s, preset: v as Preset } })} className="grid gap-4 sm:grid-cols-3">
                {(Object.keys(PRESETS) as Preset[]).map((k) => (
                  <Label key={k} htmlFor={`pre-${k}`} className="cursor-pointer font-normal">
                    <OptionCard selected={s.preset === k} className="w-full">
                      <div className="flex items-start gap-3">
                        <RadioGroupItem value={k} id={`pre-${k}`} className="mt-0.5" />
                        <div className="grid gap-0.5">
                          <span className="font-medium">{t(`preset.${k}`)}</span>
                          <span className="text-sm text-muted-foreground">{t(`preset.${k}Note`)}</span>
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
                    {/* clamped on blur: clamping per keystroke made "15" impossible to type */}
                    <Input id="holdout" type="number" min={10} max={40} value={holdout} onChange={(e) => setHoldout(e.target.value)}
                      onBlur={() => {
                        const v = Math.min(40, Math.max(10, Math.round(Number(holdout)) || 20))
                        setHoldout(String(v))
                        update({ settings: { ...s, holdout: v } })
                      }} /></div>
                  <div className="grid gap-1.5"><Label htmlFor="seed">{t("tr.seed")}</Label>
                    <Input id="seed" type="number" value={s.seed} onChange={(e) => update({ settings: { ...s, seed: Number(e.target.value) || 0 } })} /></div>
                  <p className="text-xs text-muted-foreground sm:col-span-3">{t("tr.advNote")}</p>
                </CollapsibleContent>
              </Collapsible>
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>{t(status.phase === "done" ? "tr.trained" : status.phase === "failed" ? "tr.stopped" : "tr.training")}</CardTitle>
              <CardDescription><StatusLine status={status} /></CardDescription>
              {live && <CardAction><Elapsed from={status.started_at} className="font-mono text-sm text-muted-foreground tabular" /></CardAction>}
            </CardHeader>
            <CardContent className="grid gap-4">
              {lost && <p role="status" className="text-sm text-muted-foreground">{t("tr.lost")}</p>}
              <TrainingProgress status={status} />
            </CardContent>
          </Card>
        )}
        <Card size="sm" className="self-start">
          <CardHeader><CardTitle>{t("tr.summary")}</CardTitle></CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
              <dt className="text-muted-foreground">{t("tr.model")}</dt><dd dir="auto" className="truncate">{project.name}</dd>
              <dt className="text-muted-foreground">{t("tr.startsFrom")}</dt><dd>{base.label}</dd>
              <dt className="text-muted-foreground">{t("tr.decisions")}</dt><dd className="tabular">{included(project).length}</dd>
              <dt className="text-muted-foreground">{t("tr.testedOn")}</dt><dd className="tabular">{t("tr.cases", { n: Math.round(project.analysis.n_rows * s.holdout / 100).toLocaleString() })}</dd>
              <dt className="text-muted-foreground">{t("tr.takes")}</dt><dd className="tabular">{t("tr.minutes", { n: minutes(project.base, PRESETS[s.preset].steps) })}</dd>
            </dl>
          </CardContent>
        </Card>
      </div>
      {error && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertTitle>{t("tr.cantStart")}</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {status?.phase === "failed" && settled(status) && (
        <Alert variant="destructive">
          <AlertCircle />
          <AlertTitle>{t("tr.failed")}</AlertTitle>
          <AlertDescription><p><StatusCopy code={status.code} message={status.message} /></p></AlertDescription>
        </Alert>
      )}
      <StepFooter back={back}>
        {!status && <Button onClick={start} disabled={starting}>{starting && <Loader2 data-icon="inline-start" className="animate-spin" />} {t("tr.start")}</Button>}
        {live && project.slug && <StopButton slug={project.slug} />}
        {status?.phase === "done" && <See next={next} />}
        {status?.phase === "failed" && <Button onClick={start} disabled={starting}>{t("tr.retry")}</Button>}
      </StepFooter>
    </>
  )
}
