import { useEffect, useState } from "react"
import { AlertCircle, ArrowRight, Check, ChevronDown, Circle, Loader2, X } from "lucide-react"
import { toast } from "sonner"

import { msg, OptionCard, PageHead } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { api, type TrainStatus } from "@/lib/api"
import { BASES, included, PRESETS, type Preset, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

const PHASES: { phase: TrainStatus["phase"]; label: string }[] = [
  { phase: "starting_gpu", label: "Starting a GPU" },
  { phase: "uploading", label: "Uploading your data" },
  { phase: "training", label: "Learning from your decisions" },
  { phase: "evaluating", label: "Testing on cases it hasn't seen" },
  { phase: "downloading", label: "Bringing your model back" },
]
const ORDER = ["queued", "starting_gpu", "uploading", "training", "evaluating", "downloading", "done"]

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
  const [status, setStatus] = useState<TrainStatus | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const s = project.settings
  const live = status && status.phase !== "done" && status.phase !== "failed"
  const elapsed = useElapsed(live ? status.started_at : undefined)
  const base = BASES[project.base]
  const minutes = Math.round(base.minutes * (PRESETS[s.preset].steps / 400))
  const dollars = Math.max(1, Math.round(base.dollars * (PRESETS[s.preset].steps / 400)))

  useEffect(() => {
    if (!project.slug || !live) return
    const t = setInterval(async () => {
      try {
        const st = await api.trainStatus(project.slug!)
        setStatus(st)
        if (st.phase === "done") {
          update({ run: (st.run as string) ?? `co_${project.slug}`, resultsKey: project.slug })
          toast.success(`${project.name} is trained`)
        }
      } catch (e) {
        setError(msg(e))
      }
    }, 4000)
    return () => clearInterval(t)
  }, [project.slug, live]) // eslint-disable-line react-hooks/exhaustive-deps

  async function start() {
    setStarting(true)
    setError(null)
    try {
      const built = await api.buildPlan({
        records_token: project.analysis.records_token, plan: project.plan, name: project.name, base: project.base, enrich: project.enrich,
        settings: { steps: PRESETS[s.preset].steps, holdout: s.holdout, seed: s.seed }, anthropic_key: project.anthropicKey,
      })
      update({ slug: built.job, run: undefined })
      setStatus(await api.trainV2({ name: project.name, base: project.base, steps: PRESETS[s.preset].steps }))
    } catch (e) {
      setError(msg(e))
    } finally {
      setStarting(false)
    }
  }

  if (project.sample)
    return (
      <>
        <PageHead title="Train">Northwind's model is already trained, so the sample skips the wait.</PageHead>
        <Card>
          <CardHeader>
            <CardTitle>Northwind triage is ready</CardTitle>
            <CardDescription>Trained from {BASES.small.label} on Northwind's tickets with the Balanced preset: about 10 minutes on one rented GPU. A fifth of the tickets were kept back, so the results are measured on cases it never saw.</CardDescription>
          </CardHeader>
          <CardFooter className="justify-between">
            <Button variant="ghost" onClick={back}>Back</Button>
            <Button onClick={next}>See how it does <ArrowRight data-icon="inline-end" /></Button>
          </CardFooter>
        </Card>
      </>
    )

  const at = status ? ORDER.indexOf(status.phase) : -1
  return (
    <>
      <PageHead title="Train">
        We rent a GPU, teach {base.label} your {included(project).length} decisions, test it on cases it never saw, and shut the GPU down.
      </PageHead>
      <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
        {!status ? (
          <Card>
            <CardHeader><CardTitle>How thorough</CardTitle><CardDescription>Balanced suits most data. Longer training helps large or messy data, and costs more.</CardDescription></CardHeader>
            <CardContent className="grid gap-5">
              <RadioGroup value={s.preset} onValueChange={(v) => update({ settings: { ...s, preset: v as Preset } })} className="grid gap-3 sm:grid-cols-3">
                {(Object.keys(PRESETS) as Preset[]).map((k) => (
                  <Label key={k} htmlFor={`pre-${k}`} className="cursor-pointer font-normal">
                    <OptionCard selected={s.preset === k} className="w-full">
                      <div className="flex items-start gap-3">
                        <RadioGroupItem value={k} id={`pre-${k}`} className="mt-0.5" />
                        <div className="grid gap-0.5">
                          <span className="font-medium">{PRESETS[k].label}</span>
                          <span className="text-xs text-muted-foreground">{PRESETS[k].note}</span>
                        </div>
                      </div>
                    </OptionCard>
                  </Label>
                ))}
              </RadioGroup>
              <Collapsible>
                <CollapsibleTrigger asChild><Button variant="ghost" size="sm" className="-ml-2">Advanced <ChevronDown data-icon="inline-end" /></Button></CollapsibleTrigger>
                <CollapsibleContent className="grid gap-4 pt-2 sm:grid-cols-3">
                  <div className="grid gap-1.5"><Label htmlFor="steps">Training steps</Label><Input id="steps" value={PRESETS[s.preset].steps} readOnly className="tabular" /></div>
                  <div className="grid gap-1.5"><Label htmlFor="holdout">Kept back to test (%)</Label>
                    <Input id="holdout" type="number" min={10} max={40} value={s.holdout} onChange={(e) => update({ settings: { ...s, holdout: Math.min(40, Math.max(10, Number(e.target.value) || 20)) } })} /></div>
                  <div className="grid gap-1.5"><Label htmlFor="seed">Seed</Label>
                    <Input id="seed" type="number" value={s.seed} onChange={(e) => update({ settings: { ...s, seed: Number(e.target.value) || 0 } })} /></div>
                  <p className="text-xs text-muted-foreground sm:col-span-3">Learning rates and adapter size stay at the values the released model was trained with, since your model starts from it.</p>
                </CollapsibleContent>
              </Collapsible>
            </CardContent>
            <CardFooter className="justify-between">
              <Button variant="ghost" onClick={back}>Back</Button>
              <Button onClick={start} disabled={starting}>
                {starting && <Loader2 data-icon="inline-start" className="animate-spin" />} Start training
              </Button>
            </CardFooter>
          </Card>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>{status.phase === "done" ? "Trained" : status.phase === "failed" ? "Stopped" : "Training"}</CardTitle>
              <CardDescription>{status.message}</CardDescription>
              {elapsed && <CardAction><span className="font-mono text-sm text-muted-foreground tabular">{elapsed}</span></CardAction>}
            </CardHeader>
            <CardContent>
              <ol className="space-y-4">
                {PHASES.map((p) => {
                  const idx = ORDER.indexOf(p.phase)
                  const state = status.phase === "failed" && idx === Math.max(at, 1) ? "failed" : idx < at || status.phase === "done" ? "done" : idx === at ? "current" : "pending"
                  return (
                    <li key={p.phase} className="flex gap-3">
                      <span className="mt-0.5">
                        {state === "done" ? <Check className="size-4 text-primary" /> : state === "current" ? <Loader2 className="size-4 animate-spin" /> : state === "failed" ? <X className="size-4 text-destructive" /> : <Circle className="size-4 text-muted-foreground/50" />}
                      </span>
                      <div className="flex-1">
                        <p className={cn("text-sm", state === "pending" && "text-muted-foreground")}>{p.label}</p>
                        {p.phase === "training" && state === "current" && typeof status.progress === "number" && (
                          <Progress value={(status.progress as number) * 100} className="mt-2 h-1.5" aria-label="Training progress" />
                        )}
                      </div>
                    </li>
                  )
                })}
              </ol>
            </CardContent>
            <CardFooter className="justify-end">
              {status.phase === "done" && <Button onClick={next}>See how it does <ArrowRight data-icon="inline-end" /></Button>}
              {status.phase === "failed" && <Button onClick={start}>Try again</Button>}
            </CardFooter>
          </Card>
        )}
        <Card size="sm" className="self-start">
          <CardHeader><CardTitle className="text-sm">Summary</CardTitle></CardHeader>
          <CardContent>
            <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
              <dt className="text-muted-foreground">Model</dt><dd className="truncate">{project.name}</dd>
              <dt className="text-muted-foreground">Starts from</dt><dd>{base.label}</dd>
              <dt className="text-muted-foreground">Decisions</dt><dd className="tabular">{included(project).length}</dd>
              <dt className="text-muted-foreground">Tested on</dt><dd className="tabular">{Math.round(project.analysis.n_rows * s.holdout / 100).toLocaleString()} cases</dd>
              <dt className="text-muted-foreground">Takes</dt><dd className="tabular">about {minutes} min</dd>
              <dt className="text-muted-foreground">Costs</dt><dd className="tabular">about ${dollars} of GPU time</dd>
            </dl>
          </CardContent>
        </Card>
      </div>
      {(error || status?.phase === "failed") && (
        <Alert variant="destructive" className="mt-4">
          <AlertCircle />
          <AlertTitle>Training stopped</AlertTitle>
          <AlertDescription>{error ?? status?.message} The GPU was shut down, so nothing is still running or billing.</AlertDescription>
        </Alert>
      )}
    </>
  )
}
