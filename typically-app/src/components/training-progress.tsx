import { useEffect, useRef, useState } from "react"
import { Check, Circle, Cpu, Loader2, X } from "lucide-react"

import { TrainingCurve } from "@/components/training-curve"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import type { JobEvent, TrainStatus } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import type { LANGUAGES } from "@/lib/project"
import { cn } from "@/lib/utils"

// the job's milestones in order (scripts/typically_job.py + the pod script's `mark` lines); each starts at its event's time
const STEPS = ["prepare", "gpu_request", "gpu_assigned", "upload", "env", "model_download", "model_load", "train", "testing", "fetch", "gpu_off"] as const
type Step = (typeof STEPS)[number]
type State = "done" | "current" | "failed" | "pending"
const LOG_TAIL = 6
const WITH_DETAIL = ["gpu_assigned", "upload", "model_download", "fetch"]   // their detail is data worth showing (a GPU name, a size)

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.max(0, Math.floor(s % 60))).padStart(2, "0")}`
const secs = (a: string, b: string | number) => ((typeof b === "number" ? b : Date.parse(b)) - Date.parse(a)) / 1000

function useNow(live: boolean) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!live) return
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [live])
  return now
}

/** Each milestone's state, start event and duration; `failedAt`: the milestone the job stopped in. */
function milestones(st: TrainStatus) {
  const events = st.events ?? []
  const at = new Map(events.map((e) => [e.key, e]))
  const order = STEPS.filter((k) => k !== "prepare" || at.has("prepare"))
  const fail = events.findIndex((e) => e.key === "failed")
  const failedAt = fail < 0 ? null
    : ([...events.slice(0, fail)].reverse().find((e) => (order as readonly string[]).includes(e.key) && e.key !== "gpu_off")?.key as Step | undefined) ?? order[0]
  const reached = order.filter((k) => at.has(k) && k !== "gpu_off")
  const last = reached[reached.length - 1]
  const over = st.phase === "done" || st.phase === "failed"
  const end = (k: Step) => {   // the next milestone's start (or done / failed)
    const later = [...order.slice(order.indexOf(k) + 1).filter((x) => x !== "gpu_off"), "done", "failed"]
    return later.map((x) => at.get(x)?.t).find(Boolean)
  }
  const state = (k: Step): State => {
    if (k === "gpu_off") return at.has(k) ? "done" : over ? "current" : "pending"
    if (failedAt) return k === failedAt ? "failed" : order.indexOf(k) < order.indexOf(failedAt) ? "done" : "pending"
    if (st.phase === "done") return "done"
    if (k === last) return "current"
    return last && order.indexOf(k) < order.indexOf(last) ? "done" : "pending"
  }
  return { order, at, failedAt, end, state }
}

/** The label of a milestone (or any event in the activity log), with its data filled in. */
function useLabel() {
  const { t, lang } = useI18n()
  const num = (n: number | string | undefined) => (n == null ? "" : Number(n).toLocaleString(lang === "he" ? "he-IL" : "en"))
  return (e: JobEvent | undefined, k: string, state: State = "done") => {
    if (k.startsWith("translate_")) return t("ev.translate", { lang: t(`lang.${k.slice(10) as (typeof LANGUAGES)[number]}`), n: num(e?.n), of: num(e?.of) })
    switch (k) {
      case "prepare": return e?.n ? t("ev.prepared", { n: num(e.n) }) : t("ev.prepare")
      case "synthetic": return t("ev.synthetic", { n: num(e?.n), of: num(e?.of) })
      case "gpu_request": return t("ev.gpu_request")
      case "gpu_assigned": return t(state === "current" ? "ev.booting" : "ev.gpu_assigned")
      case "upload": return e?.n ? t("ev.upload", { n: num(e.n) }) : t("ev.uploadPlain")
      case "model_load": return e?.detail ? t("ev.model_load", { n: num(e.detail) }) : t("ev.model_loadPlain")
      case "testing": return e?.detail ? t("ev.testing", { n: num(e.detail) }) : t("ev.testingPlain")
      case "gpu_off": return t(state === "done" ? "ev.gpu_off" : "ev.gpu_offing")
      case "env": case "model_download": case "train": case "fetch": case "done": case "failed": return t(`ev.${k}`)
      default: return k
    }
  }
}

/**
 * A running (or finished) training job, for someone watching it: every milestone in plain language with its time, the
 * step counter + curves while it learns, a small GPU panel, and a timestamped activity feed. The Train step and the model page share it.
 */
export function TrainingProgress({ status }: { status: TrainStatus }) {
  const { t, lang } = useI18n()
  const label = useLabel()
  const live = status.phase !== "done" && status.phase !== "failed"
  const now = useNow(live)
  const { order, at, end, state } = milestones(status)
  const s = status.series
  const total = status.steps ?? 0
  const step = s?.step.length ? s.step[s.step.length - 1] : 0
  const eta = s?.step_time && total > step ? Math.ceil(((total - step) * s.step_time) / 60) : null
  const progressKeys = (status.events ?? []).filter((e) => e.key === "synthetic" || e.key.startsWith("translate_"))

  return (
    <div className="grid gap-6">
      <ol className="grid gap-3">
        {order.map((k) => {
          const e = at.get(k), st = state(k), stop = end(k)
          const took = e && st === "done" && stop ? secs(e.t, stop) : e && st === "current" ? secs(e.t, now) : null
          return (
            <li key={k} className="flex gap-3">
              <span className="mt-0.5 shrink-0">
                {st === "done" ? <Check className="size-4 text-primary" /> : st === "current" ? <Loader2 className="size-4 animate-spin" />
                  : st === "failed" ? <X className="size-4 text-destructive" /> : <Circle className="size-4 text-muted-foreground/50" />}
              </span>
              <div className="grid min-w-0 flex-1 gap-1.5">
                <div className="flex items-baseline justify-between gap-3">
                  <p className={cn("text-sm", st === "pending" && "text-muted-foreground", st === "failed" && "font-medium text-destructive")}>
                    {label(e, k, st)}
                    {e?.detail && WITH_DETAIL.includes(k) && (
                      <span className="text-muted-foreground"> · <bdi className="font-mono text-xs">{e.detail}</bdi></span>
                    )}
                  </p>
                  {took != null && took >= 0 && k !== "gpu_off" && (
                    <span className={cn("shrink-0 font-mono text-xs tabular", st === "current" ? "text-foreground" : "text-muted-foreground")}>{clock(took)}</span>
                  )}
                </div>
                {k === "prepare" && progressKeys.length > 0 && (
                  <ul className="grid gap-1.5">
                    {progressKeys.map((p) => (
                      <li key={p.key} className="grid gap-1 text-xs text-muted-foreground">
                        <span className="tabular">{label(p, p.key)}</span>
                        {st === "current" && <Progress value={((p.n ?? 0) / (p.of || 1)) * 100} className="h-1" aria-label={label(p, p.key)} />}
                      </li>
                    ))}
                  </ul>
                )}
                {k === "train" && (st === "current" || st === "failed") && step > 0 && (
                  <>
                    <p className="text-xs text-muted-foreground tabular">
                      {t("chart.stepOf", { step: step.toLocaleString(), total: total.toLocaleString() })}
                      {eta != null && <> · {eta <= 1 ? t("chart.etaSoon") : t("chart.eta", { n: eta })}</>}
                    </p>
                    <Progress value={(step / (total || 1)) * 100} className="h-1.5" aria-label={t("tr.progress")} />
                  </>
                )}
                {k === "train" && st === "current" && !step && <p className="text-xs text-muted-foreground">{t("chart.waiting")}</p>}
              </div>
            </li>
          )
        })}
      </ol>

      {live && status.gpu && <GpuPanel status={status} />}
      {s && s.step.length > 0 && <TrainingCurve status={status} />}
      <ActivityLog events={status.events ?? []} lang={lang} label={label} />
    </div>
  )
}

function GpuPanel({ status }: { status: TrainStatus }) {
  const { t } = useI18n()
  const g = status.gpu!
  const name = status.events?.find((e) => e.key === "gpu_assigned")?.detail
  const gb = (mib: number) => (mib / 1024).toFixed(0)
  const items = [
    [t("gpu.busy"), `${g.util}%`],
    [t("gpu.memory"), t("gpu.gb", { used: gb(g.mem_used), total: gb(g.mem_total) })],
    ...(g.tok_s ? [[t("gpu.speed"), t("gpu.tokens", { n: g.tok_s.toLocaleString() })]] : []),
  ]
  return (
    <section className="grid gap-2 rounded-lg border bg-muted/30 p-3" aria-label={t("gpu.title")}>
      <p className="flex items-center gap-2 text-xs text-muted-foreground">
        <Cpu className="size-3.5" /> {t("gpu.title")}{name && <> · <bdi className="font-mono">{name}</bdi></>}
      </p>
      <dl className="grid grid-cols-3 gap-2">
        {items.map(([k, v]) => (
          <div key={k} className="grid gap-0.5">
            <dt className="text-xs text-muted-foreground">{k}</dt>
            <dd className="font-mono text-sm tabular">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function ActivityLog({ events, lang, label }: { events: JobEvent[]; lang: string; label: ReturnType<typeof useLabel> }) {
  const { t } = useI18n()
  const [all, setAll] = useState(false)
  const end = useRef<HTMLLIElement>(null)
  const shown = all ? events : events.slice(-LOG_TAIL)
  const time = new Intl.DateTimeFormat(lang === "he" ? "he-IL" : "en", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
  useEffect(() => { if (all) end.current?.scrollIntoView({ block: "nearest" }) }, [all, events.length])
  if (!events.length) return null
  return (
    <section className="grid gap-2" aria-label={t("log.title")}>
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium">{t("log.title")}</p>
        {events.length > LOG_TAIL && (
          <Button variant="ghost" size="sm" className="-me-2 h-7 text-xs" onClick={() => setAll(!all)}>
            {all ? t("log.less") : t("log.all", { n: events.length })}
          </Button>
        )}
      </div>
      <ol className={cn("grid gap-1 text-xs", all && "max-h-64 overflow-y-auto")} aria-live="polite">
        {shown.map((e, i) => (
          <li key={e.key} ref={i === shown.length - 1 ? end : undefined} className="flex gap-3">
            <time dateTime={e.t} className="shrink-0 font-mono text-muted-foreground tabular">{time.format(new Date(e.t))}</time>
            <span className="min-w-0">
              {label(e, e.key)}
              {e.detail && WITH_DETAIL.includes(e.key) && <span className="text-muted-foreground"> · <bdi>{e.detail}</bdi></span>}
            </span>
          </li>
        ))}
      </ol>
    </section>
  )
}
