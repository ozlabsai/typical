import { useEffect, useId, useState } from "react"

import { api, type TrainStatus } from "@/lib/api"
import { useI18n } from "@/lib/i18n"

/** Poll GET /train/{slug} while `active` (the model page; the Train step polls on its own). */
export function useTrainStatus(slug: string, active: boolean) {
  const [st, setSt] = useState<TrainStatus | null>(null)
  useEffect(() => {
    if (!active) return
    const load = () => void api.trainStatus(slug).then(setSt).catch(() => {})
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [slug, active])
  return st
}

const W = 560, H = 176, M = { top: 10, end: 14, bottom: 22, start: 40 }
const fmt = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3))

function niceStep(span: number) {
  const p = 10 ** Math.floor(Math.log10(span || 1)), f = span / p
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p
}

/**
 * Live training curve from the pod log: train loss (primary, solid) and validation loss (standard, dashed: identity is never
 * color alone), on a step axis that already spans the whole run, so the line grows toward the end. Hover for the values.
 */
export function TrainingCurve({ status }: { status: TrainStatus }) {
  const { t } = useI18n()
  const id = useId()
  const [hover, setHover] = useState<number | null>(null)
  const s = status.series
  if (!s) return null
  if (!s.step.length) return <p className="text-sm text-muted-foreground">{t("chart.waiting")}</p>

  const last = s.step.length - 1
  const lastOf = (ys: (number | null)[]) => [...ys].reverse().find((v) => v != null) ?? null
  const total = Math.max(status.steps ?? 0, s.step[last])
  const values = [...s.loss, ...s.val].filter((v): v is number => v != null)
  const step = niceStep((Math.max(...values) - Math.min(...values)) / 3)
  const lo = Math.max(0, Math.floor(Math.min(...values) / step) * step), hi = Math.ceil(Math.max(...values) / step) * step || lo + step
  const yTicks = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step)
  const x = (v: number) => M.start + (v / total) * (W - M.start - M.end)
  const y = (v: number) => H - M.bottom - ((v - lo) / (hi - lo)) * (H - M.top - M.bottom)
  const path = (ys: (number | null)[]) =>
    s.step.flatMap((st, i) => (ys[i] == null ? [] : [`${x(st).toFixed(1)},${y(ys[i]!).toFixed(1)}`])).map((p, i) => (i ? "L" : "M") + p).join("")

  const bestAt = s.best_on.reduce<number | null>((b, v, i) => (v != null && (b == null || v < s.best_on[b]!) ? i : b), null)
  const remaining = total - s.step[last]
  const eta = s.step_time && remaining > 0 ? Math.ceil((remaining * s.step_time) / 60) : null
  const series = [
    { key: "loss", label: t("chart.loss"), ys: s.loss, color: "var(--primary)", dash: undefined },
    { key: "val", label: t("chart.val"), ys: s.val, color: "var(--standard)", dash: "5 4" },
  ] as const
  const h = hover ?? last

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const at = ((e.clientX - box.left) / box.width) * total
    setHover(s!.step.reduce((b, st, i) => (Math.abs(st - at) < Math.abs(s!.step[b] - at) ? i : b), 0))
  }

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 text-sm">
        <p className="tabular">
          <span className="font-medium">{t("chart.stepOf", { step: s.step[last], total })}</span>
          {eta != null && <span className="text-muted-foreground"> · {eta <= 1 ? t("chart.etaSoon") : t("chart.eta", { n: eta })}</span>}
          {bestAt != null && <span className="text-muted-foreground"> · {t("chart.best", { step: s.step[bestAt] })}</span>}
        </p>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-hidden>
          {series.map((r) => (
            <li key={r.key} className="flex items-center gap-1.5">
              <svg width="18" height="6" className="shrink-0"><line x1="1" x2="17" y1="3" y2="3" stroke={r.color} strokeWidth="2" strokeDasharray={r.dash} strokeLinecap="round" /></svg>
              {r.label} <span className="font-mono text-foreground tabular">{fmt(lastOf(r.ys))}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* the step axis reads left to right in both languages */}
      <figure dir="ltr" className="relative m-0">
        <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full overflow-visible" role="img" aria-labelledby={`${id}-t ${id}-d`}>
          <title id={`${id}-t`}>{t("chart.title")}</title>
          <desc id={`${id}-d`}>{t("chart.desc", { step: s.step[last], loss: fmt(lastOf(s.loss)), val: fmt(lastOf(s.val)) })}</desc>
          <g className="text-muted-foreground" fontSize="10" fill="currentColor" style={{ fontVariantNumeric: "tabular-nums" }}>
            {yTicks.map((v) => (
              <g key={v}>
                <line x1={M.start} x2={W - M.end} y1={y(v)} y2={y(v)} stroke="var(--border)" />
                <text x={M.start - 6} y={y(v)} dy="0.32em" textAnchor="end">{v.toFixed(step < 0.1 ? 2 : 1)}</text>
              </g>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <text key={f} x={x(f * total)} y={H - 6} textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"}>{Math.round(f * total)}</text>
            ))}
          </g>
          {series.map((r) => (
            <path key={r.key} d={path(r.ys)} fill="none" stroke={r.color} strokeWidth="2" strokeDasharray={r.dash} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {hover != null && <line x1={x(s.step[h])} x2={x(s.step[h])} y1={M.top} y2={H - M.bottom} stroke="currentColor" className="text-muted-foreground/50" />}
          {series.map((r) => r.ys[h] != null && (
            <circle key={r.key} cx={x(s.step[h])} cy={y(r.ys[h]!)} r="4" fill={r.color} stroke="var(--card)" strokeWidth="2" />
          ))}
          <rect x={M.start} y={0} width={W - M.start - M.end} height={H} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
        </svg>
        {hover != null && (
          <div className="pointer-events-none absolute top-0 z-10 rounded-md border bg-popover px-2.5 py-1.5 text-xs shadow-md"
            style={{ insetInlineStart: `${(x(s.step[h]) / W) * 100}%`, translate: s.step[h] / total < 0.2 ? "8px" : s.step[h] / total > 0.8 ? "calc(-100% - 8px)" : "-50%" }}>
            <p className="font-medium tabular">{t("chart.step")} {s.step[h]}</p>
            {series.map((r) => <p key={r.key} className="flex justify-between gap-3 text-muted-foreground">{r.label}<span className="font-mono text-foreground tabular">{fmt(r.ys[h])}</span></p>)}
          </div>
        )}
      </figure>

      <table className="sr-only">
        <caption>{t("chart.title")}</caption>
        <thead><tr><th>{t("chart.step")}</th>{series.map((r) => <th key={r.key}>{r.label}</th>)}</tr></thead>
        <tbody>{s.step.map((st, i) => <tr key={st}><td>{st}</td>{series.map((r) => <td key={r.key}>{fmt(r.ys[i])}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}
