import { useEffect, useId, useRef, useState } from "react"

import type { Series, TrainStatus } from "@/lib/api"
import { useI18n } from "@/lib/i18n"

const M = { top: 10, end: 14, bottom: 22, start: 40 }
const fmt = (v: number | null | undefined) => (v == null ? "—" : v.toFixed(3))
const pctOf = (v: number | null | undefined) => (v == null ? "—" : `${Math.round(v * 100)}%`)
const lastOf = (ys: (number | null)[]) => [...ys].reverse().find((v) => v != null) ?? null

function niceStep(span: number) {
  const p = 10 ** Math.floor(Math.log10(span || 1)), f = span / p
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p
}

type Line = { key: string; label: string; ys: (number | null)[]; color: string; dash?: string }

/**
 * One step-axis chart (the axis already spans the whole run, so the lines grow toward the end). Identity is never color alone
 * (dashes + the legend); hover for the values. `best`: index of the best checkpoint, ringed on the last line.
 */
function Chart({ s, total, lines, height, fmtY, title, desc, best, domain }: {
  s: Series; total: number; lines: Line[]; height: number; fmtY: (v: number | null) => string; title: string; desc: string
  best?: number | null; domain?: [number, number]
}) {
  const { t } = useI18n()
  const id = useId()
  const [hover, setHover] = useState<number | null>(null)
  const H = height
  // drawn at the figure's real pixel width, so the 10px labels stay 10px on a phone and on a wide screen
  const fig = useRef<HTMLElement>(null)
  const [W, setW] = useState(560)
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => e.contentRect.width > 0 && setW(Math.round(e.contentRect.width)))
    if (fig.current) ro.observe(fig.current)
    return () => ro.disconnect()
  }, [])
  const values = lines.flatMap((r) => r.ys).filter((v): v is number => v != null)
  const step = niceStep(((domain?.[1] ?? Math.max(...values)) - (domain?.[0] ?? Math.min(...values))) / 3)
  const lo = domain?.[0] ?? Math.max(0, Math.floor(Math.min(...values) / step) * step)
  const hi = domain?.[1] ?? (Math.ceil(Math.max(...values) / step) * step || lo + step)
  const yTicks = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step)
  const x = (v: number) => M.start + (v / total) * (W - M.start - M.end)
  const y = (v: number) => H - M.bottom - ((v - lo) / (hi - lo)) * (H - M.top - M.bottom)
  const pts = (ys: (number | null)[]) => s.step.flatMap((st, i) => (ys[i] == null ? [] : [[x(st), y(ys[i]!)] as const]))
  const path = (ys: (number | null)[]) => pts(ys).map(([a, b], i) => `${i ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join("")
  const h = hover ?? s.step.length - 1
  const ringed = lines[lines.length - 1]

  function onMove(e: React.PointerEvent<SVGRectElement>) {
    const box = e.currentTarget.getBoundingClientRect()
    const at = ((e.clientX - box.left) / box.width) * total
    // only steps where some line has a value
    const idx = s.step.map((_, i) => i).filter((i) => lines.some((r) => r.ys[i] != null))
    setHover(idx.reduce((b, i) => (Math.abs(s.step[i] - at) < Math.abs(s.step[b] - at) ? i : b), idx[0] ?? 0))
  }

  return (
    <figure ref={fig} dir="ltr" className="relative m-0">
      {/* the step axis reads left to right in both languages */}
      <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full overflow-visible" role="img" aria-labelledby={`${id}-t ${id}-d`}>
        <title id={`${id}-t`}>{title}</title>
        <desc id={`${id}-d`}>{desc}</desc>
        <g className="text-muted-foreground" fontSize="10" fill="currentColor" style={{ fontVariantNumeric: "tabular-nums" }}>
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={M.start} x2={W - M.end} y1={y(v)} y2={y(v)} stroke="var(--border)" />
              <text x={M.start - 6} y={y(v)} dy="0.32em" textAnchor="end">{domain ? pctOf(v) : v.toFixed(step < 0.1 ? 2 : 1)}</text>
            </g>
          ))}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => (
            <text key={f} x={x(f * total)} y={H - 6} textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"}>{Math.round(f * total)}</text>
          ))}
        </g>
        {best != null && ringed.ys[best] != null && (
          <g className="text-foreground">
            <line x1={x(s.step[best])} x2={x(s.step[best])} y1={M.top} y2={H - M.bottom} stroke="currentColor" strokeDasharray="2 3" opacity="0.4" />
            <circle cx={x(s.step[best])} cy={y(ringed.ys[best]!)} r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </g>
        )}
        {lines.map((r) => (
          <g key={r.key}>
            <path d={path(r.ys)} fill="none" stroke={r.color} strokeWidth="2" strokeDasharray={r.dash} strokeLinejoin="round" strokeLinecap="round" />
            {/* sparse lines (one value every 50 steps) also get their points */}
            {r.ys.filter((v) => v != null).length < 12 && pts(r.ys).map(([a, b], i) => <circle key={i} cx={a} cy={b} r="2.5" fill={r.color} />)}
          </g>
        ))}
        {hover != null && <line x1={x(s.step[h])} x2={x(s.step[h])} y1={M.top} y2={H - M.bottom} stroke="currentColor" className="text-muted-foreground/50" />}
        {lines.map((r) => r.ys[h] != null && (
          <circle key={r.key} cx={x(s.step[h])} cy={y(r.ys[h]!)} r="4" fill={r.color} stroke="var(--card)" strokeWidth="2" />
        ))}
        <rect x={M.start} y={0} width={W - M.start - M.end} height={H} fill="transparent" onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
      {hover != null && (
        <div className="pointer-events-none absolute top-0 z-10 rounded-md border bg-popover px-2.5 py-1.5 text-xs shadow-md"
          style={{ insetInlineStart: `${(x(s.step[h]) / W) * 100}%`, translate: s.step[h] / total < 0.2 ? "8px" : s.step[h] / total > 0.8 ? "calc(-100% - 8px)" : "-50%" }}>
          <p className="font-medium tabular">{t("chart.step")} {s.step[h]}</p>
          {lines.map((r) => <p key={r.key} className="flex justify-between gap-3 text-muted-foreground">{r.label}<span className="font-mono text-foreground tabular">{fmtY(r.ys[h])}</span></p>)}
        </div>
      )}
      <table className="sr-only">
        <caption>{title}</caption>
        <thead><tr><th>{t("chart.step")}</th>{lines.map((r) => <th key={r.key}>{r.label}</th>)}</tr></thead>
        <tbody>{s.step.map((st, i) => <tr key={st}><td>{st}</td>{lines.map((r) => <td key={r.key}>{fmtY(r.ys[i])}</td>)}</tr>)}</tbody>
      </table>
    </figure>
  )
}

function Legend({ lines, fmtY }: { lines: Line[]; fmtY: (v: number | null) => string }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-hidden>
      {lines.map((r) => (
        <li key={r.key} className="flex items-center gap-1.5">
          <svg width="18" height="6" className="shrink-0"><line x1="1" x2="17" y1="3" y2="3" stroke={r.color} strokeWidth="2" strokeDasharray={r.dash} strokeLinecap="round" /></svg>
          {r.label} <span className="font-mono text-foreground tabular">{fmtY(lastOf(r.ys))}</span>
        </li>
      ))}
    </ul>
  )
}

/**
 * Live training curves from the pod log: error on the training examples (every 10 steps) and on new examples (the company's
 * held-back validation examples, every 50 steps; best.pt is picked on it, ringed), then agreement with those examples in %.
 */
export function TrainingCurve({ status }: { status: TrainStatus }) {
  const { t } = useI18n()
  const s = status.series
  if (!s) return null
  if (!s.step.length) return <p className="text-sm text-muted-foreground">{t("chart.waiting")}</p>

  const total = Math.max(status.steps ?? 0, s.step[s.step.length - 1])
  const fresh = s.best_on.some((v) => v != null) ? s.best_on : s.val   // the company's own val; older logs only had val_nll
  const best = fresh.reduce<number | null>((b, v, i) => (v != null && (b == null || v < fresh[b]!) ? i : b), null)
  const err: Line[] = [
    { key: "loss", label: t("chart.loss"), ys: s.loss, color: "var(--primary)" },
    { key: "val", label: t("chart.val"), ys: fresh, color: "var(--standard)", dash: "5 4" },
  ]
  const agree: Line[] = [{ key: "acc", label: t("chart.acc"), ys: s.acc ?? [], color: "var(--yours)" }]
  const hasAcc = agree[0].ys.some((v) => v != null)

  return (
    <div className="grid gap-5">
      <div className="grid gap-2">
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
          <p className="text-sm font-medium">{t("chart.title")}</p>
          <Legend lines={err} fmtY={fmt} />
        </div>
        <Chart s={s} total={total} lines={err} height={176} fmtY={fmt} best={best} title={t("chart.title")}
          desc={t("chart.desc", { step: s.step[s.step.length - 1], loss: fmt(lastOf(s.loss)), val: fmt(lastOf(fresh)) })} />
        <p className="text-xs text-muted-foreground">
          {t("chart.lower")}{best != null && <> {t("chart.best", { step: s.step[best] })}</>}
        </p>
      </div>
      {hasAcc && (
        <div className="grid gap-2">
          <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1">
            <p className="text-sm font-medium">{t("chart.accTitle")}</p>
            <Legend lines={agree} fmtY={pctOf} />
          </div>
          <Chart s={s} total={total} lines={agree} height={110} fmtY={pctOf} domain={[0, 1]} title={t("chart.accTitle")}
            desc={t("chart.accDesc", { pct: pctOf(lastOf(agree[0].ys)) })} />
        </div>
      )}
    </div>
  )
}
