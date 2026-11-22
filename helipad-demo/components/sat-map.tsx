"use client";
// The satellite crop with the pad, the 10/50/300 m rings and the obstacles. Geometry is SVG over the image;
// text is HTML on top, so it follows the language's direction while the map itself never mirrors.
import { motion } from "motion/react";
import { useLang } from "@/components/lang";
import { stColor } from "@/components/status";
import { obstacleText } from "@/lib/describe";
import { fracToVb, ringRect, scaleBar, sweepMax, VB } from "@/lib/geometry";
import { ltr, tr } from "@/lib/i18n";
import { RINGS, type Site } from "@/lib/protocol";

export type Highlight = { obstacles: Map<number, string>; labels: Set<number>; ring: string | null; ringStatus?: string };

const pctOf = (vb: number) => `${vb / 10}%`;

export function SatMap({ site, progress, highlight }: { site: Site; progress: number; highlight: Highlight | null }) {
  const { d } = useLang();
  const sat = site.evidence.satellite;
  const hw = sat.half_width_m;
  const [cx, cy] = fracToVb(sat.pad_xy);
  const sweep = progress * sweepMax(sat.pad_xy);
  const f = site.findings;
  const scale = scaleBar(hw);
  const shown = f.obstacles.map((o, k) => ({ o, k, p: o.xy && fracToVb(o.xy) }))
    .filter((x): x is { o: typeof x.o; k: number; p: [number, number] } => !!x.p && Math.hypot(x.p[0] - cx, x.p[1] - cy) <= sweep);
  const count = (n: number) => Math.round(n * progress);
  // video sightings have no bearing: listed, never placed on the map
  const isVideo = (k: number) => f.obstacles[k].bearing === null && !f.obstacles[k].in_pad;
  const nVideo = f.obstacles.filter((_, k) => isVideo(k)).length;
  const unmapped = [...(highlight?.obstacles.keys() ?? [])].filter(isVideo);
  const missing = Object.entries(f.layers).filter(([, ok]) => !ok).map(([k]) => k);

  return (
    <div className="relative mx-auto aspect-square w-[min(100cqw,100cqh)] overflow-hidden rounded-lg border bg-card">
      <svg viewBox={`0 0 ${VB} ${VB}`} className="absolute inset-0 size-full" role="img" aria-label={d.ui.map}>
        <image href={`/data/${sat.image}`} width={VB} height={VB} preserveAspectRatio="xMidYMid slice" />
        <rect width={VB} height={VB} fill="#0f1416" opacity={0.28} />
        {Object.entries(RINGS).reverse().map(([item, m]) => {
          const r = ringRect(sat.pad_xy, hw, m);
          const on = highlight?.ring === item;
          return (
            <motion.rect key={item} {...{ x: r.x, y: r.y, width: r.w, height: r.h, rx: r.rx }} fill="none"
              stroke={on ? stColor(highlight?.ringStatus ?? "not_checked") : "#e6ece9"} strokeOpacity={on ? 1 : 0.55}
              strokeWidth={on ? 2.5 : 1.25} strokeDasharray={on ? undefined : "6 5"} vectorEffect="non-scaling-stroke"
              initial={{ opacity: 0 }} animate={{ opacity: progress > 0.05 ? 1 : 0 }} transition={{ duration: 0.5 }} />
          );
        })}
        {(() => {
          const r = ringRect(sat.pad_xy, hw);
          return <rect x={r.x} y={r.y} width={r.w} height={r.h} fill="#e6ece9" fillOpacity={0.12} stroke="#e6ece9" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />;
        })()}
        {progress < 1 && (
          <circle cx={cx} cy={cy} r={sweep} fill="#e6ece9" fillOpacity={0.04} stroke="#e6ece9" strokeOpacity={0.5} strokeWidth={1} vectorEffect="non-scaling-stroke" />
        )}
        {shown.map(({ o, k, p }) => {
          const hl = highlight?.obstacles.get(k);
          const dim = highlight && highlight.obstacles.size > 0 && !hl;
          return (
            <motion.g key={k} initial={{ opacity: 0, scale: 0.6 }} animate={{ opacity: dim ? 0.4 : 1, scale: 1 }}
              transition={{ duration: 0.3, ease: [0.23, 1, 0.32, 1] }}>
              {hl && <circle cx={p[0]} cy={p[1]} r={17} fill="none" stroke={stColor(hl)} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
              <circle cx={p[0]} cy={p[1]} r={hl ? 8 : 6} fill={hl ? stColor(hl) : "#e6ece9"} stroke="#0f1416" strokeWidth={1.5} vectorEffect="non-scaling-stroke">
                <title>{obstacleText(o, d)}</title>
              </circle>
            </motion.g>
          );
        })}
      </svg>

      {/* ring labels, at the end of each ring's top edge */}
      {Object.entries(RINGS).map(([item, m]) => {
        const r = ringRect(sat.pad_xy, hw, m);
        if (item === "very_low_10" && highlight?.ring !== item) return null;
        return (
          <span key={item} className="pointer-events-none absolute -translate-x-full -translate-y-full pb-0.5 text-[11px] whitespace-nowrap text-foreground/80"
            style={{ left: pctOf(r.x + r.w - r.rx), top: pctOf(r.y), opacity: progress > 0.05 ? 1 : 0 }}>
            <span className="num">{m}</span> {d.ui.metres}
          </span>
        );
      })}
      {/* labels of highlighted obstacles */}
      {shown.filter(({ k }) => highlight?.labels.has(k)).map(({ o, k, p }) => (
        <span key={k} className="pointer-events-none absolute -translate-y-1/2 rounded bg-background/85 px-1.5 py-0.5 text-[11px] whitespace-nowrap"
          style={p[0] < 600 ? { left: `calc(${pctOf(p[0])} + 14px)`, top: pctOf(p[1]) } : { right: `calc(${pctOf(VB - p[0])} + 14px)`, top: pctOf(p[1]) }}>
          {obstacleText(o, d)}
        </span>
      ))}

      <div className="absolute start-3 top-3 rounded-md bg-background/80 px-2.5 py-1.5 text-xs leading-5 backdrop-blur-sm">
        <div><span className="num font-medium">{count(f.counts.obstacles_total)}</span> {d.ui.obstaclesMapped}</div>
        <div className="text-muted-foreground"><span className="num">{count(f.counts.within_300)}</span> {d.ui.within300}</div>
        {nVideo > 0 && <div className="text-muted-foreground"><span className="num">{count(nVideo)}</span> {d.ui.seenOnVideo}</div>}
        {unmapped.length > 0 && progress >= 1 && (
          <div className="mt-1.5 max-w-[17rem] border-t pt-1.5">
            <div className="text-[11px] text-muted-foreground">{d.ui.videoOnly}</div>
            <ul className="text-[11px] leading-4">
              {unmapped.slice(0, 3).map((k) => (
                <li key={k} className="flex items-baseline gap-1.5">
                  <span className="inline-block size-1.5 shrink-0 rounded-full" style={{ background: stColor(highlight!.obstacles.get(k)!) }} />
                  {obstacleText(f.obstacles[k], d)}
                </li>
              ))}
              {unmapped.length > 3 && <li className="text-muted-foreground">+{ltr(unmapped.length - 3)} {d.ui.more}</li>}
            </ul>
          </div>
        )}
      </div>
      <div aria-hidden className="absolute end-3 top-3 flex flex-col items-center rounded-md bg-background/80 px-2 py-1 text-[11px] leading-4">
        <span>↑</span>{d.ui.north}
      </div>
      <div className="absolute start-3 bottom-3 rounded-md bg-background/80 px-2.5 py-1.5 text-[11px] leading-4 text-muted-foreground">
        <div>
          {d.ui.layers}: <span className="text-foreground">{Object.entries(f.layers).filter(([, ok]) => ok).map(([k]) => tr(d.layers, k)).join(", ")}</span>
          {missing.length > 0 && <> · {d.ui.missing}: {missing.map((k) => tr(d.layers, k)).join(", ")}</>}
        </div>
        {site.touchdown.moved && (
          <div>{d.ui.moved} {ltr(site.touchdown.moved.m)} {d.ui.metres} {tr(d.dirs, site.touchdown.moved.dir)}</div>
        )}
      </div>
      <div className="absolute end-3 bottom-3 rounded-md bg-background/80 px-2.5 py-1.5 text-[11px] leading-4" style={{ width: `calc(${pctOf(scale.vb)} + 1.25rem)` }}>
        <div className="h-1.5 border-x border-b border-foreground" />
        <div className="num mt-0.5 text-center">{scale.m} {d.ui.metres}</div>
      </div>
    </div>
  );
}
