"use client";
// One site: the analysis plays in place (map, keyframes, the 11 items, the verdict), then settles on the result.
import { RotateCcw } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useMemo, useState } from "react";
import { Frames } from "@/components/frames";
import { useLang } from "@/components/lang";
import { ProbBar, ProbNumbers } from "@/components/prob-bar";
import { SatMap, type Highlight } from "@/components/sat-map";
import { arrowNav, StatusDot, stColor, useReduceMotion } from "@/components/status";
import { Button } from "@/components/ui/button";
import { decidedBy } from "@/lib/describe";
import { ltr, tr } from "@/lib/i18n";
import { FAILING, type Item, ITEMS, ordered, RINGS, type Site, STATUSES, VERDICTS } from "@/lib/protocol";
import { framesDone, itemsDone, mapProgress, schedule } from "@/lib/timeline";
import { cn } from "@/lib/utils";

const EASE = [0.23, 1, 0.32, 1] as const;

/** Milliseconds since the run started, ticking until `end`; reduced-motion users start at the end. */
function useClock(end: number) {
  const reduce = useReduceMotion();
  const [t, setT] = useState(0);
  useEffect(() => {
    if (reduce) return;
    const t0 = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const e = Math.max(0, Math.min(end, now - t0));
      setT(e);
      if (e < end) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [end, reduce]);
  return reduce ? end : t;
}

export function Inspection({ site }: { site: Site }) {
  const [run, setRun] = useState(0);
  return <Analysis key={`${site.id}-${run}`} site={site} onReplay={() => setRun((r) => r + 1)} />;
}

function Analysis({ site, onReplay }: { site: Site; onReplay: () => void }) {
  const { d, lang } = useLang();
  const frames = site.evidence.frames?.items ?? [];
  const s = useMemo(() => schedule(frames.length), [frames.length]);
  const t = useClock(s.end);
  const [hover, setHover] = useState<Item | null>(null);
  const items = useMemo(() => ITEMS.map((id) => site.items.find((i) => i.id === id)).filter((i) => !!i), [site]);
  const nItems = itemsDone(t, s);
  const nFrames = framesDone(t, s, frames.length);
  const verdictIn = t >= s.verdictAt;
  const afterIn = t >= s.afterAt;

  const highlight = useMemo<Highlight | null>(() => {
    const pick = hover ? items.filter((i) => i.id === hover) : verdictIn ? items.filter((i) => site.verdict.deciding.includes(i.id)) : [];
    if (!pick.length) return null;
    const obstacles = new Map<number, string>();
    const labels = new Set<number>();
    for (const it of pick) {
      for (const k of it.findings) obstacles.set(k, it.status);
      // one label per item, on its nearest finding, so labels do not pile up
      const near = it.findings.filter((k) => site.findings.obstacles[k].xy).sort((a, b) => site.findings.obstacles[a].dist_m - site.findings.obstacles[b].dist_m)[0];
      if (near !== undefined) labels.add(near);
    }
    const ring = pick.length === 1 && pick[0].id in RINGS ? pick[0].id : null;
    return { obstacles, labels, ring, ringStatus: pick[0].status };
  }, [hover, items, verdictIn, site.verdict.deciding, site.findings.obstacles]);

  const step = t < s.framesAt ? d.ui.stepMap : t < s.itemsAt ? d.ui.stepFrames : t < s.verdictAt ? d.ui.stepItems : d.ui.stepVerdict;
  const name = site.name[lang] || site.name.en;

  return (
    <div className={cn("grid h-full grid-cols-[minmax(0,1fr)_33rem] gap-5 p-5")}>
      {/* evidence: the map dominates; keyframes under it when the site has video */}
      <div className="flex min-h-0 min-w-0 flex-col gap-4">
        <div className="flex min-h-0 flex-1 items-center [container-type:size]">
          <SatMap site={site} progress={mapProgress(t)} highlight={highlight} />
        </div>
        {frames.length > 0 && (
          <Frames frames={frames} done={t >= s.framesAt ? nFrames : 0}
            active={t >= s.framesAt && t < s.itemsAt ? nFrames - 1 : -1} />
        )}
      </div>

      <div className="flex min-h-0 flex-col gap-4">
        {/* focal point: the verdict */}
        <section aria-live="polite" className="rounded-lg border bg-card px-4 py-3">
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <bdi className="text-sm font-medium text-foreground">{name}</bdi>
            <span>{d.ui.round} <span className="num">{site.round}</span></span>
            <span>{d.ui.aircraft} <span className="num">{site.aircraft}</span></span>
            <Button variant="ghost" size="sm" className="ms-auto -me-2 text-muted-foreground active:scale-[0.97]" onClick={onReplay}>
              <RotateCcw aria-hidden /> {d.ui.replay}
            </Button>
          </div>
          <AnimatePresence mode="wait" initial={false}>
            {verdictIn ? (
              <motion.div key="verdict" initial={{ opacity: 0, transform: "translateY(6px)", filter: "blur(4px)" }}
                animate={{ opacity: 1, transform: "translateY(0px)", filter: "blur(0px)" }} transition={{ duration: 0.45, ease: EASE }}>
                <p className="mt-1 text-[30px] leading-tight font-semibold tracking-tight" style={{ color: stColor(site.verdict.status) }}>
                  {d.verdict[site.verdict.status]}
                </p>
                <Deciding site={site} onHover={setHover} />
                <ProbBar className="mt-3" entries={ordered(site.verdict.probs, VERDICTS)} names={d.verdictOption} label={d.ui.verdict} size="lg" />
                <ProbNumbers className="mt-1.5" entries={ordered(site.verdict.probs, VERDICTS)} names={d.verdictOption} top={site.verdict.status} />
              </motion.div>
            ) : (
              <motion.div key="step" exit={{ opacity: 0 }} transition={{ duration: 0.15 }} className="mt-3">
                <p className="text-lg text-muted-foreground">{step}</p>
                <div className="mt-3 h-0.5 overflow-hidden rounded-full bg-border">
                  <div className="h-full origin-left bg-foreground/70 rtl:origin-right" style={{ transform: `scaleX(${t / s.verdictAt})` }} />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </section>

        {/* the 11 protocol items, every option with its probability */}
        <section aria-label={d.ui.items} className="flex min-h-0 flex-1 flex-col">
          <h2 className="mb-1 text-xs text-muted-foreground">{d.ui.items}</h2>
          <ul onKeyDown={arrowNav} className="thin-scroll -mx-2 min-h-0 flex-1 overflow-y-auto px-2">
            {items.map((it, i) => {
              const done = i < nItems;
              const entries = ordered(it.probs, STATUSES);
              const name = tr(d.items, it.id);
              return (
                <li key={it.id} data-nav tabIndex={0}
                  onMouseEnter={() => done && setHover(it.id)} onMouseLeave={() => setHover(null)}
                  onFocus={() => done && setHover(it.id)} onBlur={() => setHover(null)}
                  className={cn("grid grid-cols-[8.5rem_1fr] items-start gap-x-3 rounded-md px-2 py-0.5 transition-colors duration-150 outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    done && "hover:bg-accent", hover === it.id && "bg-accent")}>
                  <div className="min-w-0">
                    <div className={cn("truncate text-[13px] leading-[18px]", !done && "text-muted-foreground")}>{name}</div>
                    {done ? (
                      <div className="flex items-center gap-1.5 text-xs leading-4" style={{ color: stColor(it.status) }}>
                        <StatusDot status={it.status} className="size-1.5" />{d.status[it.status]}
                      </div>
                    ) : (
                      <div className="truncate text-xs leading-4 text-faint">{tr(d.hints, it.id)}</div>
                    )}
                  </div>
                  <div className="min-w-0 pt-1">
                    {done ? (
                      <>
                        <ProbBar entries={entries} names={d.status} label={name} />
                        <ProbNumbers className="mt-1" entries={entries} names={d.status} top={it.status} />
                      </>
                    ) : (
                      <div className={cn("h-1.5 rounded-full bg-border", i === nItems && t >= s.itemsAt && "animate-pulse")} />
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </section>

        {/* after the verdict: what to do before use, and the expert's call */}
        <motion.section initial={false} animate={{ opacity: afterIn ? 1 : 0 }} transition={{ duration: 0.35 }}
          aria-hidden={!afterIn} className="flex flex-col gap-2 border-t pt-3 text-xs">
          {site.conditions.length > 0 && (
            <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-1">
              <h2 className="w-16 shrink-0 text-muted-foreground">{d.ui.conditions}</h2>
              {site.conditions.map((c) => (
                <span key={c} className="rounded border px-1.5 leading-5">{tr(d.conditions, c)}</span>
              ))}
            </div>
          )}
          {site.prep.length > 0 && (
            <div className="flex flex-wrap items-baseline gap-x-1.5 gap-y-1">
              <h2 className="w-16 shrink-0" style={{ color: stColor("prep") }}>{d.ui.prep}</h2>
              {site.prep.map((c) => (
                <span key={c} className="rounded border px-1.5 leading-5" style={{ borderColor: "color-mix(in oklab, var(--st-prep) 45%, transparent)" }}>{tr(d.prep, c)}</span>
              ))}
            </div>
          )}
          {site.expert && (
            <p className="text-muted-foreground">
              {d.ui.expert}: <span className="text-foreground">{d.expertCall[site.expert.call]}</span>
              {site.expert.reason[lang] ? <>. {site.expert.reason[lang]}</> : null}
            </p>
          )}
        </motion.section>
      </div>
    </div>
  );
}

/** The verdict names its deciding items and their deciding findings. */
function Deciding({ site, onHover }: { site: Site; onHover: (i: Item | null) => void }) {
  const { d } = useLang();
  const deciding = site.verdict.deciding.filter((i) => FAILING.includes(site.items.find((x) => x.id === i)?.status ?? ""));
  if (!deciding.length) return <p className="mt-1 text-sm text-muted-foreground">{d.ui.noneFail}</p>;
  return (
    <div className="mt-1.5 text-sm">
      <span className="text-xs text-muted-foreground">{d.ui.decidedBy}</span>
      <ul className="mt-0.5 flex flex-col gap-0.5">
        {deciding.slice(0, 3).map((id) => {
          const it = site.items.find((x) => x.id === id)!;
          const f = decidedBy(id, site, d);
          return (
            <li key={id} className="flex items-baseline gap-2" onMouseEnter={() => onHover(id)} onMouseLeave={() => onHover(null)}>
              <StatusDot status={it.status} className="translate-y-[-1px]" />
              <span>
                <span className="font-medium">{tr(d.items, id)}</span>
                <span className="text-muted-foreground"> · {f.text}</span>
                {f.more > 0 && <span className="text-muted-foreground"> (+{ltr(f.more)} {d.ui.more})</span>}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
