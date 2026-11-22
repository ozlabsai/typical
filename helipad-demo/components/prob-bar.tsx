"use client";
// Segmented probability bar, adapted from the 21st.dev "Partition Bar" (8starlabs): one segment per option,
// sized by its probability. Every option is listed with its number, never only the top one.
import { motion } from "motion/react";
import { useLang } from "@/components/lang";
import { stColor, useReduceMotion } from "@/components/status";
import { pct, probsSentence } from "@/lib/describe";
import { tr } from "@/lib/i18n";
import { cn } from "@/lib/utils";

type Props = {
  entries: [string, number][];
  names: Record<string, string>;
  label: string;
  /** status colour per option; frame answers use neutral shades */
  color?: (option: string, i: number) => string;
  size?: "sm" | "md" | "lg";
  className?: string;
};

const NEUTRAL = ["#e6ece9", "#aab6b1", "#77847f", "#56625e", "#434d4a", "#353e3b"];
export const neutral = (_: string, i: number) => NEUTRAL[i % NEUTRAL.length];

export function ProbBar({ entries, names, label, color = stColor, size = "md", className }: Props) {
  const { lang } = useLang();
  const reduce = useReduceMotion();
  const shown = entries.map((e, i) => [...e, i] as const).filter(([, p]) => p > 0);
  return (
    <motion.ul
      role="img"
      aria-label={`${label}: ${probsSentence(entries, names)}`}
      className={cn("flex w-full gap-[2px]", size === "sm" ? "h-1" : size === "md" ? "h-1.5" : "h-2.5", className)}
      initial={reduce ? false : { clipPath: lang === "he" ? "inset(0 0 0 100%)" : "inset(0 100% 0 0)" }}
      animate={{ clipPath: "inset(0 0% 0 0%)" }}
      transition={{ duration: 0.6, ease: [0.23, 1, 0.32, 1] }}
    >
      {shown.map(([k, p, i]) => (
        <li key={k} className="h-full min-w-[2px] rounded-full" style={{ flexGrow: p, flexBasis: 0, background: color(k, i) }} />
      ))}
    </motion.ul>
  );
}

/** Every option with its percentage, in the bar's order; the reported one in full colour. */
export function ProbNumbers({ entries, names, top, color = stColor, className }:
  { entries: [string, number][]; names: Record<string, string>; top?: string; color?: Props["color"]; className?: string }) {
  return (
    <ul aria-hidden className={cn("flex flex-wrap gap-x-2 gap-y-0.5 text-[11px] leading-4", className)}>
      {entries.map(([k, p], i) => (
        <li key={k} className={cn("flex items-center gap-1", k === top ? "text-foreground" : "text-muted-foreground")}>
          <span className="inline-block size-1.5 rounded-full" style={{ background: color(k, i) }} />
          {tr(names, k)} <span className="num">{pct(p)}%</span>
        </li>
      ))}
    </ul>
  );
}
