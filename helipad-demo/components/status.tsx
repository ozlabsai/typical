"use client";
import { useSyncExternalStore } from "react";
import { cn } from "@/lib/utils";

const RM = "(prefers-reduced-motion: reduce)";
/** prefers-reduced-motion, hydration-safe: the server and the first client render both say false. */
export function useReduceMotion() {
  return useSyncExternalStore(
    (cb) => {
      const m = matchMedia(RM);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => matchMedia(RM).matches,
    () => false,
  );
}

/** Status colours live in CSS (--st-<status>); colour carries status only. */
export const stColor = (s: string) => `var(--st-${s}, var(--st-not_checked))`;

export function StatusDot({ status, className }: { status: string; className?: string }) {
  return <span aria-hidden className={cn("inline-block size-2 shrink-0 rounded-full", className)} style={{ background: stColor(status) }} />;
}

/** Arrow keys move focus between the [data-nav] elements of a list; Home/End jump to the ends. */
export function arrowNav(e: React.KeyboardEvent<HTMLElement>) {
  const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
  if (!keys.includes(e.key)) return;
  const els = [...e.currentTarget.querySelectorAll<HTMLElement>("[data-nav]")];
  const i = els.indexOf(document.activeElement as HTMLElement);
  const next = e.key === "Home" ? 0 : e.key === "End" ? els.length - 1 : i + (e.key === "ArrowDown" ? 1 : -1);
  if (els[next]) {
    e.preventDefault();
    els[next].focus();
  }
}
