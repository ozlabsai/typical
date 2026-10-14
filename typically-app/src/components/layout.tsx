import { createContext, useContext, useEffect, useState } from "react"
import { createPortal } from "react-dom"

import { cn } from "@/lib/utils"

/*
 * Layout tokens (the only sizes pages should use):
 *   header        h-14, the single top bar; page-level actions render into it via <HeaderActions>
 *   page          max-w-5xl, px-4 md:px-6, py-8;  chat column max-w-3xl
 *   rhythm        gap-6 between sections · gap-4 inside grids · gap-2 between inline controls
 *   type          page title .t-display · section label .eyebrow · section title text-base/semibold · body text-sm · meta text-xs
 *   numbers       stat values: big sans figures in a ruled band (the research site's figband)
 *   radii         cards/surfaces rounded-xl · inner tiles rounded-lg · pills rounded-full
 */

export function Page({ children, className, wide }: { children: React.ReactNode; className?: string; wide?: boolean }) {
  return <div className={cn("mx-auto grid w-full gap-6 px-4 py-8 md:px-6", wide ? "max-w-6xl" : "max-w-5xl", className)}>{children}</div>
}

export function PageHeader({ title, description, actions, badge }: { title: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="grid min-w-0 gap-1">
        <h1 className="t-display flex items-center gap-2">{title}{badge}</h1>
        {description && <div className="max-w-2xl text-sm text-muted-foreground">{description}</div>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function SectionHeader({ title, description, actions }: { title?: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="grid min-w-0 gap-1">
        {title && <h2 className="text-base font-semibold tracking-tight">{title}</h2>}
        {description && <p className="max-w-2xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Stat({ label, value, unit, tone, note }: { label: React.ReactNode; value: React.ReactNode; unit?: React.ReactNode; tone?: "standard" | "yours"; note?: React.ReactNode }) {
  return (
    <div className="grid min-w-0 content-start gap-2 border-s px-3 first:border-s-0 first:ps-0 sm:px-6">
      <span className="eyebrow line-clamp-2"><bdi>{label}</bdi></span>
      <span className={cn("text-3xl leading-none font-medium tracking-[-0.04em] tabular whitespace-nowrap sm:text-5xl", tone === "standard" && "text-standard", tone === "yours" && "text-yours")}>
        {value}{unit && <span className="ms-1.5 font-mono text-xs font-normal tracking-normal text-muted-foreground sm:text-sm">{unit}</span>}
      </span>
      {note && <span className="hidden text-xs text-muted-foreground sm:block">{note}</span>}
    </div>
  )
}

/** Typical's mark: a three-option distribution, one answer clearly ahead. Same drawing as public/favicon.svg. */
export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" aria-hidden className={cn("size-8 shrink-0", className)}>
      <rect width="16" height="16" rx="3.5" fill="#292827" />
      <rect x="3" y="9" width="2" height="3" fill="#f0eeeb" fillOpacity=".55" />
      <rect x="7" y="4" width="2" height="8" fill="#8cc5b0" />
      <rect x="11" y="7" width="2" height="5" fill="#f0eeeb" fillOpacity=".55" />
    </svg>
  )
}

/* Page-level actions rendered into the shell's single header bar (no second toolbar under it). */
const SlotCtx = createContext<HTMLElement | null>(null)
export function HeaderSlotProvider({ children }: { children: (setRef: (el: HTMLElement | null) => void) => React.ReactNode }) {
  const [el, setEl] = useState<HTMLElement | null>(null)
  return <SlotCtx.Provider value={el}>{children(setEl)}</SlotCtx.Provider>
}
export function HeaderActions({ children }: { children: React.ReactNode }) {
  const el = useContext(SlotCtx)
  const [ready, setReady] = useState(false)
  useEffect(() => { setReady(true) }, [])
  return el && ready ? createPortal(children, el) : null
}
