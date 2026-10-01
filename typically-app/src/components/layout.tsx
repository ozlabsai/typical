import { createContext, useContext, useEffect, useState } from "react"
import { createPortal } from "react-dom"

import { cn } from "@/lib/utils"

/*
 * Layout tokens (the only sizes pages should use):
 *   header        h-14, the single top bar; page-level actions render into it via <HeaderActions>
 *   page          max-w-5xl, px-4 md:px-6, py-8;  chat column max-w-3xl
 *   rhythm        gap-6 between sections · gap-4 inside grids · gap-2 between inline controls
 *   type          page title text-xl/semibold · section title text-base/semibold · body text-sm · meta text-xs
 *   numbers       stat values text-2xl font-mono tabular
 *   radii         cards/surfaces rounded-xl · inner tiles rounded-lg · pills rounded-full
 */

export function Page({ children, className, wide }: { children: React.ReactNode; className?: string; wide?: boolean }) {
  return <div className={cn("mx-auto grid w-full gap-6 px-4 py-8 md:px-6", wide ? "max-w-6xl" : "max-w-5xl", className)}>{children}</div>
}

export function PageHeader({ title, description, actions, badge }: { title: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode; badge?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="grid min-w-0 gap-1">
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">{title}{badge}</h1>
        {description && <div className="max-w-2xl text-sm text-muted-foreground">{description}</div>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function SectionHeader({ title, description, actions }: { title: React.ReactNode; description?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4">
      <div className="grid min-w-0 gap-1">
        <h2 className="text-base font-semibold tracking-tight">{title}</h2>
        {description && <p className="max-w-2xl text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </div>
  )
}

export function Stat({ label, value, tone, note, highlight }: { label: React.ReactNode; value: React.ReactNode; tone?: "standard" | "yours"; note?: React.ReactNode; highlight?: boolean }) {
  return (
    <div className={cn("grid min-w-0 content-start gap-1 rounded-xl border bg-card p-3 sm:p-4", highlight && "border-primary/40")}>
      <span className="truncate text-xs text-muted-foreground" dir="auto">{label}</span>
      <span className={cn("font-mono text-xl font-semibold tabular sm:text-2xl", tone === "standard" && "text-standard", tone === "yours" && "text-yours")}>{value}</span>
      {note && <span className="hidden text-xs text-muted-foreground sm:block">{note}</span>}
    </div>
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
