import { useState } from "react"
import { Check, Copy } from "lucide-react"
import { toast } from "sonner"

import { SectionHeader } from "@/components/layout"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/lib/i18n"
import { pct } from "@/lib/project"
import { cn } from "@/lib/utils"

export const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))

const CODES = ["queued", "preparing", "gpu_starting", "uploading", "baseline", "training", "evaluating", "downloading", "done",
  "failed_provider", "failed_infra", "failed_timeout", "failed_training", "failed_build"] as const
const DETAILED = ["failed_infra", "failed_timeout", "failed_training", "failed_build"] // generic copy; the server's message carries the cause

/** A training job's status line: the translated copy for its `code`, else the server's message (code "failed": shutdown unconfirmed). */
export function StatusCopy({ code, message }: { code?: unknown; message?: string }) {
  const { t } = useI18n()
  const c = CODES.find((x) => x === code)
  if (!c) return <>{message}</>
  return <>{t(`code.${c}`)}{DETAILED.includes(c) && message && <span dir="auto" className="mt-1 block text-xs opacity-80">{message}</span>}</>
}

/** Step / panel heading inside a page (the page itself owns the h1). Spacing comes from the parent grid. */
export function PageHead({ title, children, action }: { title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return <SectionHeader title={title} description={children} actions={action} />
}

export function Meter({ value, tone }: { value: number; tone: "standard" | "yours" }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-10 text-end font-mono text-sm tabular">{pct(value)}</span>
      <div className="h-1.5 w-full min-w-8 overflow-hidden rounded-full bg-muted sm:min-w-16">
        <div className={cn("h-full rounded-full transition-[width] duration-200", tone === "standard" ? "bg-standard" : "bg-yours")} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

export function CopyButton({ text, label }: { text: string; label?: string }) {
  const { t } = useI18n()
  const [done, setDone] = useState(false)
  return (
    <Button variant="outline" size="sm" onClick={() => navigator.clipboard.writeText(text).then(() => (setDone(true), toast.success(t("common.copied")), setTimeout(() => setDone(false), 1500)))}>
      {done ? <Check data-icon="inline-start" /> : <Copy data-icon="inline-start" />} {label ?? t("common.copy")}
    </Button>
  )
}

export function CodeBlock({ code }: { code: string }) {
  return (
    <div className="relative rounded-lg border bg-muted/40">
      <div className="absolute end-2 top-2"><CopyButton text={code} /></div>
      <pre className="max-h-80 overflow-auto p-4 pe-24 font-mono text-xs leading-relaxed"><code>{code}</code></pre>
    </div>
  )
}

/** A selectable option card (radio semantics come from the caller's RadioGroupItem). */
export function OptionCard({ selected, children, className }: { selected: boolean; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("rounded-lg border p-4 transition-colors has-[:focus-visible]:ring-3 has-[:focus-visible]:ring-ring/50", selected ? "border-primary bg-primary/5" : "hover:bg-muted/50", className)}>
      {children}
    </div>
  )
}
