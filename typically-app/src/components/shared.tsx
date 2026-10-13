import { useState } from "react"
import { AlertCircle, ArrowLeft, Check, Copy, RefreshCw } from "lucide-react"
import { toast } from "sonner"

import { SectionHeader } from "@/components/layout"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { useI18n, type T } from "@/lib/i18n"
import { pct } from "@/lib/project"
import { cn } from "@/lib/utils"

export const msg = (e: unknown) => (e instanceof Error ? e.message : String(e))

const CODES = ["queued", "preparing", "gpu_starting", "uploading", "baseline", "training", "evaluating", "downloading", "done",
  "failed_provider", "failed_infra", "failed_timeout", "failed_training", "failed_build", "failed_no_gpu", "failed_restart", "stopped", "failed"] as const
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
    <div className="flex items-center gap-2">
      <span className="w-10 text-end font-mono text-sm tabular">{pct(value)}</span>
      <div className="h-1.5 w-full min-w-8 overflow-hidden rounded-full bg-muted sm:min-w-16">
        <div className={cn("h-full rounded-full transition-[width] duration-200", tone === "standard" ? "bg-standard" : "bg-yours")} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  )
}

/** The one clipboard write: a toast either way (the clipboard is refused on http:// and in some embedded browsers). */
export const copy = (t: T, text: string, done = t("common.copied")) =>
  navigator.clipboard.writeText(text).then(() => (toast.success(done), true), () => (toast.error(t("common.copyFailed")), false))

export function CopyButton({ text, label }: { text: string; label?: string }) {
  const { t } = useI18n()
  const [done, setDone] = useState(false)
  return (
    <Button variant="outline" size="sm" onClick={() => copy(t, text).then((ok) => ok && (setDone(true), setTimeout(() => setDone(false), 1500)))}>
      {done ? <Check data-icon="inline-start" /> : <Copy data-icon="inline-start" />} {label ?? t("common.copy")}
    </Button>
  )
}

/** Code with its Copy button in a strip above it, so the button never covers the code on a phone. */
export function CodeBlock({ code, label }: { code: string; label?: string }) {
  return (
    <div className="overflow-hidden rounded-lg border bg-muted/40">
      <div className="flex items-center justify-between gap-2 border-b bg-muted/60 py-1.5 ps-4 pe-1.5">
        <span className="text-xs text-muted-foreground">{label}</span>
        <CopyButton text={code} />
      </div>
      <pre dir="ltr" className="max-h-80 overflow-auto p-4 text-start font-mono text-xs leading-relaxed"><code>{code}</code></pre>
    </div>
  )
}

/** The one footer of every wizard step: Back at the start, the step's own note + primary action at the end. Outside the cards. */
export function StepFooter({ back, children, note }: { back?: () => void; children?: React.ReactNode; note?: React.ReactNode }) {
  const { t } = useI18n()
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 border-t pt-4">
      {back ? <Button variant="ghost" onClick={back}><ArrowLeft data-icon="inline-start" className="rtl:rotate-180" /> {t("common.back")}</Button> : <span />}
      <div className="flex flex-wrap items-center justify-end gap-4">
        {note && <span className="text-sm text-muted-foreground">{note}</span>}
        {children}
      </div>
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

/** Something failed to load: what, why, and a way to try again. */
export function ErrorState({ title, detail, retry }: { title?: string; detail: string; retry: () => void }) {
  const { t } = useI18n()
  return (
    <Alert variant="destructive">
      <AlertCircle />
      {title && <AlertTitle>{title}</AlertTitle>}
      <AlertDescription className="grid gap-4">
        <p dir="auto">{detail}</p>
        <Button variant="outline" size="sm" className="justify-self-start" onClick={retry}><RefreshCw data-icon="inline-start" /> {t("common.retry")}</Button>
      </AlertDescription>
    </Alert>
  )
}
