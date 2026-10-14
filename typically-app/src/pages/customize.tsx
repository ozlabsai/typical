import { useEffect, useState } from "react"
import { Check, Pencil } from "lucide-react"

import { Page, PageHeader } from "@/components/layout"
import { useI18n } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import { NEW_INTENT, SAMPLE, suggestName, type Intent, type Project } from "@/lib/project"
import { go } from "@/lib/router"
import { cn } from "@/lib/utils"
import { DataStep } from "@/steps/data"
import { TrainStep } from "@/steps/train"
import { WhatStep } from "@/steps/what"

const STEPS = ["What", "Data", "Train"] as const
type Step = (typeof STEPS)[number]

const reachable = (s: Step, p: Project | null) => s !== "Train" || Boolean(p?.plan.decisions.some((d) => d.include))
const fileName = (s: string) => s.replace(/\.(csv|xlsx|mbox|eml|zip)( \(.*\))?$/i, "").replace(/[_-]+/g, " ")

/** New model: what to decide, the data, train. Training continues in the background; the model then lives under Models. */
export function CustomizePage({ data }: { data?: string }) {
  const { t } = useI18n()
  const { refresh } = useLibrary()
  const [step, setStep] = useState<Step>("What")
  const [intent, setIntent] = useState<Intent>(NEW_INTENT)
  const [project, setProject] = useState<Project | null>(null)
  const [typed, setTyped] = useState<string | null>(null) // null: the name follows suggestName until the user types one
  const update = (patch: Partial<Project>) => setProject((p) => (p ? { ...p, ...patch } : p))
  const name = typed ?? suggestName(intent, project ? (project.sample ? SAMPLE.name : fileName(project.analysis.name_hint)) : "")
  const settle = () => { if (!typed?.trim()) setTyped(null) }   // emptied: back to the suggestion
  const at = STEPS.indexOf(step)

  useEffect(() => { if (project?.slug) refresh() }, [project?.slug, project?.run, refresh])
  useEffect(() => { document.getElementById("scroller")?.scrollTo(0, 0) }, [step])   // Continue sits at the bottom; each step starts at its top

  return (
    <Page>
      <PageHeader
        title={
          <label className="group flex min-w-0 items-center gap-2">
            <input value={name} onChange={(e) => setTyped(e.target.value)} onBlur={settle} maxLength={80} dir="auto"
              placeholder={t("new.untitled")} aria-label={t("tr.model")}
              className="t-display field-sizing-content max-w-full min-w-[6ch] rounded-md bg-transparent outline-none placeholder:text-muted-foreground/70 focus-visible:ring-3 focus-visible:ring-ring/50" />
            <Pencil className="size-4 shrink-0 text-muted-foreground opacity-60 group-hover:opacity-100" aria-hidden />
          </label>
        }
        description={t("new.lede")}
        actions={
          <ol className="flex items-center gap-1" aria-label={t("new.progress")}>
            {STEPS.map((s, i) => {
              const done = i < at
              return (
                <li key={s} className="flex items-center">
                  {i > 0 && <span aria-hidden className={cn("mx-1 h-px w-5 bg-border", i <= at && "bg-primary/50")} />}
                  <button type="button" disabled={!reachable(s, project)} onClick={() => setStep(s)} aria-current={s === step ? "step" : undefined}
                    className={cn("flex items-center gap-1.5 rounded-md px-1.5 py-1 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed",
                      s === step ? "font-medium" : "text-muted-foreground hover:text-foreground")}>
                    <span className={cn("grid size-5 place-items-center rounded-full border text-xs tabular",
                      s === step && "border-primary bg-primary text-primary-foreground", done && "border-primary/40 bg-primary/10 text-primary")}>
                      {done ? <Check className="size-3" /> : i + 1}
                    </span>
                    <span className={cn(s === step ? "inline" : "hidden md:inline")}>{t(`step.${s}`)}</span>
                  </button>
                </li>
              )
            })}
          </ol>
        } />
      {step === "What" && <WhatStep intent={intent} setIntent={setIntent} next={() => setStep("Data")} />}
      {step === "Data" && <DataStep project={project} setProject={setProject} intent={intent} from={data} back={() => setStep("What")} next={() => setStep("Train")} />}
      {step === "Train" && project && (
        <TrainStep project={project} update={update} name={name} setName={setTyped} back={() => setStep("Data")}
          next={() => go({ name: "model", id: project.sample ? "northwind" : project.slug! })} />
      )}
    </Page>
  )
}
