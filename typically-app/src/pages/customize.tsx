import { useEffect, useState } from "react"
import { Check } from "lucide-react"

import { useI18n } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import type { Project } from "@/lib/project"
import { go } from "@/lib/router"
import { cn } from "@/lib/utils"
import { CreateStep } from "@/steps/create"
import { EnrichStep } from "@/steps/enrich"
import { TrainStep } from "@/steps/train"
import { UnderstandStep } from "@/steps/understand"

const STEPS = ["Create", "Understand", "Enrich", "Train"] as const
type Step = (typeof STEPS)[number]

const reachable = (s: Step, p: Project | null) =>
  s === "Create" || (Boolean(p) && (s !== "Train" || p!.plan.decisions.some((d) => d.include)))

/** Customize: the four-step fine-tune flow. Training continues in the background; the model then lives under Models. */
export function CustomizePage() {
  const { t } = useI18n()
  const { refresh } = useLibrary()
  const [step, setStep] = useState<Step>("Create")
  const [project, setProject] = useState<Project | null>(null)
  const update = (patch: Partial<Project>) => setProject((p) => (p ? { ...p, ...patch } : p))
  const at = STEPS.indexOf(step)

  useEffect(() => { if (project?.slug) refresh() }, [project?.slug, project?.run, refresh])

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 md:px-6">
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("new.title")}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{t("new.lede")}</p>
        </div>
        <ol className="flex items-center gap-1" aria-label="Progress">
          {STEPS.map((s, i) => {
            const done = i < at
            return (
              <li key={s} className="flex items-center">
                {i > 0 && <span aria-hidden className={cn("mx-1 h-px w-5 bg-border", i <= at && "bg-primary/50")} />}
                <button type="button" disabled={!reachable(s, project)} onClick={() => setStep(s)} aria-current={s === step ? "step" : undefined}
                  className={cn("flex items-center gap-1.5 rounded-md px-1.5 py-1 text-sm outline-none focus-visible:ring-3 focus-visible:ring-ring/50 disabled:opacity-50",
                    s === step ? "font-medium" : "text-muted-foreground hover:text-foreground")}>
                  <span className={cn("grid size-5 place-items-center rounded-full border text-[11px] tabular",
                    s === step && "border-primary bg-primary text-primary-foreground", done && "border-primary/40 bg-primary/10 text-primary")}>
                    {done ? <Check className="size-3" /> : i + 1}
                  </span>
                  <span className={cn(s === step ? "inline" : "hidden md:inline")}>{t(`step.${s}`)}</span>
                </button>
              </li>
            )
          })}
        </ol>
      </div>
      {step === "Create" && <CreateStep project={project} setProject={setProject} next={() => setStep("Understand")} />}
      {step === "Understand" && project && <UnderstandStep project={project} update={update} back={() => setStep("Create")} next={() => setStep("Enrich")} />}
      {step === "Enrich" && project && <EnrichStep project={project} update={update} back={() => setStep("Understand")} next={() => setStep("Train")} />}
      {step === "Train" && project && (
        <TrainStep project={project} update={update} back={() => setStep("Enrich")} next={() => go({ name: "model", id: project.sample ? "northwind" : project.slug! })} />
      )}
    </div>
  )
}
