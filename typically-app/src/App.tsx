import { useState } from "react"
import { Check, Moon, RotateCcw, Sun } from "lucide-react"
import { ThemeProvider, useTheme } from "next-themes"

import { Button } from "@/components/ui/button"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import type { Project } from "@/lib/project"
import { cn } from "@/lib/utils"
import { CreateStep } from "@/steps/create"
import { DeployStep } from "@/steps/deploy"
import { EnrichStep } from "@/steps/enrich"
import { EvaluateStep } from "@/steps/evaluate"
import { TrainStep } from "@/steps/train"
import { UnderstandStep } from "@/steps/understand"

const STEPS = ["Create", "Understand", "Enrich", "Train", "Evaluate", "Deploy"] as const
export type Step = (typeof STEPS)[number]

function reachable(step: Step, p: Project | null) {
  if (step === "Create") return true
  if (!p) return false
  if (step === "Understand" || step === "Enrich") return true
  if (step === "Train") return p.plan.decisions.some((d) => d.include)
  return Boolean(p.run)
}

function Stepper({ step, project, go }: { step: Step; project: Project | null; go: (s: Step) => void }) {
  const at = STEPS.indexOf(step)
  return (
    <nav aria-label="Progress" className="border-b">
      <ol className="mx-auto flex max-w-6xl items-center gap-1 overflow-x-auto px-6 py-3">
        {STEPS.map((s, i) => {
          const done = i < at && reachable(STEPS[i + 1] ?? s, project)
          const ok = reachable(s, project)
          return (
            <li key={s} className="flex items-center gap-1">
              {i > 0 && <span aria-hidden className={cn("mx-1 h-px w-6 bg-border sm:w-10", i <= at && "bg-primary/50")} />}
              <button
                type="button"
                disabled={!ok}
                onClick={() => go(s)}
                aria-current={s === step ? "step" : undefined}
                className={cn(
                  "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                  s === step ? "font-medium text-foreground" : "text-muted-foreground hover:text-foreground",
                  !ok && "pointer-events-none opacity-50",
                )}
              >
                <span
                  className={cn(
                    "grid size-6 place-items-center rounded-full border text-xs tabular",
                    s === step && "border-primary bg-primary text-primary-foreground",
                    done && s !== step && "border-primary/40 bg-primary/10 text-primary",
                  )}
                >
                  {done && s !== step ? <Check className="size-3.5" /> : i + 1}
                </span>
                <span className="whitespace-nowrap">{s}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const dark = resolvedTheme === "dark"
  return (
    <Button variant="ghost" size="icon-sm" aria-label={dark ? "Switch to light theme" : "Switch to dark theme"} onClick={() => setTheme(dark ? "light" : "dark")}>
      {dark ? <Sun /> : <Moon />}
    </Button>
  )
}

export default function App() {
  const [step, setStep] = useState<Step>("Create")
  const [project, setProject] = useState<Project | null>(null)
  const update = (patch: Partial<Project>) => setProject((p) => (p ? { ...p, ...patch } : p))

  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
      <TooltipProvider delayDuration={300}>
        <div className="min-h-svh bg-background">
          <header className="border-b">
            <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-6">
              <div className="flex min-w-0 items-center gap-3 text-sm">
                <span className="flex items-center gap-2 font-semibold">
                  <span aria-hidden className="size-2.5 rounded-full bg-primary" />
                  typically
                </span>
                {project && (
                  <>
                    <span className="text-muted-foreground" aria-hidden>/</span>
                    <span className="truncate text-muted-foreground">{project.name}</span>
                  </>
                )}
              </div>
              <div className="flex items-center gap-1">
                {project && (
                  <Button variant="ghost" size="sm" onClick={() => (setProject(null), setStep("Create"))}>
                    <RotateCcw data-icon="inline-start" /> Start over
                  </Button>
                )}
                <ThemeToggle />
              </div>
            </div>
          </header>
          <Stepper step={step} project={project} go={setStep} />
          <main className="mx-auto max-w-6xl px-6 py-8">
            {step === "Create" && <CreateStep project={project} setProject={setProject} next={() => setStep("Understand")} />}
            {step === "Understand" && project && <UnderstandStep project={project} update={update} back={() => setStep("Create")} next={() => setStep("Enrich")} />}
            {step === "Enrich" && project && <EnrichStep project={project} update={update} back={() => setStep("Understand")} next={() => setStep("Train")} />}
            {step === "Train" && project && <TrainStep project={project} update={update} back={() => setStep("Enrich")} next={() => setStep("Evaluate")} />}
            {step === "Evaluate" && project && <EvaluateStep project={project} next={() => setStep("Deploy")} />}
            {step === "Deploy" && project && <DeployStep project={project} />}
          </main>
        </div>
        <Toaster position="bottom-right" />
      </TooltipProvider>
    </ThemeProvider>
  )
}
