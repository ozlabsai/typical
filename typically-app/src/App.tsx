import { useState } from "react"
import { Check, Languages, Moon, RotateCcw, Sun } from "lucide-react"
import { ThemeProvider, useTheme } from "next-themes"

import { Button } from "@/components/ui/button"
import { Toaster } from "@/components/ui/sonner"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { I18nProvider, useI18n } from "@/lib/i18n"
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
  const { t } = useI18n()
  const at = STEPS.indexOf(step)
  return (
    <nav aria-label="Progress" className="min-w-0 flex-1">
      <ol className="flex items-center justify-center gap-0.5">
        {STEPS.map((s, i) => {
          const done = i < at && reachable(STEPS[i + 1] ?? s, project)
          const ok = reachable(s, project)
          return (
            <li key={s} className="flex items-center">
              {i > 0 && <span aria-hidden className={cn("mx-0.5 hidden h-px w-3 bg-border md:block lg:w-5", i <= at && "bg-primary/50")} />}
              <button
                type="button"
                disabled={!ok}
                onClick={() => go(s)}
                aria-current={s === step ? "step" : undefined}
                aria-label={t(`step.${s}`)}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-1.5 py-1 text-sm transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                  s === step ? "font-medium text-foreground" : "text-muted-foreground hover:text-foreground",
                  !ok && "pointer-events-none opacity-50",
                )}
              >
                <span className={cn(
                  "grid size-5 shrink-0 place-items-center rounded-full border text-[11px] tabular",
                  s === step && "border-primary bg-primary text-primary-foreground",
                  done && s !== step && "border-primary/40 bg-primary/10 text-primary",
                )}>
                  {done && s !== step ? <Check className="size-3" /> : i + 1}
                </span>
                <span className={cn("whitespace-nowrap", s === step ? "inline" : "hidden lg:inline")}>{t(`step.${s}`)}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}

function IconAction({ label, onClick, children }: { label: string; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild><Button variant="ghost" size="icon-sm" aria-label={label} onClick={onClick}>{children}</Button></TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  )
}

function Shell() {
  const { t, lang, setLang } = useI18n()
  const { resolvedTheme, setTheme } = useTheme()
  const [step, setStep] = useState<Step>("Create")
  const [project, setProject] = useState<Project | null>(null)
  const update = (patch: Partial<Project>) => setProject((p) => (p ? { ...p, ...patch } : p))
  const dark = resolvedTheme === "dark"

  return (
    <div className="min-h-svh bg-background">
      <header className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-6">
          <div className="flex min-w-0 shrink-0 items-center gap-2 text-sm sm:w-48">
            <span className="flex items-center gap-2 font-semibold"><span aria-hidden className="size-2.5 rounded-full bg-primary" />typically</span>
            {project && <><span className="hidden text-muted-foreground sm:inline" aria-hidden>/</span><span dir="auto" className="hidden truncate text-muted-foreground sm:inline">{project.name}</span></>}
          </div>
          <Stepper step={step} project={project} go={setStep} />
          <div className="flex shrink-0 items-center justify-end gap-0.5 sm:w-48">
            <Button variant="ghost" size="sm" onClick={() => setLang(lang === "en" ? "he" : "en")} aria-label={t("nav.langLabel")}>
              <Languages data-icon="inline-start" /> {t("nav.lang")}
            </Button>
            {project && <IconAction label={t("nav.startOver")} onClick={() => (setProject(null), setStep("Create"))}><RotateCcw /></IconAction>}
            <IconAction label={t(dark ? "nav.theme.light" : "nav.theme.dark")} onClick={() => setTheme(dark ? "light" : "dark")}>{dark ? <Sun /> : <Moon />}</IconAction>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">
        {step === "Create" && <CreateStep project={project} setProject={setProject} next={() => setStep("Understand")} />}
        {step === "Understand" && project && <UnderstandStep project={project} update={update} back={() => setStep("Create")} next={() => setStep("Enrich")} />}
        {step === "Enrich" && project && <EnrichStep project={project} update={update} back={() => setStep("Understand")} next={() => setStep("Train")} />}
        {step === "Train" && project && <TrainStep project={project} update={update} back={() => setStep("Enrich")} next={() => setStep("Evaluate")} />}
        {step === "Evaluate" && project && <EvaluateStep project={project} next={() => setStep("Deploy")} />}
        {step === "Deploy" && project && <DeployStep project={project} />}
      </main>
      <Toaster position="bottom-right" />
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
      <I18nProvider>
        <TooltipProvider delayDuration={300}>
          <Shell />
        </TooltipProvider>
      </I18nProvider>
    </ThemeProvider>
  )
}
