import { useRef, useState } from "react"
import { AlertCircle, ArrowRight, FileSpreadsheet, Loader2, Upload } from "lucide-react"

import { msg, OptionCard, PageHead } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type Source } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { BASES, SAMPLE, slugify, type Base, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

type Tab = "upload" | "hf" | "sheets" | "sample"

export function CreateStep({ project, setProject, next }: { project: Project | null; setProject: (p: Project) => void; next: () => void }) {
  const { t, lang } = useI18n()
  const [name, setName] = useState(project?.name ?? "")
  const [base, setBase] = useState<Base>(project?.base ?? "small")
  const [tab, setTab] = useState<Tab>("upload")
  const [file, setFile] = useState<{ name: string; text: string } | null>(null)
  const [hf, setHf] = useState({ dataset: "", split: "train" })
  const [sheet, setSheet] = useState("")
  const [over, setOver] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const source: Source | null =
    tab === "upload" ? (file ? { kind: "csv", text: file.text, name: file.name } : null)
    : tab === "hf" ? (hf.dataset.includes("/") ? { kind: "hf", dataset: hf.dataset.trim(), split: hf.split || "train", limit: 5000 } : null)
    : tab === "sheets" ? (/docs\.google\.com\/spreadsheets\/d\//.test(sheet) ? { kind: "sheets", url: sheet.trim() } : null)
    : { kind: "sample" }
  const label = tab === "upload" ? file?.name : tab === "hf" ? `Hugging Face: ${hf.dataset}` : tab === "sheets" ? "Google Sheet" : "Northwind sample"

  async function pick(f?: File) {
    if (!f) return
    if (!/\.csv$/i.test(f.name)) return setError(t("upload.csvOnly"))
    setError(null)
    setFile({ name: f.name, text: await f.text() })
    if (!name) setName(f.name.replace(/\.csv$/i, "").replace(/[_-]+/g, " "))
  }

  async function go() {
    if (!source) return
    setBusy(true)
    setError(null)
    try {
      const analysis = await api.analyze(source, lang)
      const sample = source.kind === "sample"
      if (sample)
        for (const d of analysis.plan.decisions) {
          const s = SAMPLE.questions[d.column]
          if (s) Object.assign(d, { question: s.question, type: s.type, labels: s.labels })
        }
      setProject({
        name: name.trim() || (sample ? SAMPLE.name : analysis.name_hint), base, source: label ?? "", sample, analysis, plan: analysis.plan,
        enrich: { balance: true, dedupe_soft: true, policy: {}, synthetic: false, languages: [] },
        settings: { preset: "balanced", holdout: 20, seed: 0 },
        ...(sample ? { run: SAMPLE.run, resultsKey: SAMPLE.resultsKey } : {}),
      })
      next()
    } catch (e) {
      setError(msg(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <PageHead title={t("create.title")}>{t("create.lede")}</PageHead>
      <div className="grid gap-4">
        <Card>
          <CardHeader><CardTitle>{t("create.name")}</CardTitle><CardDescription>{t("create.nameHint")}</CardDescription></CardHeader>
          <CardContent className="grid max-w-md gap-2">
            <Input dir="auto" value={name} onChange={(e) => setName(e.target.value)} placeholder={t("create.namePlaceholder")} aria-label={t("create.name")} />
            <p className="font-mono text-xs text-muted-foreground" dir="ltr">{t("create.id")}: {slugify(name) || "…"}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>{t("create.base")}</CardTitle><CardDescription>{t("create.baseHint")}</CardDescription></CardHeader>
          <CardContent>
            <RadioGroup value={base} onValueChange={(v) => setBase(v as Base)} className="grid gap-3 sm:grid-cols-2">
              {(Object.keys(BASES) as Base[]).map((b) => (
                <Label key={b} htmlFor={`base-${b}`} className="cursor-pointer font-normal">
                  <OptionCard selected={base === b} className="w-full">
                    <div className="flex items-start gap-3">
                      <RadioGroupItem value={b} id={`base-${b}`} className="mt-0.5" />
                      <div className="grid gap-1">
                        <span className="flex items-center gap-2 font-medium">
                          {BASES[b].label} <span className="font-mono text-xs text-muted-foreground">{BASES[b].size}</span>
                          {BASES[b].beta && <Badge variant="secondary">{t("base.beta")}</Badge>}
                        </span>
                        <span className="text-sm text-muted-foreground">{t(`base.${b}.note`)}</span>
                        <span className="text-xs text-muted-foreground tabular">{t("base.minutes", { n: BASES[b].minutes })}</span>
                      </div>
                    </div>
                  </OptionCard>
                </Label>
              ))}
            </RadioGroup>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>{t("create.data")}</CardTitle><CardDescription>{t("create.dataHint")}</CardDescription></CardHeader>
          <CardContent>
            <Tabs value={tab} onValueChange={(v) => (setTab(v as Tab), setError(null))}>
              <TabsList>
                <TabsTrigger value="upload">{t("tab.upload")}</TabsTrigger>
                <TabsTrigger value="hf">{t("tab.hf")}</TabsTrigger>
                <TabsTrigger value="sheets">{t("tab.sheets")}</TabsTrigger>
                <TabsTrigger value="sample">{t("tab.sample")}</TabsTrigger>
              </TabsList>
              <TabsContent value="upload" className="pt-3">
                <div
                  role="button" tabIndex={0}
                  onClick={() => input.current?.click()}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
                  onDragOver={(e) => (e.preventDefault(), setOver(true))}
                  onDragLeave={() => setOver(false)}
                  onDrop={(e) => (e.preventDefault(), setOver(false), pick(e.dataTransfer.files[0]))}
                  className={cn("flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-6 text-center transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50", over ? "border-primary bg-primary/5" : "hover:bg-muted/50")}
                >
                  {file ? <FileSpreadsheet className="size-5 text-primary" /> : <Upload className="size-5 text-muted-foreground" />}
                  <p className="text-sm font-medium">{file ? file.name : t("upload.drop")}</p>
                  <p className="text-xs text-muted-foreground">{file ? t("upload.replace") : t("upload.private")}</p>
                  <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => pick(e.target.files?.[0])} />
                </div>
              </TabsContent>
              <TabsContent value="hf" className="grid gap-3 pt-3 sm:grid-cols-[1fr_140px]">
                <div className="grid gap-2"><Label htmlFor="hf-ds">{t("hf.dataset")}</Label><Input id="hf-ds" dir="ltr" placeholder="owner/dataset" value={hf.dataset} onChange={(e) => setHf({ ...hf, dataset: e.target.value })} /></div>
                <div className="grid gap-2"><Label htmlFor="hf-split">{t("hf.split")}</Label><Input id="hf-split" dir="ltr" value={hf.split} onChange={(e) => setHf({ ...hf, split: e.target.value })} /></div>
                <p className="text-xs text-muted-foreground sm:col-span-2">{t("hf.hint")}</p>
              </TabsContent>
              <TabsContent value="sheets" className="grid gap-2 pt-3">
                <Label htmlFor="sheet">{t("sheets.link")}</Label>
                <Input id="sheet" dir="ltr" placeholder="https://docs.google.com/spreadsheets/d/…" value={sheet} onChange={(e) => setSheet(e.target.value)} />
                <p className="text-xs text-muted-foreground">{t("sheets.hint")}</p>
              </TabsContent>
              <TabsContent value="sample" className="pt-3">
                <div className="rounded-lg bg-muted/50 p-4 text-sm">
                  <p className="font-medium">{t("sample.title")}</p>
                  <p className="mt-1 text-muted-foreground">{t("sample.body")}</p>
                </div>
              </TabsContent>
            </Tabs>

          </CardContent>
          <CardFooter className="justify-between">
            {error ? <p role="alert" className="flex items-center gap-2 text-sm text-destructive"><AlertCircle className="size-4" /> {error}</p> : <span />}
            <Button onClick={go} disabled={!source || busy}>
              {busy ? <><Loader2 data-icon="inline-start" className="animate-spin" /> {t("create.reading")}</> : <>{t("common.continue")} <ArrowRight data-icon="inline-end" className="rtl:rotate-180" /></>}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </>
  )
}
