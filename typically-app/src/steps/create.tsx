import { useRef, useState } from "react"
import { AlertCircle, ArrowRight, FileSpreadsheet, KeyRound, Loader2, Sparkles, Upload } from "lucide-react"

import { msg, OptionCard, PageHead } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type Source } from "@/lib/api"
import { BASES, SAMPLE, slugify, type Base, type Project } from "@/lib/project"
import { cn } from "@/lib/utils"

type Tab = "upload" | "hf" | "sheets" | "sample"

export function CreateStep({ project, setProject, next }: { project: Project | null; setProject: (p: Project) => void; next: () => void }) {
  const [name, setName] = useState(project?.name ?? "")
  const [base, setBase] = useState<Base>(project?.base ?? "small")
  const [tab, setTab] = useState<Tab>("upload")
  const [file, setFile] = useState<{ name: string; text: string } | null>(null)
  const [hf, setHf] = useState({ dataset: "", split: "train" })
  const [sheet, setSheet] = useState("")
  const [key, setKey] = useState(project?.anthropicKey ?? "")
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
    if (!/\.csv$/i.test(f.name)) return setError("CSV only for now. In Excel or Sheets: File, then Download, then CSV.")
    setError(null)
    setFile({ name: f.name, text: await f.text() })
    if (!name) setName(f.name.replace(/\.csv$/i, "").replace(/[_-]+/g, " "))
  }

  async function go() {
    if (!source) return
    setBusy(true)
    setError(null)
    try {
      const analysis = await api.analyze(source, key || undefined)
      const sample = source.kind === "sample"
      if (sample)
        for (const d of analysis.plan.decisions) {
          const s = SAMPLE.questions[d.column]
          if (s) Object.assign(d, { question: s.question, type: s.type, labels: s.labels })
        }
      setProject({
        name: name.trim() || (sample ? SAMPLE.name : analysis.name_hint), base, source: label ?? "", sample, analysis, plan: analysis.plan,
        anthropicKey: key || undefined,
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
      <PageHead title="Create a model">Give it a name, pick the model to start from, and point us at your past decisions.</PageHead>
      <div className="grid gap-4">
        <Card>
          <CardHeader><CardTitle>Name</CardTitle><CardDescription>How you and your team will refer to it.</CardDescription></CardHeader>
          <CardContent className="grid max-w-md gap-2">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Support triage" aria-label="Model name" />
            <p className="font-mono text-xs text-muted-foreground">id: {slugify(name) || "…"}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Start from</CardTitle><CardDescription>Both read the case once and answer every question with a probability. Your data teaches it your way of deciding.</CardDescription></CardHeader>
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
                          {BASES[b].beta && <Badge variant="secondary">Beta</Badge>}
                        </span>
                        <span className="text-sm text-muted-foreground">{BASES[b].note}</span>
                        <span className="text-xs text-muted-foreground tabular">About {BASES[b].minutes} min and ${BASES[b].dollars} to train</span>
                      </div>
                    </div>
                  </OptionCard>
                </Label>
              ))}
            </RadioGroup>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Past decisions</CardTitle><CardDescription>A table with one row per case: something that describes the case, and what you decided about it.</CardDescription></CardHeader>
          <CardContent>
            <Tabs value={tab} onValueChange={(v) => (setTab(v as Tab), setError(null))}>
              <TabsList>
                <TabsTrigger value="upload">Upload</TabsTrigger>
                <TabsTrigger value="hf">Hugging Face</TabsTrigger>
                <TabsTrigger value="sheets">Google Sheets</TabsTrigger>
                <TabsTrigger value="sample">Sample</TabsTrigger>
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
                  <p className="text-sm font-medium">{file ? file.name : "Drop a CSV here, or click to choose"}</p>
                  <p className="text-xs text-muted-foreground">{file ? "Click to replace" : "Stays on this machine until you start training"}</p>
                  <input ref={input} type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => pick(e.target.files?.[0])} />
                </div>
              </TabsContent>
              <TabsContent value="hf" className="grid gap-3 pt-3 sm:grid-cols-[1fr_140px]">
                <div className="grid gap-2"><Label htmlFor="hf-ds">Dataset</Label><Input id="hf-ds" placeholder="owner/dataset" value={hf.dataset} onChange={(e) => setHf({ ...hf, dataset: e.target.value })} /></div>
                <div className="grid gap-2"><Label htmlFor="hf-split">Split</Label><Input id="hf-split" value={hf.split} onChange={(e) => setHf({ ...hf, split: e.target.value })} /></div>
                <p className="text-xs text-muted-foreground sm:col-span-2">Public datasets. We read up to 5,000 rows.</p>
              </TabsContent>
              <TabsContent value="sheets" className="grid gap-2 pt-3">
                <Label htmlFor="sheet">Share link</Label>
                <Input id="sheet" placeholder="https://docs.google.com/spreadsheets/d/…" value={sheet} onChange={(e) => setSheet(e.target.value)} />
                <p className="text-xs text-muted-foreground">Set sharing to "Anyone with the link can view". We read the first tab.</p>
              </TabsContent>
              <TabsContent value="sample" className="pt-3">
                <div className="rounded-lg bg-muted/50 p-4 text-sm">
                  <p className="font-medium">Northwind Freight support tickets</p>
                  <p className="mt-1 text-muted-foreground">277 tickets and the four calls their team made on each: which team, escalate, urgency, refund. Already trained, so you can see the whole flow in a minute.</p>
                </div>
              </TabsContent>
            </Tabs>

            <Collapsible className="mt-5">
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="-ml-2"><Sparkles data-icon="inline-start" /> AI analysis (optional)</Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="grid max-w-md gap-2 pt-2">
                <Label htmlFor="key" className="flex items-center gap-2"><KeyRound className="size-3.5" /> Anthropic API key</Label>
                <Input id="key" type="password" autoComplete="off" placeholder="sk-ant-…" value={key} onChange={(e) => setKey(e.target.value)} />
                <p className="text-xs text-muted-foreground">Lets Claude read a sample of rows to name your decisions and merge messy answers. Used for this analysis only and never stored. Without it we use built-in rules.</p>
              </CollapsibleContent>
            </Collapsible>
          </CardContent>
          <CardFooter className="justify-between">
            {error ? <p role="alert" className="flex items-center gap-2 text-sm text-destructive"><AlertCircle className="size-4" /> {error}</p> : <span />}
            <Button onClick={go} disabled={!source || busy}>
              {busy ? <><Loader2 data-icon="inline-start" className="animate-spin" /> Reading your data…</> : <>Continue <ArrowRight data-icon="inline-end" /></>}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </>
  )
}
