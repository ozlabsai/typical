import { useEffect, useRef, useState } from "react"
import { AlertCircle, ArrowRight, Database, FileSpreadsheet, Loader2, Mail, Upload } from "lucide-react"

import { msg, OptionCard, PageHead } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, type Dataset, type Source } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { BASES, SAMPLE, slugify, type Analysis, type Base, type Project } from "@/lib/project"
import { go as navigate } from "@/lib/router"
import { cn } from "@/lib/utils"

type Tab = "upload" | "mail" | "hf" | "sheets" | "sample"
type Picked = { name: string; text: string } | { name: string; b64: string }
type SampleName = "northwind" | "enron"
const MAIL = /\.(mbox|eml|zip)$/i
const MAIL_MB = 50 // site/typically_analyze.py MAX_MAIL_MB
const SAMPLES: Record<SampleName, { title: "sample.title" | "sample.enronTitle"; body: "sample.body" | "sample.enronBody" }> = {
  northwind: { title: "sample.title", body: "sample.body" },
  enron: { title: "sample.enronTitle", body: "sample.enronBody" },
}

const base64 = (f: File) => new Promise<string>((ok, fail) => {
  const r = new FileReader()
  r.onload = () => ok(String(r.result).split(",")[1] ?? "")
  r.onerror = () => fail(r.error)
  r.readAsDataURL(f)
})

function DropZone({ picked, icon: Icon, drop, hint, accept, onPick }: {
  picked?: string; icon: typeof Upload; drop: string; hint: string; accept: string; onPick: (f?: File) => void
}) {
  const { t } = useI18n()
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  return (
    <div
      role="button" tabIndex={0}
      onClick={() => input.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      onDragOver={(e) => (e.preventDefault(), setOver(true))}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => (e.preventDefault(), setOver(false), onPick(e.dataTransfer.files[0]))}
      className={cn("flex min-h-40 cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed p-6 text-center transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50", over ? "border-primary bg-primary/5" : "hover:bg-muted/50")}
    >
      {picked ? <Icon className="size-5 text-primary" /> : <Upload className="size-5 text-muted-foreground" />}
      <p className="text-sm font-medium" dir="auto">{picked ?? drop}</p>
      <p className="text-xs text-muted-foreground">{picked ? t("upload.replace") : hint}</p>
      <input ref={input} type="file" accept={accept} className="sr-only" onChange={(e) => onPick(e.target.files?.[0])} />
    </div>
  )
}

/** `from`: start from a dataset already uploaded (its records_token, "sample" or "sample-enron"), as opened from the Data page. */
export function CreateStep({ project, setProject, next, from }: { project: Project | null; setProject: (p: Project) => void; next: () => void; from?: string }) {
  const { t, lang } = useI18n()
  const [name, setName] = useState(project?.name ?? "")
  const [base, setBase] = useState<Base>(project?.base ?? "small")
  const [tab, setTab] = useState<Tab>(from?.startsWith("sample") ? "sample" : "upload")
  const [sampleName, setSampleName] = useState<SampleName>(from === "sample-enron" ? "enron" : "northwind")
  const [file, setFile] = useState<Picked | null>(null)
  const [mail, setMail] = useState<{ name: string; b64: string } | null>(null)
  const [owner, setOwner] = useState("") // mail: the owner field; `asked` is the owner `book` was read as
  const [asked, setAsked] = useState("")
  const [book, setBook] = useState<Analysis | null>(null) // the last xlsx / mail analysis: its `sheets` / `mail` drive the panels below
  const [sheetName, setSheetName] = useState<string>()
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [hf, setHf] = useState({ dataset: "", split: "train" })
  const [sheet, setSheet] = useState("")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {   // the page is keyed on `from`, so this runs once per dataset
    if (!from || from.startsWith("sample")) return
    api.datasets().then((all) => {
      const d = all.find((x) => x.token === from)
      if (!d) return navigate({ name: "new" })   // deleted since: plain Create
      setDataset(d)
      setName((n) => n || d.name.replace(/\.(csv|xlsx|mbox|eml|zip)( \(.*\))?$/i, "").replace(/[_-]+/g, " "))
    }).catch((e) => setError(msg(e)))
  }, [from])

  const source: Source | null =
    dataset ? { kind: "upload", token: dataset.token }
    : tab === "upload" ? (!file ? null : "text" in file ? { kind: "csv", text: file.text, name: file.name } : { kind: "xlsx", data_base64: file.b64, name: file.name, sheet: sheetName })
    : tab === "mail" ? (!mail ? null : { kind: "mail", data_base64: mail.b64, name: mail.name, ...(owner.trim() && owner.trim() !== asked ? { owner: owner.trim() } : {}) })
    : tab === "hf" ? (hf.dataset.includes("/") ? { kind: "hf", dataset: hf.dataset.trim(), split: hf.split || "train", limit: 5000 } : null)
    : tab === "sheets" ? (/docs\.google\.com\/spreadsheets\/d\//.test(sheet) ? { kind: "sheets", url: sheet.trim() } : null)
    : sampleName === "enron" ? { kind: "sample", name: "enron" } : { kind: "sample" }
  const label = dataset ? dataset.name : tab === "upload" ? (file && sheetName ? `${file.name} (${sheetName})` : file?.name) : tab === "mail" ? mail?.name
    : tab === "hf" ? `Hugging Face: ${hf.dataset}` : tab === "sheets" ? "Google Sheet" : sampleName === "enron" ? "Enron email sample" : "Northwind sample"

  /** A table goes to Upload, a mailbox to Email, whichever tab it was dropped on. */
  async function pick(f?: File) {
    if (!f) return
    const xlsx = /\.xlsx$/i.test(f.name)
    const isMail = MAIL.test(f.name)
    if (!xlsx && !isMail && !/\.csv$/i.test(f.name)) return setError(t("upload.fileTypes"))
    if (isMail && f.size > MAIL_MB * 1024 * 1024) return setError(t("mail.tooBig", { n: MAIL_MB }))
    setError(null)
    setBook(null)
    setSheetName(undefined)
    setTab(isMail ? "mail" : "upload")
    if (isMail) {
      setOwner("")
      setAsked("")
      setMail({ name: f.name, b64: await base64(f) })
    } else setFile(xlsx ? { name: f.name, b64: await base64(f) } : { name: f.name, text: await f.text() })
    if (!name) setName(f.name.replace(/\.(csv|xlsx|mbox|eml|zip)$/i, "").replace(/[_-]+/g, " "))
  }

  async function go() {
    if (!source) return
    setBusy(true)
    setError(null)
    try {
      const reuse = book && (book.mail ? source.kind === "mail" && owner.trim() === asked : source.kind === "xlsx" && book.sheet === sheetName)
      const analysis = reuse ? book : await api.analyze(source, lang)
      if (source.kind === "xlsx") {
        setBook(analysis)
        // a workbook with more sheets: stop once on the first sheet's result, so the picker is seen before moving on
        if (!book && (analysis.sheets?.length ?? 0) > 1) return setSheetName(analysis.sheet)
      }
      if (source.kind === "mail" && !reuse) {   // stop once on whose mailbox we read it as, so a wrong guess is fixed before moving on
        setBook(analysis)
        setOwner(analysis.mail?.owner ?? "")
        return setAsked(analysis.mail?.owner ?? "")
      }
      const sample = source.kind === "sample" && !source.name   // Northwind: already trained
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
            {dataset ? (
              <div className="flex items-start gap-3 rounded-lg border p-4">
                <Database className="mt-0.5 size-5 shrink-0 text-primary" />
                <div className="grid min-w-0 flex-1 gap-0.5">
                  <p className="text-sm font-medium"><span className="text-muted-foreground">{t("create.fromData")}: </span><span dir="auto">{dataset.name}</span></p>
                  <p className="text-xs text-muted-foreground tabular">{t("create.fromDataBody", { rows: dataset.rows.toLocaleString(), cols: dataset.columns.length })}</p>
                </div>
                <Button variant="ghost" size="sm" onClick={() => navigate({ name: "new" })}>{t("create.otherData")}</Button>
              </div>
            ) : (
            <Tabs value={tab} onValueChange={(v) => (setTab(v as Tab), setError(null))}>
              <TabsList className="max-w-full justify-start overflow-x-auto">
                <TabsTrigger value="upload">{t("tab.upload")}</TabsTrigger>
                <TabsTrigger value="mail">{t("tab.mail")}</TabsTrigger>
                <TabsTrigger value="hf">{t("tab.hf")}</TabsTrigger>
                <TabsTrigger value="sheets">{t("tab.sheets")}</TabsTrigger>
                <TabsTrigger value="sample">{t("tab.sample")}</TabsTrigger>
              </TabsList>
              <TabsContent value="upload" className="pt-3">
                <DropZone picked={file?.name} icon={FileSpreadsheet} drop={t("upload.drop")} hint={t("upload.private")} onPick={pick}
                  accept=".csv,text/csv,.xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" />
                {book?.sheets && book.sheets.length > 1 && (
                  <div className="mt-3 grid gap-2 sm:grid-cols-[220px_1fr] sm:items-end sm:gap-4">
                    <div className="grid gap-1.5">
                      <Label htmlFor="xlsx-sheet">{t("create.sheet")}</Label>
                      <Select value={sheetName} onValueChange={setSheetName}>
                        <SelectTrigger id="xlsx-sheet" className="w-full"><SelectValue /></SelectTrigger>
                        <SelectContent>{book.sheets.map((n) => <SelectItem key={n} value={n}><span dir="auto">{n}</span></SelectItem>)}</SelectContent>
                      </Select>
                    </div>
                    <p className="text-xs text-muted-foreground sm:pb-2">{t("create.sheetHint", { n: book.sheets.length, sheet: book.sheet ?? "" })}</p>
                  </div>
                )}
              </TabsContent>
              <TabsContent value="mail" className="grid gap-3 pt-3">
                <DropZone picked={mail?.name} icon={Mail} drop={t("mail.drop")} hint={t("mail.private")} onPick={pick}
                  accept=".mbox,.eml,.zip,message/rfc822,application/mbox,application/zip" />
                {book?.mail && (
                  <div className="grid gap-3 rounded-lg bg-muted/50 p-4">
                    <p className="text-sm font-medium tabular">
                      {t("mail.stats", { owner: `⁨${book.mail.owner}⁩`, messages: book.mail.messages.toLocaleString(), threads: book.mail.threads.toLocaleString(), inbound: book.mail.inbound.toLocaleString() })}
                    </p>
                    <div className="grid max-w-md gap-1.5">
                      <Label htmlFor="mail-owner">{t("mail.owner")}</Label>
                      <Input id="mail-owner" dir="ltr" type="text" inputMode="email" value={owner} onChange={(e) => setOwner(e.target.value)} />
                      <p className="text-xs text-muted-foreground">{t("mail.ownerHint")}</p>
                    </div>
                    {(book.warnings?.length ?? 0) > 0 && (
                      <ul className="list-disc ps-4 text-xs text-muted-foreground">{book.warnings!.map((w) => <li key={w} dir="auto">{w}</li>)}</ul>
                    )}
                  </div>
                )}
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
                <RadioGroup value={sampleName} onValueChange={(v) => setSampleName(v as SampleName)} className="grid gap-3 sm:grid-cols-2">
                  {(Object.keys(SAMPLES) as SampleName[]).map((s) => (
                    <Label key={s} htmlFor={`sample-${s}`} className="cursor-pointer font-normal">
                      <OptionCard selected={sampleName === s} className="h-full w-full">
                        <div className="flex items-start gap-3">
                          <RadioGroupItem value={s} id={`sample-${s}`} className="mt-0.5" />
                          <div className="grid gap-1">
                            <span className="font-medium">{t(SAMPLES[s].title)}</span>
                            <span className="text-sm text-muted-foreground">{t(SAMPLES[s].body)}</span>
                          </div>
                        </div>
                      </OptionCard>
                    </Label>
                  ))}
                </RadioGroup>
              </TabsContent>
            </Tabs>
            )}

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
