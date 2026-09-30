import { useEffect, useState } from "react"
import { AlertCircle, Download, ExternalLink, KeyRound, Loader2, Lock, UploadCloud } from "lucide-react"
import { toast } from "sonner"

import { CodeBlock, CopyButton, msg, PageHead } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, downloadUrl, type Snippets } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { modelId, slugify, type Project } from "@/lib/project"

const TABS: (keyof Snippets)[] = ["curl", "python", "javascript", "sdk"]

export function DeployStep({ project }: { project: Project }) {
  const { t } = useI18n()
  const id = modelId(project)
  const url = `${window.location.origin}/v1/models/${id}/decide`
  const [key, setKey] = useState<string | null>(null)
  const [snips, setSnips] = useState<Snippets | null>(null)
  const [snipError, setSnipError] = useState<string | null>(null)
  const [busy, setBusy] = useState<"key" | "push" | null>(null)
  const [ns, setNs] = useState<string | null>(null)
  const [hf, setHf] = useState({ repo: "", private: true })
  const [pushed, setPushed] = useState<string | null>(null)
  const [pushError, setPushError] = useState<string | null>(null)

  useEffect(() => {
    api.snippets(id).then(setSnips).catch((e) => setSnipError(msg(e)))
    api.capabilities().then((c) => {
      setNs(c.hf_namespace)
      if (c.hf_namespace) setHf((h) => (h.repo ? h : { ...h, repo: `${c.hf_namespace}/${slugify(project.name).replace(/_/g, "-")}` }))
    }).catch(() => setNs(null))
  }, [id, project.name])

  async function createKey() {
    setBusy("key")
    try {
      setKey((await api.createKey(id)).key)
    } catch (e) {
      toast.error(msg(e))
    } finally {
      setBusy(null)
    }
  }

  async function push() {
    setBusy("push")
    setPushError(null)
    try {
      const r = await api.push({ run: project.run!, repo: hf.repo.trim() || undefined, private: hf.private })
      setPushed(r.url)
      toast.success(t("dep.pushed"))
    } catch (e) {
      setPushError(msg(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <PageHead title={t("dep.title")} action={<Button variant="outline" asChild><a href={downloadUrl(project.run!)} download><Download data-icon="inline-start" /> {t("dep.weights")}</a></Button>}>
        {t("dep.lede", { name: project.name })}
      </PageHead>

      <div className="grid gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">{t("dep.endpoint")} <Badge variant="secondary" className="gap-1"><span className="size-1.5 rounded-full bg-primary" />{t("dep.live")}</Badge></CardTitle>
            <CardDescription>{t("dep.endpointBody")}</CardDescription>
            <CardAction><Badge variant="outline" className="gap-1 font-normal"><Lock className="size-3" /> {t("dep.protected")}</Badge></CardAction>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-center gap-2">
              <Input readOnly dir="ltr" value={url} className="font-mono text-xs" aria-label={t("dep.urlLabel")} />
              <CopyButton text={url} />
            </div>
            <p className="text-xs text-muted-foreground">{t("dep.localNote")}</p>
            {key ? (
              <Alert>
                <KeyRound />
                <AlertTitle>{t("dep.key")}</AlertTitle>
                <AlertDescription className="grid gap-2">
                  <span>{t("dep.keyOnce")}</span>
                  <span className="flex items-center gap-2"><code dir="ltr" className="rounded bg-muted px-2 py-1 font-mono text-xs break-all">{key}</code><CopyButton text={key} /></span>
                </AlertDescription>
              </Alert>
            ) : (
              <div><Button onClick={createKey} disabled={busy === "key"}>{busy === "key" ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <KeyRound data-icon="inline-start" />} {t("dep.createKey")}</Button></div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>{t("dep.use")}</CardTitle><CardDescription>{t("dep.useHint", { env: "TYPICAL_API_KEY" })}</CardDescription></CardHeader>
          <CardContent>
            {snipError ? (
              <p className="text-sm text-destructive">{snipError}</p>
            ) : !snips ? (
              <Skeleton className="h-40" />
            ) : (
              <Tabs defaultValue="curl">
                <TabsList>{TABS.map((k) => <TabsTrigger key={k} value={k}>{t(`dep.snippet.${k}`)}</TabsTrigger>)}</TabsList>
                {TABS.map((k) => <TabsContent key={k} value={k} className="pt-3" dir="ltr"><CodeBlock code={snips[k]} /></TabsContent>)}
              </Tabs>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t("dep.hf")}</CardTitle>
            <CardDescription>{t("dep.hfBody", { ns: ns ?? "Hugging Face" })}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5 sm:col-span-2 sm:max-w-md"><Label htmlFor="repo">{t("dep.repo")}</Label><Input id="repo" dir="ltr" value={hf.repo} onChange={(e) => setHf({ ...hf, repo: e.target.value })} className="font-mono text-xs" /></div>
            <Label className="flex items-center gap-3 font-normal sm:col-span-2">
              <Switch checked={hf.private} onCheckedChange={(v) => setHf({ ...hf, private: v })} /> {t("dep.private")}
            </Label>
            {pushError && <p role="alert" className="flex items-center gap-2 text-sm text-destructive sm:col-span-2"><AlertCircle className="size-4" /> {pushError}</p>}
          </CardContent>
          <CardFooter className="justify-between">
            {pushed ? <a href={pushed} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-sm text-primary underline-offset-4 hover:underline">{pushed.replace("https://", "")} <ExternalLink className="size-3.5" /></a> : <span />}
            <Button onClick={push} disabled={!ns || !hf.repo.includes("/") || busy === "push"}>
              {busy === "push" ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <UploadCloud data-icon="inline-start" />} {t("dep.push")}
            </Button>
          </CardFooter>
        </Card>
      </div>
    </>
  )
}
