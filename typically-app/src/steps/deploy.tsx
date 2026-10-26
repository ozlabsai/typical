import { useEffect, useState } from "react"
import { AlertCircle, Download, ExternalLink, KeyRound, Loader2, Lock, UploadCloud } from "lucide-react"
import { toast } from "sonner"

import { SectionHeader } from "@/components/layout"
import { CodeBlock, CopyButton, msg } from "@/components/shared"
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
import { useAuth } from "@/lib/auth"
import { useI18n } from "@/lib/i18n"
import type { ModelRef } from "@/lib/library"
import { slugify } from "@/lib/project"

const TABS: (keyof Snippets)[] = ["curl", "python", "javascript", "sdk"]

export function DeployPanel({ model }: { model: ModelRef }) {
  const { t } = useI18n()
  const { user } = useAuth()
  const id = model.id
  const url = `${window.location.origin}/v1/models/${id}/decide`
  const [key, setKey] = useState<string | null>(null)
  const [snips, setSnips] = useState<Snippets | null>(null)
  const [snipError, setSnipError] = useState<string | null>(null)
  const [busy, setBusy] = useState<"key" | "push" | null>(null)
  const [ns, setNs] = useState<string | null>(null) // set only when this user may push (else the Hugging Face card is not shown)
  const [hf, setHf] = useState({ repo: "", private: true })
  const [pushed, setPushed] = useState<string | null>(null)
  const [pushError, setPushError] = useState<string | null>(null)
  const [instead, setInstead] = useState<string | null>(null) // the org's storage is full: the token owner's own repo

  useEffect(() => {
    api.snippets(id).then(setSnips).catch((e) => setSnipError(msg(e)))
    api.capabilities().then((c) => {
      // hosted: a push lands in the operator's org, so only admins get it (the server's can_push; before it existed, the admin flag)
      const may = c.can_push ?? (!user || user.admin)
      const to = may ? c.hf_namespace : null
      setNs(to)
      if (to) setHf((h) => (h.repo ? h : { ...h, repo: `${to}/${slugify(model.name).replace(/_/g, "-")}` }))
    }).catch(() => setNs(null))
  }, [id, model.name, user])

  // the SDK snippet needs a published repo; without one the server fills in a placeholder, which is not shown
  const tabs = TABS.filter((k) => snips?.[k] && !snips[k]!.includes("<hf-user>"))

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

  async function push(repo = hf.repo.trim()) {
    setBusy("push")
    setPushError(null)
    setInstead(null)
    try {
      const r = await api.push({ run: model.run, repo: repo || undefined, private: hf.private })
      setPushed(r.url)
      toast.success(t("dep.pushed"))
    } catch (e) {
      setPushError(msg(e))
      setInstead((e as { detail?: { repo?: string | null } }).detail?.repo ?? null)
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <SectionHeader description={t("dep.lede", { name: model.name })}
        actions={<Button variant="outline" size="sm" asChild><a href={downloadUrl(model.run)} download><Download data-icon="inline-start" /> {t("dep.download")}</a></Button>} />

      <div className="grid gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">{t("dep.endpoint")} <Badge variant="secondary"><span className="size-1.5 rounded-full bg-success" />{t("dep.live")}</Badge></CardTitle>
            <CardDescription>{t("dep.endpointBody")}</CardDescription>
            <CardAction><Badge variant="outline" className="font-normal"><Lock /> {t("dep.protected")}</Badge></CardAction>
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
                  <span className="flex items-center gap-2"><code dir="ltr" className="rounded-md bg-muted px-2 py-1 font-mono text-xs break-all">{key}</code><CopyButton text={key} /></span>
                </AlertDescription>
              </Alert>
            ) : (
              <div><Button variant="outline" size="sm" onClick={createKey} disabled={busy === "key"}>{busy === "key" ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <KeyRound data-icon="inline-start" />} {t("dep.createKey")}</Button></div>
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
                <TabsList>{tabs.map((k) => <TabsTrigger key={k} value={k}>{t(`dep.snippet.${k}`)}</TabsTrigger>)}</TabsList>
                {tabs.map((k) => <TabsContent key={k} value={k} className="pt-3"><CodeBlock code={snips[k]!} /></TabsContent>)}
              </Tabs>
            )}
          </CardContent>
        </Card>

        {ns && <Card>
          <CardHeader>
            <CardTitle>{t("dep.hf")}</CardTitle>
            <CardDescription>{t("dep.hfBody", { ns })}</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5 sm:col-span-2 sm:max-w-md"><Label htmlFor="repo">{t("dep.repo")}</Label><Input id="repo" dir="ltr" value={hf.repo} onChange={(e) => setHf({ ...hf, repo: e.target.value })} className="font-mono text-xs" /></div>
            <Label className="flex items-center gap-3 font-normal sm:col-span-2">
              <Switch checked={hf.private} onCheckedChange={(v) => setHf({ ...hf, private: v })} /> {t("dep.private")}
            </Label>
            {pushError && <p role="alert" className="flex items-center gap-2 text-sm text-destructive sm:col-span-2"><AlertCircle className="size-4" /> {pushError}</p>}
            {instead && (
              <div className="sm:col-span-2">
                <Button variant="outline" size="sm" disabled={busy === "push"} onClick={() => { setHf({ ...hf, repo: instead }); push(instead) }}>
                  <UploadCloud data-icon="inline-start" /> {t("dep.pushInstead", { ns: instead.split("/")[0] })}
                </Button>
              </div>
            )}
          </CardContent>
          <CardFooter className="justify-between">
            {pushed ? <a href={pushed} target="_blank" rel="noreferrer" dir="ltr" className="inline-flex items-center gap-1.5 text-sm text-primary underline-offset-4 hover:underline">{pushed.replace("https://", "")} <ExternalLink className="size-4" /></a> : <span />}
            <Button variant="outline" size="sm" onClick={() => push()} disabled={!hf.repo.includes("/") || busy === "push"}>
              {busy === "push" ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <UploadCloud data-icon="inline-start" />} {t("dep.push")}
            </Button>
          </CardFooter>
        </Card>}
      </div>
    </>
  )
}
