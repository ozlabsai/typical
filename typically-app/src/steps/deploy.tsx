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
import { modelId, slugify, type Project } from "@/lib/project"

const TABS: [keyof Snippets, string][] = [["curl", "cURL"], ["python", "Python"], ["javascript", "JavaScript"], ["sdk", "Python SDK"]]

export function DeployStep({ project }: { project: Project }) {
  const id = modelId(project)
  const url = `${window.location.origin}/v1/models/${id}/decide`
  const [key, setKey] = useState<string | null>(null)
  const [snips, setSnips] = useState<Snippets | null>(null)
  const [snipError, setSnipError] = useState<string | null>(null)
  const [busy, setBusy] = useState<"key" | "push" | null>(null)
  const [hf, setHf] = useState({ repo: `your-username/${slugify(project.name).replace(/_/g, "-")}`, token: "", private: true })
  const [pushed, setPushed] = useState<string | null>(null)
  const [pushError, setPushError] = useState<string | null>(null)

  useEffect(() => {
    api.snippets(id).then(setSnips).catch((e) => setSnipError(msg(e)))
  }, [id])

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
      const r = await api.push({ run: project.run!, repo: hf.repo.trim(), token: hf.token, private: hf.private })
      setPushed(r.url)
      setHf((h) => ({ ...h, token: "" }))
      toast.success("Pushed to Hugging Face")
    } catch (e) {
      setPushError(msg(e))
    } finally {
      setBusy(null)
    }
  }

  return (
    <>
      <PageHead title="Deploy" action={<Button variant="outline" asChild><a href={downloadUrl(project.run!)} download><Download data-icon="inline-start" /> Download weights</a></Button>}>
        Call {project.name} from your own software, or take the weights with you.
      </PageHead>

      <div className="grid gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">Endpoint <Badge variant="secondary" className="gap-1"><span className="size-1.5 rounded-full bg-primary" />Live</Badge></CardTitle>
            <CardDescription>Send a case and your questions; get a probability for every answer. Requests need an API key.</CardDescription>
            <CardAction><Badge variant="outline" className="gap-1 font-normal"><Lock className="size-3" /> Protected</Badge></CardAction>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-center gap-2">
              <Input readOnly value={url} className="font-mono text-xs" aria-label="Endpoint URL" />
              <CopyButton text={url} />
            </div>
            <p className="text-xs text-muted-foreground">This prototype serves from this machine, so the URL works where the app runs. Hosted endpoints come next.</p>
            {key ? (
              <Alert>
                <KeyRound />
                <AlertTitle>Your API key</AlertTitle>
                <AlertDescription className="grid gap-2">
                  <span>Copy it now. We only store a fingerprint, so we can't show it again.</span>
                  <span className="flex items-center gap-2"><code className="rounded bg-muted px-2 py-1 font-mono text-xs break-all">{key}</code><CopyButton text={key} /></span>
                </AlertDescription>
              </Alert>
            ) : (
              <div><Button onClick={createKey} disabled={busy === "key"}>{busy === "key" ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <KeyRound data-icon="inline-start" />} Create API key</Button></div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle>Use it</CardTitle><CardDescription>Set <code className="font-mono text-xs">TYPICAL_API_KEY</code> in your environment, then:</CardDescription></CardHeader>
          <CardContent>
            {snipError ? (
              <p className="text-sm text-destructive">{snipError}</p>
            ) : !snips ? (
              <Skeleton className="h-40" />
            ) : (
              <Tabs defaultValue="curl">
                <TabsList>{TABS.map(([k, l]) => <TabsTrigger key={k} value={k}>{l}</TabsTrigger>)}</TabsList>
                {TABS.map(([k]) => <TabsContent key={k} value={k} className="pt-3"><CodeBlock code={snips[k]} /></TabsContent>)}
              </Tabs>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Hugging Face</CardTitle>
            <CardDescription>Keep a copy in your own Hugging Face account. The repo is created under your account; the token is used once and not stored.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div className="grid gap-1.5"><Label htmlFor="repo">Repository</Label><Input id="repo" value={hf.repo} onChange={(e) => setHf({ ...hf, repo: e.target.value })} className="font-mono text-xs" /></div>
            <div className="grid gap-1.5"><Label htmlFor="hf-token">Access token (write)</Label><Input id="hf-token" type="password" autoComplete="off" placeholder="hf_…" value={hf.token} onChange={(e) => setHf({ ...hf, token: e.target.value })} /></div>
            <Label className="flex items-center gap-3 font-normal sm:col-span-2">
              <Switch checked={hf.private} onCheckedChange={(v) => setHf({ ...hf, private: v })} /> Private repository
            </Label>
            {pushError && <p role="alert" className="flex items-center gap-2 text-sm text-destructive sm:col-span-2"><AlertCircle className="size-4" /> {pushError}</p>}
          </CardContent>
          <CardFooter className="justify-between">
            {pushed ? <a href={pushed} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-sm text-primary underline-offset-4 hover:underline">{pushed.replace("https://", "")} <ExternalLink className="size-3.5" /></a> : <span />}
            <Button onClick={push} disabled={!hf.token || !hf.repo.includes("/") || busy === "push"}>
              {busy === "push" ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <UploadCloud data-icon="inline-start" />} Push to Hugging Face
            </Button>
          </CardFooter>
        </Card>
      </div>
    </>
  )
}
