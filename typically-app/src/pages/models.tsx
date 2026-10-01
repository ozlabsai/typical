import { AlertCircle, Archive, Boxes, MessageSquare, MoreHorizontal, Plus, Rocket } from "lucide-react"
import { toast } from "sonner"

import { StatusDot } from "@/components/app-sidebar"
import { CodeBlock, msg } from "@/components/shared"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { library, type LibraryModel } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { refOf, useLibrary } from "@/lib/library"
import { BASES, pct } from "@/lib/project"
import { go, href } from "@/lib/router"
import { DeployPanel } from "@/steps/deploy"
import { Results } from "@/steps/evaluate"

const when = (iso: string | undefined, lang: string) =>
  iso ? new Intl.DateTimeFormat(lang === "he" ? "he-IL" : "en", { dateStyle: "medium" }).format(new Date(iso)) : "—"

function Agreement({ m }: { m: LibraryModel }) {
  if (!m.metrics) return <span className="text-muted-foreground">—</span>
  return (
    <span className="inline-flex items-center gap-2 font-mono text-sm tabular">
      <span className="text-standard">{pct(m.metrics.standard)}</span>
      <span aria-hidden className="text-muted-foreground">→</span>
      <span className="font-medium text-yours">{pct(m.metrics.yours)}</span>
    </span>
  )
}

function StatusLabel({ m }: { m: LibraryModel }) {
  const { t } = useI18n()
  return (
    <span className="inline-flex items-center gap-2 text-sm">
      <StatusDot status={m.status} /> {t(`status.${m.status}`)}
      {m.status === "training" && typeof m.progress === "number" && <span className="font-mono text-xs text-muted-foreground tabular">{Math.round(m.progress * 100)}%</span>}
    </span>
  )
}

export function ModelsPage() {
  const { t, lang } = useI18n()
  const { lib, refresh } = useLibrary()

  async function archive(m: LibraryModel) {
    try {
      await library.archive(m.id)
      toast.success(`${m.name}: ${t("models.archived")}`)
      refresh()
    } catch (e) {
      toast.error(msg(e))
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-6">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{t("models.title")}</h1>
          <p className="mt-1 text-sm text-muted-foreground">{t("models.lede")}</p>
        </div>
        <Button asChild><a href={href({ name: "new" })}><Plus data-icon="inline-start" /> {t("app.new")}</a></Button>
      </div>

      <Card>
        <CardHeader><CardTitle>{t("models.custom")}</CardTitle></CardHeader>
        <CardContent className="px-0">
          {!lib ? (
            <div className="grid gap-2 px-6">{[0, 1].map((i) => <Skeleton key={i} className="h-10" />)}</div>
          ) : !lib.custom.length ? (
            <div className="grid justify-items-center gap-3 px-6 py-10 text-center">
              <Boxes className="size-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">{t("models.empty")}</p>
              <Button asChild><a href={href({ name: "new" })}>{t("models.train")}</a></Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="ps-6">{t("models.name")}</TableHead>
                  <TableHead>{t("models.from")}</TableHead>
                  <TableHead>{t("models.status")}</TableHead>
                  <TableHead>{t("models.agreement")}</TableHead>
                  <TableHead>{t("models.created")}</TableHead>
                  <TableHead className="w-28 pe-6"><span className="sr-only">Actions</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {lib.custom.map((m) => (
                  <TableRow key={m.id} className="cursor-pointer" onClick={() => go({ name: "model", id: m.id })}>
                    <TableCell className="ps-6">
                      <span className="flex items-center gap-2 font-medium">
                        <span dir="auto">{m.name}</span>
                        {m.sample && <Badge variant="secondary">{t("app.sample")}</Badge>}
                      </span>
                      <span className="text-xs text-muted-foreground">{t("models.questions", { n: m.decisions?.length ?? 0 })}</span>
                    </TableCell>
                    <TableCell className="font-mono text-xs">{BASES[m.base].label}</TableCell>
                    <TableCell><StatusLabel m={m} /></TableCell>
                    <TableCell><Agreement m={m} /></TableCell>
                    <TableCell className="text-sm text-muted-foreground">{when(m.created_at, lang)}</TableCell>
                    <TableCell className="pe-6" onClick={(e) => e.stopPropagation()}>
                      <div className="flex justify-end gap-1">
                        <Button variant="ghost" size="sm" disabled={m.status !== "ready"} onClick={() => go({ name: "chat", model: m.id })}>
                          <MessageSquare data-icon="inline-start" /> {t("models.chat")}
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild><Button variant="ghost" size="icon-sm" aria-label="More"><MoreHorizontal /></Button></DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => go({ name: "model", id: m.id })}>{t("models.open")}</DropdownMenuItem>
                            <DropdownMenuItem disabled={m.sample || m.status === "training"} onClick={() => archive(m)}><Archive /> {t("models.archive")}</DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <h2 className="mt-8 mb-3 text-sm font-medium text-muted-foreground">{t("models.baseTitle")}</h2>
      <div className="grid gap-4 sm:grid-cols-2">
        {(lib?.base ?? []).map((m) => (
          <Card key={m.id} size="sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 font-mono text-sm">{m.name} <span className="text-muted-foreground">{m.params}</span></CardTitle>
              <CardDescription>{t(`base.${m.base}.note`)}</CardDescription>
              <CardAction>
                <Button variant="outline" size="sm" onClick={() => go({ name: "chat", model: m.id })}><MessageSquare data-icon="inline-start" /> {t("models.chat")}</Button>
              </CardAction>
            </CardHeader>
          </Card>
        ))}
      </div>
    </div>
  )
}

export function ModelPage({ id, tab }: { id: string; tab?: string }) {
  const { t, lang } = useI18n()
  const { lib, find } = useLibrary()
  const m = find(id)
  if (!lib) return <div className="mx-auto max-w-6xl px-6 py-8"><Skeleton className="h-40" /></div>
  if (!m) return <div className="mx-auto max-w-6xl px-6 py-8"><Alert variant="destructive"><AlertCircle /><AlertTitle>404</AlertTitle><AlertDescription>{id}</AlertDescription></Alert></div>
  const ready = m.status === "ready"
  const ref = refOf(m)

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-6">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="grid gap-1.5">
          <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight"><span dir="auto">{m.name}</span>{m.sample && <Badge variant="secondary">{t("app.sample")}</Badge>}</h1>
          <div className="flex flex-wrap items-center gap-3 text-sm text-muted-foreground">
            <StatusLabel m={m} />
            <span aria-hidden>·</span><span className="font-mono text-xs">{BASES[m.base].label}</span>
            {m.steps && <><span aria-hidden>·</span><span>{t("models.steps", { n: m.steps })}</span></>}
            {m.created_at && <><span aria-hidden>·</span><span>{when(m.created_at, lang)}</span></>}
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" disabled={!ready} onClick={() => go({ name: "model", id, tab: "deploy" })}><Rocket data-icon="inline-start" /> {t("models.deploy")}</Button>
          <Button disabled={!ready} onClick={() => go({ name: "chat", model: id })}><MessageSquare data-icon="inline-start" /> {t("models.chat")}</Button>
        </div>
      </div>

      {m.status === "training" && (
        <Card className="mb-4">
          <CardHeader><CardTitle>{t("models.training")}</CardTitle><CardDescription>{m.message ?? t("models.notReady")}</CardDescription></CardHeader>
          <CardContent><Progress value={(m.progress ?? 0) * 100} className="h-1.5" /></CardContent>
        </Card>
      )}
      {m.status === "failed" && (
        <Alert variant="destructive" className="mb-4"><AlertCircle /><AlertTitle>{t("models.failed")}</AlertTitle><AlertDescription>{m.message}</AlertDescription></Alert>
      )}

      <Tabs value={ready ? tab ?? "overview" : "overview"} onValueChange={(v) => go({ name: "model", id, tab: v === "overview" ? undefined : v })}>
        <TabsList className="mb-4">
          <TabsTrigger value="overview">{t("models.overview")}</TabsTrigger>
          <TabsTrigger value="results" disabled={!ready}>{t("models.results")}</TabsTrigger>
          <TabsTrigger value="deploy" disabled={!ready}>{t("models.deploy")}</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="grid gap-4">
          {m.metrics && (
            <div className="grid gap-4 sm:grid-cols-3">
              <Card size="sm"><CardHeader><CardDescription>{t("ev.standard")}</CardDescription><p className="font-mono text-3xl font-medium tabular text-standard">{pct(m.metrics.standard)}</p></CardHeader></Card>
              <Card size="sm" className="ring-1 ring-primary/30"><CardHeader><CardDescription dir="auto">{m.name}</CardDescription><p className="font-mono text-3xl font-medium tabular text-yours">{pct(m.metrics.yours)}</p></CardHeader></Card>
              <Card size="sm"><CardHeader><CardDescription>{t("ev.difference")}</CardDescription><p className="font-mono text-3xl font-medium tabular">+{Math.round((m.metrics.yours - m.metrics.standard) * 100)} {t("ev.pts")}</p></CardHeader></Card>
            </div>
          )}
          <Card>
            <CardHeader><CardTitle>{t("models.decisions")}</CardTitle></CardHeader>
            <CardContent className="px-0">
              <Table>
                <TableBody>
                  {(m.decisions ?? []).map((d) => (
                    <TableRow key={d.column}>
                      <TableCell className="ps-6 whitespace-normal"><span dir="auto" className="font-medium">{d.question}</span></TableCell>
                      <TableCell className="text-sm text-muted-foreground">{t(`type.${d.type}`)}</TableCell>
                      <TableCell className="pe-6">
                        <div className="flex flex-wrap gap-1">{d.labels.map((l) => <Badge key={l} variant="outline" className="font-normal" dir="auto">{l}</Badge>)}</div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
          {ready && m.hf_repo && (
            <Card>
              <CardHeader><CardTitle>{t("dep.use")}</CardTitle><CardDescription>Python</CardDescription></CardHeader>
              <CardContent dir="ltr">
                <CodeBlock code={`from typical_ai import Typical\n\nm = Typical.from_pretrained("${m.hf_repo}")\nm.choice(case, "${m.decisions?.find((d) => d.type === "choice")?.question ?? "Which team?"}", ${JSON.stringify(m.decisions?.find((d) => d.type === "choice")?.labels ?? ["a", "b"])})`} />
              </CardContent>
            </Card>
          )}
        </TabsContent>
        <TabsContent value="results">{ready && <Results model={ref} next={() => go({ name: "model", id, tab: "deploy" })} />}</TabsContent>
        <TabsContent value="deploy">{ready && <DeployPanel model={ref} />}</TabsContent>
      </Tabs>
    </div>
  )
}
