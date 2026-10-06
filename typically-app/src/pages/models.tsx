import { useEffect, useState } from "react"
import { AlertCircle, Archive, Boxes, Loader2, MessageSquare, MoreHorizontal, Plus, RefreshCw, Rocket, X } from "lucide-react"
import { toast } from "sonner"

import { StatusDot } from "@/components/app-sidebar"
import { ShareButton } from "@/components/share-dialog"
import { Page, PageHeader, SectionHeader, Stat } from "@/components/layout"
import { CodeBlock, msg, StatusCopy } from "@/components/shared"
import { useTrainStatus } from "@/components/training-curve"
import { TrainingProgress } from "@/components/training-progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { corrections, library, type CorrectionItem, type LibraryModel } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { refOf, useLibrary } from "@/lib/library"
import { BASES, pct, say, signed, snippet } from "@/lib/project"
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
      <span aria-hidden className="inline-block text-muted-foreground rtl:rotate-180">→</span>
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
    <Page>
      <PageHeader title={t("models.title")} description={t("models.lede")}
        actions={<Button asChild><a href={href({ name: "new" })}><Plus data-icon="inline-start" /> {t("app.new")}</a></Button>} />

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
                  <TableHead className="hidden md:table-cell">{t("models.from")}</TableHead>
                  <TableHead>{t("models.status")}</TableHead>
                  <TableHead className="whitespace-normal">{t("models.agreement")}</TableHead>
                  <TableHead className="hidden md:table-cell">{t("models.created")}</TableHead>
                  <TableHead className="hidden w-28 pe-6 md:table-cell"><span className="sr-only">{t("common.actions")}</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {lib.custom.map((m) => (
                  <TableRow key={m.id} className="cursor-pointer" onClick={() => go({ name: "model", id: m.id })}>
                    <TableCell className="ps-6 whitespace-normal">
                      <span className="flex flex-wrap items-center gap-x-2 gap-y-1 font-medium">
                        <span dir="auto">{m.name}</span>
                        {m.sample && <Badge variant="secondary">{t("app.sample")}</Badge>}
                      </span>
                      <span className="text-xs text-muted-foreground">{t("models.questions", { n: m.decisions?.length ?? 0 })}</span>
                    </TableCell>
                    <TableCell className="hidden md:table-cell font-mono text-xs">{BASES[m.base].label}</TableCell>
                    <TableCell><StatusLabel m={m} /></TableCell>
                    <TableCell><Agreement m={m} /></TableCell>
                    <TableCell className="hidden md:table-cell text-sm text-muted-foreground">{when(m.created_at, lang)}</TableCell>
                    <TableCell className="hidden pe-6 md:table-cell" onClick={(e) => e.stopPropagation()}>
                      <div className="flex justify-end gap-1">
                        <Button variant="ghost" size="sm" disabled={m.status !== "ready"} onClick={() => go({ name: "chat", model: m.id })}>
                          <MessageSquare data-icon="inline-start" /> <span className="hidden md:inline">{t("models.chat")}</span>
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild><Button variant="ghost" size="icon-sm" aria-label={t("common.more")}><MoreHorizontal /></Button></DropdownMenuTrigger>
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

      <SectionHeader title={t("models.baseTitle")} />
      <div className="-mt-2 grid gap-4 sm:grid-cols-2">
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
    </Page>
  )
}

/** Chat's "Wrong?" answers for this model, and the button that retrains them into the next version (<root>_v<N>). */
function Corrections({ m }: { m: LibraryModel }) {
  const { t } = useI18n()
  const { lib, refresh } = useLibrary()
  const [items, setItems] = useState<CorrectionItem[]>([])
  const [busy, setBusy] = useState(false)
  useEffect(() => { corrections.list(m.id).then((r) => setItems(r.items)).catch(() => setItems([])) }, [m.id, m.corrections])
  if (!m.corrections) return null
  // the server names versions <root>_v<N>, N = next free; mirror it for the label
  const root = (m.version ?? 1) > 1 ? m.id.replace(/_v\d+$/, "") : m.id
  const next = 1 + Math.max(1, ...(lib?.custom ?? []).filter((x) => x.id === root || x.id.startsWith(`${root}_v`)).map((x) => x.version ?? 1))
  const jobRunning = lib?.custom.some((x) => x.status === "training" || x.status === "queued")

  async function remove(n: number) {
    try { await corrections.remove(m.id, n); refresh() } catch (e) { toast.error(msg(e)) }
  }
  async function retrain() {
    setBusy(true)
    try {
      const r = await corrections.retrain(m.id)
      toast.success(t("corr.started", { name: r.name }))
      await refresh() // the new version must be in the library before its page opens
      go({ name: "model", id: r.id })
    } catch (e) {
      toast.error(msg(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{m.corrections === 1 ? t("corr.titleOne") : t("corr.title", { n: m.corrections })}</CardTitle>
        <CardDescription>{t("corr.lede")}</CardDescription>
        <CardAction>
          <Button onClick={retrain} disabled={busy || jobRunning || m.status !== "ready"}>
            {busy ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <RefreshCw data-icon="inline-start" />} {t("corr.retrain", { n: next })}
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="ps-6">{t("corr.case")}</TableHead>
              <TableHead className="hidden md:table-cell">{t("corr.question")}</TableHead>
              <TableHead>{t("corr.change")}</TableHead>
              <TableHead className="w-12 pe-6"><span className="sr-only">{t("corr.remove")}</span></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {items.map((c) => (
              <TableRow key={c.n}>
                <TableCell className="max-w-72 ps-6 whitespace-normal"><span dir="auto" className="line-clamp-2 text-sm text-muted-foreground">{snippet(c.case)}</span></TableCell>
                <TableCell className="hidden md:table-cell whitespace-normal"><span dir="auto" className="text-sm">{c.question}</span></TableCell>
                <TableCell>
                  <span className="inline-flex items-center gap-2 text-sm whitespace-nowrap">
                    <span dir="auto" className="text-muted-foreground line-through decoration-muted-foreground/50">{c.model_answer ? say(t, c.type, c.model_answer) : "—"}</span>
                    <span aria-hidden className="inline-block text-muted-foreground rtl:rotate-180">→</span>
                    <span dir="auto" className="font-medium text-yours">{say(t, c.type, c.answer)}</span>
                  </span>
                </TableCell>
                <TableCell className="pe-6">
                  <Button variant="ghost" size="icon-sm" onClick={() => remove(c.n)} aria-label={t("corr.remove")} title={t("corr.remove")}><X /></Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  )
}

/** The live card while it trains; after a failure, the same milestones show where it stopped. */
function TrainingCard({ m }: { m: LibraryModel }) {
  const { t } = useI18n()
  const status = useTrainStatus(m.id, true)
  const failed = m.status === "failed"
  if (failed && !status?.events?.length) return null   // failed before this card existed: the alert says it all
  return (
    <Card>
      <CardHeader>
        <CardTitle>{t(failed ? "tr.stopped" : "models.training")}</CardTitle>
        {!failed && <CardDescription>{m.code || m.message ? <StatusCopy code={m.code} message={m.message} /> : t("models.notReady")}</CardDescription>}
      </CardHeader>
      <CardContent>
        {status ? <TrainingProgress status={status} /> : <Progress value={(m.progress ?? 0) * 100} className="h-1.5" aria-label={t("tr.progress")} />}
      </CardContent>
    </Card>
  )
}

export function ModelPage({ id, tab }: { id: string; tab?: string }) {
  const { t, lang } = useI18n()
  const { lib, find } = useLibrary()
  const m = find(id)
  if (!lib) return <Page><Skeleton className="h-40" /></Page>
  if (!m) return <Page><Alert variant="destructive"><AlertCircle /><AlertTitle>404</AlertTitle><AlertDescription>{id}</AlertDescription></Alert></Page>
  const ready = m.status === "ready"
  const ref = refOf(m)

  return (
    <Page>
      <PageHeader
        title={<span dir="auto">{m.name}</span>}
        badge={m.sample ? <Badge variant="secondary">{t("app.sample")}</Badge> : undefined}
        description={
          <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
            {/* each separator travels with the item after it, so a wrapped line never ends on a dangling dot */}
            {[
              <StatusLabel m={m} />,
              (m.version ?? 1) > 1 && m.parent && (
                <a href={href({ name: "model", id: m.parent })} className="underline-offset-4 hover:text-foreground hover:underline" dir="auto">
                  {t("models.versionOf", { n: m.version!, name: find(m.parent)?.name ?? m.parent })}
                </a>
              ),
              <span className="font-mono text-xs">{BASES[m.base].label}</span>,
              m.steps && <span>{t("models.steps", { n: m.steps })}</span>,
              m.created_at && <span>{when(m.created_at, lang)}</span>,
            ].filter(Boolean).map((x, i) => <span key={i} className="inline-flex items-center gap-3">{i > 0 && <span aria-hidden>·</span>}{x}</span>)}
          </span>
        }
        actions={<>
          <ShareButton id={id} disabled={!ready || !m.metrics} />
          <Button variant="outline" disabled={!ready} onClick={() => go({ name: "model", id, tab: "deploy" })}><Rocket data-icon="inline-start" /> {t("models.deploy")}</Button>
          <Button disabled={!ready} onClick={() => go({ name: "chat", model: id })}><MessageSquare data-icon="inline-start" /> {t("models.chat")}</Button>
        </>}
      />

      {m.status === "failed" && (
        <Alert variant="destructive"><AlertCircle /><AlertTitle>{t("models.failed")}</AlertTitle><AlertDescription><p><StatusCopy code={m.code} message={m.message} /></p></AlertDescription></Alert>
      )}
      {(m.status === "training" || m.status === "failed") && <TrainingCard key={m.id} m={m} />}

      <Tabs value={ready ? tab ?? "overview" : "overview"} onValueChange={(v) => go({ name: "model", id, tab: v === "overview" ? undefined : v })}>
        <TabsList className="mb-2">
          <TabsTrigger value="overview">{t("models.overview")}</TabsTrigger>
          <TabsTrigger value="results" disabled={!ready}>{t("models.results")}</TabsTrigger>
          <TabsTrigger value="deploy" disabled={!ready}>{t("models.deploy")}</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="grid gap-4">
          {m.metrics && (
            <div className="grid grid-cols-3 gap-2 sm:gap-4">
              <Stat label={t("ev.standard")} value={pct(m.metrics.standard)} tone="standard" />
              <Stat label={m.name} value={pct(m.metrics.yours)} tone="yours" highlight />
              <Stat label={t("ev.difference")} value={`${signed(Math.round((m.metrics.yours - m.metrics.standard) * 100))} ${t("ev.pts")}`} />
            </div>
          )}
          <Corrections m={m} />
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
                        <div className="flex flex-wrap gap-1">{d.labels.map((l) => <Badge key={l} variant="outline" className="font-normal" dir="auto">{say(t, d.type, l)}</Badge>)}</div>
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
        <TabsContent value="results" className="grid gap-6">{ready && <Results model={ref} />}</TabsContent>
        <TabsContent value="deploy" className="grid gap-6">{ready && <DeployPanel model={ref} />}</TabsContent>
      </Tabs>
    </Page>
  )
}
