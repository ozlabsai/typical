import { useEffect, useState } from "react"
import { AlertCircle, Archive, Boxes, Loader2, MessageSquare, MoreHorizontal, Plus, RefreshCw, SearchX, X } from "lucide-react"
import { toast } from "sonner"

import { StatusDot } from "@/components/app-sidebar"
import { ShareButton } from "@/components/share-dialog"
import { Page, PageHeader, SectionHeader } from "@/components/layout"
import { msg, StatusCopy } from "@/components/shared"
import { StatusLine, StopButton, TrainingProgress, useTrainStatus } from "@/components/training-progress"
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
import { BASES, pct, presetOf, say, snippet } from "@/lib/project"
import { go, href } from "@/lib/router"
import { DeployPanel } from "@/steps/deploy"
import { Results, ScoreStats } from "@/steps/evaluate"

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

/** Icon + sentence + one action: every empty or missing state. */
function Empty({ icon: Icon, text, action }: { icon: typeof Boxes; text: string; action: React.ReactNode }) {
  return (
    <div className="grid justify-items-center gap-4 px-6 py-8 text-center">
      <Icon className="size-5 text-muted-foreground" />
      <p className="text-sm text-muted-foreground">{text}</p>
      {action}
    </div>
  )
}

export function ModelsPage() {
  const { t, date } = useI18n()
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

      <section className="grid gap-4">
        <SectionHeader title={t("models.custom")} />
        <Card className="gap-0 py-0">
          {!lib ? (
            <div className="grid gap-2 p-6">{[0, 1].map((i) => <Skeleton key={i} className="h-10" />)}</div>
          ) : !lib.custom.length ? (
            <Empty icon={Boxes} text={t("models.empty")} action={<Button asChild><a href={href({ name: "new" })}>{t("models.train")}</a></Button>} />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="ps-6">{t("models.name")}</TableHead>
                  <TableHead className="hidden md:table-cell">{t("models.from")}</TableHead>
                  <TableHead className="hidden sm:table-cell">{t("models.status")}</TableHead>
                  <TableHead className="whitespace-normal">{t("models.agreement")}</TableHead>
                  <TableHead className="hidden md:table-cell">{t("models.created")}</TableHead>
                  <TableHead className="w-0 pe-6"><span className="sr-only">{t("common.actions")}</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {lib.custom.map((m) => (
                  // the name is a real link stretched over the row: keyboard, middle-click and screen readers open it like any link
                  <TableRow key={m.id} className="relative">
                    <TableCell className="ps-6 whitespace-normal">
                      <span className="flex flex-wrap items-center gap-x-2 gap-y-1 font-medium">
                        <a href={href({ name: "model", id: m.id })} dir="auto"
                          className="outline-none after:absolute after:inset-0 after:rounded-md focus-visible:after:ring-3 focus-visible:after:ring-ring/50">{m.name}</a>
                        {m.sample && <Badge variant="secondary">{t("app.sample")}</Badge>}
                      </span>
                      <span className="flex flex-wrap items-center gap-x-2 text-xs text-muted-foreground">
                        <span className="sm:hidden"><StatusLabel m={m} /></span>
                        {t("models.questions", { n: m.decisions?.length ?? 0 })}
                      </span>
                    </TableCell>
                    <TableCell className="hidden md:table-cell text-sm">{BASES[m.base].label}</TableCell>
                    <TableCell className="hidden sm:table-cell"><StatusLabel m={m} /></TableCell>
                    <TableCell><Agreement m={m} /></TableCell>
                    <TableCell className="hidden md:table-cell text-sm text-muted-foreground">{date(m.created_at)}</TableCell>
                    <TableCell className="relative z-10 pe-6">
                      <div className="flex justify-end gap-1">
                        <Button variant="ghost" size="sm" className="hidden md:inline-flex" disabled={m.status !== "ready"} onClick={() => go({ name: "chat", model: m.id })}>
                          <MessageSquare data-icon="inline-start" /> {t("models.chat")}
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild><Button variant="ghost" size="icon-sm" aria-label={t("common.more")}><MoreHorizontal /></Button></DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => go({ name: "model", id: m.id })}>{t("models.open")}</DropdownMenuItem>
                            <DropdownMenuItem disabled={m.status !== "ready"} onClick={() => go({ name: "chat", model: m.id })}><MessageSquare /> {t("models.chat")}</DropdownMenuItem>
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
        </Card>
      </section>

      <section className="grid gap-4">
        <SectionHeader title={t("models.baseTitle")} />
        <div className="grid gap-4 sm:grid-cols-2">
        {(lib?.base ?? []).map((m) => (
          <Card key={m.id} size="sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">{m.name}</CardTitle>
              <CardDescription>{t(`base.${m.base}.note`)}</CardDescription>
              <CardAction>
                <Button variant="outline" size="sm" onClick={() => go({ name: "chat", model: m.id })}><MessageSquare data-icon="inline-start" /> {t("models.chat")}</Button>
              </CardAction>
            </CardHeader>
          </Card>
        ))}
        </div>
      </section>
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
  const jobRunning = lib?.custom.some((x) => x.status === "training")

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
          <Button size="sm" onClick={retrain} disabled={busy || jobRunning || m.status !== "ready"}>
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

/** The live card while it trains; after a failure, the same milestones show where it stopped; once ready, the run's record (folded). */
function TrainingCard({ m }: { m: LibraryModel }) {
  const { t } = useI18n()
  const { status } = useTrainStatus(m.id, !m.sample)
  const [open, setOpen] = useState(false)
  const failed = m.status === "failed", ready = m.status === "ready"
  if ((failed || ready) && !status?.events?.length) return null   // a run from before milestones were recorded
  return (
    <Card>
      <CardHeader>
        <CardTitle>{t(ready ? "models.howTrained" : failed ? "tr.stopped" : "models.training")}</CardTitle>
        {!failed && !ready && <CardDescription>{status ? <StatusLine status={status} /> : m.code || m.message ? <StatusCopy code={m.code} message={m.message} /> : t("models.notReady")}</CardDescription>}
        {!ready && !failed && <CardAction><StopButton slug={m.id} /></CardAction>}
        {ready && <CardAction><Button variant="ghost" size="sm" aria-expanded={open} onClick={() => setOpen(!open)}>{t(open ? "models.hideRun" : "models.showRun")}</Button></CardAction>}
      </CardHeader>
      {(!ready || open) && (
        <CardContent>
          {status ? <TrainingProgress status={status} /> : <Progress value={(m.progress ?? 0) * 100} className="h-1.5" aria-label={t("tr.progress")} />}
        </CardContent>
      )}
    </Card>
  )
}

export function ModelPage({ id, tab }: { id: string; tab?: string }) {
  const { t, date } = useI18n()
  const { lib, find } = useLibrary()
  const m = find(id)
  if (!lib) return <Page><Skeleton className="h-40" /></Page>
  if (!m)
    return (
      <Page>
        <Card><Empty icon={SearchX} text={t("models.notFoundBody")} action={<Button asChild><a href={href({ name: "models" })}>{t("models.backToModels")}</a></Button>} /></Card>
      </Page>
    )
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
              <span>{BASES[m.base].label}</span>,
              presetOf(m.steps) && <span>{t("models.steps", { preset: t(`preset.${presetOf(m.steps)!}`) })}</span>,
              m.created_at && <span>{date(m.created_at)}</span>,
            ].filter(Boolean).map((x, i) => <span key={i} className="inline-flex items-center gap-3">{i > 0 && <span aria-hidden>·</span>}{x}</span>)}
          </span>
        }
        actions={<>
          <ShareButton id={id} disabled={!ready || !m.metrics} />
          <Button disabled={!ready} onClick={() => go({ name: "chat", model: id })}><MessageSquare data-icon="inline-start" /> {t("models.chat")}</Button>
        </>}
      />

      {m.status === "failed" && (
        <Alert variant="destructive"><AlertCircle /><AlertTitle>{t("models.failed")}</AlertTitle><AlertDescription><p><StatusCopy code={m.code} message={m.message} /></p></AlertDescription></Alert>
      )}
      {m.status !== "ready" && <TrainingCard key={m.id} m={m} />}

      <Tabs value={ready ? tab ?? "overview" : "overview"} onValueChange={(v) => go({ name: "model", id, tab: v === "overview" ? undefined : v })}>
        <TabsList className="mb-2">
          <TabsTrigger value="overview">{t("models.overview")}</TabsTrigger>
          <TabsTrigger value="results" disabled={!ready}>{t("models.results")}</TabsTrigger>
          <TabsTrigger value="deploy" disabled={!ready}>{t("models.deploy")}</TabsTrigger>
        </TabsList>
        <TabsContent value="overview" className="grid gap-6">
          {m.metrics && <ScoreStats standard={m.metrics.standard} yours={m.metrics.yours} />}
          <Corrections m={m} />
          {ready && <TrainingCard key={m.id} m={m} />}
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
        </TabsContent>
        <TabsContent value="results" className="grid gap-6">{ready && <Results model={ref} />}</TabsContent>
        <TabsContent value="deploy" className="grid gap-6">{ready && <DeployPanel model={ref} />}</TabsContent>
      </Tabs>
    </Page>
  )
}
