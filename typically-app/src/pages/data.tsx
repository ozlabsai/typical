import { useCallback, useEffect, useState } from "react"
import { Database, Plus, Trash2 } from "lucide-react"
import { toast } from "sonner"

import { Page, PageHeader } from "@/components/layout"
import { ErrorState, msg } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { api, type Dataset } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { href } from "@/lib/router"

/** Data: every table uploaded through Customize, the models trained from each, and a way to start again from one. */
export function DataPage() {
  const { t, date } = useI18n()
  const [rows, setRows] = useState<Dataset[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  // samples first: they never change, and every row then keeps the same action column
  const load = useCallback(() => void api.datasets().then((all) => {
    setRows([...all.filter((d) => d.sample), ...all.filter((d) => !d.sample)])
    setError(null)
  }).catch((e) => setError(msg(e))), [])
  useEffect(load, [load])

  async function remove(d: Dataset) {
    if (!window.confirm(t("data.confirm", { name: d.name }))) return
    try {
      await api.deleteDataset(d.token)
      toast.success(`${d.name}: ${t("data.deleted")}`)
      load()
    } catch (e) {
      toast.error(msg(e))
    }
  }

  return (
    <Page>
      <PageHeader title={t("data.title")} description={t("data.lede")}
        actions={<Button asChild><a href={href({ name: "new" })}><Plus data-icon="inline-start" /> {t("data.upload")}</a></Button>} />
      {error && !rows && <ErrorState detail={error} retry={load} />}
      <Card className="gap-0 py-0">
          {!rows ? (
            <div className="grid gap-2 p-6">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-10" />)}</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="ps-6">{t("data.name")}</TableHead>
                  <TableHead className="hidden text-end sm:table-cell">{t("data.rows")}</TableHead>
                  <TableHead className="hidden md:table-cell">{t("data.models")}</TableHead>
                  <TableHead className="hidden md:table-cell">{t("data.created")}</TableHead>
                  <TableHead className="w-0 pe-6"><span className="sr-only">{t("common.actions")}</span></TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((d) => (
                  <TableRow key={d.token}>
                    <TableCell className="max-w-0 ps-6 md:max-w-none">
                      <span className="flex min-w-0 items-center gap-2 font-medium">
                        <Database className="size-4 shrink-0 text-muted-foreground" />
                        <span dir="auto" className="truncate">{d.name}</span>
                        {d.sample && <Badge variant="secondary">{t("app.sample")}</Badge>}
                      </span>
                      <span className="flex min-w-0 gap-1 ps-6 text-xs text-muted-foreground md:max-w-96" title={d.columns.join(", ")}>
                        <span className="shrink-0"><span className="sm:hidden">{t("data.rowsN", { n: d.rows.toLocaleString() })} · </span>{t("data.cols", { n: d.columns.length })}:</span>
                        {/* column names are LTR identifiers: truncate their own end, also inside a Hebrew line */}
                        <bdi dir="ltr" className="block min-w-0 truncate">{d.columns.join(", ")}</bdi>
                      </span>
                    </TableCell>
                    <TableCell className="hidden text-end font-mono text-sm tabular sm:table-cell">{d.rows.toLocaleString()}</TableCell>
                    <TableCell className="hidden md:table-cell">
                      <div className="flex flex-wrap gap-1">
                        {d.models.length ? d.models.map((m) => (
                          <Badge key={m.id} variant="outline" asChild><a href={href({ name: "model", id: m.id })} dir="auto">{m.name}</a></Badge>
                        )) : <span className="text-muted-foreground">—</span>}
                      </div>
                    </TableCell>
                    <TableCell className="hidden md:table-cell text-sm text-muted-foreground">{date(d.created)}</TableCell>
                    <TableCell className="pe-6">
                      <div className="flex justify-end gap-1">
                        <Button variant="ghost" size="sm" asChild><a href={href({ name: "new", data: d.token })} aria-label={t("data.new")} title={t("data.new")}><Plus data-icon="inline-start" /> <span className="hidden sm:inline">{t("data.new")}</span></a></Button>
                        {d.sample ? <span aria-hidden className="size-8 shrink-0" /> : (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              {/* a span: a disabled button gets no pointer events, so the reason would never show */}
                              <span tabIndex={d.models.length ? 0 : -1}>
                                <Button variant="ghost" size="icon-sm" aria-label={t("data.delete")} disabled={d.models.length > 0} onClick={() => remove(d)}><Trash2 /></Button>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent>{d.models.length ? t("data.inUse") : t("data.delete")}</TooltipContent>
                          </Tooltip>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
                {rows.every((d) => d.sample) && (
                  <TableRow><TableCell colSpan={5} className="ps-6 text-sm text-muted-foreground">{t("data.empty")}</TableCell></TableRow>
                )}
              </TableBody>
            </Table>
          )}
      </Card>
    </Page>
  )
}
