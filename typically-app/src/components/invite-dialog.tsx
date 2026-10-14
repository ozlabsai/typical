import { useCallback, useEffect, useState } from "react"
import { Copy, Download, Ellipsis, Loader2, Plus, Trash2 } from "lucide-react"
import { toast } from "sonner"

import { CopyButton, copy, msg } from "@/components/shared"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { inviteLink, invites, ticketUrl, type Invite, type Invites } from "@/lib/api"
import { useAuth } from "@/lib/auth"
import { locale, useI18n } from "@/lib/i18n"
import { cn } from "@/lib/utils"

/** The name cut into 1-3 lines at spaces and its size in cqw: the cut that sets it largest, fewer lines on a tie
 *  (typically_share.name_lines draws the PNG the same way). 62cqw: the main panel's text width; 0.62em: a Geist Mono cell + spacing. */
export function nameLines(name: string, widest = 62, largest = 6): { lines: string[]; size: number } {
  const words = name.split(/\s+/).filter(Boolean), n = words.length
  const cuts: number[][] = [[]]
  for (let i = 1; i < n; i++) cuts.push([i])
  for (let i = 1; i < n; i++) for (let j = i + 1; j < n; j++) cuts.push([i, j])
  let best = { lines: [name], size: 0 }
  for (const cut of cuts) {
    const b = [0, ...cut, n]
    const lines = b.slice(1).map((end, k) => words.slice(b[k], end).join(" "))
    const size = Math.min(largest, widest / (0.62 * Math.max(...lines.map((l) => [...l].length))))
    if (size > best.size + 1e-9) best = { lines, size }
  }
  return best
}

/** The invite as an "ADMIT ONE" ticket; the downloadable PNG (ticket.png) draws the same. Never shows the code or the email. */
function Ticket({ invite, from }: { invite: Invite; from: string }) {
  const { t, lang } = useI18n()
  const name = invite.name ?? ""
  const { lines, size } = nameLines(name)
  const day = new Intl.DateTimeFormat(lang === "he" ? "he-IL" : "en-GB", { dateStyle: "medium" }).format(new Date(invite.created))
  return (
    <div role="img" aria-label={t("ticket.label", { name })} className="@container">
      <div className="ticket">
        <div className="grid min-w-0 grid-rows-[auto_1fr_auto] p-[7cqw]">
          <div className="ticket-small">
            <p>{t("ticket.presents")}</p>
            <p className="truncate">{t("ticket.from", { name: "" })}<bdi>{from}</bdi></p>
          </div>
          <p className="self-center uppercase" style={{ fontSize: `${size}cqw`, lineHeight: 1, letterSpacing: "0.02em" }}>
            {lines.map((l, i) => <span key={i} className="block"><bdi>{l}</bdi></span>)}
          </p>
          <p className="ticket-small"><bdi>{window.location.host}</bdi> · {day}</p>
        </div>
        <div className="ticket-stub">
          <span className="ticket-digits">{[...invite.no].map((d, i) => <span key={i} className="block">{d}</span>)}</span>
          <span className="ticket-vertical">{t("ticket.admit")}</span>
        </div>
      </div>
    </div>
  )
}

const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/   // the server's check (typically_auth.EMAIL)
type Field = "name" | "email"

/** Invite people: a name + email form, then the new invite as a ticket to copy or download. Below the form, the invites you made
 *  (5 active; admins unlimited, and can switch to every invite). */
export function InviteDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t, lang } = useI18n()
  const { user } = useAuth()
  const [all, setAll] = useState(false)   // admins: every invite, with who made it
  const [data, setData] = useState<Invites | undefined>(undefined) // undefined = loading
  const [busy, setBusy] = useState(false)
  const [form, setForm] = useState({ name: "", email: "" })
  const [errors, setErrors] = useState<Partial<Record<Field, string>>>({})
  const [fresh, setFresh] = useState<Invite | null>(null)   // the invite just made: shown as the ticket

  const load = useCallback(() => invites.list(all).then(setData).catch((e) => toast.error(msg(e))), [all])
  useEffect(() => {
    if (open) void load()
    else { setFresh(null); setErrors({}) }
  }, [open, load])

  const day = (iso: string) => new Intl.DateTimeFormat(locale(lang), { month: "short", day: "numeric" }).format(new Date(iso))
  const full = data?.left === 0
  const count = !data ? null : data.limit === null ? t("invite.unlimited") : t("invite.used", { used: data.limit - (data.left ?? 0), limit: data.limit })

  async function create(e: React.FormEvent) {
    e.preventDefault()
    const name = form.name.trim().replace(/\s+/g, " "), email = form.email.trim()
    const errs = { name: name ? undefined : t("invite.nameRequired"), email: EMAIL.test(email) && email.length <= 254 ? undefined : t("invite.emailInvalid") }
    setErrors(errs)
    if (errs.name || errs.email) return
    setBusy(true)
    try {
      const r = await invites.create(name, email)
      setFresh(r.invites.find((i) => i.code === r.code) ?? null)
      setForm({ name: "", email: "" })
      await load()
    } catch (err) { toast.error(msg(err)) } finally { setBusy(false) }
  }
  async function revoke(code: string) {
    if (!window.confirm(t("invite.confirm"))) return
    try { await invites.revoke(code); toast.success(t("invite.revoked")); await load() } catch (err) { toast.error(msg(err)) }
  }

  const field = (k: Field, type: string, autoComplete: string, maxLength: number) => (
    <div className="grid content-start gap-1.5">
      <Label htmlFor={`invite-${k}`}>{t(`invite.${k}`)}</Label>
      <Input id={`invite-${k}`} type={type} value={form[k]} required maxLength={maxLength} autoComplete={autoComplete} dir={k === "email" ? "ltr" : "auto"}
        placeholder={t(`invite.${k}Placeholder`)} disabled={!data || full} aria-invalid={Boolean(errors[k])}
        aria-describedby={errors[k] ? `invite-${k}-error` : undefined}
        onChange={(e) => { setForm({ ...form, [k]: e.target.value }); setErrors({ ...errors, [k]: undefined }) }} />
      {errors[k] && <p id={`invite-${k}-error`} className="text-xs text-destructive">{errors[k]}</p>}
    </div>
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100svh-2rem)] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pe-8">
            <DialogTitle>{t("invite.title")}</DialogTitle>
            {count && <span className="font-mono text-xs text-muted-foreground tabular">{count}</span>}
          </div>
          <DialogDescription>{t("invite.lede")}</DialogDescription>
        </DialogHeader>

        {fresh ? (
          <div className="grid gap-4">
            <div className="ticket-in"><Ticket invite={fresh} from={fresh.inviter ?? user?.name ?? ""} /></div>
            <div className="flex flex-wrap gap-2">
              <CopyButton text={inviteLink(fresh)} label={t("invite.copy")} variant="default" size="default" />
              <Button variant="outline" asChild>
                <a href={ticketUrl(fresh.code)} download={`invite-ticket-${fresh.no}.png`}><Download data-icon="inline-start" /> {t("invite.download")}</a>
              </Button>
              <Button variant="ghost" className="sm:ms-auto" onClick={() => setFresh(null)}>{t("invite.another")}</Button>
            </div>
          </div>
        ) : (
          <>
            <form onSubmit={create} noValidate className="grid gap-3">
              <div className="grid gap-3 sm:grid-cols-2">
                {field("name", "text", "off", 60)}
                {field("email", "email", "off", 254)}
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="max-w-80 text-xs text-muted-foreground">{full ? t("invite.full", { limit: data?.limit ?? 0 }) : t("invite.privacy")}</p>
                <Button type="submit" disabled={busy || !data || full} className="ms-auto">
                  {busy ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Plus data-icon="inline-start" />} {t("invite.create")}
                </Button>
              </div>
            </form>

            <section className="grid gap-1 border-t pt-4">
              <div className="flex min-h-7 items-center justify-between gap-2">
                <h3 className="eyebrow">{t(all ? "invite.everyone" : "invite.yours")}</h3>
                {user?.admin && (
                  <div role="group" aria-label={t("invite.everyone")} className="inline-flex rounded-md bg-muted p-0.5 text-xs">
                    {([false, true] as const).map((v) => (
                      <button key={String(v)} type="button" aria-pressed={all === v} onClick={() => setAll(v)}
                        className={cn("rounded-[5px] px-2.5 py-1 font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50 focus-visible:outline-none",
                          all === v && "bg-background text-foreground shadow-xs")}>
                        {t(v ? "invite.all" : "invite.mine")}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {!data ? (
                <div className="grid h-16 place-items-center"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>
              ) : data.invites.length ? (
                <ul className="divide-y">
                  {data.invites.map((i) => (
                    <li key={i.code} className="flex items-center gap-3 py-2.5">
                      <div className="grid min-w-0 flex-1 gap-0.5 leading-tight">
                        <span className="truncate text-sm font-medium">{i.name ? <bdi>{i.name}</bdi> : "—"}</span>
                        <span className="truncate text-xs text-muted-foreground">
                          {i.email ? <bdi>{i.email}</bdi> : "—"}
                          {all && <span className="sm:hidden"> · <bdi>{i.inviter ?? "—"}</bdi></span>}
                        </span>
                      </div>
                      {all && <span className="hidden w-28 truncate text-xs text-muted-foreground sm:block">{t("invite.by", { name: "" })}<bdi>{i.inviter ?? "—"}</bdi></span>}
                      <div className="grid shrink-0 justify-items-end gap-1 sm:w-32">
                        {i.joined
                          ? <Badge variant="outline" className="border-success/30 bg-success/10 font-normal text-success">{t("invite.joined", { date: day(i.joined) })}</Badge>
                          : <Badge variant="outline" className="font-normal text-muted-foreground">{t("invite.pending")}</Badge>}
                        <span className="text-xs text-muted-foreground tabular">{t("invite.created", { date: day(i.created) })}</span>
                      </div>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon-sm" aria-label={t("invite.actions", { name: i.name ?? "—" })}><Ellipsis /></Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onSelect={() => void copy(t, inviteLink(i))}><Copy /> {t("invite.copy")}</DropdownMenuItem>
                          <DropdownMenuItem asChild>
                            <a href={ticketUrl(i.code)} download={`invite-ticket-${i.no}.png`}><Download /> {t("invite.download")}</a>
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem variant="destructive" onSelect={() => void revoke(i.code)}><Trash2 /> {t("invite.revoke")}</DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="py-2 text-sm text-muted-foreground">{t("invite.none")}</p>
              )}
            </section>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
