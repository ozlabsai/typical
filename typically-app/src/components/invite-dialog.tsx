import { useEffect, useState } from "react"
import { Download, Loader2, Plus, Ticket as TicketIcon } from "lucide-react"
import { toast } from "sonner"

import { CopyButton, msg } from "@/components/shared"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { inviteLink, invites, ticketUrl, type Invite, type Invites } from "@/lib/api"
import { useAuth } from "@/lib/auth"
import { useI18n } from "@/lib/i18n"

/** Split at the space nearest the middle (typically_share.two_lines draws the PNG the same way). */
function twoLines(name: string): string[] {
  const spaces = [...name].flatMap((ch, i) => (ch === " " ? [i] : []))
  if (!spaces.length) return [name]
  const cut = spaces.reduce((a, b) => (Math.abs(b - name.length / 2) < Math.abs(a - name.length / 2) ? b : a))
  return [name.slice(0, cut), name.slice(cut + 1)]
}

/** The invite as an "ADMIT ONE" ticket; the downloadable PNG (ticket.png) draws the same. Never shows the code. */
function Ticket({ invite, from }: { invite: Invite; from: string }) {
  const { t, lang } = useI18n()
  const name = invite.for?.trim() || t("ticket.bearer")
  const lines = twoLines(name)
  const longest = Math.max(...lines.map((l) => [...l].length))
  const size = Math.min(10, 66 / (0.62 * longest))   // cqw: the name fills the main panel (about 66cqw wide) and never overflows it
  const day = new Intl.DateTimeFormat(lang === "he" ? "he-IL" : "en-GB", { dateStyle: "medium" }).format(new Date(invite.created))
  return (
    <div role="img" aria-label={t("ticket.label", { name })} className="@container">
      <div className="ticket">
        <div className="flex min-w-0 flex-col justify-between p-[4.67cqw]">
          <p className="ticket-small">{t("ticket.from", { name: "" })}<bdi>{from}</bdi></p>
          <p className="font-semibold uppercase" style={{ fontSize: `${size}cqw`, lineHeight: 0.98, letterSpacing: "-0.02em" }}>
            {lines.map((l, i) => <span key={i} className="block"><bdi>{l}</bdi></span>)}
          </p>
          <p className="ticket-small"><bdi>{window.location.host}</bdi> · {day}</p>
        </div>
        <div className="ticket-stub flex items-center justify-center gap-[2.5cqw]">
          <span className="ticket-vertical text-[5cqw] uppercase">{t("ticket.admit")}</span>
          <span className="ticket-vertical text-[7.67cqw] opacity-30">{t("ticket.no", { n: invite.no })}</span>
        </div>
      </div>
    </div>
  )
}

/** Make an invite (for someone, optionally): it arrives as a ticket to copy or download. Below, the ones you made (5 active; admins unlimited). */
export function InviteDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t, date } = useI18n()
  const { user } = useAuth()
  const [data, setData] = useState<Invites | undefined>(undefined) // undefined = loading
  const [busy, setBusy] = useState(false)
  const [name, setName] = useState("")
  const [fresh, setFresh] = useState<string | null>(null)   // the code just made: shown as the ticket
  const ticket = data?.invites.find((i) => i.code === fresh)

  useEffect(() => {
    if (open) invites.list().then(setData).catch((e) => toast.error(msg(e)))
    else setFresh(null)
  }, [open])

  async function run(f: () => Promise<Invites>) {
    setBusy(true)
    try { setData(await f()) } catch (e) { toast.error(msg(e)) } finally { setBusy(false) }
  }
  const create = (e: React.FormEvent) => {
    e.preventDefault()
    void run(async () => { const r = await invites.create(name.trim()); setFresh(r.code); setName(""); return r })
  }
  const revoke = (code: string) => window.confirm(t("invite.confirm")) &&
    run(async () => { const r = await invites.revoke(code); toast.success(t("invite.revoked")); return r })
  const download = (i: Invite, label: string, variant: "outline" | "ghost") => (
    <Button variant={variant} size="sm" asChild>
      <a href={ticketUrl(i.code)} download={`invite-ticket-${i.no}.png`}><Download data-icon="inline-start" /> {label}</a>
    </Button>
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100svh-2rem)] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("invite.title")}</DialogTitle>
          <DialogDescription>{t("invite.lede")}</DialogDescription>
        </DialogHeader>

        {ticket ? (
          <div className="grid gap-4">
            <div className="ticket-in py-1"><Ticket invite={ticket} from={user?.name ?? ""} /></div>
            <div className="flex flex-wrap items-center gap-2">
              <CopyButton text={inviteLink(ticket)} label={t("invite.copy")} />
              {download(ticket, t("invite.download"), "outline")}
              <Button variant="ghost" size="sm" className="ms-auto" onClick={() => setFresh(null)}>{t("invite.another")}</Button>
            </div>
          </div>
        ) : (
          <form onSubmit={create} className="grid gap-2">
            <Label htmlFor="invite-for">{t("invite.for")} <span className="font-normal text-muted-foreground">{t("auth.optional")}</span></Label>
            <div className="flex gap-2">
              <Input id="invite-for" value={name} onChange={(e) => setName(e.target.value)} maxLength={40} dir="auto" autoComplete="off"
                placeholder={t("invite.forPlaceholder")} disabled={!data || data.left === 0} />
              <Button type="submit" disabled={busy || !data || data.left === 0}>
                {busy ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Plus data-icon="inline-start" />} {t("invite.create")}
              </Button>
            </div>
            {data?.limit != null && <span className="text-xs text-muted-foreground tabular">{t("invite.left", { left: data.left ?? 0, limit: data.limit })}</span>}
          </form>
        )}

        <div className="grid gap-2 border-t pt-4">
          <h3 className="eyebrow">{t("invite.yours")}</h3>
          {!data ? (
            <div className="grid h-16 place-items-center"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>
          ) : data.invites.length ? (
            <ul className="grid gap-1">
              {data.invites.map((i) => (
                <li key={i.code} className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md py-1.5 ps-1">
                  <TicketIcon aria-hidden className="size-4 shrink-0 -rotate-12 text-standard" />
                  <div className="grid min-w-48 flex-1 leading-tight">  {/* phones: the actions wrap under it */}
                    <span className="truncate text-sm font-medium">{i.for ? <bdi>{i.for}</bdi> : <span className="font-normal text-muted-foreground">{t("invite.unnamed")}</span>}</span>
                    <span className="text-xs text-muted-foreground">{t("invite.created", { date: date(i.created) })}</span>
                  </div>
                  <CopyButton text={inviteLink(i)} label={t("invite.copy")} />
                  {download(i, t("invite.downloadShort"), "ghost")}
                  <Button variant="ghost" size="sm" className="text-destructive" disabled={busy} onClick={() => revoke(i.code)}>{t("invite.revoke")}</Button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">{t("invite.none")}</p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
