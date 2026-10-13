import { useEffect, useState } from "react"
import { Loader2, Plus } from "lucide-react"
import { toast } from "sonner"

import { CopyButton, msg } from "@/components/shared"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { inviteLink, invites, type Invites } from "@/lib/api"
import { useI18n } from "@/lib/i18n"

/** The invite codes you made: copy a link, revoke one (signs its holder out), make another (5 active; admins unlimited). */
export function InviteDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { t, date } = useI18n()
  const [data, setData] = useState<Invites | undefined>(undefined) // undefined = loading
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (open) invites.list().then(setData).catch((e) => toast.error(msg(e)))
  }, [open])

  async function run(f: () => Promise<Invites>) {
    setBusy(true)
    try { setData(await f()) } catch (e) { toast.error(msg(e)) } finally { setBusy(false) }
  }
  const revoke = (code: string) => window.confirm(t("invite.confirm")) &&
    run(async () => { const r = await invites.revoke(code); toast.success(t("invite.revoked")); return r })

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("invite.title")}</DialogTitle>
          <DialogDescription>{t("invite.lede")}</DialogDescription>
        </DialogHeader>
        {!data ? (
          <div className="grid h-24 place-items-center"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>
        ) : data.invites.length ? (
          <ul className="grid max-h-80 gap-2 overflow-y-auto">
            {data.invites.map((i) => (
              <li key={i.code} className="flex flex-wrap items-center gap-2 rounded-lg border px-3 py-2">
                <div className="grid min-w-0 flex-1">
                  {/* the start of the code only: enough to tell invites apart, never the whole credential on screen */}
                  <span dir="ltr" className="truncate text-start font-mono text-sm">{i.code.slice(0, 6)}…</span>
                  <span className="text-xs text-muted-foreground">{t("invite.created", { date: date(i.created) })}</span>
                </div>
                <CopyButton text={inviteLink(i.code)} label={t("invite.copy")} />
                <Button variant="ghost" size="sm" className="text-destructive" disabled={busy} onClick={() => revoke(i.code)}>{t("invite.revoke")}</Button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-muted-foreground">{t("invite.none")}</p>
        )}
        <DialogFooter className="items-center gap-2 sm:justify-between">
          <span className="text-sm text-muted-foreground tabular">{data?.limit != null && t("invite.left", { left: data.left ?? 0, limit: data.limit })}</span>
          <Button onClick={() => run(invites.create)} disabled={busy || !data || data.left === 0}>
            {busy ? <Loader2 data-icon="inline-start" className="animate-spin" /> : <Plus data-icon="inline-start" />} {t("invite.create")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
