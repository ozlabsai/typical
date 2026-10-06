import { useEffect, useState } from "react"
import { ExternalLink, Loader2, Share2 } from "lucide-react"
import { toast } from "sonner"

import { CopyButton, msg } from "@/components/shared"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { shares, type Share } from "@/lib/api"
import { useI18n } from "@/lib/i18n"

/** Opt-in public results card for one model: create the link, copy it, preview the image, turn it off. */
export function ShareButton({ id, disabled }: { id: string; disabled?: boolean }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [share, setShare] = useState<Share | null | undefined>(undefined) // undefined = loading
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (open) shares.get(id).then(setShare).catch((e) => { setShare(null); toast.error(msg(e)) })
  }, [open, id])

  async function run(f: () => Promise<void>) {
    setBusy(true)
    try { await f() } catch (e) { toast.error(msg(e)) } finally { setBusy(false) }
  }
  const create = () => run(async () => setShare(await shares.create(id)))
  const revoke = () => run(async () => { await shares.revoke(id); setShare(null); toast.success(t("share.revoked")) })

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" disabled={disabled}><Share2 data-icon="inline-start" /> {t("share.button")}</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("share.title")}</DialogTitle>
          <DialogDescription>{t("share.lede")}</DialogDescription>
        </DialogHeader>
        {share === undefined ? (
          <div className="grid h-24 place-items-center"><Loader2 className="size-5 animate-spin text-muted-foreground" /></div>
        ) : share ? (
          <div className="grid gap-4">
            <div className="flex items-center gap-2">
              <Input readOnly value={share.url} dir="ltr" className="font-mono text-xs" onFocus={(e) => e.currentTarget.select()} aria-label={t("share.link")} />
              <CopyButton text={share.url} />
            </div>
            <a href={share.url} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-lg border">
              <img src={share.image} alt={t("share.preview")} width={1200} height={630} className="aspect-[1200/630] w-full bg-muted" />
            </a>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">{t("share.private")}</p>
        )}
        <DialogFooter>
          {share ? <>
            <Button variant="destructive" onClick={revoke} disabled={busy}>{t("share.revoke")}</Button>
            <Button asChild><a href={share.url} target="_blank" rel="noreferrer"><ExternalLink data-icon="inline-start" /> {t("share.open")}</a></Button>
          </> : (
            <Button onClick={create} disabled={busy || share === undefined}>
              {busy && <Loader2 data-icon="inline-start" className="animate-spin" />} {t("share.create")}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
