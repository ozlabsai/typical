import { useEffect, useState } from "react"
import { AlertCircle, Languages, Loader2 } from "lucide-react"

import { BrandMark } from "@/components/layout"
import { msg } from "@/components/shared"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { session, type User } from "@/lib/api"
import { useI18n } from "@/lib/i18n"

/** Invite-code sign-in (TYPICALLY_AUTH=1). */
export function SignIn({ done }: { done: (u: User) => void }) {
  const { t, lang, setLang } = useI18n()
  // an invite link (#/?invite=<code>&for=<name>) prefills the code and name; the code leaves the address bar once read
  const [link] = useState(() => new URLSearchParams(window.location.hash.split("?")[1]))
  const [code, setCode] = useState(link.get("invite") ?? "")
  useEffect(() => {
    if (window.location.hash.includes("invite=")) history.replaceState(null, "", `${window.location.pathname}${window.location.search}#/`)
  }, [])
  const [name, setName] = useState((link.get("for") ?? "").slice(0, 60))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const me = await session.login(code.trim(), name.trim())
      done(me.user!)
    } catch (err) {
      setError(msg(err).includes("not valid") ? t("auth.invalid") : msg(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="grid min-h-svh place-items-center bg-background px-4 py-8">
      <div className="absolute end-4 top-4">
        <Button variant="ghost" size="sm" onClick={() => setLang(lang === "en" ? "he" : "en")} aria-label={t("nav.langLabel")}>
          <Languages data-icon="inline-start" /> {t("nav.lang")}
        </Button>
      </div>
      <div className="grid w-full max-w-sm gap-6">
        <div className="flex items-center justify-center gap-2">
          <BrandMark />
          <span className="text-lg font-semibold tracking-tight">{t("brand.name")}</span>
        </div>
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">{t("auth.title")}</CardTitle>
            <CardDescription>{t("auth.lede")}</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={submit} className="grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="invite">{t("auth.code")}</Label>
                <Input id="invite" value={code} onChange={(e) => setCode(e.target.value)} required autoFocus autoComplete="off"
                  spellCheck={false} dir="ltr" className="font-mono" aria-invalid={Boolean(error)} />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="name">{t("auth.name")} <span className="font-normal text-muted-foreground">{t("auth.optional")}</span></Label>
                <Input id="name" value={name} onChange={(e) => setName(e.target.value)} maxLength={60} autoComplete="name" dir="auto"
                  placeholder={t("auth.namePlaceholder")} />
              </div>
              {error && <Alert variant="destructive"><AlertCircle /><AlertDescription>{error}</AlertDescription></Alert>}
              <Button type="submit" disabled={busy || !code.trim()}>
                {busy && <Loader2 data-icon="inline-start" className="animate-spin" />} {t("auth.submit")}
              </Button>
            </form>
          </CardContent>
        </Card>
        <p className="text-center text-xs text-muted-foreground">{t("auth.noCode")}</p>
      </div>
    </main>
  )
}
