import { useEffect } from "react"
import { Boxes, Database, Languages, MessageSquare, Moon, Plus, SlidersHorizontal, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import { StatusDot } from "@/components/app-sidebar"
import { Command, CommandDialog, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList, CommandSeparator } from "@/components/ui/command"
import { useI18n } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import { go, type Route } from "@/lib/router"

export const SHORTCUT = /Mac|iPhone|iPad/.test(navigator.platform) ? "⌘K" : "Ctrl K"

/** ⌘K / Ctrl+K: jump to a page or a model, or run an action. Open state lives in the shell (the sidebar's Search row opens it too). */
export function CommandPalette({ open, setOpen }: { open: boolean; setOpen: (o: boolean | ((o: boolean) => boolean)) => void }) {
  const { t, lang, setLang } = useI18n()
  const { lib } = useLibrary()
  const { resolvedTheme, setTheme } = useTheme()
  const dark = resolvedTheme === "dark"

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault()
        setOpen((o) => !o)
      }
    }
    window.addEventListener("keydown", on)
    return () => window.removeEventListener("keydown", on)
  }, [setOpen])

  const run = (f: () => void) => () => (setOpen(false), f())
  const nav = (r: Route) => run(() => go(r))
  const pages = [
    { label: t("app.chat"), icon: MessageSquare, route: { name: "chat" } },
    { label: t("app.customize"), icon: SlidersHorizontal, route: { name: "new" } },
    { label: t("app.models"), icon: Boxes, route: { name: "models" } },
    { label: t("app.data"), icon: Database, route: { name: "data" } },
  ] as const

  return (
    <CommandDialog open={open} onOpenChange={setOpen} title={t("cmd.title")} description={t("cmd.description")}>
      <Command>
      <CommandInput placeholder={t("cmd.placeholder")} />
      <CommandList className="max-h-96" label={t("cmd.description")}>
        <CommandEmpty>{t("cmd.empty")}</CommandEmpty>
        <CommandGroup heading={t("cmd.go")}>
          {pages.map((p) => (
            <CommandItem key={p.route.name} value={`page ${p.label}`} onSelect={nav(p.route)}><p.icon /> {p.label}</CommandItem>
          ))}
        </CommandGroup>
        {!!lib?.custom.length && (
          <>
            <CommandSeparator />
            <CommandGroup heading={t("app.yourModels")}>
              {lib.custom.flatMap((m) => [
                <CommandItem key={`open-${m.id}`} value={`open ${m.id} ${m.name}`} keywords={[m.name]} onSelect={nav({ name: "model", id: m.id })}>
                  <StatusDot status={m.status} className="mx-1" /> <span dir="auto" className="truncate">{t("cmd.open", { name: m.name })}</span>
                </CommandItem>,
                m.status === "ready" && (
                  <CommandItem key={`chat-${m.id}`} value={`chat ${m.id} ${m.name}`} keywords={[m.name]} onSelect={nav({ name: "chat", model: m.id })}>
                    <MessageSquare /> <span dir="auto" className="truncate">{t("cmd.chatWith", { name: m.name })}</span>
                  </CommandItem>
                ),
              ])}
            </CommandGroup>
          </>
        )}
        {!!lib?.base.length && (
          <CommandGroup heading={t("app.base")}>
            {lib.base.map((m) => (
              <CommandItem key={m.id} value={`chat ${m.id}`} onSelect={nav({ name: "chat", model: m.id })}>
                <MessageSquare /> <span className="truncate">{t("cmd.chatWith", { name: m.name })}</span>
              </CommandItem>
            ))}
          </CommandGroup>
        )}
        <CommandSeparator />
        <CommandGroup heading={t("cmd.actions")}>
          <CommandItem value="action new model" keywords={[t("app.new")]} onSelect={nav({ name: "new" })}><Plus /> {t("app.new")}</CommandItem>
          <CommandItem value="action language" keywords={["language", "שפה", "hebrew", "english", t("nav.langLabel")]} onSelect={run(() => setLang(lang === "en" ? "he" : "en"))}>
            <Languages /> {t("nav.langLabel")}
          </CommandItem>
          <CommandItem value="action theme" keywords={["theme", "dark", "light", "ערכת נושא", t(dark ? "nav.theme.light" : "nav.theme.dark")]} onSelect={run(() => setTheme(dark ? "light" : "dark"))}>
            {dark ? <Sun /> : <Moon />} {t(dark ? "nav.theme.light" : "nav.theme.dark")}
          </CommandItem>
        </CommandGroup>
      </CommandList>
      </Command>
    </CommandDialog>
  )
}
