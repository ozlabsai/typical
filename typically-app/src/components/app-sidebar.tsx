import { Boxes, Database, Languages, MessageSquare, Moon, Plus, Search, SlidersHorizontal, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupAction, SidebarGroupContent, SidebarGroupLabel,
  SidebarHeader, SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarMenuSkeleton,
} from "@/components/ui/sidebar"
import { SHORTCUT } from "@/components/command-palette"
import type { LibraryModel } from "@/lib/api"
import { useI18n } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import { href, type Route } from "@/lib/router"
import { cn } from "@/lib/utils"

export function StatusDot({ status, className }: { status: LibraryModel["status"]; className?: string }) {
  return (
    <span aria-hidden className={cn("relative inline-flex size-2 shrink-0 rounded-full",
      status === "ready" ? "bg-emerald-500" : status === "failed" ? "bg-destructive" : "bg-amber-500", className)}>
      {(status === "training" || status === "queued") && <span className="absolute inset-0 animate-ping rounded-full bg-amber-500/60" />}
    </span>
  )
}

export function AppSidebar({ route, onSearch }: { route: Route; onSearch: () => void }) {
  const { t, lang, setLang } = useI18n()
  const { lib } = useLibrary()
  const { resolvedTheme, setTheme } = useTheme()
  const dark = resolvedTheme === "dark"
  const activeId = route.name === "model" ? route.id : undefined

  return (
    <Sidebar side="left" collapsible="icon">  {/* "left" = inline-start: sidebar.tsx uses logical start/end, so this mirrors in RTL */}
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href={href({ name: "chat" })}>
                <span className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground">
                  <span className="size-2.5 rounded-full bg-primary-foreground" />
                </span>
                <span className="grid leading-tight">
                  <span className="font-semibold">typically</span>
                  <span className="text-xs text-muted-foreground">by Typical</span>
                </span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={onSearch} tooltip={`${t("app.search")} (${SHORTCUT})`} className="text-muted-foreground">
              <Search /> <span>{t("app.search")}</span>
              <kbd dir="ltr" className="ms-auto rounded border bg-muted px-1.5 font-mono text-[10px] leading-4 group-data-[collapsible=icon]:hidden">{SHORTCUT}</kbd>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={route.name === "chat"} tooltip={t("app.chat")}>
                  <a href={href({ name: "chat" })}><MessageSquare /> <span>{t("app.chat")}</span></a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={route.name === "new"} tooltip={t("app.customize")}>
                  <a href={href({ name: "new" })}><SlidersHorizontal /> <span>{t("app.customize")}</span></a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={route.name === "models"} tooltip={t("app.models")}>
                  <a href={href({ name: "models" })}><Boxes /> <span>{t("app.models")}</span></a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={route.name === "data"} tooltip={t("app.data")}>
                  <a href={href({ name: "data" })}><Database /> <span>{t("app.data")}</span></a>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="group-data-[collapsible=icon]:hidden">
          <SidebarGroupLabel>{t("app.yourModels")}</SidebarGroupLabel>
          <SidebarGroupAction asChild title={t("app.new")}><a href={href({ name: "new" })}><Plus /></a></SidebarGroupAction>
          <SidebarGroupContent>
            <SidebarMenu>
              {!lib && [0, 1, 2].map((i) => <SidebarMenuItem key={i}><SidebarMenuSkeleton /></SidebarMenuItem>)}
              {lib?.custom.map((m) => (
                <SidebarMenuItem key={m.id}>
                  <SidebarMenuButton asChild isActive={activeId === m.id}>
                    <a href={href({ name: "model", id: m.id })}>
                      <StatusDot status={m.status} />
                      <span dir="auto" className="truncate">{m.name}</span>
                    </a>
                  </SidebarMenuButton>
                  {m.status === "training" && typeof m.progress === "number"
                    ? <SidebarMenuBadge className="tabular">{Math.round(m.progress * 100)}%</SidebarMenuBadge>
                    : m.sample ? <SidebarMenuBadge className="text-muted-foreground">{t("app.sample")}</SidebarMenuBadge> : null}
                </SidebarMenuItem>
              ))}
              {lib && !lib.custom.length && <p className="px-2 py-1 text-xs text-muted-foreground">{t("app.noModels")}</p>}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton tooltip={t("nav.langLabel")} onClick={() => setLang(lang === "en" ? "he" : "en")}>
              <Languages /> <span>{t("nav.lang")}</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton tooltip={t(dark ? "nav.theme.light" : "nav.theme.dark")} onClick={() => setTheme(dark ? "light" : "dark")}>
              {dark ? <Sun /> : <Moon />} <span>{t(dark ? "nav.theme.light" : "nav.theme.dark")}</span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
  )
}
