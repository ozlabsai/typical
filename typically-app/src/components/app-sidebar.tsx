import { useEffect, useState } from "react"
import { Boxes, Database, Languages, LogOut, MessageSquare, Moon, Plus, Search, Sun, UserPlus, UserRound } from "lucide-react"
import { useTheme } from "next-themes"

import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupAction, SidebarGroupContent, SidebarGroupLabel,
  SidebarHeader, SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarMenuSkeleton, useSidebar,
} from "@/components/ui/sidebar"
import { SHORTCUT } from "@/components/command-palette"
import { InviteDialog } from "@/components/invite-dialog"
import { Badge } from "@/components/ui/badge"
import type { LibraryModel } from "@/lib/api"
import { useAuth } from "@/lib/auth"
import { useI18n } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import { href, type Route } from "@/lib/router"
import { cn } from "@/lib/utils"

export function StatusDot({ status, className }: { status: LibraryModel["status"]; className?: string }) {
  return (
    <span aria-hidden className={cn("relative inline-flex size-2 shrink-0 rounded-full",
      status === "ready" ? "bg-success" : status === "failed" ? "bg-destructive" : "bg-warning", className)}>
      {status === "training" && <span className="absolute inset-0 animate-ping rounded-full bg-warning/60" />}
    </span>
  )
}

export function AppSidebar({ route, onSearch }: { route: Route; onSearch: () => void }) {
  const { t, lang, setLang } = useI18n()
  const { lib, error } = useLibrary()
  const { user, signOut } = useAuth()
  const { resolvedTheme, setTheme } = useTheme()
  const dark = resolvedTheme === "dark"
  const activeId = route.name === "model" ? route.id : undefined
  const { setOpenMobile } = useSidebar()
  const [inviting, setInviting] = useState(false)
  const at = href(route)
  useEffect(() => setOpenMobile(false), [at, setOpenMobile]) // phones: a tapped link must not leave the sheet over the page

  return (
    <>
    {/* outside the Sidebar: on phones it is a sheet, and closing the sheet would unmount the dialog with it */}
    <InviteDialog open={inviting} onOpenChange={setInviting} />
    <Sidebar side="left" collapsible="icon">  {/* "left" = inline-start: sidebar.tsx uses logical start/end, so this mirrors in RTL */}
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href={href({ name: "chat" })}>
                <span className="grid size-8 place-items-center rounded-md bg-primary text-primary-foreground">
                  <span className="size-2.5 rounded-full bg-primary-foreground" />
                </span>
                <span className="grid leading-tight group-data-[collapsible=icon]:hidden">
                  <span className="font-semibold">{t("brand.name")}</span>
                  <span className="text-xs text-muted-foreground">{t("brand.credit")}</span>
                </span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <SidebarMenuButton onClick={() => (setOpenMobile(false), onSearch())} tooltip={`${t("app.search")} (${SHORTCUT})`} className="text-muted-foreground">
              <Search /> <span>{t("app.search")}</span>
              <kbd dir="ltr" className="ms-auto rounded-md border bg-muted px-1.5 font-mono text-xs group-data-[collapsible=icon]:hidden">{SHORTCUT}</kbd>
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
                <SidebarMenuButton asChild isActive={route.name === "new"} tooltip={t("app.new")}>
                  <a href={href({ name: "new" })}><Plus /> <span>{t("app.new")}</span></a>
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
              {!lib && !error && [0, 1, 2].map((i) => <SidebarMenuItem key={i}><SidebarMenuSkeleton /></SidebarMenuItem>)}
              {lib?.custom.map((m) => (
                <SidebarMenuItem key={m.id}>
                  <SidebarMenuButton asChild isActive={activeId === m.id}>
                    <a href={href({ name: "model", id: m.id })}>
                      <StatusDot status={m.status} />
                      {/* a version's name ends in " vN": the badge carries it, so the list reads "Northwind triage [v2]" */}
                      <span dir="auto" className="truncate">{(m.version ?? 1) > 1 ? m.name.replace(new RegExp(` v${m.version}$`), "") : m.name}</span>
                      {(m.version ?? 1) > 1 && <Badge variant="outline" className="font-normal text-muted-foreground">v{m.version}</Badge>}
                    </a>
                  </SidebarMenuButton>
                  {m.status === "training" && typeof m.progress === "number"
                    ? <SidebarMenuBadge className="tabular">{Math.round(m.progress * 100)}%</SidebarMenuBadge>
                    : m.sample ? <SidebarMenuBadge><Badge variant="secondary">{t("app.sample")}</Badge></SidebarMenuBadge> : null}
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
          {user && <>
            <SidebarMenuItem>
              <div className="flex h-8 min-w-0 items-center gap-2 px-2 text-sm group-data-[collapsible=icon]:hidden" title={t("auth.signedInAs", { name: user.name })}>
                <UserRound className="size-4 shrink-0 text-muted-foreground" />
                <span dir="auto" className="truncate font-medium">{user.name}</span>
                {user.admin && <Badge variant="outline" className="font-normal text-muted-foreground">{t("auth.admin")}</Badge>}
              </div>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip={t("invite.button")} onClick={() => (setOpenMobile(false), setInviting(true))}>
                <UserPlus /> <span>{t("invite.button")}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
            <SidebarMenuItem>
              <SidebarMenuButton tooltip={t("auth.signOut")} onClick={() => void signOut()}>
                <LogOut className="rtl:-scale-x-100" /> <span>{t("auth.signOut")}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          </>}
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
    </>
  )
}
