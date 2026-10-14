import { useEffect, useState } from "react"
import { Boxes, Compass, Database, Ellipsis, Languages, LogOut, MessageSquare, Moon, Plus, Search, Sun, UserPlus } from "lucide-react"
import { useTheme } from "next-themes"

import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupAction, SidebarGroupContent, SidebarGroupLabel,
  SidebarHeader, SidebarMenu, SidebarMenuBadge, SidebarMenuButton, SidebarMenuItem, SidebarMenuSkeleton, useSidebar,
} from "@/components/ui/sidebar"
import { SHORTCUT } from "@/components/command-palette"
import { InviteDialog } from "@/components/invite-dialog"
import { BrandMark } from "@/components/layout"
import { Badge } from "@/components/ui/badge"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { invites, type Invites, type LibraryModel } from "@/lib/api"
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

/** Invites: a mini ticket (the invite dialog's, in miniature: .ticket-mini in index.css), the count on its stub ("2/5", admins "∞").
 *  Quiet paper at rest; on hover / focus its orange fill sweeps in from the stub. */
function InviteButton({ onOpen, refresh }: { onOpen: () => void; refresh: number }) {
  const { t } = useI18n()
  const [data, setData] = useState<Invites | null>(null)
  useEffect(() => { invites.list().then(setData).catch(() => setData(null)) }, [refresh])
  const used = data?.invites.length ?? 0, limit = data?.limit ?? null
  const spelled = !data ? "" : limit === null ? t("invite.unlimited") : t("invite.used", { used, limit })
  const label = spelled ? `${t("invite.button")}, ${spelled}` : undefined
  return (
    <SidebarMenuItem>
      <SidebarMenuButton tooltip={t("invite.button")} aria-label={label} onClick={onOpen} data-tour="invite" className="hidden group-data-[collapsible=icon]:flex">
        <span aria-hidden className="ticket-mini ticket-glyph"><span className="ticket-mini-fill" /></span>
      </SidebarMenuButton>
      <button type="button" onClick={onOpen} aria-label={label} data-tour="invite" className="ticket-btn group-data-[collapsible=icon]:hidden">
        <span className="ticket-mini">
          <span aria-hidden className="ticket-mini-fill" />
          <span className="flex min-w-0 flex-1 items-center gap-2 ps-3 font-mono text-[11px] font-medium tracking-[0.12em] uppercase rtl:font-sans rtl:text-xs rtl:tracking-normal">
            <UserPlus aria-hidden className="size-3.5 shrink-0" /> <span className="truncate">{t("invite.button")}</span>
          </span>
          <span aria-hidden className="ticket-mini-stub font-mono text-xs font-medium tabular">
            {data && (limit === null ? <span className="text-base leading-none">∞</span> : `${used}/${limit}`)}
          </span>
        </span>
      </button>
    </SidebarMenuItem>
  )
}

export function AppSidebar({ route, onSearch, onTour }: { route: Route; onSearch: () => void; onTour: () => void }) {
  const { t, lang, setLang } = useI18n()
  const { lib, error } = useLibrary()
  const { user, signOut } = useAuth()
  const { resolvedTheme, setTheme } = useTheme()
  const dark = resolvedTheme === "dark"
  const activeId = route.name === "model" ? route.id : undefined
  const { setOpenMobile } = useSidebar()
  const [inviting, setInviting] = useState(false)
  const [invitesSeen, setInvitesSeen] = useState(0)   // bumped when the dialog closes: the card recounts
  const at = href(route)
  useEffect(() => setOpenMobile(false), [at, setOpenMobile]) // phones: a tapped link must not leave the sheet over the page

  return (
    <>
    {/* outside the Sidebar: on phones it is a sheet, and closing the sheet would unmount the dialog with it */}
    <InviteDialog open={inviting} onOpenChange={(o) => { setInviting(o); if (!o) setInvitesSeen((n) => n + 1) }} />
    <Sidebar side="left" collapsible="icon">  {/* "left" = inline-start: sidebar.tsx uses logical start/end, so this mirrors in RTL */}
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <a href={href({ name: "chat" })}>
                <BrandMark className="size-8!" />  {/* the menu button sizes its svgs to 16px */}
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
                  <a href={href({ name: "new" })} data-tour="new-model"><Plus /> <span>{t("app.new")}</span></a>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={route.name === "models"} tooltip={t("app.models")}>
                  <a href={href({ name: "models" })} data-tour="models"><Boxes /> <span>{t("app.models")}</span></a>
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
          <SidebarGroupLabel className="eyebrow">{t("app.yourModels")}</SidebarGroupLabel>
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
          {user ? <>
            <InviteButton onOpen={() => (setOpenMobile(false), setInviting(true))} refresh={invitesSeen} />
            <SidebarMenuItem>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <SidebarMenuButton size="lg" tooltip={user.name} aria-label={t("nav.account")}>
                    <span aria-hidden className="grid size-8 shrink-0 place-items-center rounded-full bg-foreground text-sm font-semibold text-background">
                      {Array.from(user.name.trim())[0]?.toUpperCase() ?? "?"}
                    </span>
                    <span className="flex min-w-0 items-center gap-2">
                      <span dir="auto" className="truncate font-medium">{user.name}</span>
                      {user.admin && <Badge variant="outline" className="font-normal text-muted-foreground">{t("auth.admin")}</Badge>}
                    </span>
                    <Ellipsis className="ms-auto text-muted-foreground" />
                  </SidebarMenuButton>
                </DropdownMenuTrigger>
                <DropdownMenuContent side="top" align="start" className="w-(--radix-dropdown-menu-trigger-width) min-w-56">
                  <DropdownMenuLabel className="text-xs font-normal text-muted-foreground" dir="auto">{t("auth.signedInAs", { name: user.name })}</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={() => setLang(lang === "en" ? "he" : "en")}><Languages /> {t("nav.lang")}</DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => setTheme(dark ? "light" : "dark")}>{dark ? <Sun /> : <Moon />} {t(dark ? "nav.theme.light" : "nav.theme.dark")}</DropdownMenuItem>
                  <DropdownMenuItem onSelect={() => (setOpenMobile(false), onTour())}><Compass /> {t("tour.take")}</DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem onSelect={() => void signOut()}><LogOut className="rtl:-scale-x-100" /> {t("auth.signOut")}</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </SidebarMenuItem>
          </> : (
            <SidebarMenuItem className="flex gap-1 group-data-[collapsible=icon]:flex-col">  {/* sign-in off (local): no account, two quiet controls */}
              <SidebarMenuButton tooltip={t("nav.langLabel")} onClick={() => setLang(lang === "en" ? "he" : "en")} className="flex-1">
                <Languages /> <span>{t("nav.lang")}</span>
              </SidebarMenuButton>
              <SidebarMenuButton tooltip={t(dark ? "nav.theme.light" : "nav.theme.dark")} aria-label={t(dark ? "nav.theme.light" : "nav.theme.dark")} onClick={() => setTheme(dark ? "light" : "dark")} className="w-8 shrink-0 justify-center">
                {dark ? <Sun /> : <Moon />}
              </SidebarMenuButton>
            </SidebarMenuItem>
          )}
        </SidebarMenu>
      </SidebarFooter>
    </Sidebar>
    </>
  )
}
