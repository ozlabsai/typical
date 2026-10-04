import { useState } from "react"
import { ThemeProvider } from "next-themes"

import { AppSidebar } from "@/components/app-sidebar"
import { CommandPalette } from "@/components/command-palette"
import { SignIn } from "@/components/sign-in"
import { HeaderSlotProvider } from "@/components/layout"
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { AuthGate } from "@/lib/auth"
import { I18nProvider, useI18n } from "@/lib/i18n"
import { LibraryProvider, useLibrary } from "@/lib/library"
import { href, useRoute } from "@/lib/router"
import { ChatPage } from "@/pages/chat"
import { CustomizePage } from "@/pages/customize"
import { DataPage } from "@/pages/data"
import { ModelPage, ModelsPage } from "@/pages/models"

function Shell() {
  const { t, lang } = useI18n()
  const { find } = useLibrary()
  const route = useRoute()
  const [search, setSearch] = useState(false)
  const crumbs: { label: string; to?: string }[] =
    route.name === "chat" ? [{ label: t("app.chat") }]
    : route.name === "new" ? [{ label: t("app.new") }]
    : route.name === "models" ? [{ label: t("app.models") }]
    : route.name === "data" ? [{ label: t("app.data") }]
    : [{ label: t("app.models"), to: href({ name: "models" }) }, { label: find(route.id)?.name ?? route.id }]

  return (
    <HeaderSlotProvider>
      {(setSlot) => (
        <SidebarProvider className="h-svh overflow-hidden">
          <AppSidebar route={route} onSearch={() => setSearch(true)} />
          <CommandPalette open={search} setOpen={setSearch} />
          {/* the window never scrolls; one scroll container with a reserved gutter, so width never jumps between routes */}
          <SidebarInset className="h-svh min-w-0 overflow-hidden">
            <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background px-4">
              <SidebarTrigger className="-ms-1" />
              <Separator orientation="vertical" className="me-1 data-[orientation=vertical]:h-4" />
              <Breadcrumb className="min-w-0">
                <BreadcrumbList className="flex-nowrap">
                  {crumbs.map((c, i) => (
                    <span key={i} className="contents">
                      {i > 0 && <BreadcrumbSeparator className="rtl:rotate-180" />}
                      <BreadcrumbItem className="min-w-0">{c.to ? <BreadcrumbLink href={c.to}>{c.label}</BreadcrumbLink> : <BreadcrumbPage dir="auto" className="truncate">{c.label}</BreadcrumbPage>}</BreadcrumbItem>
                    </span>
                  ))}
                </BreadcrumbList>
              </Breadcrumb>
              <div ref={setSlot} className="ms-auto flex min-w-0 items-center gap-2" />
            </header>
            <div id="scroller" className="min-h-0 flex-1 overflow-y-auto [scrollbar-gutter:stable]">
              {route.name === "chat" && <ChatPage modelId={route.model} />}
              {route.name === "models" && <ModelsPage />}
              {route.name === "model" && <ModelPage id={route.id} tab={route.tab} />}
              {route.name === "new" && <CustomizePage key={route.data} data={route.data} />}
              {route.name === "data" && <DataPage />}
            </div>
          </SidebarInset>
          <Toaster position={lang === "he" ? "bottom-left" : "bottom-right"} containerAriaLabel={t("common.notifications")} />
        </SidebarProvider>
      )}
    </HeaderSlotProvider>
  )
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
      <I18nProvider>
        <AuthGate signIn={(done) => <SignIn done={done} />}>
          <LibraryProvider>
            <TooltipProvider delayDuration={300}>
              <Shell />
            </TooltipProvider>
          </LibraryProvider>
        </AuthGate>
      </I18nProvider>
    </ThemeProvider>
  )
}
