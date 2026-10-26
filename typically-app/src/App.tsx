import { Component, useState } from "react"
import { ThemeProvider } from "next-themes"

import { AppSidebar } from "@/components/app-sidebar"
import { CommandPalette } from "@/components/command-palette"
import { SignIn } from "@/components/sign-in"
import { useTour } from "@/components/tour"
import { HeaderSlotProvider, Page } from "@/components/layout"
import { ErrorState } from "@/components/shared"
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

/** A page that throws shows this instead of a blank screen; it is keyed on the route, so navigating away recovers. */
class PageBoundary extends Component<{ children: React.ReactNode; fallback: (error: Error) => React.ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  render() { return this.state.error ? this.props.fallback(this.state.error) : this.props.children }
}

function Shell() {
  const { t, lang } = useI18n()
  const { lib, error, refresh, find } = useLibrary()
  const route = useRoute()
  const [search, setSearch] = useState(false)
  const tour = useTour()
  const crumbs: { label: string; to?: string }[] =
    route.name === "chat" ? [{ label: t("app.chat") }]
    : route.name === "new" ? [{ label: t("app.new") }]
    : route.name === "models" ? [{ label: t("app.models") }]
    : route.name === "data" ? [{ label: t("app.data") }]
    : [{ label: t("app.models"), to: href({ name: "models" }) }, { label: find(route.id)?.name ?? (lib ? t("models.notFound") : "") }]

  return (
    <HeaderSlotProvider>
      {(setSlot) => (
        <SidebarProvider className="h-svh overflow-hidden">
          <a href="#" onClick={(e) => { e.preventDefault(); document.getElementById("scroller")?.focus() }}
            className="sr-only z-50 rounded-md bg-background px-3 py-2 text-sm font-medium shadow-md ring-3 ring-ring/50 focus:not-sr-only focus:absolute focus:start-2 focus:top-2">
            {t("app.skip")}
          </a>
          <AppSidebar route={route} onSearch={() => setSearch(true)} onTour={tour} />
          <CommandPalette open={search} setOpen={setSearch} onTour={tour} />
          {/* the window never scrolls; one scroll container with a reserved gutter, so width never jumps between routes */}
          <SidebarInset className="h-svh min-w-0 overflow-hidden">
            <header className="flex h-14 shrink-0 items-center gap-2 border-b bg-background px-4">
              <SidebarTrigger className="-ms-1" />
              <Separator orientation="vertical" className="me-1 data-vertical:h-4 data-vertical:self-center" />
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
            <div id="scroller" tabIndex={-1} className="min-h-0 flex-1 overflow-y-auto outline-none [scrollbar-gutter:stable]">
              {!lib && error ? <Page><ErrorState title={t("app.loadFail")} detail={error} retry={() => void refresh()} /></Page> : (
                <PageBoundary key={route.name === "model" ? `model:${route.id}` : route.name} fallback={(e) => <Page><ErrorState title={t("app.crashed")} detail={e.message} retry={() => window.location.reload()} /></Page>}>
                  {route.name === "chat" && <ChatPage modelId={route.model} />}
                  {route.name === "models" && <ModelsPage />}
                  {route.name === "model" && <ModelPage id={route.id} tab={route.tab} />}
                  {route.name === "new" && <CustomizePage key={route.data} data={route.data} />}
                  {route.name === "data" && <DataPage />}
                </PageBoundary>
              )}
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
