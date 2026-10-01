import { ThemeProvider } from "next-themes"

import { AppSidebar } from "@/components/app-sidebar"
import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar"
import { Toaster } from "@/components/ui/sonner"
import { TooltipProvider } from "@/components/ui/tooltip"
import { I18nProvider, useI18n } from "@/lib/i18n"
import { LibraryProvider, useLibrary } from "@/lib/library"
import { href, useRoute } from "@/lib/router"
import { ChatPage } from "@/pages/chat"
import { CustomizePage } from "@/pages/customize"
import { ModelPage, ModelsPage } from "@/pages/models"

function Shell() {
  const { t, lang } = useI18n()
  const { find } = useLibrary()
  const route = useRoute()
  const crumbs: { label: string; to?: string }[] =
    route.name === "chat" ? [{ label: t("app.chat") }]
    : route.name === "new" ? [{ label: t("app.new") }]
    : route.name === "models" ? [{ label: t("app.models") }]
    : [{ label: t("app.models"), to: href({ name: "models" }) }, { label: find(route.id)?.name ?? route.id }]

  return (
    <SidebarProvider>
      <AppSidebar route={route} />
      <SidebarInset>
        <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-2 border-b bg-background/95 px-4 backdrop-blur">
          <SidebarTrigger className="-ms-1" />
          <Separator orientation="vertical" className="me-1 data-[orientation=vertical]:h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              {crumbs.map((c, i) => (
                <span key={i} className="contents">
                  {i > 0 && <BreadcrumbSeparator className="rtl:rotate-180" />}
                  <BreadcrumbItem>{c.to ? <BreadcrumbLink href={c.to}>{c.label}</BreadcrumbLink> : <BreadcrumbPage dir="auto">{c.label}</BreadcrumbPage>}</BreadcrumbItem>
                </span>
              ))}
            </BreadcrumbList>
          </Breadcrumb>
        </header>
        {route.name === "chat" && <ChatPage modelId={route.model} />}
        {route.name === "models" && <ModelsPage />}
        {route.name === "model" && <ModelPage id={route.id} tab={route.tab} />}
        {route.name === "new" && <CustomizePage />}
      </SidebarInset>
      <Toaster position={lang === "he" ? "bottom-left" : "bottom-right"} />
    </SidebarProvider>
  )
}

export default function App() {
  return (
    <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false} disableTransitionOnChange>
      <I18nProvider>
        <LibraryProvider>
          <TooltipProvider delayDuration={300}>
            <Shell />
          </TooltipProvider>
        </LibraryProvider>
      </I18nProvider>
    </ThemeProvider>
  )
}
