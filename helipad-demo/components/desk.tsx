"use client";
import { Plus } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LangToggle, useLang } from "@/components/lang";
import { arrowNav, StatusDot, stColor } from "@/components/status";
import type { Index } from "@/lib/protocol";
import { cn } from "@/lib/utils";

export function Desk({ index, children }: { index: Index; children: React.ReactNode }) {
  const { d } = useLang();
  const path = usePathname();
  return (
    <div className="grid h-dvh grid-cols-[14.5rem_1fr] grid-rows-[3rem_1fr]">
      <header className="col-span-2 flex items-center gap-4 border-b px-5">
        <h1 className="text-[15px] font-semibold tracking-tight">{d.ui.appTitle}</h1>
        <span className="text-xs text-muted-foreground">
          <bdi>{index.model.name}</bdi> <span className="num">{index.model.version}</span>
        </span>
        <div className="ms-auto">
          <LangToggle />
        </div>
      </header>
      <nav aria-label={d.ui.siteList} className="flex min-h-0 flex-col border-e">
        <Link href="/new" aria-current={path === "/new" ? "page" : undefined}
          className={cn("m-3 flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors duration-150 hover:bg-accent active:scale-[0.98]",
            path === "/new" && "bg-accent")}>
          <Plus className="size-4" aria-hidden /> {d.ui.newInspection}
        </Link>
        <h2 className="px-5 pt-2 pb-1.5 text-xs text-muted-foreground">{d.ui.sites}</h2>
        <ul onKeyDown={arrowNav} className="thin-scroll min-h-0 flex-1 overflow-y-auto px-2 pb-3">
          {index.sites.map((s) => {
            const on = path === `/sites/${s.id}`;
            return (
              <li key={s.id}>
                <Link data-nav href={`/sites/${s.id}`} aria-current={on ? "page" : undefined}
                  className={cn("group flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors duration-150 hover:bg-accent",
                    on && "bg-accent")}>
                  <span className="num w-7 font-medium">{s.id}</span>
                  <span className="num text-xs whitespace-nowrap text-muted-foreground">{s.round}</span>
                  <span className="ms-auto flex items-center gap-1.5 text-xs whitespace-nowrap" style={{ color: stColor(s.verdict) }}>
                    <StatusDot status={s.verdict} />
                    {d.verdictOption[s.verdict]}
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <main className="min-h-0 min-w-0">{children}</main>
    </div>
  );
}
