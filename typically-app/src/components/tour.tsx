import { useCallback, useEffect, useRef } from "react"
import type { DriveStep } from "driver.js"

import { buttonVariants } from "@/components/ui/button"
import { useAuth } from "@/lib/auth"
import { useI18n, type Key } from "@/lib/i18n"
import { useLibrary } from "@/lib/library"
import { go, parse } from "@/lib/router"
import { cn } from "@/lib/utils"

/** The visible element marked data-tour="name" (the sidebar's invite has a collapsed and an expanded twin; phones keep the sidebar in a closed sheet). */
const visible = (name: string) =>
  [...document.querySelectorAll<HTMLElement>(`[data-tour="${name}"]`)].find((el) => el.getClientRects().length > 0)

/** Resolves once the chat composer is on screen and no menu or dialog (the account menu, ⌘K) is still open or animating out; false after 5 s. */
function ready(): Promise<boolean> {
  const t0 = performance.now()
  return new Promise((done) => {
    const check = () => {
      if (visible("composer") && !document.querySelector('[role="dialog"], [role="menu"]')) done(true)
      else if (performance.now() - t0 > 5000) done(false)
      else requestAnimationFrame(check)
    }
    requestAnimationFrame(check)
  })
}

const X = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>'

/**
 * The first-run product tour (driver.js). Starts by itself once per user when the library has loaded (so after sign-in, never on
 * the server-rendered /s/* share pages); returns `start` for the replay entries (account menu, ⌘K). Every exit marks it done.
 */
export function useTour() {
  const { t, lang } = useI18n()
  const { user } = useAuth()
  const { lib } = useLibrary()
  const key = `df.tour.v1.${user?.id ?? "local"}`   // ponytail: per browser; a server-side flag if it matters
  const running = useRef(false)

  const start = useCallback(async () => {
    if (running.current) return
    running.current = true
    if (parse(window.location.hash).name !== "chat") go({ name: "chat" })
    const [{ driver }, ok] = await Promise.all([import("driver.js"), ready()])   // its own chunk: returning users never download it
    if (!ok) { running.current = false; return }
    const back = document.activeElement as HTMLElement | null   // after the menu that started it has handed focus back
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches
    const rtl = lang === "he"
    const dark = document.documentElement.classList.contains("dark")   // graphite under a 55% veil barely separates the highlight
    const aside = rtl ? "left" : "right"   // the sidebar sits inline-start
    const defs: [string, string?, DriveStep["popover"]?][] = [
      ["welcome"],
      ["demo", "demo", { side: "bottom" }],
      ["composer", "composer", { side: "top" }],
      ["new", "new-model", { side: aside }],
      ["models", "models", { side: aside }],
      ...(user ? [[user.admin ? "inviteAdmin" : "invite", "invite", { side: aside, align: "end" }] as [string, string, DriveStep["popover"]]] : []),
    ]
    // steps whose element isn't on screen are dropped up front, so "Step 2 of 3" counts what is shown (phones: no sidebar steps)
    const steps: DriveStep[] = defs.flatMap(([k, name, pop]) => {
      const element = name ? visible(name) : undefined
      return name && !element ? [] : [{ element, popover: { title: t(`tour.${k}.title` as Key), description: t(`tour.${k}.body` as Key), align: "start", ...pop } }]
    })
    const watch = new ResizeObserver(() => tour.refresh())   // re-measure when the page reflows under the highlight (examples loading, etc.)
    const tour = driver({
      steps, animate: !still, smoothScroll: !still, overlayColor: "#1c1a18", overlayOpacity: dark ? 0.7 : 0.55, stagePadding: 6, stageRadius: 12, popoverOffset: 6,
      showProgress: true, progressText: t("tour.progress"), nextBtnText: t("tour.next"), prevBtnText: t("tour.back"), doneBtnText: t("tour.done"),
      closeBtnLabel: t("common.close"),
      popoverClass: "w-[min(22rem,calc(100vw-2rem))] max-w-none min-w-0 rounded-xl border bg-card p-5 font-sans text-card-foreground shadow-lg",
      onPopoverRender: (p) => {
        p.wrapper.dir = rtl ? "rtl" : "ltr"
        p.arrow.style.visibility = "hidden"   // kept in the layout: driver.js measures it to keep the card 10px off the screen edges
        p.progress.className = "eyebrow"
        p.wrapper.insertBefore(p.progress, p.title)
        p.title.className = "mt-2 pe-6 text-base font-semibold tracking-tight"
        p.description.className = "mt-1 text-sm text-muted-foreground"
        p.footer.className = "mt-5 flex items-center gap-2"
        p.footerButtons.className = "ms-auto flex gap-2"
        p.nextButton.className = cn("driver-popover-next-btn", buttonVariants({ size: "sm" }), "px-3")
        p.previousButton.className = cn("driver-popover-prev-btn", buttonVariants({ variant: "ghost", size: "sm" }))
        if (p.previousButton.disabled) p.previousButton.remove()
        p.closeButton.className = cn("driver-popover-close-btn", buttonVariants({ variant: "ghost", size: "icon-xs" }), "absolute start-auto end-3 top-3 text-muted-foreground")
        p.closeButton.innerHTML = X
        p.wrapper.append(p.closeButton)   // tab order: the buttons first, the ✕ last
        const skip = Object.assign(document.createElement("button"), { type: "button", textContent: t("tour.skip"), onclick: () => tour.destroy() })
        skip.className = cn(buttonVariants({ variant: "ghost", size: "sm" }), "-ms-2.5 text-muted-foreground")
        p.footer.prepend(skip)
        queueMicrotask(() => p.nextButton.focus())   // driver.js focuses the first button it finds; Next is the one to land on
      },
      onHighlighted: (el) => {
        watch.disconnect()
        if (el) [el, el.parentElement].forEach((x) => x && watch.observe(x))
      },
      onDestroyed: () => {
        watch.disconnect()
        localStorage.setItem(key, "1")
        running.current = false
        back?.focus()
      },
    })
    tour.drive()
  }, [t, lang, user, key])

  useEffect(() => { if (lib && !localStorage.getItem(key)) void start() }, [lib, key, start])
  return start
}
