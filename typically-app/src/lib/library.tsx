import { createContext, useCallback, useContext, useEffect, useState } from "react"

import { library, type Library, type LibraryModel } from "@/lib/api"

/** What Results / Deploy need to know about a trained model. */
export interface ModelRef { id: string; name: string; run: string; sample: boolean; resultsKey: string }
export const refOf = (m: LibraryModel): ModelRef => ({ id: m.id, name: m.name, run: m.run ?? `co_${m.id}`, sample: Boolean(m.sample), resultsKey: m.sample ? "northwind" : m.id })

const Ctx = createContext<{ lib: Library | null; refresh: () => Promise<void>; find: (id?: string) => LibraryModel | undefined }>(null!)

export function LibraryProvider({ children }: { children: React.ReactNode }) {
  const [lib, setLib] = useState<Library | null>(null)
  const refresh = useCallback(() => library.list().then(setLib).catch(() => {}), [])
  useEffect(() => { void refresh() }, [refresh])
  // poll while anything is training, so the sidebar dots and progress stay live
  const busy = lib?.custom.some((m) => m.status === "training" || m.status === "queued")
  useEffect(() => {
    if (!busy) return
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [busy, refresh])
  const find = (id?: string) => (id ? [...(lib?.custom ?? []), ...(lib?.base ?? [])].find((m) => m.id === id) : undefined)
  return <Ctx.Provider value={{ lib, refresh, find }}>{children}</Ctx.Provider>
}

export const useLibrary = () => useContext(Ctx)
