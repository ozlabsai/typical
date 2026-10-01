import { createContext, useCallback, useContext, useEffect, useState } from "react"

import { session, SIGNED_OUT, type User } from "@/lib/api"

/** null user = the server runs without sign-in (local dev). */
const Ctx = createContext<{ user: User | null; signOut: () => Promise<void> }>(null!)

/** Renders `signIn` until there is a session (or the server has sign-in off), then the app. */
export function AuthGate({ children, signIn }: { children: React.ReactNode; signIn: (done: (u: User) => void) => React.ReactNode }) {
  const [state, setState] = useState<"loading" | "out" | { user: User | null }>("loading")
  useEffect(() => {
    session.me().then((m) => setState(m ? { user: m.user } : "out")).catch(() => setState({ user: null }))
    const out = () => setState("out")
    window.addEventListener(SIGNED_OUT, out)
    return () => window.removeEventListener(SIGNED_OUT, out)
  }, [])
  const signOut = useCallback(async () => { await session.logout().catch(() => {}); setState("out") }, [])
  if (state === "loading") return null
  if (state === "out") return <>{signIn((user) => setState({ user }))}</>
  return <Ctx.Provider value={{ user: state.user, signOut }}>{children}</Ctx.Provider>
}

export const useAuth = () => useContext(Ctx)
