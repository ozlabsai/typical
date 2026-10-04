import { useEffect, useState } from "react"

// ponytail: hash routing (#/chat, #/models/northwind, #/new, #/data) — five routes don't need a router dependency;
// the server's SPA fallback keeps working because the hash never reaches it.
export type Route =
  | { name: "chat"; model?: string }
  | { name: "models" }
  | { name: "model"; id: string; tab?: string }
  | { name: "new"; data?: string } // data: start from a dataset already uploaded (its records_token, "sample" or "sample-enron")
  | { name: "data" }

export function parse(hash: string): Route {
  const [path, query = ""] = hash.replace(/^#\/?/, "").split("?")
  const [a, b, c] = path.split("/").map(decodeURIComponent)
  const q = new URLSearchParams(query)
  if (a === "models" && b) return { name: "model", id: b, tab: c }
  if (a === "models") return { name: "models" }
  if (a === "new") return { name: "new", data: q.get("data") ?? undefined }
  if (a === "data") return { name: "data" }
  return { name: "chat", model: q.get("model") ?? undefined }
}

export function href(r: Route): string {
  if (r.name === "model") return `#/models/${encodeURIComponent(r.id)}${r.tab ? `/${r.tab}` : ""}`
  if (r.name === "models") return "#/models"
  if (r.name === "new") return `#/new${r.data ? `?data=${encodeURIComponent(r.data)}` : ""}`
  if (r.name === "data") return "#/data"
  return `#/chat${r.model ? `?model=${encodeURIComponent(r.model)}` : ""}`
}

export const go = (r: Route) => (window.location.hash = href(r))

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash))
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash))
    window.addEventListener("hashchange", on)
    return () => window.removeEventListener("hashchange", on)
  }, [])
  return route
}
