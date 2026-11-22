// Reads the recorded results of DATA.md from HELIPAD_DEMO_DATA (default: fixtures/). Server only.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { ITEMS, VERDICTS, type Index, type Site } from "./protocol";

export function dataDir(): string {
  return path.resolve(/*turbopackIgnore: true*/ process.env.HELIPAD_DEMO_DATA || path.join(process.cwd(), "fixtures"));
}

const ID = /^[A-Za-z0-9_-]+$/;

function need(ok: unknown, what: string): asserts ok {
  if (!ok) throw new Error(`helipad data: ${what}`);
}

const isProbs = (p: unknown) =>
  !!p && typeof p === "object" && Object.values(p).every((v) => typeof v === "number" && v >= 0 && v <= 1);

export function parseIndex(raw: unknown): Index {
  const j = raw as Index;
  need(j && typeof j.model?.name === "string" && typeof j.model?.version === "string", "index.model");
  need(Array.isArray(j.sites), "index.sites");
  for (const s of j.sites) {
    need(ID.test(s.id), `index site id ${s.id}`);
    need((VERDICTS as readonly string[]).includes(s.verdict), `index ${s.id} verdict ${s.verdict}`);
  }
  return j;
}

export function parseSite(raw: unknown): Site {
  const s = raw as Site;
  need(s && ID.test(s.id), "site.id");
  const at = (w: string) => `${s.id}: ${w}`;
  need(typeof s.name?.en === "string", at("name"));
  need((VERDICTS as readonly string[]).includes(s.verdict?.status), at("verdict.status"));
  need(isProbs(s.verdict.probs), at("verdict.probs"));
  need(Array.isArray(s.verdict.deciding), at("verdict.deciding"));
  need(Array.isArray(s.items), at("items"));
  for (const it of s.items) {
    need((ITEMS as readonly string[]).includes(it.id), at(`unknown item ${it.id}`));
    need(isProbs(it.probs), at(`${it.id}.probs`));
    need(it.status in it.probs, at(`${it.id}.status not among its options`));
  }
  const n = s.findings?.obstacles?.length;
  need(typeof n === "number", at("findings.obstacles"));
  for (const it of s.items) need(it.findings.every((k) => k >= 0 && k < n), at(`${it.id}.findings index`));
  const sat = s.evidence?.satellite;
  need(sat && typeof sat.image === "string" && sat.half_width_m > 0, at("evidence.satellite"));
  for (const f of s.evidence.frames?.items ?? []) {
    need(typeof f.image === "string" && Object.values(f.answers).every(isProbs), at(`frame ${f.image}`));
  }
  return s;
}

const readJson = async (rel: string) => JSON.parse(await readFile(path.join(dataDir(), rel), "utf8"));

export async function loadIndex(): Promise<Index> {
  return parseIndex(await readJson("index.json"));
}

/** null when the id is malformed or there is no record for it. */
export async function loadSite(id: string): Promise<Site | null> {
  if (!ID.test(id)) return null;
  try {
    return parseSite(await readJson(`sites/${id}.json`));
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw e;
  }
}

/** Resolve an asset path under the data dir; null if it escapes it. */
export function assetPath(rel: string): string | null {
  const root = dataDir();
  const p = path.resolve(root, rel);
  return p.startsWith(root + path.sep) ? p : null;
}
