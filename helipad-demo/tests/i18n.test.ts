import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { DICT } from "@/lib/i18n";
import { COMPASS, ITEMS, NULL_OPTION, QUESTION_OPTIONS, QUESTIONS, STATUSES, VERDICTS } from "@/lib/protocol";

const { en, he } = DICT;

function leaves(o: object, prefix = ""): [string, unknown][] {
  return Object.entries(o).flatMap(([k, v]) => (v && typeof v === "object" ? leaves(v, `${prefix}${k}.`) : [[`${prefix}${k}`, v]]));
}

describe("i18n", () => {
  it("EN and HE have the same keys, all non-empty", () => {
    const e = leaves(en), h = leaves(he);
    expect(h.map(([k]) => k)).toEqual(e.map(([k]) => k));
    for (const [k, v] of [...e, ...h]) expect(v, k).toMatch(/\S/);
  });

  it("every protocol id has a word in both languages", () => {
    for (const d of [en, he]) {
      for (const i of ITEMS) expect(d.items[i] && d.hints[i], i).toBeTruthy();
      for (const s of STATUSES) expect(d.status[s], s).toBeTruthy();
      for (const v of VERDICTS) expect(d.verdict[v] && d.verdictOption[v], v).toBeTruthy();
      for (const q of QUESTIONS) {
        expect(d.questions[q], q).toBeTruthy();
        for (const o of [...QUESTION_OPTIONS[q], NULL_OPTION]) expect((d.options as Record<string, string>)[o], `${q}.${o}`).toBeTruthy();
      }
      for (const c of COMPASS) expect(d.dirs[c], c).toBeTruthy();
    }
  });

  it("the required Hebrew status words", () => {
    expect(he.status).toMatchObject({ passes: "עומד", fails: "לא עומד", probably_fails: "כנראה לא עומד",
      not_checked: "לא נבדק", prep: "הכנה", coordination: "תיאום" });
    expect(he.verdict.approved).toBe("מאושר");
  });

  it("every id used by the fixtures is in the dictionary", () => {
    const dir = path.join(__dirname, "..", "fixtures", "sites");
    for (const f of readdirSync(dir)) {
      const s = JSON.parse(readFileSync(path.join(dir, f), "utf8"));
      const need: [Record<string, string>, string][] = [
        ...s.findings.obstacles.map((o: { kind: string }) => [en.kinds, o.kind]),
        ...s.findings.zones.map((z: { kind: string }) => [en.zones, z.kind]),
        ...s.conditions.map((c: string) => [en.conditions, c]),
        ...s.prep.map((c: string) => [en.prep, c]),
        ...Object.keys(s.findings.layers).map((k) => [en.layers, k]),
      ];
      for (const [map, id] of need) expect(map[id], `${f}: ${id}`).toBeTruthy();
    }
  });
});
