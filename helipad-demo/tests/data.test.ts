import { readFileSync } from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { assetPath, dataDir, loadIndex, loadSite, parseSite } from "@/lib/data";
import { compass, ordered, STATUSES } from "@/lib/protocol";

const fixture = (id: string) => JSON.parse(readFileSync(path.join(__dirname, "..", "fixtures", "sites", `${id}.json`), "utf8"));

describe("data", () => {
  afterEach(() => delete process.env.HELIPAD_DEMO_DATA);

  it("loads the three fixture sites: approved, probably fails, fails", async () => {
    const idx = await loadIndex();
    expect(idx.sites.map((s) => [s.id, s.verdict])).toEqual([["S2", "approved"], ["S4", "probably_fails"], ["O7", "fails"]]);
    for (const s of idx.sites) {
      const site = (await loadSite(s.id))!;
      expect(site.verdict.status).toBe(s.verdict);
      expect(site.items).toHaveLength(11);
      for (const p of [site.verdict.probs, ...site.items.map((i) => i.probs)]) {
        expect(Object.values(p).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 2);
      }
    }
    expect((await loadSite("S4"))!.evidence.frames?.items.length).toBeGreaterThan(0);
    expect((await loadSite("O7"))!.evidence.frames).toBeUndefined();
  });

  it("reads HELIPAD_DEMO_DATA when set", () => {
    process.env.HELIPAD_DEMO_DATA = "/tmp/elsewhere";
    expect(dataDir()).toBe("/tmp/elsewhere");
  });

  it("rejects malformed records and ids", async () => {
    const bad = fixture("S4");
    bad.items[0].id = "runway";
    expect(() => parseSite(bad)).toThrow(/unknown item/);
    const bad2 = fixture("S4");
    bad2.items[3].findings = [99];
    expect(() => parseSite(bad2)).toThrow(/findings index/);
    expect(await loadSite("../index")).toBeNull();
    expect(await loadSite("ZZ9")).toBeNull();
  });

  it("asset paths cannot escape the data dir", () => {
    expect(assetPath("img/S4_sat.jpg")).toMatch(/fixtures\/img\/S4_sat\.jpg$/);
    expect(assetPath("../package.json")).toBeNull();
  });

  it("orders options canonically whatever the record's key order", () => {
    expect(ordered({ not_checked: 0, fails: 0.01, passes: 0.03, probably_fails: 0.96 }, STATUSES).map(([k]) => k))
      .toEqual(["passes", "probably_fails", "fails", "not_checked"]);
    expect([0, 8, 22.6, 359, 180, 270].map(compass)).toEqual(["north", "north", "north-east", "north", "south", "west"]);
  });
});
