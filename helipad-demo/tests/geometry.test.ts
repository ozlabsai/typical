import { describe, expect, it } from "vitest";
import { obstacleText, pct, probsSentence } from "@/lib/describe";
import { m2vb, ringRect, scaleBar, sweepMax } from "@/lib/geometry";
import { DICT } from "@/lib/i18n";
import { schedule } from "@/lib/timeline";

describe("overlay geometry", () => {
  it("scales metres by half_width_m", () => {
    expect(m2vb(400, 400)).toBe(500);
    expect(m2vb(35, 400)).toBeCloseTo(43.75);
  });

  it("the pad is 35 m square, centred on pad_xy", () => {
    const r = ringRect([0.5, 0.5], 400);
    expect(r.w).toBeCloseTo(43.75);
    expect(r.x + r.w / 2).toBeCloseTo(500);
    expect(r.rx).toBe(0);
  });

  it("a ring is the pad grown by its distance, with corners of that radius", () => {
    const r = ringRect([0.5, 0.5], 400, 300);
    expect(r.w).toBeCloseTo(m2vb(35 + 600, 400));
    expect(r.rx).toBeCloseTo(m2vb(300, 400));
    expect(ringRect([0.25, 0.5], 400, 50).x + ringRect([0.25, 0.5], 400, 50).w / 2).toBeCloseTo(250);
  });

  it("scale bar and sweep", () => {
    expect(scaleBar(400)).toEqual({ m: 200, vb: 250 });
    expect(sweepMax([0.5, 0.5])).toBeCloseTo(Math.SQRT1_2 * 1000);
  });
});

describe("words and timing", () => {
  it("names a deciding finding plainly", () => {
    const o = { kind: "power_line", label: "", height_m: 24, height_src: "unknown", dist_m: 142, bearing: 8, in_pad: false, source: "map", xy: null };
    expect(obstacleText(o, DICT.en).replace(/[⁦⁩]/g, "")).toBe("power line, height unknown, 142 m north");
    expect(obstacleText({ ...o, height_src: "measured", bearing: null }, DICT.en).replace(/[⁦⁩]/g, "")).toBe("power line seen on video, 24 m, 142 m from the camera");
  });

  it("lists every option in the aria sentence, most likely first", () => {
    const s = probsSentence([["passes", 0.03], ["probably_fails", 0.96], ["fails", 0.01], ["not_checked", 0]], DICT.en.status);
    expect(s).toBe("probably fails 96%, passes 3%, fails 1%, not checked 0%");
    expect(pct(0.002)).toBe("<1");
  });

  it("the replay runs about 20 s with keyframes", () => {
    expect(schedule(3).end).toBeGreaterThan(17000);
    expect(schedule(3).end).toBeLessThan(22000);
    expect(schedule(0).itemsAt).toBeLessThan(schedule(3).itemsAt);
  });
});
