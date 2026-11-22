import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { decidedBy, obstacleText } from "@/lib/describe";
import { DICT } from "@/lib/i18n";
import type { Obstacle, Site } from "@/lib/protocol";

const plain = (s: string) => s.replace(/[⁦⁩]/g, "");
const site = (): Site => JSON.parse(readFileSync(path.join(__dirname, "..", "fixtures", "sites", "S4.json"), "utf8"));
const wire: Obstacle = { kind: "wire", label: "overhead wire (video; height and distance unknown, treated as within 50 m)",
  height_m: 8, height_src: "unknown", dist_m: 25, bearing: null, in_pad: false, source: "video", xy: null };

describe("video sightings", () => {
  it("read 'seen on video' with no direction", () => {
    expect(plain(obstacleText(wire, DICT.en))).toBe("overhead wire seen on video, distance unknown");
    const pole = { ...wire, kind: "pole", height_m: 30.2, height_src: "estimated", dist_m: 117.9 };
    expect(plain(obstacleText(pole, DICT.en))).toBe("pole seen on video, 30 m est., 118 m from the camera");
    expect(plain(obstacleText(wire, DICT.he))).toBe("תיל עילי נראה בווידאו, מרחק לא ידוע");
  });
});

describe("decided by", () => {
  it("names the nearest deciding finding", () => {
    const s = site();
    s.findings.obstacles.push(wire);
    s.items.find((i) => i.id === "low_wires")!.findings = [s.findings.obstacles.length - 1];
    expect(plain(decidedBy("low_wires", s, DICT.en).text)).toBe("overhead wire seen on video, distance unknown");
  });

  it("a wire item with no listed finding still names the wire seen on video", () => {
    const s = site();
    s.findings.obstacles.push(wire);
    s.items.find((i) => i.id === "low_wires")!.findings = [];
    expect(plain(decidedBy("low_wires", s, DICT.en).text)).toBe("overhead wire seen on video, distance unknown");
  });

  it("a pad decided by tree canopy says so", () => {
    const s = site();
    s.findings.canopy = { pad_max_m: 9, max_10_m: 9, max_50_m: 15, pad_over_rotor: true };
    s.items.find((i) => i.id === "pad")!.findings = [];
    expect(plain(decidedBy("pad", s, DICT.en).text)).toBe("tree canopy in the pad, 9 m");
  });

  it("otherwise the item's reported status with its probability", () => {
    const s = site();
    const it = s.items.find((i) => i.id === "very_low_10")!;
    Object.assign(it, { status: "probably_fails", findings: [], probs: { passes: 0.1, probably_fails: 0.87, fails: 0.01, not_checked: 0.02 } });
    expect(plain(decidedBy("very_low_10", s, DICT.en).text)).toBe("probably fails 87%");
    expect(decidedBy("very_low_10", s, DICT.en).more).toBe(0);
  });
});
