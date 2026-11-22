// Vocabulary and record types of helipad-demo/DATA.md. Pure: safe to import from client and server.

export const ITEMS = ["pad", "very_low_10", "low_50", "high_300", "ultra_1km", "slope", "zones",
  "belly", "dust", "low_wires", "minefield_500"] as const;
export const STATUSES = ["passes", "probably_fails", "fails", "coordination", "prep", "not_checked"] as const;
export const VERDICTS = ["approved", "probably_fails", "fails", "not_checked"] as const;
export const QUESTIONS = ["wires", "pole", "tall_structure", "tallest_height", "tallest_distance",
  "tallest_type", "ground", "vegetation_belly", "dust"] as const;
// Option ids per frame question (docs/superpowers/plans/2026-11-21-helipad-plan2-vision-student.md), plus the null option.
export const QUESTION_OPTIONS: Record<Question, readonly string[]> = {
  wires: ["yes", "no"],
  pole: ["yes", "no"],
  tall_structure: ["yes", "no"],
  tallest_height: ["below_1_2", "1_2_to_4_5", "4_5_to_24", "above_24"],
  tallest_distance: ["below_10", "10_to_50", "50_to_300", "above_300"],
  tallest_type: ["tree", "building", "wall_or_fence", "pole", "vehicle", "none"],
  ground: ["paved", "hard_earth", "stony", "ploughed", "vegetation", "powdery"],
  vegetation_belly: ["yes", "no"],
  dust: ["light", "medium", "heavy"],
};
export const NULL_OPTION = "cant_tell";
export const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"] as const;
// The three obstacle rings of the protocol, in metres from the pad edge, keyed by their item.
export const RINGS = { very_low_10: 10, low_50: 50, high_300: 300 } as const;
export const PAD_SIDE_M = 35;
export const FAILING: readonly string[] = ["fails", "probably_fails"];

export type Item = (typeof ITEMS)[number];
export type Status = (typeof STATUSES)[number];
export type Verdict = (typeof VERDICTS)[number];
export type Question = (typeof QUESTIONS)[number];
export type Probs = Record<string, number>;
export type Bi = { en: string; he: string };

export type Obstacle = {
  kind: string; label: string; height_m: number; height_src: string; dist_m: number;
  bearing: number | null; in_pad: boolean; source: string; xy: [number, number] | null;
};
export type Zone = { kind: string; dist_m: number; inside: boolean; bearing: number | null };
export type SiteItem = { id: Item; status: Status; probs: Probs; findings: number[] };
export type Frame = { image: string; t: number; answers: Record<string, Probs> };

export type Site = {
  id: string;
  name: Bi;
  round: string;
  aircraft: number;
  touchdown: { lat?: number; lon?: number; moved: { m: number; dir: string } | null };
  expert: { call: "approved" | "not_approved"; reason: Bi } | null;
  verdict: { status: Verdict; probs: Probs; deciding: Item[] };
  items: SiteItem[];
  conditions: string[];
  prep: string[];
  findings: {
    obstacles: Obstacle[];
    zones: Zone[];
    slope: { deg: number; unc: number } | null;
    canopy: Record<string, number | boolean> | null;
    dust: string | null;
    belly: string | null;
    layers: Record<string, boolean>;
    counts: { obstacles_total: number; within_300: number };
  };
  evidence: {
    satellite: { image: string; half_width_m: number; pad_xy: [number, number] };
    frames?: { source: string; items: Frame[] };
  };
};

export type SiteSummary = { id: string; round: string; verdict: Verdict };
export type Index = { model: { name: string; version: string }; sites: SiteSummary[] };

/** Options in a fixed display order (the data's key order is shuffled); unknown keys go last. */
export function ordered(probs: Probs, order: readonly string[]): [string, number][] {
  const rank = (k: string) => (order.includes(k) ? order.indexOf(k) : order.length);
  return Object.entries(probs).sort((a, b) => rank(a[0]) - rank(b[0]));
}

export function compass(bearing: number): (typeof COMPASS)[number] {
  return COMPASS[Math.floor((((bearing % 360) + 360) % 360 + 22.5) / 45) % 8];
}
