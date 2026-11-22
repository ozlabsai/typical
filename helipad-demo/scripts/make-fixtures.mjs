// Writes fixtures/: three synthetic sites (S2 approved, S4 probably fails, O7 fails) in the DATA.md format,
// with generated placeholder imagery. No real imagery, coordinates or place names. Run: npm run fixtures
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const OUT = path.join(path.dirname(new URL(import.meta.url).pathname), "..", "fixtures");
const HW = 400;
const ALLOWED = {
  pad: ["passes", "fails", "probably_fails", "not_checked"],
  very_low_10: ["passes", "fails", "probably_fails", "not_checked"],
  low_50: ["passes", "fails", "probably_fails", "not_checked"],
  high_300: ["passes", "fails", "probably_fails", "not_checked"],
  ultra_1km: ["passes", "coordination", "not_checked"],
  slope: ["passes", "fails", "not_checked"],
  zones: ["passes", "coordination", "not_checked"],
  belly: ["passes", "prep", "not_checked"],
  dust: ["passes", "prep", "not_checked"],
  low_wires: ["not_checked", "probably_fails", "fails"],
  minefield_500: ["not_checked", "fails"],
};
const SPILL = ["not_checked", "passes", "prep", "coordination", "probably_fails", "fails"];

const r4 = (x) => Math.round(x * 1e4) / 1e4;

function rng(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The reported status gets pTop; the rest spills onto the other options, the milder ones first. */
function probs(item, status, pTop) {
  const rest = SPILL.filter((s) => s !== status && ALLOWED[item].includes(s));
  const w = [0.62, 0.26, 0.09, 0.03].slice(0, rest.length);
  const sum = w.reduce((a, b) => a + b, 0);
  const p = { [status]: pTop };
  rest.forEach((s, i) => (p[s] = r4(((1 - pTop) * w[i]) / sum)));
  return p;
}

/** DATA.md: the verdict distribution follows from the items by the verdict rule, items independent. */
function verdictProbs(items) {
  const noFail = items.reduce((a, it) => a * (1 - (it.probs.fails ?? 0)), 1);
  const clear = items.reduce((a, it) => a * (1 - (it.probs.fails ?? 0) - (it.probs.probably_fails ?? 0)), 1);
  return { approved: r4(clear), probably_fails: r4(noFail - clear), fails: r4(1 - noFail), not_checked: 0 };
}

function xy(o) {
  if (o.in_pad) return [0.5, 0.5];
  const c = o.dist_m + 17.5;
  if (o.bearing === null || c > HW) return null;
  const b = (o.bearing * Math.PI) / 180;
  return [r4(0.5 + (c * Math.sin(b)) / (2 * HW)), r4(0.5 - (c * Math.cos(b)) / (2 * HW))];
}

const ob = (kind, label, height_m, height_src, dist_m, bearing) =>
  ({ kind, label, height_m, height_src, dist_m, bearing, in_pad: false, source: "map" });

const SITES = [
  {
    id: "S2", round: "2026-10", aircraft: 1, seed: 2,
    expert: { call: "approved", reason: { en: "Open field; the wires and buildings sit well outside the rings.",
      he: "שטח פתוח; התילים והמבנים רחוקים מהטבעות." } },
    obstacles: [
      ob("woodland", "woodland", 9, "estimated", 64, 250), ob("building", "building", 6, "estimated", 88, 120),
      ob("power_pole", "power pole", 12, "unknown", 210, 32), ob("building", "building", 7, "estimated", 140, 300),
      ob("orchard", "orchard", 4, "estimated", 35, 185), ob("fence", "fence", 1.5, "estimated", 22, 90),
      ob("road", "road", 0, "estimated", 118, 340), ob("power_line", "power line (voltage unknown)", 24, "unknown", 420, 20),
    ],
    status: { dust: "prep", belly: "passes", low_wires: "not_checked", minefield_500: "not_checked" },
    findings: {}, zones: [], slope: { deg: 1.8, unc: 6.0 }, dust: "medium", belly: "clear",
    layers: { osm: true, dem: true, govmap: true, corridors: false }, counts: { obstacles_total: 64, within_300: 9 },
    conditions: ["minefield_check", "low_wire_walk", "approach_heading", "notam"], prep: ["wetting"],
    frames: [{ t: 2.0, scene: "open" }, { t: 7.5, scene: "trees" }, { t: 12.0, scene: "open" }],
  },
  {
    id: "S4", round: "2026-10", aircraft: 1, seed: 4,
    expert: { call: "not_approved", reason: { en: "Power line north of the pad, inside the 300 m ring.",
      he: "קו חשמל צפונית למשטח, בתוך טבעת 300 מ'." } },
    obstacles: [
      ob("power_line", "power line (voltage unknown)", 24, "unknown", 142, 8), ob("power_pole", "power pole", 24, "unknown", 151, 352),
      ob("building", "building", 8, "estimated", 96, 210), ob("woodland", "woodland", 11, "estimated", 72, 140),
      ob("lit_road", "lit road", 10, "unknown", 260, 262), ob("tree", "tree", 6, "estimated", 58, 75),
      ob("building", "building", 5, "estimated", 180, 110),
      { ...ob("wire", "overhead wire (video; height and distance unknown, treated as within 50 m)", 8, "unknown", 25, null), source: "video" },
    ],
    status: { high_300: "probably_fails", zones: "coordination", belly: "passes", low_wires: "probably_fails", minefield_500: "not_checked" },
    findings: { high_300: [0, 1], low_wires: [7] }, zones: [{ kind: "military_area", dist_m: 2100, inside: false, bearing: null }],
    slope: { deg: 2.6, unc: 6.0 }, dust: "light", belly: "clear",
    layers: { osm: true, dem: true, govmap: true, corridors: false }, counts: { obstacles_total: 112, within_300: 14 },
    conditions: ["minefield_check", "low_wire_walk", "approach_heading", "notam"], prep: [],
    frames: [{ t: 1.5, scene: "wires" }, { t: 4.0, scene: "wires" }, { t: 6.5, scene: "trees" },
      { t: 9.0, scene: "open" }, { t: 11.5, scene: "open" }, { t: 14.0, scene: "wires" }],
  },
  {
    id: "O7", round: "2026-11", aircraft: 2, seed: 7, expert: null,
    obstacles: [
      ob("water_tower", "water tower", 18, "measured", 32, 225), ob("tree", "tree", 3, "estimated", 6, 100),
      ob("building", "building", 9, "estimated", 44, 300), ob("power_line", "power line (voltage unknown)", 24, "unknown", 372, 45),
      ob("antenna", "antenna mast", 82, "tagged", 760, 15), ob("building", "building", 6, "estimated", 120, 160),
    ],
    status: { very_low_10: "probably_fails", low_50: "fails", ultra_1km: "coordination", zones: "not_checked",
      belly: "prep", dust: "prep", low_wires: "not_checked", minefield_500: "not_checked" },
    findings: { very_low_10: [1], low_50: [0, 2], ultra_1km: [4] }, zones: [],
    slope: { deg: 7.5, unc: 2.0 }, dust: "heavy", belly: "obstacles",
    layers: { osm: true, dem: true, govmap: false, corridors: false }, counts: { obstacles_total: 41, within_300: 6 },
    conditions: ["minefield_check", "low_wire_walk", "approach_heading", "notam"], prep: ["wetting", "clear_belly_area"],
    frames: null,
  },
];

const OPTIONS = {
  wires: ["yes", "no"], pole: ["yes", "no"], tall_structure: ["yes", "no"],
  tallest_height: ["below_1_2", "1_2_to_4_5", "4_5_to_24", "above_24"],
  tallest_distance: ["below_10", "10_to_50", "50_to_300", "above_300"],
  tallest_type: ["tree", "building", "wall_or_fence", "pole", "vehicle", "none"],
  ground: ["paved", "hard_earth", "stony", "ploughed", "vegetation", "powdery"],
  vegetation_belly: ["yes", "no"], dust: ["light", "medium", "heavy"],
};
// The most likely answer to each frame question, per kind of scene in the placeholder frame.
const SCENES = {
  wires: { wires: "yes", pole: "yes", tall_structure: "no", tallest_height: "above_24", tallest_distance: "50_to_300",
    tallest_type: "pole", ground: "hard_earth", vegetation_belly: "no", dust: "light" },
  trees: { wires: "no", pole: "no", tall_structure: "no", tallest_height: "4_5_to_24", tallest_distance: "50_to_300",
    tallest_type: "tree", ground: "vegetation", vegetation_belly: "yes", dust: "light" },
  open: { wires: "no", pole: "no", tall_structure: "no", tallest_height: "1_2_to_4_5", tallest_distance: "above_300",
    tallest_type: "none", ground: "ploughed", vegetation_belly: "no", dust: "medium" },
};

/** Every option of every question plus cant_tell: the scene's answer gets most of the mass. */
function answers(scene, rand) {
  return Object.fromEntries(Object.entries(OPTIONS).map(([q, opts]) => {
    const top = SCENES[scene][q];
    const pTop = 0.62 + 0.33 * rand();
    const rest = [...opts.filter((o) => o !== top), "cant_tell"];
    const w = rest.map(() => 0.05 + rand());
    const sum = w.reduce((x, y) => x + y, 0);
    const p = { [top]: r4(pTop) };
    rest.forEach((o, i) => (p[o] = r4(((1 - pTop) * w[i]) / sum)));
    return [q, p];
  }));
}

function satSvg(site, obstacles) {
  const rand = rng(site.seed * 97);
  const S = 1024;
  const px = (f) => f * S;
  const m = (v) => (v / (2 * HW)) * S;
  const fields = Array.from({ length: 9 }, () => {
    const x = rand() * S, y = rand() * S, w = 160 + rand() * 260, h = 120 + rand() * 240, a = rand() * 40 - 20;
    const fill = ["#7b7650", "#5f6a43", "#8b7d59", "#6c7449", "#93865f"][Math.floor(rand() * 5)];
    return `<rect x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" fill="${fill}" opacity="0.55" transform="rotate(${a} ${x} ${y})"/>`;
  }).join("");
  const roadY = 140 + rand() * 120;
  const road = `<path d="M -20 ${roadY} C 300 ${roadY + 90}, 620 ${roadY - 60}, 1050 ${roadY + 40}" stroke="#b8b19c" stroke-width="9" fill="none" opacity="0.85"/>
    <path d="M ${560 + rand() * 80} 1040 C 540 800, 600 650, 548 ${S / 2 + 28}" stroke="#a9a28c" stroke-width="5" fill="none" opacity="0.7"/>`;
  const marks = obstacles.map((o) => {
    if (!o.xy) return "";
    const [x, y] = [px(o.xy[0]), px(o.xy[1])];
    if (["building", "water_tower"].includes(o.kind)) {
      const w = m(o.kind === "water_tower" ? 9 : 14 + rand() * 10), h = m(10 + rand() * 8);
      return `<rect x="${x - w / 2 + 3}" y="${y - h / 2 + 3}" width="${w}" height="${h}" fill="#2c2a22" opacity="0.5"/>
        <rect x="${x - w / 2}" y="${y - h / 2}" width="${w}" height="${h}" fill="#cfc8b6"/>`;
    }
    if (["woodland", "orchard", "tree"].includes(o.kind)) {
      const n = o.kind === "tree" ? 1 : 14;
      return Array.from({ length: n }, () => {
        const dx = (rand() - 0.5) * m(o.kind === "tree" ? 0 : 50), dy = (rand() - 0.5) * m(o.kind === "tree" ? 0 : 36);
        return `<circle cx="${x + dx}" cy="${y + dy}" r="${m(3 + rand() * 2.5)}" fill="#34402a"/>`;
      }).join("");
    }
    return "";
  }).join("");
  const line = obstacles.filter((o) => o.xy && ["power_pole", "power_line"].includes(o.kind)).map((o) => o.xy).sort((a, b) => a[0] - b[0]);
  const wire = line.length >= 1
    ? `<path d="M ${px(line[0][0]) - 600} ${px(line[0][1]) - 40} L ${line.map(([a, b]) => `${px(a)} ${px(b)}`).join(" L ")} L ${px(line.at(-1)[0]) + 600} ${px(line.at(-1)[1]) + 60}" stroke="#262620" stroke-width="2" fill="none" opacity="0.8"/>`
    : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${S}" height="${S}">
    <defs>
      <filter id="t" x="0" y="0" width="100%" height="100%">
        <feTurbulence type="fractalNoise" baseFrequency="0.006" numOctaves="5" seed="${site.seed}"/>
        <feColorMatrix values="0 0 0 0 0.22  0 0 0 0 0.2  0 0 0 0 0.12  0 0 0 1.1 -0.35"/>
      </filter>
      <filter id="g" x="0" y="0" width="100%" height="100%">
        <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="2" seed="${site.seed + 3}"/>
        <feColorMatrix values="0 0 0 0 0.9  0 0 0 0 0.88  0 0 0 0 0.8  0 0 0 0.35 0"/>
      </filter>
    </defs>
    <rect width="100%" height="100%" fill="#77744f"/>
    ${fields}
    <rect width="100%" height="100%" filter="url(#t)"/>
    ${road}
    <rect x="${S / 2 - m(24)}" y="${S / 2 - m(24)}" width="${m(48)}" height="${m(48)}" fill="#a69d7c" opacity="0.7" transform="rotate(8 ${S / 2} ${S / 2})"/>
    ${marks}${wire}
    <rect width="100%" height="100%" filter="url(#g)"/>
  </svg>`;
}

function frameSvg(scene, seed) {
  const rand = rng(seed);
  const W = 360, H = 640, hz = 300 + rand() * 40;
  const trees = Array.from({ length: scene === "trees" ? 18 : 6 }, () => {
    const x = rand() * W, r = 10 + rand() * (scene === "trees" ? 34 : 14);
    return `<ellipse cx="${x}" cy="${hz - r * 0.5}" rx="${r * 1.3}" ry="${r}" fill="#3e4a33"/>`;
  }).join("");
  const poles = scene === "wires"
    ? [40, 190, 330].map((x, i) => `<rect x="${x}" y="${hz - 150 + i * 30}" width="6" height="${150 - i * 30}" fill="#2a2a26"/>`).join("") +
      `<path d="M 0 ${hz - 140} Q 20 ${hz - 130} 43 ${hz - 148} Q 118 ${hz - 100} 193 ${hz - 118} Q 262 ${hz - 78} 333 ${hz - 88} Q 350 ${hz - 80} 360 ${hz - 84}" stroke="#1e1e1a" stroke-width="1.6" fill="none"/>`
    : "";
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}">
    <defs>
      <linearGradient id="s" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8fa7b8"/><stop offset="1" stop-color="#d9dcd6"/></linearGradient>
      <filter id="n" x="0" y="0" width="100%" height="100%">
        <feTurbulence type="fractalNoise" baseFrequency="0.02 0.12" numOctaves="4" seed="${seed}"/>
        <feColorMatrix values="0 0 0 0 0.3  0 0 0 0 0.26  0 0 0 0 0.18  0 0 0 0.9 -0.2"/>
      </filter>
    </defs>
    <rect width="${W}" height="${hz}" fill="url(#s)"/>
    ${trees}
    <rect y="${hz}" width="${W}" height="${H - hz}" fill="#8f8160"/>
    <rect y="${hz}" width="${W}" height="${H - hz}" filter="url(#n)"/>
    ${poles}
  </svg>`;
}

async function main() {
  await mkdir(path.join(OUT, "sites"), { recursive: true });
  await mkdir(path.join(OUT, "img"), { recursive: true });
  const sites = [];
  for (const s of SITES) {
    const rand = rng(s.seed);
    const obstacles = s.obstacles.map((o) => ({ ...o, xy: xy(o) }));
    const items = Object.keys(ALLOWED).map((id) => {
      const status = s.status[id] ?? "passes";
      const pTop = ["fails", "probably_fails"].includes(status) ? 0.86 + rand() * 0.11 : 0.985 + rand() * 0.012;
      return { id, status, probs: probs(id, status, r4(pTop)), findings: s.findings[id] ?? [] };
    });
    const deciding = items.filter((i) => ["fails", "probably_fails"].includes(i.status)).map((i) => i.id);
    const statuses = items.map((i) => i.status);
    const verdict = statuses.includes("fails") ? "fails" : statuses.includes("probably_fails") ? "probably_fails" : "approved";
    const rec = {
      id: s.id, name: { en: s.id, he: s.id }, round: s.round, aircraft: s.aircraft,
      touchdown: { moved: null }, expert: s.expert,
      verdict: { status: verdict, probs: verdictProbs(items), deciding },
      items, conditions: s.conditions, prep: s.prep,
      findings: { obstacles, zones: s.zones, slope: s.slope, canopy: { pad_max_m: 0.3, max_10_m: 1.2, max_50_m: 8.1 },
        dust: s.dust, belly: s.belly, layers: s.layers, counts: s.counts },
      evidence: { satellite: { image: `img/${s.id}_sat.jpg`, half_width_m: HW, pad_xy: [0.5, 0.5] } },
    };
    await sharp(Buffer.from(satSvg(s, obstacles))).jpeg({ quality: 82 }).toFile(path.join(OUT, rec.evidence.satellite.image));
    if (s.frames) {
      rec.evidence.frames = { source: "model", items: [] };
      for (const [k, f] of s.frames.entries()) {
        const image = `img/${s.id}_f${String(k + 1).padStart(2, "0")}.jpg`;
        await sharp(Buffer.from(frameSvg(f.scene, s.seed * 10 + k))).jpeg({ quality: 80 }).toFile(path.join(OUT, image));
        rec.evidence.frames.items.push({ image, t: f.t,
          answers: answers(f.scene, rand) });
      }
    }
    await writeFile(path.join(OUT, "sites", `${s.id}.json`), JSON.stringify(rec, null, 1) + "\n");
    sites.push({ id: s.id, round: s.round, verdict });
    console.log(`${s.id}: ${verdict}`, rec.verdict.probs);
  }
  await writeFile(path.join(OUT, "index.json"),
    JSON.stringify({ model: { name: "typical-medium · helipad", version: "plan1-800" }, sites }, null, 1) + "\n");
}

main();
