// Findings as words, from ids and numbers through the dictionary.
import { type Dict, ltr, tr } from "./i18n";
import { compass, type Item, type Obstacle, type Site } from "./protocol";

const CERTAIN = ["tagged", "levels", "iec", "measured"];

export function obstacleText(o: Obstacle, d: Dict): string {
  const u = d.ui.metres;
  const kind = tr(d.kinds, o.kind);
  const height = o.height_src === "unknown" ? d.ui.heightUnknown
    : CERTAIN.includes(o.height_src) ? `${ltr(Math.round(o.height_m))} ${u}`
    : `${ltr(Math.round(o.height_m))} ${u} ${d.ui.estimated}`;
  if (o.bearing === null && !o.in_pad) {
    // A video sighting has no direction. With no height either, the recorded distance is a placeholder
    // (the bake's "treated as within 50 m"), so it reads as unknown.
    return o.height_src === "unknown"
      ? `${kind} ${d.ui.seenOnVideo}, ${d.ui.distanceUnknown}`
      : `${kind} ${d.ui.seenOnVideo}, ${height}, ${ltr(Math.round(o.dist_m))} ${u} ${d.ui.fromCamera}`;
  }
  const where = o.in_pad ? d.ui.inPad : `${ltr(Math.round(o.dist_m))} ${u} ${tr(d.dirs, compass(o.bearing!))}`;
  return `${kind}, ${height}, ${where}`;
}

/** What decided an item: its nearest deciding finding (and how many more), else the evidence behind it,
 *  else its reported status with that status's probability. */
export function decidedBy(item: Item, site: Site, d: Dict): { text: string; more: number } {
  const it = site.items.find((i) => i.id === item);
  return evidence(item, site, d) ?? { text: it ? `${d.status[it.status]} ${ltr(`${pct(it.probs[it.status] ?? 0)}%`)}` : "", more: 0 };
}

function evidence(item: Item, site: Site, d: Dict): { text: string; more: number } | null {
  const f = site.findings;
  const it = site.items.find((i) => i.id === item);
  const nearest = (ks: number[]) => ks.map((k) => f.obstacles[k]).sort((a, b) => a.dist_m - b.dist_m);
  const obs = nearest(it?.findings ?? []);
  if (obs.length) return { text: obstacleText(obs[0], d), more: obs.length - 1 };
  if (item === "pad" && f.canopy?.pad_over_rotor) {
    return { text: `${d.ui.canopyInPad}, ${ltr(Math.round(Number(f.canopy.pad_max_m)))} ${d.ui.metres}`, more: 0 };
  }
  if (item === "low_wires") {
    const wires = nearest(f.obstacles.flatMap((o, k) => (o.kind === "wire" ? [k] : [])));
    if (wires.length) return { text: obstacleText(wires[0], d), more: wires.length - 1 };
  }
  if (item === "slope" && f.slope) return { text: `${d.ui.slope} ${ltr(`${f.slope.deg.toFixed(1)}° ± ${f.slope.unc.toFixed(1)}°`)}`, more: 0 };
  if (item === "dust" && f.dust) return { text: tr(d.options, f.dust), more: 0 };
  const zones = f.zones.filter((z) => (item === "minefield_500") === (z.kind === "minefield"));
  if ((item === "zones" || item === "minefield_500" || item === "high_300") && zones.length) {
    const z = [...zones].sort((a, b) => a.dist_m - b.dist_m)[0];
    const dir = z.bearing === null ? "" : ` ${tr(d.dirs, compass(z.bearing))}`;
    return { text: `${tr(d.zones, z.kind)}, ${ltr(Math.round(z.dist_m))} ${d.ui.metres}${dir}`, more: zones.length - 1 };
  }
  return null;
}

/** "probably fails 96%, passes 3%, …": every option, most likely first (for aria labels). */
export function probsSentence(entries: [string, number][], names: Record<string, string>): string {
  return [...entries].sort((a, b) => b[1] - a[1]).map(([k, p]) => `${tr(names, k)} ${pct(p)}%`).join(", ");
}

/** Whole percent; a non-zero mass that rounds to 0 reads "<1". */
export function pct(p: number): string {
  const r = Math.round(p * 100);
  return r === 0 && p > 0 ? "<1" : String(r);
}
