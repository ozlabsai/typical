// Overlay geometry on the satellite image, in a 1000×1000 SVG viewBox over a north-up square crop.
import { PAD_SIDE_M } from "./protocol";

export const VB = 1000;

/** Metres to viewBox units: the crop spans 2 × half_width_m metres. */
export const m2vb = (m: number, halfWidthM: number) => (m / (2 * halfWidthM)) * VB;

export type Rect = { x: number; y: number; w: number; h: number; rx: number };

/** The area within `ringM` metres of the 35×35 m pad edge: the pad grown by ringM, with corners of radius ringM. */
export function ringRect(padXY: [number, number], halfWidthM: number, ringM = 0): Rect {
  const half = m2vb(PAD_SIDE_M / 2 + ringM, halfWidthM);
  const [cx, cy] = [padXY[0] * VB, padXY[1] * VB];
  return { x: cx - half, y: cy - half, w: 2 * half, h: 2 * half, rx: m2vb(ringM, halfWidthM) };
}

/** An image fraction (x right, y down) to viewBox units. */
export const fracToVb = (xy: [number, number]): [number, number] => [xy[0] * VB, xy[1] * VB];

/** A round scale-bar length (metres) close to a fifth of the crop, and its viewBox length. */
export function scaleBar(halfWidthM: number): { m: number; vb: number } {
  const target = (2 * halfWidthM) / 5;
  const m = [10, 20, 25, 50, 100, 200, 250, 500, 1000].reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
  return { m, vb: m2vb(m, halfWidthM) };
}

/** Distance from the pad centre (viewBox units) to the farthest point the image shows, for the scan sweep. */
export const sweepMax = (padXY: [number, number]) =>
  Math.hypot(Math.max(padXY[0], 1 - padXY[0]), Math.max(padXY[1], 1 - padXY[1])) * VB;
