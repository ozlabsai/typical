// The replay schedule (ms from opening a site): map, then video keyframes, then the 11 items, then the verdict.
import { ITEMS } from "./protocol";

export const MAP_MS = 4500;
const FRAME_MS = 1300;
const ITEM_MS = 800;

export type Schedule = { framesAt: number; itemsAt: number; verdictAt: number; afterAt: number; end: number };

export function schedule(nFrames: number): Schedule {
  const framesAt = MAP_MS + 300;
  const itemsAt = framesAt + nFrames * FRAME_MS + (nFrames ? 300 : 0);
  const verdictAt = itemsAt + ITEMS.length * ITEM_MS + 500;
  const afterAt = verdictAt + 900;
  return { framesAt, itemsAt, verdictAt, afterAt, end: afterAt + 600 };
}

/** 0..1 progress of the map sweep at time t. */
export const mapProgress = (t: number) => Math.min(1, t / MAP_MS);
/** How many keyframes have been answered at time t. */
export const framesDone = (t: number, s: Schedule, n: number) =>
  Math.max(0, Math.min(n, Math.floor((t - s.framesAt) / FRAME_MS) + 1));
/** How many protocol items have been filled in at time t. */
export const itemsDone = (t: number, s: Schedule) =>
  Math.max(0, Math.min(ITEMS.length, Math.floor((t - s.itemsAt) / ITEM_MS) + 1));
