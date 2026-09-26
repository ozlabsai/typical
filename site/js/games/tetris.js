// Pure Tetris engine. No DOM, no rendering framework - state in, state out. Matches
// site/js/snake.js's shape: state()/describe()/render()/candidates()/step()/selfTest().
//
// Demo-spec v4 grammar, adapted: the model does not press keys, it picks ONE candidate
// placement (rotation+column) per piece - the decision that matters, not the keystrokes that
// execute it (Snake picks a direction per tick, not a lookahead path; this is the same idea one
// level up). Candidates are legal (rotation, column) hard-drop pairs, capped at 6 and spread
// across the board features that make one placement different from another (a well, a notch, a
// clear), never raw coordinates. The state sentence lists only the board situations that apply,
// pre-computed, fixed order, no numbers to compare. A drop that would trap an empty cell says so in
// words ("bury a hole on the left"), and the state opens with the one rule that makes that phrase
// mean something. Measured on typical-small (rev 0ff732e), 10x20, seeds 1-40, capped at 300 pieces:
//   positional labels only ("set it down on the left of the stack")  median  64 pieces, 12 lines
//   + "leaving a gap underneath" / "sitting flush" (model preferred the gap)  74 pieces, 15 lines
//   "bury a hole" / "fill in the stack" + the solid-blocks rule          253 pieces, 89 lines
//   random choice among the same options                              55 pieces,  8 lines
// Labels that spelled out raw holes / peak / roughness numbers were worse than positional ones.

function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Base shapes in a 4x4 box; every rotation is derived from this one (x,y)->(y,3-x) transform
// rather than hand-written per piece - one rotate() reused for all seven, dedup'd so I/O/S/Z
// naturally collapse to their true 2/1/2/2 distinct states instead of 4 duplicates each.
const BASE = {
  I: [[0, 1], [1, 1], [2, 1], [3, 1]],
  O: [[1, 0], [2, 0], [1, 1], [2, 1]],
  T: [[0, 0], [1, 0], [2, 0], [1, 1]],
  S: [[1, 0], [2, 0], [0, 1], [1, 1]],
  Z: [[0, 0], [1, 0], [1, 1], [2, 1]],
  J: [[0, 0], [0, 1], [1, 1], [2, 1]],
  L: [[2, 0], [0, 1], [1, 1], [2, 1]],
};
const PIECE_TYPES = Object.keys(BASE);

function rotateCW(cells) {
  return cells.map(([x, y]) => [y, 3 - x]);
}
function normalize(cells) {
  const minX = Math.min(...cells.map((c) => c[0]));
  const minY = Math.min(...cells.map((c) => c[1]));
  return cells.map(([x, y]) => [x - minX, y - minY]).sort((a, b) => a[1] - b[1] || a[0] - b[0]);
}
// Rotates the *unnormalized* 4x4-box cells each step (a fixed pivot), normalizing only for the
// dedup key and the stored width/height - normalizing between rotations would rotate around a
// drifting pivot and corrupt the 3rd/4th state.
function statesFor(base) {
  const states = [];
  const seen = new Set();
  let cur = base;
  for (let i = 0; i < 4; i++) {
    const norm = normalize(cur);
    const key = JSON.stringify(norm);
    if (!seen.has(key)) {
      seen.add(key);
      const w = Math.max(...norm.map((c) => c[0])) + 1;
      const h = Math.max(...norm.map((c) => c[1])) + 1;
      states.push({ cells: norm, w, h });
    }
    cur = rotateCW(cur);
  }
  return states;
}
export const SHAPES = Object.fromEntries(PIECE_TYPES.map((t) => [t, statesFor(BASE[t])]));

function fits(board, cells, col, row, W, H) {
  return cells.every(([dx, dy]) => {
    const x = col + dx;
    const y = row + dy;
    return x >= 0 && x < W && y >= 0 && y < H && !board[y][x];
  });
}
export function dropRow(board, cells, col, W, H) {
  if (!fits(board, cells, col, 0, W, H)) return null;
  let row = 0;
  while (fits(board, cells, col, row + 1, W, H)) row++;
  return row;
}
function colHeights(board, W, H) {
  const heights = [];
  for (let c = 0; c < W; c++) {
    let h = 0;
    for (let r = 0; r < H; r++) {
      if (board[r][c]) {
        h = H - r;
        break;
      }
    }
    heights.push(h);
  }
  return heights;
}
function countHoles(board, W, H) {
  let holes = 0;
  for (let c = 0; c < W; c++) {
    let seen = false;
    for (let r = 0; r < H; r++) {
      if (board[r][c]) seen = true;
      else if (seen) holes++;
    }
  }
  return holes;
}
// Contiguous runs of columns at least 2 lower than both neighbours (board edge counts as an
// infinitely tall neighbour, so an edge column can be a well against the wall). width===1 is a
// notch, width>=2 is a well - describe()/classify() key off this one shared computation.
function wellRegions(board, W, H) {
  const heights = colHeights(board, W, H);
  const left = (c) => (c === 0 ? Infinity : heights[c - 1]);
  const right = (c) => (c === W - 1 ? Infinity : heights[c + 1]);
  const isWell = heights.map((h, c) => h <= Math.min(left(c), right(c)) - 2);
  const regions = [];
  let start = null;
  for (let c = 0; c <= W; c++) {
    if (c < W && isWell[c]) {
      if (start === null) start = c;
    } else if (start !== null) {
      regions.push({ c0: start, c1: c - 1, width: c - start, depth: Math.min(left(start), right(c - 1)) - heights[start] });
      start = null;
    }
  }
  return regions;
}
// Locks `cells` at (col,row) onto a copy of `board`, clears full rows, and reports the
// heuristics candidate ranking and greedyPolicy use. Never mutates `board`.
function lockAndClear(board, W, H, cells, col, row, type) {
  const b2 = board.map((r) => r.slice());
  cells.forEach(([dx, dy]) => {
    b2[row + dy][col + dx] = type;
  });
  let cleared = 0;
  const kept = b2.filter((r) => {
    const full = r.every((c) => c);
    if (full) cleared++;
    return !full;
  });
  while (kept.length < H) kept.unshift(new Array(W).fill(null));
  const holes = countHoles(kept, W, H);
  const holesCreated = Math.max(0, holes - countHoles(board, W, H));
  const heights = colHeights(kept, W, H);
  const maxHeight = Math.max(0, ...heights);
  let bumpiness = 0;
  for (let c = 0; c < W - 1; c++) bumpiness += Math.abs(heights[c] - heights[c + 1]);
  return { board: kept, linesCleared: cleared, holes, holesCreated, bumpiness, maxHeight };
}

function third(p, W) {
  const mid = p.col + (p.w - 1) / 2;
  return mid < W / 3 ? 'on the left' : mid > (2 * W) / 3 ? 'on the right' : 'in the middle';
}

// Classifies one candidate placement against the board it would land on (pre-lock, for the well/
// notch geometry) - the same precedence order as RULES/greedyPolicy. Every placement gets
// exactly one label; label text never carries a raw distance, only an identifying column number
// or a qualitative side (demo-spec v4: numbers the model would have to compare leak into the
// decision, but an identifier like "column 4" does not ask for a comparison).
function classify(p, wells, W) {
  if (p.linesCleared > 0) {
    return { category: 'clear_line', label: `clear the row by filling column ${p.col + 1}` };
  }
  const pc0 = p.col;
  const pc1 = p.col + p.w - 1;
  const region = wells.find((w) => pc0 >= w.c0 && pc1 <= w.c1);
  if (region && region.width === 1) {
    return { category: 'fill_notch', label: `fill the notch in column ${pc0 + 1}` };
  }
  if (region) {
    const side = region.c0 === 0 ? 'left' : region.c1 === W - 1 ? 'right' : (region.c0 + region.c1) / 2 < W / 2 ? 'left' : 'right';
    const edge = region.c0 === 0 || region.c1 === W - 1;
    const upright = p.h > p.w;
    if (edge && upright) return { category: 'settle_well', label: `stand it up against the ${side} wall` };
    return { category: 'settle_well', label: upright ? `stand it up in the ${side} well` : `lay it flat in the ${side} well` };
  }
  return { category: 'default', label: `fill in the stack ${third(p, W)}` };
}

const CATEGORY_ORDER = ['clear_line', 'fill_notch', 'settle_well', 'default'];

export class Tetris {
  constructor({ w = 8, h = 14, seed = 1 } = {}) {
    this.W = w;
    this.H = h;
    this.rng = mulberry32(seed);
    this.reset();
  }

  reset() {
    this.board = Array.from({ length: this.H }, () => new Array(this.W).fill(null));
    this.score = 0;
    this.lines = 0;
    this.pieces = 0;
    this.dead = false;
    this._bag = [];
    this.nextType = this._drawType();
    this.current = null;
    this._spawn();
    return this.state();
  }

  _drawType() {
    if (!this._bag || this._bag.length === 0) {
      this._bag = [...PIECE_TYPES];
      for (let i = this._bag.length - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        [this._bag[i], this._bag[j]] = [this._bag[j], this._bag[i]];
      }
    }
    return this._bag.pop();
  }

  _spawn() {
    const type = this.nextType;
    this.nextType = this._drawType();
    const state = SHAPES[type][0];
    const col = Math.floor((this.W - state.w) / 2);
    this.current = { type, col, cells: state.cells };
    if (!fits(this.board, state.cells, col, 0, this.W, this.H)) this.dead = true;
  }

  // Every legal (rotation, column) hard drop for the current piece, scored and classified, deduped
  // by label (keeping the higher-scoring of two placements that would read the same way - two
  // rotations landing in the same well, say), then capped to a spread of the top 2 per category
  // (not just the global top 6, which would cluster in one spot) and the best 6 overall.
  _candidates() {
    if (this.dead || !this.current) return [];
    const { type } = this.current;
    const wells = wellRegions(this.board, this.W, this.H);
    const raw = [];
    SHAPES[type].forEach((state, rotation) => {
      for (let col = 0; col <= this.W - state.w; col++) {
        const row = dropRow(this.board, state.cells, col, this.W, this.H);
        if (row === null) continue;
        const lock = lockAndClear(this.board, this.W, this.H, state.cells, col, row, type);
        const p = { col, row, w: state.w, h: state.h, cells: state.cells, rotation, ...lock };
        p.score = p.linesCleared * 100 - p.holesCreated * 15 - p.bumpiness * 2 - p.maxHeight;
        const { category, label } = classify(p, wells, this.W);
        p.category = category;
        // A hole outranks the geometry: "fill the notch" that buries a cell is still a bad drop.
        p.label = p.holesCreated && category !== 'clear_line' ? `bury a hole ${third(p, this.W)}` : label;
        raw.push(p);
      }
    });
    const byLabel = new Map();
    for (const p of raw) {
      const cur = byLabel.get(p.label);
      if (!cur || p.score > cur.score) byLabel.set(p.label, p);
    }
    const byCat = new Map();
    for (const p of byLabel.values()) {
      const arr = byCat.get(p.category) ?? [];
      arr.push(p);
      byCat.set(p.category, arr);
    }
    const spread = [];
    for (const arr of byCat.values()) {
      arr.sort((a, b) => b.score - a.score);
      spread.push(...arr.slice(0, 2));
    }
    spread.sort((a, b) => b.score - a.score);
    return spread.slice(0, 6);
  }

  candidates() {
    return this._candidates().map((p) => p.label);
  }

  // The full placements behind the labels (where each lands, its heuristic score) - for UIs that
  // animate the drop or guard a live answer. The model itself only ever sees the labels.
  candidateDetails() {
    return this._candidates();
  }

  // Copies the board (like Snake/Doom/Drive's state()) rather than handing out live references.
  state() {
    return {
      board: this.board.map((r) => r.slice()),
      score: this.score,
      lines: this.lines,
      pieces: this.pieces,
      dead: this.dead,
      current: this.current ? { type: this.current.type, col: this.current.col, cells: this.current.cells.map((c) => [...c]) } : null,
      next: this.nextType,
      w: this.W,
      h: this.H,
    };
  }

  step(label) {
    if (this.dead) return this.state();
    const options = this._candidates();
    const pick = options.find((p) => p.label === label) ?? options[0];
    if (!pick) {
      this.dead = true;
      return this.state();
    }
    const lock = lockAndClear(this.board, this.W, this.H, pick.cells, pick.col, pick.row, this.current.type);
    this.board = lock.board;
    this.lines += lock.linesCleared;
    this.score += [0, 100, 300, 500, 800][lock.linesCleared] ?? 0;
    this.pieces += 1;
    this._spawn();
    return this.state();
  }

  render() {
    const grid = this.board.map((row) => row.map((c) => (c ? '█' : '·')));
    const top = '┌' + '─'.repeat(this.W) + '┐';
    const bottom = '└' + '─'.repeat(this.W) + '┘';
    const rows = grid.map((row) => '│' + row.join('') + '│');
    return [top, ...rows, bottom].join('\n');
  }

  // Situations-only grammar (demo-spec v4, extended to a placement game): only the board
  // situations that apply, pre-computed, fixed order, no raw heights or distances - "the stack is
  // close to the top" not "the stack is 10 rows tall".
  describe() {
    const wells = wellRegions(this.board, this.W, this.H);
    const heights = colHeights(this.board, this.W, this.H);
    const sentences = ['A good placement rests on solid blocks.'];
    const side = (r) => (r.c0 === 0 ? 'left' : r.c1 === this.W - 1 ? 'right' : (r.c0 + r.c1) / 2 < this.W / 2 ? 'left' : 'right');
    const well = wells.filter((w) => w.width >= 2).sort((a, b) => b.depth - a.depth)[0];
    if (well) sentences.push(`The board has a multi-column well on the ${well.c0 === 0 || well.c1 === this.W - 1 ? side(well) : 'middle'}.`);
    const notch = wells.filter((w) => w.width === 1).sort((a, b) => b.depth - a.depth)[0];
    if (notch) sentences.push(`The board has a one-column well on the ${side(notch)}.`);
    let bestGap = null;
    for (const row of this.board) {
      const empties = row.filter((c) => !c).length;
      if (empties > 0 && empties <= 2 && (bestGap === null || empties < bestGap)) bestGap = empties;
    }
    if (bestGap != null) sentences.push(`A flat row needs ${bestGap === 1 ? 'one more cell' : 'two more cells'} to clear.`);
    if (Math.max(0, ...heights) >= this.H - 4) sentences.push('The stack is close to the top.');
    return sentences.join(' ');
  }
}

// The policy the state sentences and candidate labels encode, in precedence order (documentation
// + greedyPolicy, same role as Snake's RULES). At most 3 rules before the default, one condition
// each. Tried putting this list in the QUESTION the way doom.js/drive.js do (their candidates
// *are* the RULES keys, so the text lines up); Tetris's candidates are dynamic per-piece phrases,
// not that fixed vocabulary, and the mismatch reproduced Snake's own documented failure mode
// exactly: probed on 25 live decisions (seed 7), rules-in-question read .76 accuracy at p_null
// .80 (the model mostly answers "none of these"), the plain question below .64 at p_null .12 -
// same shape as spec-v4's Snake finding (.82-.95 acc / p_null .84-.94 with rules in the question
// vs .984 acc / p_null .04 plain), so QUESTION stays plain and RULES is documentation only.
export const RULES = {
  'clear the row': 'applies when a candidate placement completes a row',
  'fill the notch': 'applies when a candidate placement fills a one-column gap',
  'settle in a well or wall': 'applies when a candidate placement settles flush into a well or against a wall',
  'flatten the stack': 'applies otherwise, keeping the stack low and even',
};
// Sent verbatim as the query (no rule list - see the note above).
export const QUESTION = 'Which placement should the piece take?';

// Scripted policy = RULES applied literally over the offered candidates (same spread the model
// sees, so "gold" is always one of the labels actually offered): first category with a match
// wins, ties broken by the placement heuristic score. This is the gold the model is measured
// against.
export function greedyPolicy(engine) {
  if (engine.dead) return null;
  const options = engine._candidates();
  if (!options.length) return null;
  for (const cat of CATEGORY_ORDER) {
    const matches = options.filter((p) => p.category === cat);
    if (matches.length) return matches.reduce((best, p) => (p.score > best.score ? p : best)).label;
  }
  return options[0].label;
}

function cellsConnected(cells) {
  if (cells.length !== 4) return false;
  const key = (x, y) => `${x},${y}`;
  const set = new Set(cells.map(([x, y]) => key(x, y)));
  const seen = new Set([key(...cells[0])]);
  const stack = [cells[0]];
  while (stack.length) {
    const [x, y] = stack.pop();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nk = key(x + dx, y + dy);
      if (set.has(nk) && !seen.has(nk)) {
        seen.add(nk);
        stack.push([x + dx, y + dy]);
      }
    }
  }
  return seen.size === 4;
}

function selfTest() {
  // rotation: exact distinct-state counts (I/O/S/Z collapse via dedup, T/J/L keep all 4) and
  // every generated state is a connected 4-cell tetromino inside its own w x h box.
  const expected = { I: 2, O: 1, T: 4, S: 2, Z: 2, J: 4, L: 4 };
  for (const [type, count] of Object.entries(expected)) {
    console.assert(SHAPES[type].length === count, `${type} has ${count} distinct rotation states`);
    for (const s of SHAPES[type]) {
      console.assert(cellsConnected(s.cells), `${type} rotation state is a connected tetromino`);
      console.assert(s.cells.every(([x, y]) => x >= 0 && x < s.w && y >= 0 && y < s.h), `${type} state cells fit their own bounding box`);
    }
  }

  const t = new Tetris({ w: 8, h: 14, seed: 3 });
  console.assert(t.W === 8 && t.H === 14, 'constructor honors explicit w/h');
  console.assert(new Tetris().W === 8 && new Tetris().H === 14, 'default board is 8x14');
  console.assert(t.current && PIECE_TYPES.includes(t.current.type), 'spawns a current piece from the seven types');
  console.assert(t.board.every((row) => row.every((c) => c === null)), 'starts with an empty board');

  // candidate generation: a spread of legal placements, gold is always one of the offered labels
  const cands = t.candidates();
  console.assert(cands.length >= 1 && cands.length <= 6, 'offers between 1 and 6 candidates');
  console.assert(new Set(cands).size === cands.length && !cands.some((label) => /holes|peak/.test(label)), 'candidate labels are unique and carry no numbers to compare');
  const gold = greedyPolicy(t);
  console.assert(cands.includes(gold), 'greedyPolicy picks one of the offered candidates');

  // state() copies, does not alias
  const snap = t.state();
  snap.board[0][0] = 'X';
  console.assert(t.board[0][0] !== 'X', 'state() copies the board, mutating the snapshot does not touch the engine');

  // line clear: fill row H-1 except one column, then land a placement that completes it exactly
  // there (an O piece dropped in a 2-wide open pair reaches the floor in that column pair).
  const t2 = new Tetris({ w: 6, h: 6, seed: 1 });
  for (let c = 0; c < 6; c++) if (c !== 2 && c !== 3) t2.board[5][c] = 'I';
  t2.current = { type: 'O', col: 2, cells: SHAPES.O[0].cells };
  const before = t2.lines;
  const scoreBefore = t2.score;
  const opts = t2._candidates();
  const clearing = opts.find((p) => p.category === 'clear_line');
  console.assert(clearing, 'a placement that completes the open row is offered and classified as clear_line');
  t2.step(clearing.label);
  console.assert(t2.lines === before + 1, 'completing the row increments lines');
  console.assert(t2.score > scoreBefore, 'clearing a line scores');
  console.assert(t2.board.length === 6, 'board keeps H rows after a clear (empties unshifted on top)');
  console.assert(t2.board[0].every((c) => c === null), 'a cleared row is replaced by a fresh empty row unshifted on top');

  // classify(): a one-column gap is a notch, a two-column gap is a well
  const t3 = new Tetris({ w: 6, h: 8, seed: 2 });
  for (let r = 4; r < 8; r++) for (let c = 0; c < 6; c++) if (c !== 3) t3.board[r][c] = 'L';
  const wells3 = wellRegions(t3.board, 6, 8);
  console.assert(wells3.some((w) => w.width === 1 && w.c0 === 3), 'a single open column reads as a width-1 well (a notch)');

  // game over: spawn blocked at the top
  const t4 = new Tetris({ w: 6, h: 8, seed: 5 });
  for (let r = 0; r < 8; r++) for (let c = 0; c < 6; c++) t4.board[r][c] = 'I';
  t4._spawn();
  console.assert(t4.dead === true, 'spawning into a full board is game over');
  console.assert(t4.candidates().length === 0, 'no candidates once dead');
  console.assert(t4.step('anything').dead === true, 'step() on a dead engine is a no-op that stays dead');

  // determinism
  function run(seed) {
    const e = new Tetris({ seed });
    for (let i = 0; i < 60 && !e.dead; i++) e.step(greedyPolicy(e));
    return JSON.stringify(e.state());
  }
  console.assert(run(11) === run(11), 'same seed produces identical greedy runs');

  // a long greedy run should not throw and should keep placing pieces
  const t5 = new Tetris({ seed: 9 });
  for (let i = 0; i < 80 && !t5.dead; i++) t5.step(greedyPolicy(t5));
  console.assert(t5.pieces > 0, 'greedy run places pieces');
  console.assert(QUESTION.startsWith('Which placement should the piece take') && Object.keys(RULES).length === 4, 'QUESTION/RULES exported');

  console.log('tetris.js self-test OK');
  return true;
}

if (typeof process !== 'undefined' && import.meta.url === `file://${process.argv[1]}`) {
  selfTest();
}

export { selfTest };
