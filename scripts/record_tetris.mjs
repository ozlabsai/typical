#!/usr/bin/env node
// Records a seeded, deterministic playthrough of Tetris driven by the live server, one tick per
// piece (a full hard-drop decision, not a keystroke). Modelled on scripts/record_snake.mjs; used
// by site/js/games/render-tetris.js's static-mode "Model" playback so the demo works without a
// server. Requires server.py running on :8787 (uv run uvicorn server:app --port 8787).
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { Tetris, greedyPolicy, QUESTION } from '../site/js/games/tetris.js';

const SERVER = 'http://localhost:8787';
const SOLO = process.argv.includes('--solo');
const SEED = SOLO ? 8 : 7;
const MAX_TICKS = 400; // pieces, not frames
const BOARD = SOLO ? { w: 10, h: 20 } : { w: 8, h: 14 };
const OUTPUT = SOLO ? 'tetris-solo.json' : 'tetris.json';
const OUTPUT_URL = new URL(`../site/data/replays/${OUTPUT}`, import.meta.url);

function checkReplay({ frames, summary }) {
  const engine = new Tetris({ ...BOARD, seed: SEED });
  for (const frame of frames.filter((f) => f.move)) {
    const state = engine.state();
    assert.equal(frame.tick, state.pieces);
    assert.deepEqual(frame.state.board, state.board);
    assert.deepEqual(frame.candidates, engine.candidates());
    assert.ok(frame.candidates.includes(frame.move));
    engine.step(frame.move);
  }
  assert.equal(engine.dead, true);
  assert.equal(engine.lines, summary.lines_cleared);
  console.log(`[tetris] replay check OK: ${summary.decisions} decisions, ${summary.lines_cleared} lines`);
}

async function decide(state, queries) {
  const res = await fetch(`${SERVER}/api/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'typical-small', state, queries }),
  });
  if (!res.ok) throw new Error(`decide failed: ${res.status} ${await res.text()}`);
  return res.json();
}

async function main() {
  const engine = new Tetris({ ...BOARD, seed: SEED });
  const frames = [];
  let tick = 0;

  for (; tick < MAX_TICKS; tick++) {
    const candidates = engine.candidates();
    if (candidates.length === 0) {
      // spawn blocked - game over; record the death frame so the replay flashes it
      frames.push({ tick, state: engine.state(), desc: engine.describe(), candidates: [], probs: [], p_null: 0, move: null, ms: 0 });
      break;
    }

    const desc = engine.describe();
    const gold = greedyPolicy(engine);
    let move, probs, p_null, ms;
    if (candidates.length === 1) {
      [move, probs, p_null, ms] = [candidates[0], { [candidates[0]]: 1 }, 0, 0];
    } else {
      const t0 = Date.now();
      const res = await decide(desc, [{ type: 'choice', question: QUESTION, labels: candidates }]);
      ms = Date.now() - t0;
      const r = res.results[0];
      probs = r.probs;
      p_null = r.p_null;
      move = candidates.reduce((best, m) => ((r.probs[m] ?? 0) > (r.probs[best] ?? 0) ? m : best), candidates[0]);
    }

    const linesBefore = engine.lines;
    frames.push({ tick, state: engine.state(), desc, candidates, probs: candidates.map((m) => probs[m] ?? 0), p_null, move, ms, gold });
    engine.step(move);
    if (engine.lines > linesBefore) frames[frames.length - 1].linesCleared = engine.lines - linesBefore;
    if (engine.dead) {
      frames.push({ tick: tick + 1, state: engine.state(), desc: engine.describe(), candidates: [], probs: [], p_null: 0, move: null, ms: 0 });
      break;
    }
  }

  const decided = frames.filter((f) => f.move);
  const summary = {
    game: 'tetris',
    seed: SEED,
    ...BOARD,
    pieces: frames.length,
    lines_cleared: frames[frames.length - 1].state.lines,
    decisions: decided.length,
    rule_agreement: decided.filter((f) => f.move === f.gold).length,
    mean_p_null: decided.reduce((s, f) => s + f.p_null, 0) / (decided.length || 1),
    mean_ms: decided.reduce((s, f) => s + f.ms, 0) / (decided.length || 1),
  };
  const replay = { frames, summary };
  checkReplay(replay);
  fs.writeFileSync(OUTPUT_URL, JSON.stringify(replay));
  console.log(`[tetris] pieces=${frames.length} lines_cleared=${summary.lines_cleared} rule_agreement=${summary.rule_agreement}/${summary.decisions}`);
  console.log(`[tetris] mean p_null=${summary.mean_p_null.toFixed(3)} mean ms=${summary.mean_ms.toFixed(1)}`);
  console.log(`STATS_JSON ${JSON.stringify(summary)}`);
}

if (process.argv.includes('--check')) checkReplay(JSON.parse(fs.readFileSync(OUTPUT_URL)));
else main();
