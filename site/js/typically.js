// typically.js -- import a CSV of past decisions, build the dataset, compare base vs your model.
// Four steps, one page; each step unlocks the next. All model calls go through /api/typically/*.
import { bars } from './bars.js';
import { mountCopyButtons } from './copycode.js';

const $ = (id) => document.getElementById(id);
const BASE = 'typical-small';
const YESNO = ['yes', 'no', 'true', 'false', '1', '0'];
let csvText = '';
let pre = null; // last /preview response
let sampled = false; // true while the bundled Northwind sample is loaded
let decs = []; // [{column, question, type, labels, on}] straight from the step-2 rows

function h(tag, props = {}, ...kids) {
  const el = Object.assign(document.createElement(tag), props); // textContent only: cells are user text
  el.append(...kids);
  return el;
}

async function api(path, body) {
  const res = await fetch(`/api/typically/${path}`, body && {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || 'Something went wrong.');
  return data;
}

// ponytail: the questions local:co_a was trained on and one held-out ticket (first row of data_co_a/eval/a_oneliner.jsonl), baked in
// so the sample demo hits the trained wording; regenerate if the sample or data_co_a changes.
const SAMPLE = {
  model: 'local:co_a',
  questions: {
    team: 'Which team should handle this ticket?', escalate: 'Should we escalate this to a manager?',
    urgency: 'How urgent is this ticket?', refund: 'Can the agent refund this without approval?',
  },
  ticket: "Customer: Acme Supply (business plan, EU). Tickets from this customer in the last 30 days: 0. Shipment value: $53, shipped 23 days ago.\n\nYOUR COMPANY OVERBILLED US ON NW-2024106! AGAIN! This is fraud at this point. Refund the overcharge NOW or we're switching carriers tomorrow.",
};

function say(id, msg) { $(id).textContent = msg || ''; }

// ---- 1. import -------------------------------------------------------------------------
async function load(text, isSample = false) {
  sampled = isSample;
  say('s1-note', 'Reading…');
  try {
    pre = await api('preview', { csv_text: (csvText = text) });
  } catch (e) { return say('s1-note', e.message); }
  say('s1-note', `${pre.n} cases, ${pre.columns.length} columns.`);
  const t = h('table');
  t.append(h('thead', {}, h('tr', {}, ...pre.columns.map((c) => h('th', { textContent: c })))),
    h('tbody', {}, ...pre.rows.map((r) => h('tr', {}, ...pre.columns.map((c) => h('td', { textContent: r[c], title: r[c] }))))));
  $('preview').replaceChildren(t);
  showDecisions();
}

$('file').onchange = async (e) => e.target.files[0] && load(await e.target.files[0].text());
$('sample').onclick = async () => load(await (await fetch('data/typically_sample.csv')).text(), true);

// ---- 2. decisions ----------------------------------------------------------------------
// two yes/no-like values -> yes/no; whole numbers -> a scale; anything else -> pick one
function detect(vals) {
  if (vals.length === 2 && vals.every((v) => YESNO.includes(v.toLowerCase()))) return 'noul';
  return vals.every((v) => /^-?\d+$/.test(v)) ? 'score' : 'choice';
}
// default wording per answer type: "Which team?", "Should we escalate?", "How much urgency?"
const ASK = { choice: 'Which', noul: 'Should we', score: 'How much' };
const question = (column, type) => `${ASK[type]} ${column.replace(/_/g, ' ').toLowerCase()}?`;
const TYPES = { choice: 'Pick one', noul: 'Yes or no', score: 'A scale' };

function showDecisions() {
  // the case text is the column with the longest entries
  const len = (c) => Math.max(...pre.rows.map((r) => r[c].length));
  const text = pre.columns.reduce((a, c) => (len(c) > len(a) ? c : a));
  $('textcol').replaceChildren(...pre.columns.map((c) => h('option', { value: c, textContent: c, selected: c === text })));
  $('textcol').onchange = drawDecisions;
  drawDecisions();
  ['s2', 's3'].forEach((id) => ($(id).hidden = false));
  $('case').value = sampled ? SAMPLE.ticket : pre.rows[0][text];
  $('built').hidden = true;
  refreshModels();
}

function drawDecisions() {
  decs = pre.columns.filter((c) => c !== $('textcol').value).map((column) => {
    const vals = pre.values[column]; // absent when the column has too many different values to be a decision
    const type = vals ? detect(vals) : 'choice';
    const labels = type === 'score' ? [...vals].sort((a, b) => a - b) : vals; // same order the server trains on
    const d = { column, type, labels, question: (sampled && SAMPLE.questions[column]) || question(column, type), on: !!vals };
    const box = h('input', { type: 'checkbox', checked: d.on, disabled: !vals, ariaLabel: `Learn ${column}`, onchange: () => { d.on = box.checked; row.classList.toggle('off', !d.on); } });
    const q = h('input', { className: 'tryit-input', value: d.question, ariaLabel: `Question for ${column}`, oninput: () => (d.question = q.value) });
    q.style.margin = 0;
    const sel = h('select', { className: 'tryit-input', ariaLabel: `Answer type for ${column}`, onchange: () => {
      if (d.question === question(column, d.type)) q.value = d.question = question(column, sel.value); // keep an edited question
      d.type = sel.value;
    } },
      ...Object.entries(TYPES).map(([v, t]) => h('option', { value: v, textContent: t, selected: v === type })));
    sel.style.margin = 0;
    const row = h('div', { className: 'dec' + (d.on ? '' : ' off') }, box,
      h('b', { textContent: column }), q, vals ? sel : h('span', { className: 'note', textContent: 'too many different values' }));
    d.row = row;
    return d;
  });
  $('decs').replaceChildren(...decs.map((d) => d.row));
}

// ---- 3. teach --------------------------------------------------------------------------
const chosen = () => decs.filter((d) => d.on);

function balanceTable({ before, after }) {
  const t = h('table');
  t.append(h('thead', {}, h('tr', {}, ...['Decision', 'Answers before', 'Answers after'].map((x) => h('th', { textContent: x })))),
    h('tbody', {}, ...Object.keys(before).map((k) => h('tr', {}, h('td', { textContent: k }),
      ...[before, after].map((s) => h('td', { textContent: Object.entries(s[k]).map(([l, n]) => `${l} ${n}`).join(' · ') }))))));
  return t;
}

$('build').onclick = async () => {
  if (!chosen().length) return say('s3-note', 'Tick at least one decision above.');
  say('s3-note', 'Building…');
  try {
    const r = await api('build', {
      csv_text: csvText, text_col: $('textcol').value, name: $('name').value,
      decisions: chosen().map(({ column, question, type }) => ({ column, question, type })),
    });
    say('s3-note', '');
    $('counts').textContent = `${r.splits.train} examples to learn from · ${r.splits.val} to check as it learns · ${r.splits.import_oneliner} kept back to test.`;
    $('balance').replaceChildren(h('h3', { className: 't-eyebrow muted', textContent: 'Balance of answers in the learning examples' }), balanceTable(r.balance));
    $('cmd').textContent = r.command;
    $('built').hidden = false;
    mountCopyButtons($('built'));
  } catch (e) { say('s3-note', e.message); }
};

// ---- 4. try it -------------------------------------------------------------------------
async function refreshModels() {
  const { tuned } = await api('models');
  $('tuned').replaceChildren(...tuned.map((m) => h('option', { value: m, textContent: m.replace('local:', '') })));
  if (sampled && tuned.includes(SAMPLE.model)) $('tuned').value = SAMPLE.model;
  $('s4').hidden = false;
  say('s4-note', tuned.length ? '' : 'No tuned model yet. Train one in step 3 first.');
}

$('ask').onclick = async () => {
  if (!$('tuned').value || !chosen().length) return say('s4-note', 'Pick a model and tick at least one decision.');
  say('s4-note', 'Thinking… the first question loads the models and can take a minute.');
  try {
    const { models } = await api('compare', {
      state: $('case').value, models: [BASE, $('tuned').value],
      decisions: chosen().map(({ question, type, labels }) => ({ question, type, labels })),
    });
    say('s4-note', '');
    const cols = [[BASE, 'Standard Typical'], [$('tuned').value, 'Yours']].map(([m, title]) => {
      const col = h('div', {}, h('h3', { className: 't-eyebrow muted', textContent: title }));
      models[m].results.forEach((r, i) => {
        const bx = h('div');
        bars(bx, Object.entries(r.probs).map(([label, p]) => ({ label, p })));
        col.append(h('p', { textContent: chosen()[i].question, style: 'margin:14px 0 6px;font-weight:600' }), bx);
      });
      return col;
    });
    $('answers').replaceChildren(h('div', { className: 'side' }, ...cols));
  } catch (e) { say('s4-note', e.message); }
};
