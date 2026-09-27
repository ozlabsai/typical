"""Architecture diagram for Typical, generated so every coordinate is computed.

Hand-placing y-values against a programmatically stacked layer column does not stay
aligned; the first version had the tap line 80px off its own layer. Everything here
derives from LAYERS/TAP/LORA and the stack geometry.
"""
N, TAP, LORA = 28, 20, 8                      # Qwen3-1.7B-Base; tap 20/28 = 71%; LoRA on top 8 kept
W, H = 1320, 900
SX, SW = 150, 430                             # stack x, width
BH, BG = 15, 2.4                              # bar height, gap
SB = 690                                      # stack bottom (L1 baseline)
def ly(i): return SB - (i - 1) * (BH + BG)    # top-y of layer i
STACK_TOP = ly(N)

o = []
A = o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  f'font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
A('''<defs>
<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker>
<marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#b45309"/></marker>
<marker id="ag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#0d9488"/></marker>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#16a34a"/></marker>
</defs>''')
A(f'<rect width="{W}" height="{H}" fill="#fff"/>')
A('<text x="40" y="42" font-size="22" font-weight="600" fill="#0f172a">Typical — the decision pass</text>')
A('<text x="40" y="64" font-size="13" fill="#475569">State in, typed probabilities out. One causal forward pass; no token is generated.</text>')

# ---- layer stack ----
A(f'<text x="{SX}" y="{STACK_TOP-34}" font-size="12.5" font-weight="600" fill="#0f172a">Backbone — Qwen3-1.7B-Base, 28 layers, d = 2048</text>')
for i in range(1, N + 1):
    y = ly(i)
    if i <= N - LORA - (N - TAP):  st = ('#e2e8f0', '#94a3b8', '', '1')          # frozen kept
    elif i <= TAP:                 st = ('#ccfbf1', '#0d9488', '', '1')          # LoRA kept
    else:                          st = ('#fafafa', '#cbd5e1', ' stroke-dasharray="4 3"', '0.7')
    f, s, d, op = st
    A(f'<rect x="{SX}" y="{y:.1f}" width="{SW}" height="{BH}" rx="2.5" fill="{f}" stroke="{s}" stroke-width="1"{d} opacity="{op}"/>')
for i, lab in ((1,'L1'), (TAP-LORA+1,f'L{TAP-LORA+1}'), (TAP,f'L{TAP}'), (TAP+1,f'L{TAP+1}'), (N,f'L{N}')):
    A(f'<text x="{SX-8}" y="{ly(i)+11.5:.1f}" font-size="10" fill="#64748b" text-anchor="end">{lab}</text>')

# group brackets (left of the labels)
def bracket(i_lo, i_hi, x, color, title, sub):
    top, bot = ly(i_hi), ly(i_lo) + BH
    A(f'<path d="M{x+10},{top:.1f} H{x} V{bot:.1f} H{x+10}" fill="none" stroke="{color}" stroke-width="1.3"/>')
    mid = (top + bot) / 2
    A(f'<text x="{x-6}" y="{mid-3:.1f}" font-size="11.5" font-weight="600" fill="{color}" text-anchor="end">{title}</text>')
    A(f'<text x="{x-6}" y="{mid+12:.1f}" font-size="10.5" fill="{color}" text-anchor="end">{sub}</text>')
bracket(1, TAP-LORA, 118, '#64748b', 'L1–L12', 'frozen')
bracket(TAP-LORA+1, TAP, 118, '#0d9488', 'L13–L20', 'LoRA r=16')
bracket(TAP+1, N, 118, '#94a3b8', 'L21–L28', 'discarded')

# ---- tap line, exactly at the top of layer TAP ----
TAPY = ly(TAP) - 5
A(f'<line x1="{SX-58}" y1="{TAPY:.1f}" x2="{SX+SW+150}" y2="{TAPY:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 4"/>')
A(f'<text x="{SX+SW+156}" y="{TAPY-6:.1f}" font-size="12.5" font-weight="700" fill="#dc2626">TAP — read here, at layer {TAP} of {N} ({TAP/N:.0%})</text>')
A(f'<text x="{SX+SW+156}" y="{TAPY+10:.1f}" font-size="10.5" fill="#dc2626">the final layer is optimised for next-token prediction,</text>')
A(f'<text x="{SX+SW+156}" y="{TAPY+24:.1f}" font-size="10.5" fill="#dc2626">which is not the best representation for deciding</text>')

# ---- token lanes ----
TY = 742
lanes = [(SX,      168, '#eff6ff', '#3b82f6', '#1e40af', 'State prefix', 'encoded ONCE, KV-cached', 'a'),
         (SX+176,  104, '#f0fdfa', '#0d9488', '#115e59', 'Question',     'asked at call time',       'ag'),
         (SX+288,  150, '#fffbeb', '#b45309', '#92400e', 'Candidates',   'defined at call time',     'ao'),
         (SX+446,   84, '#f1f5f9', '#334155', '#0f172a', 'terminal',     'decision position',        'a')]
A(f'<text x="{SX}" y="{TY-14}" font-size="12.5" font-weight="600" fill="#0f172a">Token sequence — one causal pass</text>')
for x, w, fill, stroke, tc, t1, t2, mk in lanes:
    A(f'<rect x="{x}" y="{TY}" width="{w}" height="46" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    A(f'<text x="{x+10}" y="{TY+19}" font-size="11.5" font-weight="600" fill="{tc}">{t1}</text>')
    A(f'<text x="{x+10}" y="{TY+35}" font-size="10" fill="{tc}">{t2}</text>')
    A(f'<line x1="{x+w/2}" y1="{TY-2}" x2="{x+w/2}" y2="{ly(1)+BH+8:.1f}" stroke="{stroke}" stroke-width="1.3" marker-end="url(#{mk})"/>')
A(f'<text x="{SX}" y="{TY+70}" font-size="10.5" fill="#64748b">O(state) once, then O(question + candidates) per decision — ~22 ms state encode, ~2.7 ms per further question at K ≤ 32.</text>')

# ---- pickups from the tap ----
CX, HX = SX + 288 + 75, SX + 446 + 42
A(f'<line x1="{CX}" y1="{TAPY:.1f}" x2="{CX}" y2="{STACK_TOP-24:.1f}" stroke="#b45309" stroke-width="1.6" marker-end="url(#ao)"/>')
A(f'<line x1="{HX}" y1="{TAPY:.1f}" x2="{HX}" y2="{STACK_TOP-24:.1f}" stroke="#334155" stroke-width="1.6" marker-end="url(#a)"/>')
A(f'<text x="{CX-4}" y="{STACK_TOP-30:.1f}" font-size="10.5" fill="#b45309" text-anchor="middle">c₃ᵏ</text>')
A(f'<text x="{HX+4}" y="{STACK_TOP-30:.1f}" font-size="10.5" fill="#334155" text-anchor="middle">h_D</text>')
A(f'<text x="{SX}" y="{STACK_TOP-14:.1f}" font-size="10.5" fill="#475569">each candidate\'s own contextual state, and the terminal decision state</text>')

# ---- readout / heads / output, all right of the stack ----
RX, RY = 700, 96
A(f'<rect x="{RX}" y="{RY}" width="404" height="150" rx="8" fill="#fef2f2" stroke="#dc2626" stroke-width="1.6"/>')
A(f'<text x="{RX+16}" y="{RY+24}" font-size="13.5" font-weight="700" fill="#991b1b">N3 contextual readout</text>')
for k, t in enumerate(['z-score both sides, then LayerNorm + Linear → 256-d',
                       'u = proj_h(h_D)        v_k = proj_3(c₃ᵏ)',
                       's_k = ⟨u,v_k⟩·d^−½ + w(u⊙v_k) + res(u⊙v_k)',
                       'bilinear + linear + GELU-MLP residual, per candidate']):
    mono = ' font-family="ui-monospace,Menlo,monospace"' if k in (1, 2) else ''
    A(f'<text x="{RX+16}" y="{RY+46+k*19}" font-size="11"{mono} fill="#7f1d1d">{t}</text>')
A(f'<text x="{RX+16}" y="{RY+130}" font-size="10.5" font-style="italic" fill="#991b1b">A candidate-blind state fails here: priors and calibration transfer, question-conditioned knowledge does not.</text>')

A(f'<rect x="{RX}" y="{RY+164}" width="404" height="44" rx="7" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.4"/>')
A(f'<text x="{RX+16}" y="{RY+182}" font-size="12" font-weight="700" fill="#3730a3">Factored ∅ gate — abstention is a gate, not a candidate</text>')
A(f'<text x="{RX+16}" y="{RY+199}" font-size="10.3" fill="#3730a3">MLP(4 + 4·d_v → 256 → 1) over set statistics; preserves pairwise odds among real candidates</text>')

heads = [('Choice', 'categorical', 'softmax over your K labels'),
         ('Score',  'ordinal-smoothed', 'τ = 0.7; levels are ordered'),
         ('Noul',   'Bernoulli on h_D', 'exactly order-invariant')]
for k, (t, s1, s2) in enumerate(heads):
    x = RX + k * 137
    A(f'<rect x="{x}" y="{RY+226}" width="128" height="72" rx="7" fill="#f8fafc" stroke="#334155" stroke-width="1.3"/>')
    A(f'<text x="{x+13}" y="{RY+247}" font-size="12.5" font-weight="700" fill="#0f172a">{t}</text>')
    A(f'<text x="{x+13}" y="{RY+265}" font-size="10.3" fill="#475569">{s1}</text>')
    A(f'<text x="{x+13}" y="{RY+281}" font-size="10" fill="#64748b">{s2}</text>')
A(f'<text x="{RX}" y="{RY+318}" font-size="10.5" fill="#475569">Typed heads are contracts on the output distribution, not three separate models.</text>')

OX = RX + 424
A(f'<rect x="{OX}" y="{RY}" width="172" height="150" rx="8" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.6"/>')
A(f'<text x="{OX+15}" y="{RY+24}" font-size="13" font-weight="700" fill="#166534">Output</text>')
for k, (lab, p) in enumerate([('refund','.82'),('replacement','.11'),('repair','.05')]):
    A(f'<text x="{OX+15}" y="{RY+48+k*19}" font-size="11" font-family="ui-monospace,Menlo,monospace" fill="#14532d">{lab:<12}{p}</text>')
A(f'<text x="{OX+15}" y="{RY+112}" font-size="11" font-family="ui-monospace,Menlo,monospace" font-weight="700" fill="#166534">∅ (none)    .02</text>')
A(f'<text x="{OX+15}" y="{RY+134}" font-size="10.3" fill="#166534">nothing to parse</text>')
A(f'<line x1="{RX+404}" y1="{RY+75}" x2="{OX-4}" y2="{RY+75}" stroke="#16a34a" stroke-width="1.8" marker-end="url(#ar)"/>')
A(f'<line x1="{CX+6}" y1="{STACK_TOP-40:.1f}" x2="{RX-6}" y2="{RY+120}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="3 3"/>')

A(f'<text x="40" y="{H-18}" font-size="10.3" fill="#94a3b8">Depths shown for typical-small (Qwen3-1.7B-Base, tap 20/28). typical-medium taps 23/36 on Qwen3.5-4B; typical-large-preview taps 28/40 on Qwen3-14B.</text>')
A('</svg>')
open("figures/fig_architecture_detail.svg","w").write("\n".join(o))
print("wrote figures/fig_architecture_detail.svg")
print(f"  stack spans y={STACK_TOP:.0f}..{SB+BH:.0f}, tap line at y={TAPY:.0f} (top of L{TAP})")
