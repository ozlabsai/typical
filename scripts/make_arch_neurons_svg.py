"""Neuron-level architecture diagram for Typical.

Every dimension is the real one (Qwen3-1.7B-Base: d=2048, ffn=6144, 16 heads / 8 kv,
head_dim=128; head: d_proj=256). Node counts are SAMPLED for drawing -- each column
says how many units it really has -- and connections are drawn densely so the picture
reads as a network rather than a block diagram.
"""
import math
W, H = 1700, 790
o=[]; A=o.append
A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'font-family="ui-sans-serif,-apple-system,Segoe UI,Helvetica,Arial,sans-serif">')
A('''<defs>
<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#334155"/></marker>
<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#16a34a"/></marker>
<marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#b45309"/></marker>
</defs>''')
A(f'<rect width="{W}" height="{H}" fill="#fff"/>')
A('<text x="36" y="40" font-size="23" font-weight="600" fill="#0f172a">Typical — neurons, connections, and where we tap</text>')
A('<text x="36" y="62" font-size="12.5" fill="#475569">Real dimensions throughout. Circles are sampled units; each column is labelled with its true width.</text>')

def col(cx, cy, n, r=6.5, gap=19, fill="#e2e8f0", stroke="#94a3b8", ell=False):
    """vertical column of n circles centred on cy; returns their y positions"""
    ys=[cy + (i-(n-1)/2)*gap for i in range(n)]
    for i,y in enumerate(ys):
        if ell and i==n//2:
            A(f'<text x="{cx}" y="{y+4}" font-size="13" fill="#94a3b8" text-anchor="middle">⋮</text>')
        else:
            A(f'<circle cx="{cx}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    return ys

def wire(xs, ys_a, xt, ys_b, color="#cbd5e1", w=0.5, op=0.75, skip_a=(), skip_b=()):
    for i,ya in enumerate(ys_a):
        if i in skip_a: continue
        for j,yb in enumerate(ys_b):
            if j in skip_b: continue
            A(f'<line x1="{xs+7}" y1="{ya:.1f}" x2="{xt-7}" y2="{yb:.1f}" stroke="{color}" stroke-width="{w}" opacity="{op}"/>')

def lab(x, y, t, s=10.5, c="#475569", anc="middle", w=None):
    A(f'<text x="{x}" y="{y}" font-size="{s}" fill="{c}" text-anchor="{anc}"'
      + (f' font-weight="{w}"' if w else '') + f'>{t}</text>')

# ───────────────── panel 1: one transformer block, expanded ─────────────────
A('<rect x="28" y="86" width="720" height="470" rx="10" fill="#fbfdff" stroke="#cbd5e1" stroke-width="1.2"/>')
lab(48, 112, 'One decoder block (×20 kept) — layers 13–20 carry LoRA', 13, '#0f172a', 'start', 700)
BY = 320
x_in, x_q, x_at, x_f1, x_out = 110, 250, 390, 530, 680
ys_in = col(x_in, BY, 7, fill="#e2e8f0", ell=True)
ys_q  = col(x_q,  BY, 5, fill="#dbeafe", stroke="#3b82f6", ell=True)
ys_at = col(x_at, BY, 5, fill="#dbeafe", stroke="#3b82f6", ell=True)
ys_f1 = col(x_f1, BY, 7, fill="#fae8ff", stroke="#a855f7", ell=True)
ys_out= col(x_out,BY, 7, fill="#e2e8f0", ell=True)
wire(x_in, ys_in, x_q, ys_q, "#93c5fd", .55, .8, skip_a=(3,), skip_b=(2,))
wire(x_q, ys_q, x_at, ys_at, "#93c5fd", .55, .55, skip_a=(2,), skip_b=(2,))
wire(x_at, ys_at, x_f1, ys_f1, "#d8b4fe", .55, .8, skip_a=(2,), skip_b=(3,))
wire(x_f1, ys_f1, x_out, ys_out, "#d8b4fe", .55, .8, skip_a=(3,), skip_b=(3,))
lab(x_in, BY+92, 'hidden', 11, '#334155', w=600); lab(x_in, BY+107, 'd = 2048')
lab(x_q, BY+92, 'q / k / v', 11, '#1e40af', w=600); lab(x_q, BY+107, '16 q · 8 kv heads', 10, '#1e40af'); lab(x_q, BY+121, 'head_dim 128', 10, '#1e40af')
lab(x_at, BY+92, 'attention out', 11, '#1e40af', w=600); lab(x_at, BY+107, 'o_proj → 2048', 10, '#1e40af')
lab(x_f1, BY+92, 'MLP', 11, '#7e22ce', w=600); lab(x_f1, BY+107, 'gate/up → 6144', 10, '#7e22ce'); lab(x_f1, BY+121, 'SwiGLU', 10, '#7e22ce')
lab(x_out, BY+92, 'hidden', 11, '#334155', w=600); lab(x_out, BY+107, 'down → 2048')
# residuals
A(f'<path d="M{x_in},{BY-118} C{(x_in+x_at)/2},{BY-168} {(x_in+x_at)/2},{BY-168} {x_at},{BY-118}" fill="none" stroke="#64748b" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#a)"/>')
lab((x_in+x_at)/2, BY-172, 'residual + RMSNorm', 10, '#64748b')
A(f'<path d="M{x_at},{BY+128} C{(x_at+x_out)/2},{BY+178} {(x_at+x_out)/2},{BY+178} {x_out},{BY+128}" fill="none" stroke="#64748b" stroke-width="1.1" stroke-dasharray="4 3" marker-end="url(#a)"/>')
lab((x_at+x_out)/2, BY+196, 'residual + RMSNorm', 10, '#64748b')
# LoRA annotation
A(f'<rect x="196" y="{BY-152}" width="360" height="26" rx="5" fill="#ccfbf1" stroke="#0d9488" stroke-width="1.1"/>')
lab(376, BY-134, 'LoRA r=16 on q,k,v,o,gate,up,down — 0.6% of weights trained', 10.5, '#115e59')

# ───────────────── panel 2: depth + tap ─────────────────
A('<rect x="768" y="86" width="256" height="470" rx="10" fill="#fbfdff" stroke="#cbd5e1" stroke-width="1.2"/>')
lab(788, 112, 'Depth', 13, '#0f172a', 'start', 700)
N, TAP = 28, 20
bh, bg, sb = 11.5, 2.2, 520
def ly(i): return sb - (i-1)*(bh+bg)
for i in range(1, N+1):
    y=ly(i)
    if i<=12:   f,s,d,op = "#e2e8f0","#94a3b8","","1"
    elif i<=TAP:f,s,d,op = "#ccfbf1","#0d9488","","1"
    else:       f,s,d,op = "#fafafa","#cbd5e1",' stroke-dasharray="3 2.5"',"0.75"
    A(f'<rect x="812" y="{y:.1f}" width="150" height="{bh}" rx="2" fill="{f}" stroke="{s}" stroke-width="1"{d} opacity="{op}"/>')
for i,t in ((1,'L1'),(13,'L13'),(TAP,'L20'),(N,'L28')):
    lab(806, ly(i)+9, t, 9.5, '#64748b', 'end')
ty = ly(TAP)-4
A(f'<line x1="778" y1="{ty:.1f}" x2="1010" y2="{ty:.1f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 3.5"/>')
lab(894, ty-9, 'TAP · 20 of 28 (71%)', 11, '#dc2626', w=700)
lab(894, ly(N)-14, 'L21–L28 discarded', 10, '#94a3b8')
lab(894, 548, 'the last layer is tuned for', 9.5, '#64748b')
lab(894, 560, 'next-token, not for deciding', 9.5, '#64748b')

# ───────────────── panel 3: the N3 head, fully wired ─────────────────
A(f'<rect x="1044" y="86" width="628" height="470" rx="10" fill="#fff7f7" stroke="#dc2626" stroke-width="1.4"/>')
lab(1064, 112, 'N3 contextual readout — the actual head', 13, '#991b1b', 'start', 700)
hx1, hx2, hx3 = 1112, 1258, 1408
ys_h = col(hx1, 210, 6, fill="#f1f5f9", stroke="#334155", ell=True)
ys_c = col(hx1, 400, 6, fill="#fed7aa", stroke="#b45309", ell=True)
ys_u = col(hx2, 210, 5, fill="#e0e7ff", stroke="#4f46e5", ell=True)
ys_v = col(hx2, 400, 5, fill="#e0e7ff", stroke="#4f46e5", ell=True)
wire(hx1, ys_h, hx2, ys_u, "#a5b4fc", .55, .85, skip_a=(2,), skip_b=(2,))
wire(hx1, ys_c, hx2, ys_v, "#a5b4fc", .55, .85, skip_a=(2,), skip_b=(2,))
lab(hx1, 128, 'h_D', 11.5, '#0f172a', w=700); lab(hx1, 142, 'terminal state', 9.5, '#334155'); lab(hx1, 154, '2048', 9.5, '#64748b')
lab(hx1, 318, 'c₃ᵏ', 11.5, '#b45309', w=700); lab(hx1, 332, "candidate k's own", 9.5, '#92400e'); lab(hx1, 344, 'contextual state · 2048', 9.5, '#92400e')
lab(hx2, 128, 'u = proj_h', 11, '#3730a3', w=600); lab(hx2, 142, 'LayerNorm+Linear', 9.5, '#4338ca'); lab(hx2, 154, '2048 → 256', 9.5, '#4338ca')
lab(hx2, 318, 'v_k = proj_3', 11, '#3730a3', w=600); lab(hx2, 332, 'LayerNorm+Linear', 9.5, '#4338ca'); lab(hx2, 344, '2048 → 256', 9.5, '#4338ca')
# elementwise product node
A(f'<circle cx="{hx3}" cy="305" r="17" fill="#fff" stroke="#334155" stroke-width="1.5"/>')
lab(hx3, 310, '⊙', 16, '#0f172a')
for y in ys_u:
    A(f'<line x1="{hx2+7}" y1="{y:.1f}" x2="{hx3-17}" y2="{300:.1f}" stroke="#94a3b8" stroke-width=".5" opacity=".7"/>')
for y in ys_v:
    A(f'<line x1="{hx2+7}" y1="{y:.1f}" x2="{hx3-17}" y2="{310:.1f}" stroke="#94a3b8" stroke-width=".5" opacity=".7"/>')
lab(hx3, 338, 'u ⊙ v_k · 256', 9.5, '#475569')
# three scoring paths
paths=[(232,'⟨u,v_k⟩ · d^−½','dot product'),(288,'w( · )','Linear 256→1'),(344,'res( · )','256→GELU→1')]
for y,t1,t2 in paths:
    A(f'<rect x="1462" y="{y-16}" width="104" height="32" rx="5" fill="#f8fafc" stroke="#334155" stroke-width="1.1"/>')
    lab(1514, y-2, t1, 9.5, '#0f172a', w=600); lab(1514, y+10, t2, 8.5, '#64748b')
    A(f'<line x1="{hx3+17}" y1="305" x2="1458" y2="{y}" stroke="#94a3b8" stroke-width=".9" marker-end="url(#a)"/>')
A('<circle cx="1514" cy="410" r="16" fill="#fff" stroke="#16a34a" stroke-width="1.6"/>')
lab(1514, 415, '+', 16, '#16a34a')
for y,_,_ in paths:
    A(f'<line x1="1514" y1="{y+16}" x2="1514" y2="394" stroke="#86efac" stroke-width="1.1"/>')
lab(1514, 444, 's_k · one logit per candidate', 10, '#166534', w=600)
# null gate
A('<rect x="1064" y="466" width="360" height="40" rx="6" fill="#eef2ff" stroke="#4f46e5" stroke-width="1.2"/>')
lab(1078, 483, 'Factored ∅ gate: MLP(4 + 4·256 → 256 → 1)', 10.5, '#3730a3', 'start', 600)
lab(1078, 498, 'from set statistics — abstention is a gate, not a candidate', 9.5, '#3730a3', 'start')
A(f'<line x1="1424" y1="486" x2="1490" y2="430" stroke="#818cf8" stroke-width="1.1" marker-end="url(#a)"/>')
lab(1064, 528, 'per-candidate scores + ∅ → softmax → calibrated distribution', 10, '#475569', 'start')

# ───────────────── bottom: tokens and typed heads ─────────────────
A(f'<rect x="28" y="586" width="996" height="150" rx="10" fill="#fbfdff" stroke="#cbd5e1" stroke-width="1.2"/>')
lab(48, 612, 'What enters the single causal pass', 13, '#0f172a', 'start', 700)
toks=[(60,300,'#eff6ff','#3b82f6','#1e40af','State prefix','encoded once · KV-cached · reused by every later question'),
      (372,180,'#f0fdfa','#0d9488','#115e59','Question','rendered into the suffix'),
      (564,260,'#fffbeb','#b45309','#92400e','Candidates','A. refund  B. replacement  C. repair'),
      (836,170,'#f1f5f9','#334155','#0f172a','terminal','gives h_D')]
for x,w_,f,s,tc,t1,t2 in toks:
    A(f'<rect x="{x}" y="640" width="{w_}" height="52" rx="6" fill="{f}" stroke="{s}" stroke-width="1.4"/>')
    lab(x+12, 660, t1, 11.5, tc, 'start', 600); lab(x+12, 677, t2, 9.5, tc, 'start')
lab(48, 716, 'Cost: O(state) paid once, then O(question + candidates) per decision — ~22 ms state encode, ~2.7 ms per further question at K ≤ 32.', 10.3, '#64748b', 'start')

A(f'<rect x="1044" y="586" width="628" height="150" rx="10" fill="#fbfdff" stroke="#cbd5e1" stroke-width="1.2"/>')
lab(1064, 612, 'Typed heads — contracts on the output distribution', 13, '#0f172a', 'start', 700)
th=[('Choice','softmax over your K labels','categorical'),
    ('Score','ordinal-smoothed τ = 0.7','levels are ordered'),
    ('Noul','Bernoulli w_n: Linear(2048→1)','exactly order-invariant')]
for k,(t,s1,s2) in enumerate(th):
    x=1064+k*204
    A(f'<rect x="{x}" y="632" width="192" height="64" rx="6" fill="#f8fafc" stroke="#334155" stroke-width="1.2"/>')
    lab(x+12, 652, t, 12, '#0f172a', 'start', 700); lab(x+12, 669, s1, 9.3, '#475569', 'start'); lab(x+12, 684, s2, 9, '#64748b', 'start')
lab(1064, 718, 'One backbone, one readout; the head changes what the distribution means.', 10.3, '#64748b', 'start')

# ───────────────── flow arrows between panels ─────────────────
A('<line x1="700" y1="666" x2="1036" y2="666" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 4"/>')
A(f'<line x1="962" y1="{ly(TAP)+6:.1f}" x2="1040" y2="300" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#a)"/>')
lab(1000, 286, 'tapped states', 9.5, '#dc2626')
A(f'<line x1="748" y1="320" x2="806" y2="320" stroke="#94a3b8" stroke-width="1.2" marker-end="url(#a)"/>')

A(f'<text x="36" y="{H-22}" font-size="10.3" fill="#94a3b8">Dimensions are Qwen3-1.7B-Base (typical-small): d=2048, ffn=6144, 16 query / 8 KV heads, head_dim 128; readout d_proj=256. Circle counts are illustrative; widths are labelled.</text>')
A('</svg>')
open("figures/fig_architecture_neurons.svg","w").write("\n".join(o))
print("wrote figures/fig_architecture_neurons.svg")
