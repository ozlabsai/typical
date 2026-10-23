"""Showcase figures for Typical (R044 → R062). Numbers copied from the committed result docs:
docs/research/R056_RESULTS.md, R060_RESULTS.md, R060PRE_BLEND_PROXY.md, hebrew/HEJEV_V0_RESULTS.md, hebrew/R062_RESULTS.md.
Run: uv run --with matplotlib python scripts/showcase_plots.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("docs/research/figures/showcase")
BLUE, ORANGE, GRAY, LIGHT = "#2a78d6", "#eb6834", "#8a8984", "#c9c8c3"   # Typical, Jev, others, faint
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8,
    "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titleweight": "semibold", "axes.titlesize": 13,
    "axes.titlecolor": INK,
})


def finish(fig, name, note):
    fig.text(0.01, 0.005, note, fontsize=8.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def wf_progress():
    rows = [("Frozen Gemma-4-12B (Cygnet readout)", .521, GRAY), ("R044 (old recipe)", .528, GRAY),
            ("Frozen base, our readout", .550, GRAY), ("Gemma + thinking (teacher, slow)", .612, LIGHT),
            ("R056: distilled single pass", .649, BLUE), ("R060: + GPT agent-trace labels", .648, BLUE)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(rows))
    for i, (_, v, c) in enumerate(rows):
        ax.plot([.50, v], [i, i], color=GRID, lw=1.2, zorder=1)
        ax.scatter(v, i, s=90 if c == BLUE else 60, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.text(v - .004, i + .28, f"{v:.3f}", ha="center", color=INK, fontsize=9.5)
    ax.axvline(.678, color=ORANGE, lw=2, ls="--")
    ax.text(.676, -.75, "Jev 1.13  .678", color=INK, fontsize=10, ha="right")
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.set_xlim(.50, .70)
    ax.set_ylim(-1, len(rows) - .4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("WorkflowEvals mean (equal-weight, 4 workflows, 705 cases)")
    fig.suptitle("Distillation lifted WorkflowEvals from .53 to .65 (Jev .68)", x=.01, ha="left",
                 fontsize=13, fontweight="semibold", color=INK)
    finish(fig, "01_workflowevals_progress", "Single-pass models unless noted. Source: R056/R060 result docs.")


def wf_per_workflow():
    wfs = ["Security", "Agent trace", "Invoice", "Customer"]
    series = [("R044 (old recipe)", [.338, .550, .518, .708], LIGHT), ("Frozen base", [.392, .550, .547, .711], GRAY),
              ("R060 (ours, best)", [.613, .644, .600, .735], BLUE)]
    jev = [.617, .716, .618, .760]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = .26
    for k, (name, vals, c) in enumerate(series):
        xs = [i + (k - 1) * w for i in range(4)]
        ax.bar(xs, vals, width=w - .03, color=c, label=name)
        if c == BLUE:
            for x, v in zip(xs, vals):
                ax.text(x, v - .045, f"{v:.3f}", ha="center", fontsize=9, color="white", fontweight="bold")
    for i, v in enumerate(jev):
        ax.plot([i - 1.5 * w, i + 1.5 * w], [v, v], color=ORANGE, lw=2.2, ls="--", label="Jev 1.13" if i == 0 else None)
    ax.set_xticks(range(4), wfs)
    ax.set_ylim(0, .9)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Score vs consensus gold")
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=9.5, bbox_to_anchor=(0, 1.02))
    ax.set_title("Security is at Jev's level; agent trace (−7 pts) is the biggest gap")
    finish(fig, "02_workflowevals_by_workflow", "Security/customer: exact action set; agent trace: primary action; invoice: exact. Source: R060_RESULTS.md.")


def hebrew_peers():
    models = [("StartLux-9B", (.878, .874, .916), GRAY), ("torchcast-12b", (.865, .865, .899), GRAY),
              ("Winnow-12B", (.865, .843, .840), GRAY), ("Frozen Gemma base", (.852, .835, .815), GRAY),
              ("Typical R056 (λ=.5)", (.852, .843, .899), BLUE), ("Typical R062 (Hebrew)", (.852, .870, .840), BLUE),
              ("Clef-flash", (.774, .752, .782), GRAY)]
    cols = ["Hebrew (translated)", "Mixed (HE doc, EN question)", "Native Hebrew"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.2), sharey=True)
    for j, (ax, col) in enumerate(zip(axes, cols)):
        for i, (name, vals, c) in enumerate(models):
            ax.plot([.74, vals[j]], [i, i], color=GRID, lw=1, zorder=1)
            ax.scatter(vals[j], i, s=70 if c == BLUE else 45, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            ax.text(vals[j] + .004, i, f"{vals[j]:.3f}", va="center", fontsize=8.5, color=INK if c == BLUE else INK2)
        ax.set_title(col, fontsize=11, loc="left")
        ax.set_xlim(.74, .95)
        ax.set_xticks([.75, .80, .85, .90])
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis() if j == 0 else None
    axes[0].set_yticks(range(len(models)), [m[0] for m in models])
    fig.suptitle("Hebrew, 9–14B decision models: near StartLux on mixed (.870 vs .874), behind on pure Hebrew",
                 x=.01, ha="left", fontsize=12.5, fontweight="semibold", color=INK)
    finish(fig, "03_hebrew_peers", "Hebrew JevBench v0 (eval-only; 230/230/119 items). Native labels are Claude-consensus. Source: hebrew/R062_RESULTS.md.")


def blend_tradeoff():
    pts = [("base (λ=0)", .550, 202), ("λ=.5", .601, 202), ("λ=.75", .608, 199), ("R056 (λ=1)", .649, 196)]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot([p[1] for p in pts], [p[2] for p in pts], color=BLUE, lw=2, zorder=2)
    for name, x, y in pts:
        ax.scatter(x, y, s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 7), fontsize=9.5, color=INK)
    ax.scatter(.648, 198, s=60, color=BLUE, marker="s", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("R060 (KL anchors)", (.648, 198), textcoords="offset points", xytext=(8, -14), fontsize=9.5, color=INK)
    ax.scatter(.678, 200, s=90, color=ORANGE, marker="D", zorder=3, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("Jev 1.13", (.678, 200), textcoords="offset points", xytext=(-22, 9), fontsize=9.5, color=INK)
    ax.set_xlabel("WorkflowEvals mean (specialist skill)")
    ax.set_ylabel("Public JevBench, correct of 231 (general skill)")
    ax.set_xlim(.53, .70)
    ax.set_ylim(194, 205)
    ax.set_title("Scaling the adapter trades workflow skill for general skill")
    finish(fig, "04_blend_tradeoff", "λ = fraction of the R056 LoRA delta applied. Source: R060PRE_BLEND_PROXY.md.")


def contested():
    rows = [("Typical R056", 66.5, 35.0, BLUE), ("JEV 1.13", 65.5, 52.5, ORANGE),
            ("StartLux-9B", 63.1, 49.7, GRAY), ("LiquidAI d1", 60.3, 59.6, GRAY)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, j, title in ((axes[0], 1, "Single-label decisions"), (axes[1], 2, "Multi-label decisions")):
        ax.barh(range(len(rows)), [r[j] for r in rows], color=[r[3] for r in rows], height=.55)
        for i, r in enumerate(rows):
            ax.text(r[j] + .8, i, f"{r[j]:.1f}%", va="center", fontsize=9.5, color=INK)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlim(0, 78)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Contested decisions: ahead of JEV 1.13 on single-label; multi-label is the gap",
                 x=.01, ha="left", fontsize=13, fontweight="semibold", color=INK)
    finish(fig, "05_contested_decisions", "zoom-ai/contested-fast-decisions (818). Peer numbers from its leaderboard. Source: R060PRE_BLEND_PROXY.md.")


def images():
    rows = [("Frozen base", 85, GRAY), ("R056", 114, BLUE), ("R060", 120, BLUE), ("R062 (Hebrew)", 110, BLUE)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(rows)), [r[1] / 128 for r in rows], color=[r[2] for r in rows], width=.55)
    for i, r in enumerate(rows):
        ax.text(i, r[1] / 128 + .015, f"{r[1]}/128", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ScienceQA real-image accuracy")
    ax.grid(axis="x", visible=False)
    ax.set_title("Image decisions improved with training, not just retained")
    finish(fig, "06_image_decisions", "128-item ScienceQA retention suite; all retention gates pass for R056/R060/R062.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (wf_progress, wf_per_workflow, hebrew_peers, blend_tradeoff, contested, images):
        f()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
