"""Showcase figures for Typical in two house styles: a dark benchmark board (Prime Intellect-like)
and a light card on sage (TypeSafe-like). Numbers are copied from the committed result docs:
docs/research/R056_RESULTS.md, R060_RESULTS.md, R060PRE_BLEND_PROXY.md, hebrew/HEJEV_V0_RESULTS.md,
hebrew/R062_RESULTS.md, and JEV_PARITY_STRATEGY_20261004.md (peer WorkflowEvals rows).
Run: uv run --with matplotlib python scripts/showcase_plots.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("docs/research/figures/showcase")

THEMES = {
    "dark": dict(page="#0b0b0b", card=None, ink="#f2f2f2", ink2="#8c8c8c", track="#1c1c1c", other="#3b3b3b",
                 ours="#f2f2f2", ours_ink="#0b0b0b", jev="#eb6834", grid="#1f1f1f", mono="Space Mono", sans="Inter"),
    "light": dict(page="#abbab9", card="#ffffff", ink="#2b2c2c", ink2="#6b6d6c", track="#f1f1ef", other="#b9bcbb",
                  ours="#2a78d6", ours_ink="#ffffff", jev="#eb6834", grid="#ececea", mono="Space Mono", sans="Inter"),
}

# (label, value in %, kind) — kind: ours | jev | peer.  Typical rows name the exact model.
BOARD = [
    ("WORKFLOWEVALS · MEAN", [("TYPICAL R056", 64.9, "ours"), ("JEV 1.13", 67.8, "jev"), ("WINNOW-12B", 55.8, "peer"),
                              ("CYGNET 12B", 52.1, "peer"), ("INTERN-DECISION-4B", 49.9, "peer")]),
    ("WORKFLOWEVALS · SECURITY", [("TYPICAL R056", 60.4, "ours"), ("JEV 1.13", 61.7, "jev"), ("WINNOW-12B", 39.2, "peer"),
                                  ("CYGNET 12B", 33.3, "peer"), ("INTERN-DECISION-4B", 52.1, "peer")]),
    ("WORKFLOWEVALS · INVOICE", [("TYPICAL R056", 62.7, "ours"), ("JEV 1.13", 61.8, "jev"), ("WINNOW-12B", 47.8, "peer"),
                                 ("CYGNET 12B", 53.1, "peer"), ("INTERN-DECISION-4B", 44.7, "peer")]),
    ("WORKFLOWEVALS · AGENT TRACE", [("TYPICAL R056", 63.5, "ours"), ("JEV 1.13", 71.6, "jev"), ("WINNOW-12B", 62.6, "peer"),
                                     ("CYGNET 12B", 52.3, "peer"), ("INTERN-DECISION-4B", 46.8, "peer")]),
    ("WORKFLOWEVALS · CUSTOMER", [("TYPICAL R056", 72.9, "ours"), ("JEV 1.13", 76.0, "jev"), ("WINNOW-12B", 73.5, "peer"),
                                  ("CYGNET 12B", 69.8, "peer"), ("INTERN-DECISION-4B", 55.9, "peer")]),
    ("CONTESTED DECISIONS · SINGLE-LABEL", [("TYPICAL R056", 66.5, "ours"), ("JEV 1.13", 65.5, "jev"),
                                            ("LIQUIDAI D1", 60.3, "peer"), ("CLEF-FLASH", 51.3, "peer")]),
    ("HEBREW JEVBENCH · TRANSLATED", [("TYPICAL R062", 85.2, "ours"), ("TORCHCAST-12B", 86.5, "peer"),
                                      ("WINNOW-12B", 86.5, "peer"), ("CYGNET 12B", 83.9, "peer"), ("CLEF-FLASH", 77.4, "peer")]),
    ("HEBREW JEVBENCH · MIXED HE/EN", [("TYPICAL R062", 87.0, "ours"), ("TORCHCAST-12B", 86.5, "peer"),
                                       ("WINNOW-12B", 84.3, "peer"), ("CYGNET 12B", 83.9, "peer"), ("CLEF-FLASH", 75.2, "peer")]),
    ("HEBREW JEVBENCH · NATIVE", [("TYPICAL R062", 84.0, "ours"), ("TORCHCAST-12B", 89.9, "peer"),
                                  ("WINNOW-12B", 84.0, "peer"), ("CYGNET 12B", 88.2, "peer"), ("CLEF-FLASH", 78.2, "peer")]),
]

PROGRESS = [("CYGNET 12B (FROZEN, ITS READOUT)", 52.1, "peer"), ("TYPICAL R044 (OLD RECIPE)", 52.8, "peer"),
            ("GEMMA-12B + THINKING (TEACHER)", 61.2, "peer"), ("TYPICAL R056 (DISTILLED)", 64.9, "ours"),
            ("TYPICAL R060 (+GPT TRACE LABELS)", 64.8, "ours"), ("JEV 1.13", 67.8, "jev")]

BLEND = [("BASE (SCALE 0)", 55.0, 202), ("SCALE 0.5", 60.1, 202), ("SCALE 0.75", 60.8, 199), ("R056 (SCALE 1)", 64.9, 196)]

IMAGES = [("FROZEN BASE", 85, "peer"), ("TYPICAL R056", 114, "ours"), ("TYPICAL R060", 120, "ours"),
          ("TYPICAL R062", 110, "ours")]


def page(t, w, h):
    fig = plt.figure(figsize=(w, h), facecolor=t["page"])
    if t["card"]:
        fig.patches.append(FancyBboxPatch((.02, .03), .96, .94, boxstyle="round,pad=0,rounding_size=.012",
                                          transform=fig.transFigure, facecolor=t["card"], edgecolor="none", zorder=-10))
    return fig


def header(fig, t, title, sub):
    """Title + mono subtitle at fixed inch offsets from the top, so short figures don't collide."""
    h = fig.get_figheight()
    fig.text(.05, 1 - .4 / h, title, color=t["ink"], fontsize=17, family=t["sans"], weight="semibold", va="top")
    fig.text(.05, 1 - .82 / h, sub, color=t["ink2"], fontsize=10, family=t["mono"], va="top")


def body(fig, left=.05, right=.95, bottom_in=.35):
    """Axes rectangle filling the space below the header (1.2 in from the top)."""
    h = fig.get_figheight()
    return fig.add_axes((left, bottom_in / h, right - left, 1 - (1.2 + bottom_in) / h))


def color(t, kind):
    return {"ours": t["ours"], "jev": t["jev"], "peer": t["other"]}[kind]


def bar_panel(ax, t, title, rows, vmax, fmt="{:.1f}", label_w=.42):
    """One Prime-style panel: label chips on the left, bars from zero, value at the bar end."""
    ax.set_facecolor(t["card"] or t["page"])
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - .4, -1.1)
    ax.axis("off")
    ax.text(0, -.95, title, color=t["ink"], fontsize=9.5, family=t["mono"], va="center", weight="bold")
    span = 1 - label_w - .1
    for i, (name, v, kind) in enumerate(rows):
        c = color(t, kind)
        ours = kind == "ours"
        ax.text(label_w - .02, i, name, ha="right", va="center", fontsize=7.6, family=t["mono"],
                color=t["ours_ink"] if ours else (t["ink"] if kind == "jev" else t["ink2"]),
                bbox=dict(boxstyle="square,pad=.28", facecolor=t["ours"], edgecolor="none") if ours else None)
        ax.barh(i, span, left=label_w, height=.62, color=t["track"], zorder=1)
        ax.barh(i, span * v / vmax, left=label_w, height=.62, color=c, zorder=2)
        ax.text(label_w + span * v / vmax + .012, i, fmt.format(v), va="center", fontsize=7.8, family=t["mono"],
                color=t["ink"] if kind != "peer" else t["ink2"])


def board(t, name):
    fig = page(t, 13, 8.6)
    header(fig, t, "Typical vs decision models of similar size",
           "WORKFLOWEVALS (705 CASES) · CONTESTED-FAST-DECISIONS · HEBREW JEVBENCH V0 · ACCURACY %")
    gs = fig.add_gridspec(3, 3, left=.05, right=.97, top=.84, bottom=.08, wspace=.12, hspace=.35)
    for k, (title, rows) in enumerate(BOARD):
        bar_panel(fig.add_subplot(gs[k // 3, k % 3]), t, title, rows, vmax=100)
    fig.text(.05, .06, "Highlighted: Typical (Gemma-4-12B + LoRA, single forward pass). Orange: JEV 1.13. Peers measured on our harness;",
             color=t["ink2"], fontsize=7.5, family=t["mono"])
    fig.text(.05, .04, "contested-decision peers from its public leaderboard. Hebrew native labels are Claude-consensus, not human-verified.",
             color=t["ink2"], fontsize=7.5, family=t["mono"])
    fig.savefig(OUT / f"{name}_board.png", dpi=220, facecolor=t["page"])
    plt.close(fig)


def progress(t, name):
    fig = page(t, 10, 4.2)
    header(fig, t, "Thinking-teacher distillation: WorkflowEvals .53 → .65",
           "WORKFLOWEVALS MEAN · 4 WORKFLOWS · 705 CASES · ACCURACY %")
    bar_panel(body(fig), t, "", PROGRESS, vmax=80, label_w=.36)
    fig.savefig(OUT / f"{name}_progress.png", dpi=220, facecolor=t["page"])
    plt.close(fig)


def tradeoff(t, name):
    fig = page(t, 9, 5.6)
    header(fig, t, "Scaling the adapter trades workflow skill for general skill",
           "SCALE = FRACTION OF THE R056 LORA DELTA APPLIED")
    ax = body(fig, left=.1, right=.94, bottom_in=.8)
    ax.set_facecolor(t["card"] or t["page"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color=t["grid"], lw=.8)
    ax.tick_params(colors=t["ink2"], labelsize=8.5, length=0)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family(t["mono"])
    xs, ys = [b[1] for b in BLEND], [b[2] for b in BLEND]
    ax.plot(xs, ys, color=t["ours"], lw=2, zorder=2)
    ax.scatter(xs, ys, s=55, color=t["ours"], zorder=3, edgecolor=t["card"] or t["page"], linewidth=1.5)
    for lab, x, y in BLEND:
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(7, 7), fontsize=8.5, family=t["mono"], color=t["ink"])
    ax.scatter(64.8, 198, s=55, marker="s", color=t["ours"], zorder=3)
    ax.annotate("R060 (KL ANCHORS)", (64.8, 198), textcoords="offset points", xytext=(8, -13), fontsize=8.5,
                family=t["mono"], color=t["ink"])
    ax.scatter(67.8, 200, s=80, marker="D", color=t["jev"], zorder=3)
    ax.annotate("JEV 1.13", (67.8, 200), textcoords="offset points", xytext=(-20, 9), fontsize=8.5,
                family=t["mono"], color=t["ink"])
    ax.set_xlim(53, 70)
    ax.set_ylim(194, 205)
    ax.set_xlabel("WORKFLOWEVALS MEAN %", color=t["ink2"], fontsize=8.5, family=t["mono"])
    ax.set_ylabel("PUBLIC JEVBENCH, CORRECT / 231", color=t["ink2"], fontsize=8.5, family=t["mono"])
    fig.savefig(OUT / f"{name}_tradeoff.png", dpi=220, facecolor=t["page"])
    plt.close(fig)


def images(t, name):
    fig = page(t, 9, 3.3)
    header(fig, t, "Image decisions improved with training",
           "SCIENCEQA REAL-IMAGE ACCURACY · 128-ITEM RETENTION SUITE · ALL RETENTION GATES PASS")
    bar_panel(body(fig), t, "", [(n, v / 128 * 100, k) for n, v, k in IMAGES], vmax=100, label_w=.3)
    fig.savefig(OUT / f"{name}_images.png", dpi=220, facecolor=t["page"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.png"):
        old.unlink()
    for tname, t in THEMES.items():
        for f in (board, progress, tradeoff, images):
            f(t, tname)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
