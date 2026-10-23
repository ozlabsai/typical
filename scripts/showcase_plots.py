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

# (label, value in %, kind) — kind: ours | jev | peer. "TYPICAL" = best Typical checkpoint per benchmark:
# WF mean/invoice R056, security/trace R060, customer + native R056 at 0.5 scale, contested R056 at 0.75, mixed R062.
BOARD = [
    ("WORKFLOWEVALS · MEAN", [("TYPICAL", 64.9, "ours"), ("JEV 1.13", 67.8, "jev"), ("WINNOW-12B", 55.8, "peer"),
                              ("CYGNET 12B", 52.1, "peer"), ("INTERN-DECISION-4B", 49.9, "peer")]),
    ("WORKFLOWEVALS · SECURITY", [("TYPICAL", 61.3, "ours"), ("JEV 1.13", 61.7, "jev"), ("WINNOW-12B", 39.2, "peer"),
                                  ("CYGNET 12B", 33.3, "peer"), ("INTERN-DECISION-4B", 52.1, "peer")]),
    ("WORKFLOWEVALS · INVOICE", [("TYPICAL", 62.7, "ours"), ("JEV 1.13", 61.8, "jev"), ("WINNOW-12B", 47.8, "peer"),
                                 ("CYGNET 12B", 53.1, "peer"), ("INTERN-DECISION-4B", 44.7, "peer")]),
    ("WORKFLOWEVALS · AGENT TRACE", [("TYPICAL", 64.4, "ours"), ("JEV 1.13", 71.6, "jev"), ("WINNOW-12B", 62.6, "peer"),
                                     ("CYGNET 12B", 52.3, "peer"), ("INTERN-DECISION-4B", 46.8, "peer")]),
    ("WORKFLOWEVALS · CUSTOMER", [("TYPICAL", 74.8, "ours"), ("JEV 1.13", 76.0, "jev"), ("WINNOW-12B", 73.5, "peer"),
                                  ("CYGNET 12B", 69.8, "peer"), ("INTERN-DECISION-4B", 55.9, "peer")]),
    ("CONTESTED DECISIONS · SINGLE-LABEL", [("TYPICAL", 67.4, "ours"), ("JEV 1.13", 65.5, "jev"),
                                            ("LIQUIDAI D1", 60.3, "peer"), ("CLEF-FLASH", 51.3, "peer")]),
    ("HEBREW JEVBENCH · TRANSLATED", [("TYPICAL", 85.2, "ours"), ("TORCHCAST-12B", 86.5, "peer"),
                                      ("WINNOW-12B", 86.5, "peer"), ("CYGNET 12B", 83.9, "peer"), ("CLEF-FLASH", 77.4, "peer")]),
    ("HEBREW JEVBENCH · MIXED HE/EN", [("TYPICAL", 87.0, "ours"), ("TORCHCAST-12B", 86.5, "peer"),
                                       ("WINNOW-12B", 84.3, "peer"), ("CYGNET 12B", 83.9, "peer"), ("CLEF-FLASH", 75.2, "peer"),
          ("LIQUIDAI D1-3B", 66.1, "peer")]),
    ("HEBREW JEVBENCH · NATIVE", [("TYPICAL", 89.9, "ours"), ("TORCHCAST-12B", 89.9, "peer"),
                                  ("WINNOW-12B", 84.0, "peer"), ("CYGNET 12B", 88.2, "peer"), ("CLEF-FLASH", 78.2, "peer")]),
]


BLEND = [("BASE (SCALE 0)", 55.0, 202), ("SCALE 0.5", 60.1, 202), ("SCALE 0.75", 60.8, 199), ("R056 (SCALE 1)", 64.9, 196)]

IMAGES = [("BASE MODEL", 85, "peer"), ("TYPICAL", 120, "ours")]


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


def keep(rows, jev):
    return [r for r in rows if jev or r[2] != "jev"]


def leads(rows):
    ours = max(v for _, v, k in rows if k == "ours")
    return all(ours > v for _, v, k in rows if k == "peer")


def board(t, name, jev=True, lead_only=False):
    panels = [(title, keep(rows, jev)) for title, rows in BOARD if not lead_only or leads(rows)]
    cols = 3
    nrows = -(-len(panels) // cols)
    fig = page(t, 4.4 * cols, 1.3 + 2.45 * nrows)
    header(fig, t, "Typical vs decision models of similar size",
           "WORKFLOWEVALS (705 CASES) · CONTESTED-FAST-DECISIONS · HEBREW JEVBENCH V0 · ACCURACY %")
    h = fig.get_figheight()
    gs = fig.add_gridspec(nrows, cols, left=.05, right=.97, top=1 - 1.25 / h, bottom=.45 / h, wspace=.12, hspace=.35)
    for k, (title, rows) in enumerate(panels):
        bar_panel(fig.add_subplot(gs[k // cols, k % cols]), t, title, rows, vmax=100)
    fig.savefig(OUT / f"{name}_board.png", dpi=220, facecolor=t["page"])
    plt.close(fig)




def tradeoff(t, name, jev=True):
    fig = page(t, 9, 5.6)
    header(fig, t, "Scaling the adapter trades workflow skill for general skill",
           "SCALE = STRENGTH OF THE TYPICAL FINE-TUNE")
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
    if jev:
        ax.scatter(67.8, 200, s=80, marker="D", color=t["jev"], zorder=3)
        ax.annotate("JEV 1.13", (67.8, 200), textcoords="offset points", xytext=(-20, 9), fontsize=8.5,
                    family=t["mono"], color=t["ink"])
    ax.set_xlim(53, 70)
    ax.set_ylim(194, 205)
    ax.set_xlabel("WORKFLOWEVALS MEAN %", color=t["ink2"], fontsize=8.5, family=t["mono"])
    ax.set_ylabel("PUBLIC JEVBENCH, CORRECT / 231", color=t["ink2"], fontsize=8.5, family=t["mono"])
    fig.savefig(OUT / f"{name}_tradeoff.png", dpi=220, facecolor=t["page"])
    plt.close(fig)


# mean accuracy of the translated / mixed / native splits (HEJEV_V0_RESULTS.md); Typical = R056 at scale 0.5,
# one model on all three splits (.852 / .843 / .899)
HEBREW = [("TYPICAL", 86.5, "ours"), ("CYGNET 12B", 85.3, "peer"), ("WINNOW-12B", 84.9, "peer"),
          ("DECIDER-12B", 83.3, "peer"), ("D3-FLASH", 81.5, "peer"), ("D3-MINI", 80.5, "peer"),
          ("INTERN-DECISION-4B", 80.4, "peer"), ("CLEF-FLASH", 76.9, "peer"), ("KEV-9B", 75.3, "peer"),
          ("INTERN-DECISION-2B", 72.1, "peer"), ("LIQUIDAI D1-3B", 65.0, "peer")]


def hebrew(t, name):
    fig = page(t, 10, 2.2 + .45 * len(HEBREW))
    header(fig, t, "Typical leads in Hebrew decisions",
           "HEBREW JEVBENCH V0 · TRANSLATED, MIXED AND NATIVE · MEAN ACCURACY %")
    bar_panel(body(fig), t, "", HEBREW, vmax=100, label_w=.25)
    fig.savefig(OUT / f"{name}_hebrew.png", dpi=220, facecolor=t["page"])
    plt.close(fig)


def images(t, name, jev=True):
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
        for f in (tradeoff, images):
            f(t, tname)
            f(t, f"{tname}_nojev", jev=False)
        hebrew(t, tname)
        board(t, tname)
        board(t, f"{tname}_nojev_leads", jev=False, lead_only=True)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
