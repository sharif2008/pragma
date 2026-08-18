"""Apply (Execution Layer) flowchart for PRAGMA (IEEE-friendly)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).with_name("Flow.png")

fig, ax = plt.subplots(figsize=(4.4, 5.0), dpi=220)
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 11.2)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


def box(x, y, w, h, text, fc="#f4f4f4", ec="#222222", lw=1.1):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.18",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.4,
        color="#111111",
        zorder=3,
        linespacing=1.25,
    )
    return (x + w / 2, y + h, x + w / 2, y)


def arrow(x1, y1, x2, y2, label=None):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.15,
            color="#222222",
            zorder=1,
        )
    )
    if label:
        ax.text(x1 - 0.32, (y1 + y2) / 2, label, fontsize=7.2, color="#333", ha="right", va="center")


ax.add_patch(
    FancyBboxPatch(
        (0.35, 0.3),
        9.75,
        10.5,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#f4f7f4",
        edgecolor="#2f5d32",
        linewidth=1.35,
        zorder=0,
    )
)
ax.text(0.55, 10.35, "Apply (Execution Layer)", fontsize=9.5, fontweight="bold", color="#1b3d1d")

a0 = box(1.15, 8.55, 5.15, 1.25, "Apply request\n(attack, action, tier)", fc="#ffffff")
a1 = box(1.15, 6.35, 5.15, 1.25, "Policy ground\nwhitelisted for this attack?", fc="#e8eef6")
a2 = box(1.15, 4.15, 5.15, 1.25, "Plan binding\naction and tier match plan?", fc="#e8eef6")
ok = box(1.15, 1.55, 5.15, 1.35, "applyAction\nreceipt + dispatch to tier", fc="#dcecdc")
no = box(6.85, 5.05, 2.9, 1.35, "Block\noperator queue", fc="#f6dcdc", ec="#8a3030")

arrow(a0[2], a0[3] - 0.02, a1[0], a1[1] + 0.02)
arrow(a1[2], a1[3] - 0.02, a2[0], a2[1] + 0.02, label="yes")
arrow(a2[2], a2[3] - 0.02, ok[0], ok[1] + 0.02, label="yes")

ax.add_patch(
    FancyArrowPatch(
        (6.3, 6.97),
        (6.85, 6.15),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.15,
        color="#8a3030",
        zorder=1,
    )
)
ax.text(6.45, 7.12, "no", fontsize=7.2, color="#8a3030")
ax.add_patch(
    FancyArrowPatch(
        (6.3, 4.77),
        (6.85, 5.55),
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.15,
        color="#8a3030",
        zorder=1,
    )
)
ax.text(6.45, 4.45, "no", fontsize=7.2, color="#8a3030")

ax.text(1.15, 0.55, "Restore tier from Detect / Reason, then gate dispatch.", fontsize=7, color="#333")

fig.tight_layout(pad=0.12)
fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
