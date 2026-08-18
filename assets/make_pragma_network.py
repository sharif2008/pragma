"""PRAGMA network diagram: left domains, center 2x2 core, right stores."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).with_name("Pragma_network.png")

BLUE = "#0b60bd"
BLUE_DK = "#0a3d73"
BLUE_LT = "#e8f1fb"
RED = "#c42020"
BLACK = "#1a1a1a"
GRAY = "#5a5a5a"
WHITE = "#ffffff"
FILL = "#f7f9fc"

fig, ax = plt.subplots(figsize=(13.2, 7.2), dpi=200)
ax.set_xlim(0, 132)
ax.set_ylim(0, 72)
ax.axis("off")
fig.patch.set_facecolor(WHITE)
ax.set_facecolor(WHITE)


def rbox(x, y, w, h, fc=FILL, ec=BLUE, lw=1.2, rad=0.35):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.12,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
    )
    ax.add_patch(p)
    return p


def txt(x, y, s, size=8, color=BLACK, weight="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va=va, zorder=4, linespacing=1.25)


def arr(x1, y1, x2, y2, color=BLUE, lw=1.4):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=11, linewidth=lw,
        color=color, zorder=3, shrinkA=0, shrinkB=0,
    ))


# ----- LEFT: domain agents -----
rbox(3, 12, 26, 50, fc=BLUE_LT, ec=BLUE, lw=1.4)
txt(16, 58.5, "Domain agents", 10, BLUE_DK, "bold")
txt(16, 55.8, "(no raw features leave the party)", 6.5, GRAY)

domains = [
    (42, r"$D_1$  Access / ISP\n23 flow features"),
    (31, r"$D_2$  Perimeter / IDS\n32 flow features"),
    (20, r"$D_3$  Endpoint / EDR\n42 flow features"),
]
for y, label in domains:
    rbox(5.5, y, 21, 8.2, fc=WHITE, ec=BLUE)
    txt(16, y + 4.1, label, 7.4, BLACK)

txt(16, 14.5, "CICIDS-2017 aligned\nsample IDs, disjoint columns", 6.6, GRAY)

# ----- CENTER: PRAGMA core -----
rbox(34, 10, 58, 54, fc="#f0f5fb", ec=BLUE_DK, lw=1.6)
txt(63, 60.2, "PRAGMA core", 11, BLUE_DK, "bold")
txt(63, 57.6, "detect  —  reason  —  commit  —  apply", 7, GRAY)

tools = [
    (36.5, 34.5, "1  Detect  (VFL Layer)", "Local MLP encoders\n64-d embeddings, concat\nSHAP names driving tier"),
    (64.0, 34.5, "2  Reason  (LLM Layer)", "RAG over policy corpus\nGPT-4o-mini JSON plan\ntier-assigned actions"),
    (36.5, 13.5, "3  Commit  (Blockchain Layer)", "Canonical JSON payload P\nSHA-256 digest only\nanchor() write-once"),
    (64.0, 13.5, "4  Apply  (Execution Layer)", "Whitelist for attack\nPlan + tier binding\napplyAction or block"),
]
for x, y, title, body in tools:
    rbox(x, y, 25.5, 18.5, fc=WHITE, ec=BLUE)
    rbox(x, y + 14.2, 25.5, 4.3, fc=BLUE, ec=BLUE, rad=0.25)
    txt(x + 12.75, y + 16.35, title, 7.3, WHITE, "bold")
    txt(x + 12.75, y + 7.2, body, 6.8, BLACK)

# pipeline: 1→2 (top), 2→3 (across the gap), 3→4 (bottom)
arr(62.0, 43.7, 64.0, 43.7)
ax.add_patch(FancyArrowPatch(
    (76.75, 34.5), (49.25, 32.0),
    arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
    color=BLUE, zorder=3, connectionstyle="bar,angle=180,fraction=-0.18",
))
arr(62.0, 22.7, 64.0, 22.7)

# left -> center
arr(29.0, 46.1, 34.0, 46.1)
arr(29.0, 35.1, 34.0, 43.5)
arr(29.0, 24.1, 34.0, 41.0)

# ----- RIGHT: stores / actuation -----
rbox(97, 45.5, 32, 16.5, fc=WHITE, ec=BLUE)
txt(113, 59.2, "Policy corpus", 8.2, BLUE_DK, "bold")
txt(113, 52.2, "NIST SP 800-53 / 800-207\nCIS Controls v8  ·  ATT&CK\nFAISS + MiniLM retrieval", 6.8, BLACK)

rbox(97, 27.5, 32, 15.5, fc=WHITE, ec=BLUE)
txt(113, 40.4, "AgenticTrustRegistry", 8.2, BLUE_DK, "bold")
txt(113, 33.4, "Hardhat  ·  Solidity 0.8.20\nhash-only commitments\nper-attack action whitelist", 6.8, BLACK)

rbox(97, 10.0, 32, 15.0, fc=WHITE, ec=BLUE)
txt(113, 22.4, "Network / operator", 8.2, BLUE_DK, "bold")
txt(107.5, 15.2, "dispatch to tier  τ", 7, BLUE, "bold")
txt(118.5, 15.2, "block + alert", 7, RED, "bold")
txt(113, 12.0, "Access · Perimeter · Endpoint", 6.4, GRAY)

arr(92.0, 52.5, 97.0, 52.5)   # reason -> corpus (conceptually RAG)
arr(92.0, 35.0, 97.0, 35.0)   # commit/apply -> chain
arr(92.0, 17.5, 97.0, 17.5)   # apply -> network

# downward on right
arr(113, 45.5, 113, 43.0)
arr(113, 27.5, 113, 25.0)

# ----- bottom banner -----
ax.add_patch(Rectangle((3, 1.2), 126, 6.6, facecolor=BLUE_DK, edgecolor=BLUE_DK, zorder=2))
txt(66, 4.5, "PRAGMA  ·  Privacy-preserving multi-domain intrusion response", 10, WHITE, "bold")

fig.tight_layout(pad=0.25)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=WHITE)
print(f"wrote {OUT}")
