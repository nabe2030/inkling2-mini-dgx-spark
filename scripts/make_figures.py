"""Figures of the README. The numbers are transcribed from the measurements.

  pip install matplotlib
  python3 scripts/make_figures.py      # from the repository root
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Noto Sans CJK was used for the published figures; matplotlib's default font is the fallback.
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
# Figures are written to ./figures, so run this from the repository root (or a copy of it).
os.makedirs("figures", exist_ok=True)

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#b9b8b2"


def base(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


C = [1, 2, 4, 8, 16, 32]
series = [
    ("Inkling 2 Mini (DSpark on)", BLUE, "-", "o", [39.86, 58.41, 84.84, 130.55, 201.13, 298.80], C),
    ("Inkling 2 Mini (DSpark off)", "#86b6ef", "--", "o", [36.64, 55.91, 79.75, 110.87, 152.40, 210.09], C),
    ("Qwen3.8-27B (vLLM, MTP)", YELLOW, "-", "s", [16.60, 105.00, 168.49, 235.74], [1, 8, 16, 32]),
    ("Nemotron 3.5 Super VL (vLLM, no spec. decoding)", AQUA, "-", "^", [19.2, 30.22, 44.12, 59.06, 78.55, 105.67], C),
    ("Qwen3.8-Flash-Next (SGLang)", ORANGE, "-", "D", [26.97, 38.31, 51.64, 67.68, 77.48, 89.35], C),
]
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
fig.patch.set_facecolor(SURF)
base(ax)
for name, col, ls, mk, ys, xs in series:
    if name.startswith("Qwen3.8-27B"):
        ax.plot(xs[:2], ys[:2], color=col, linestyle=":", linewidth=1.5)
        ax.plot(xs[1:], ys[1:], color=col, linestyle=ls, linewidth=2, marker=mk, markersize=8,
                markeredgecolor=SURF, markeredgewidth=1.5, label=name)
        ax.plot(xs[:1], ys[:1], color=col, linestyle="none", marker=mk, markersize=8,
                markeredgecolor=SURF, markeredgewidth=1.5)
        continue
    ax.plot(xs, ys, color=col, linestyle=ls, linewidth=2, marker=mk, markersize=8,
            markeredgecolor=SURF, markeredgewidth=1.5, label=name)
ax.set_xscale("log", base=2)
ax.set_xticks(C)
ax.set_xticklabels([str(c) for c in C])
ax.set_xlim(0.8, 32 * 3.4)
ax.set_ylim(0, 330)
ax.set_xlabel("Concurrent requests c (c ≤ 4: personal use; c ≥ 8: shared team/enterprise use)", color=INK2)
ax.set_ylabel("Aggregate output throughput (tok/s)", color=INK2)
ax.set_title("Aggregate output throughput on one DGX Spark (vllm bench serve, ShareGPT, 200 prompts)",
             color=INK, loc="left", fontsize=12)
ends = [("Inkling (DSpark on) 298.8", 298.80, BLUE),
        ("Qwen3.8-27B 235.7", 235.74, INK2),
        ("Inkling (DSpark off) 210.1", 210.09, BLUE),
        ("Nemotron 3.5 105.7", 105.67, INK2),
        ("Flash-Next 89.4", 89.35, INK2)]
label_y = {298.80: 298.80, 235.74: 235.74, 210.09: 210.09, 105.67: 110.0, 89.35: 84.0}
for txt, y, col in ends:
    ax.text(32 * 1.12, label_y[y], txt, va="center", ha="left", fontsize=9.5, color=INK)
ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=INK)
fig.text(0.01, 0.01,
         "Note: tok/s counts all output tokens including reasoning (Inkling ran at its default reasoning effort; most output is reasoning).\n"
         "Runtime and quantization differ between series. Qwen3.8-27B was not measured at c=2 and c=4 (dotted).",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("figures/fig2_speed.png", facecolor=SURF)
plt.close(fig)

gain = [8.8, 4.5, 6.4, 17.7, 32.0, 42.2]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
fig.patch.set_facecolor(SURF)
base(ax)
ax.grid(axis="x", visible=False)
xs = range(len(C))
bars = ax.bar(xs, gain, width=0.6, color=BLUE)
for x, g in zip(xs, gain):
    ax.text(x, g + 1.0, f"+{g:.1f}%", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(list(xs))
ax.set_xticklabels([f"c={c}" for c in C])
ax.set_ylim(0, 50)
ax.set_ylabel("Aggregate throughput gain vs. DSpark off (%)", color=INK2)
ax.set_title("DSpark (bundled speculative decoder) pays off most at c ≥ 8",
             color=INK, loc="left", fontsize=12)
fig.text(0.01, 0.01, "Draft acceptance rate ≈21% at every c. Including the bonus token, ≈2.5 tokens are committed per verification step (7 draft tokens, temp 1.0).",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/fig3_dspark_gain.png", facecolor=SURF)
plt.close(fig)

rows = [
    ("Qwen3.8-Flash-Next (RadixArk build)", 70.00),
    ("Nemotron 3.5 Super VL", 70.00),
    ("Gemma4-31B (Q4_K_M)", 69.33),
    ("Qwen3.8-Flash-Next (senfu build)", 69.00),
    ("DeepSeek-V4-Flash-0731 (IQ3_XXS)", 69.00),
    ("Gemma4-31B (NVFP4)", 68.00),
    ("Gemma4-26B-A4B (Q4_K_M)", 62.67),
    ("Nemotron 3 Super 120B (Q4_K)", 62.33),
    ("Qwen3.6-35B-A3B (MXFP4)", 60.00),
    ("Qwen3.8-27B (Q4_K_M)", 60.00),
    ("Muse Glimmer 30B", 58.33),
    ("Inkling 2 Mini (reference: reasoning on, temp 1.0)", 54.67),
    ("Inkling 2 Mini (no reasoning)", 45.67),
]
fig, ax = plt.subplots(figsize=(9, 6.2), dpi=150)
fig.patch.set_facecolor(SURF)
base(ax)
ax.grid(axis="y", visible=False)
names = [r[0] for r in rows][::-1]
vals = [r[1] for r in rows][::-1]
cols = []
for n in names:
    if n == "Inkling 2 Mini (no reasoning)":
        cols.append(BLUE)
    elif n.startswith("Inkling"):
        cols.append("#86b6ef")
    else:
        cols.append(GRAY)
y = range(len(names))
ax.barh(list(y), vals, height=0.62, color=cols)
for yy, v in zip(y, vals):
    ax.text(v + 0.6, yy, f"{v:.2f}%", va="center", ha="left", fontsize=9.5, color=INK)
ax.set_yticks(list(y))
ax.set_yticklabels(names, fontsize=9.5, color=INK)
ax.set_xlim(0, 80)
ax.axvline(25, color=INK2, linewidth=1, linestyle=":", zorder=0.9)
ax.text(25.5, len(names) - 0.4, "Random guess (4 choices) 25%", fontsize=8.5, color=INK2, va="center")
ax.set_xlabel("Accuracy (%)", color=INK2)
fig.suptitle("Japanese knowledge: JamC-QA accuracy on the same 300 questions (3-shot, 4-choice)", x=0.01, ha="left", color=INK, fontsize=12)
fig.text(0.01, 0.01,
         "300 questions: category-stratified random sample (seed 20261123). Earlier models were re-scored on the same 300 from their per-question results.", fontsize=8.0, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("figures/fig4_jamcqa.png", facecolor=SURF)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.set_xlim(0, 110)
ax.set_ylim(0, 62)
ax.axis("off")


def box(x, y, w, h, text, fc, ec=INK2, fs=10.5, color=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                       facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color)


def arrow(x1, y1, x2, y2, color=INK2, rad=0.0, lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        color=color, linewidth=lw, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)


ax.text(2, 59, "How Inkling 2 Mini works: a MoE that loops its middle layers", fontsize=13, color=INK, va="center")

box(2, 40, 15, 7, "Text", "#f1f0ec")
box(2, 29, 15, 7, "Image\n(ViT encoder)", "#f1f0ec", fs=9.5)
arrow(17.8, 43.5, 23.2, 39.2)
arrow(17.8, 32.5, 23.2, 36.0)
box(24, 33, 13, 9, "Early layers", "#cde2fb")
arrow(37.8, 37.5, 43.2, 37.5)
box(44, 31, 22, 13, "Middle 12 layers\n(3 passes,\nshared weights)", "#86b6ef", fs=10.5)
arrow(58, 44.6, 52, 44.6, color=BLUE, rad=1.6, lw=1.8)
ax.text(55, 53.3, "Passes 2 and 3 reuse the weights,\neach with its own KV cache", ha="center", va="center",
        fontsize=9, color=INK2)
arrow(66.8, 37.5, 72.2, 37.5)
box(73, 33, 13, 9, "Late layers", "#cde2fb")
arrow(86.8, 37.5, 92.2, 37.5)
box(93, 33, 14, 9, "Output\n(text only)", "#f1f0ec", fs=9.5)
ax.text(55, 27.5, "Early + late = 12 layers; with the middle 12, the weights cover 24 layers.\nCompute per token: 48 layer passes (middle 12 × 3 + 12).",
        ha="center", va="center", fontsize=9.5, color=INK)

box(14, 3, 82, 17, "", "#ffffff", ec=GRAY)
ax.text(17, 17.0, "Every layer's feed-forward is MoE", fontsize=10.5, color=INK, va="center")
for i in range(16):
    x = 18 + i * 4.6
    sel = i in (1, 4, 6, 9, 10, 12, 14, 15)
    ax.add_patch(FancyBboxPatch((x, 7.5), 3.3, 4.5, boxstyle="round,pad=0.1,rounding_size=0.5",
                                facecolor=BLUE if sel else "#e4e3df", edgecolor="none"))
ax.text(18, 5.3, "Each token is routed to 8 of 128 experts (16 shown here)",
        fontsize=9, color=INK2, va="center")
ax.text(92, 9.8, "…", fontsize=14, color=INK2, va="center")

ax.text(2, -4.5,
        "26.4B total parameters; 3.4B active per pass (~5.4B effective over 3 loops).\n"
        "Up to 1M-token context (the DGX Spark recipe uses 262,144). Ships with the DSpark speculative decoder.",
        fontsize=8.5, color=INK2, va="bottom")
fig.savefig("figures/fig1_architecture.png", facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
plt.close(fig)
print("ok")
