#!/usr/bin/env python3
"""Charts of the decode benchmark (27B, 9B) from aggregate.py's JSON (static PNGs for the
report and the README; the table beside them is the accessible view).

  1. decode_2k_<theme>.png   grouped bars: decode tok/s per system at 2K context, 256 tokens,
                             one panel per mode (greedy | sampling), groups = prompt class
  2. decode_ctx_<theme>.png  lines: greedy decode tok/s vs context (2K .. 80K), one panel per class

Colors follow the system (a fixed slot per system, never by rank). matplotlib is not in the
repo's envs; run it from uv's cache without a download:

  uv run --offline --no-project --with matplotlib python scripts/perf/bench27/chart.py \\
      --agg agg.json --out docs/perf/2026-11-16/ [--title "..."] [--prefix decode_9b]

The title defaults to "<model> decode on <--machine>", the model named by the systems present
(MODEL); --title replaces the whole prefix.
"""
from __future__ import annotations

import argparse
import json
import os

# reference categorical palette (light, dark), fixed slot per system
SLOTS = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"), ("#eda100", "#c98500"),
         ("#e87ba4", "#d55181"), ("#008300", "#0ca30c"), ("#6250d6", "#9085e9"), ("#e34948", "#e66767")]
ORDER = ["ours-dflash", "llama-q4_0", "mlx-vlm", "ours-plain", "llama-q4_0-mtp", "llama-q4kxl",
         "mlx-vlm-dflash", "llama-q4_0-dflash"]
LABEL = {"ours-dflash": "ours, DFlash", "ours-plain": "ours, plain", "llama-q4_0": "llama.cpp Q4_0",
         "llama-q4kxl": "llama.cpp UD-Q4_K_XL", "llama-q4_0-mtp": "llama.cpp Q4_0 + MTP",
         "llama-q4_0-dflash": "llama.cpp Q4_0 + DFlash", "mlx-vlm": "mlx-vlm 0.7.6",
         "mlx-vlm-dflash": "mlx-vlm 0.7.6 + DFlash", "ours-oq4e-dflash": "ours oQ4e, DFlash",
         "ours-oq4e-plain": "ours oQ4e, plain", "ours-oq4e-auto": "ours oQ4e, auto (DFlash + MTP)",
         "ours-oq4e-mtp": "ours oQ4e, MTP",
         # the 9B (Ornith 1.5, MLX 8-bit / GGUF Q8_0)
         "ours9-auto": "ours, auto", "ours9-plain": "ours, plain", "ours9-dflash": "ours, DFlash",
         "ours9-mtp": "ours, MTP head", "mlx-vlm9": "mlx-vlm", "llama9-q8_0": "llama.cpp (Q8_0)",
         "llama9-q8_0-mtp": "llama.cpp + MTP (Q8_0)",
         # oMLX (the 27B: oQ4e; the 9B: the same MLX 8-bit)
         "omlx": "oMLX oQ4e", "omlx-mtp": "oMLX oQ4e + MTP", "omlx-dflash": "oMLX oQ4e + DFlash",
         "omlx9": "oMLX", "omlx9-mtp": "oMLX + MTP", "omlx9-dflash": "oMLX + DFlash"}
# the oQ4e rows reuse their 4-bit twin's hue; hatched (secondary encoding, not a 9th hue)
# only in a chart that also shows another system of that hue
TWIN = {"ours-oq4e-dflash": "ours-dflash", "ours-oq4e-plain": "ours-plain", "ours-oq4e-auto": "ours-dflash",
        "ours-oq4e-mtp": "llama-q4_0-dflash",
        # the 9B systems take the hue of their 27B counterpart (the charts never mix the two models)
        "ours9-auto": "ours-dflash", "ours9-dflash": "ours-dflash", "ours9-plain": "ours-plain",
        "ours9-mtp": "llama-q4_0-dflash", "mlx-vlm9": "mlx-vlm", "llama9-q8_0": "llama-q4_0",
        "llama9-q8_0-mtp": "llama-q4_0-mtp",
        # oMLX takes the slots the 9B charts leave free (green, purple; DFlash = MTP's hue, hatched)
        "omlx": "llama-q4kxl", "omlx-mtp": "mlx-vlm-dflash", "omlx-dflash": "mlx-vlm-dflash",
        "omlx9": "llama-q4kxl", "omlx9-mtp": "mlx-vlm-dflash", "omlx9-dflash": "mlx-vlm-dflash"}
# the model a system serves, for the default title
MODEL = {**{s: "Qwen3.8-27B (4-bit class)" for s in ORDER + ["ours-oq4e-dflash", "ours-oq4e-plain",
                                                             "ours-oq4e-auto", "ours-oq4e-mtp", "omlx", "omlx-mtp",
                                                             "omlx-dflash"]},
         **{s: "Ornith-1.5-9B (8-bit)" for s in ("ours9-auto", "ours9-plain", "ours9-dflash", "ours9-mtp",
                                                 "mlx-vlm9", "llama9-q8_0", "llama9-q8_0-mtp", "omlx9", "omlx9-mtp",
                                                 "omlx9-dflash")}}
# the 9B charts: bars (and the legend) grouped by family, ours first; the 27B keeps ORDER
SEQ = ["ours9-auto", "ours9-mtp", "ours9-dflash", "ours9-plain", "llama9-q8_0-mtp", "llama9-q8_0", "omlx9-mtp",
       "omlx9-dflash", "omlx9", "mlx-vlm9"]
THEMES = {"light": {"bg": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"},
          "dark": {"bg": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#3a3a37"}}
CLASSES = [("code", "code"), ("en", "English prose"), ("ru", "Russian prose")]


def base(system):
    return system.split("[")[0]


def style(system, theme, systems=()):
    """(color, hatch): the hue of the system's slot; hatched when a system drawn before it in the
    same chart has that hue too (the first of a hue stays solid)."""
    b = base(system)
    key = TWIN.get(b, b)
    i = ORDER.index(key) if key in ORDER else len(ORDER) - 1
    shared = any(base(x) != b and TWIN.get(base(x), base(x)) == key and sort_key(x) < sort_key(system)
                 for x in systems)
    return SLOTS[i][0 if theme == "light" else 1], ("///" if b in TWIN and shared else None)


def sort_key(system):
    b = base(system)
    key = TWIN.get(b, b)
    if b in SEQ:
        return (0, SEQ.index(b), False, system)
    return (1, ORDER.index(key) if key in ORDER else 99, b in TWIN, system)


def model_title(rows, machine):
    models = []
    for s in sorted({base(r["system"]) for r in rows}, key=sort_key):
        m = MODEL.get(s, s)
        if m not in models:
            models.append(m)
    return f"{' / '.join(models)} decode on {machine}"


def setup(plt, th):
    plt.rcParams.update({"figure.facecolor": th["bg"], "axes.facecolor": th["bg"], "savefig.facecolor": th["bg"],
                         "text.color": th["ink"], "axes.labelcolor": th["ink2"], "xtick.color": th["ink2"],
                         "ytick.color": th["ink2"], "axes.edgecolor": th["grid"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": th["grid"], "grid.linewidth": 0.8,
                         "hatch.color": th["bg"], "hatch.linewidth": 1.5})


def bars(plt, rows, theme, out, title, prefix="decode"):
    th = THEMES[theme]
    setup(plt, th)
    sel = [r for r in rows if r["ctx"] == 2048 and r["max_tokens"] == 256 and not r["thinking"]]
    systems = sorted({r["system"] for r in sel}, key=sort_key)
    if not systems:
        return None
    modes = [m for m in ("greedy", "sample") if any(r["mode"] == m for r in sel)]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes) + 1.5, 4.2), sharey=True, squeeze=False)
    w = 0.8 / len(systems)
    pad = max(0.6, 0.02 * max(r["max"] for r in sel))     # value label above the whisker
    for ax, mode in zip(axes[0], modes):
        for j, s in enumerate(systems):
            color, hatch = style(s, theme, systems)
            for i, (cls, _) in enumerate(CLASSES):
                r = next((x for x in sel if x["system"] == s and x["cls"] == cls and x["mode"] == mode), None)
                if not r:
                    continue
                x = i - 0.4 + w * (j + 0.5)
                ax.bar(x, r["decode_tps"], w * 0.92, color=color, hatch=hatch, linewidth=0,
                       label=LABEL.get(base(s), s) if i == 0 else None, zorder=2)
                ax.errorbar(x, r["decode_tps"], yerr=[[r["decode_tps"] - r["min"]], [r["max"] - r["decode_tps"]]],
                            fmt="none", ecolor=th["ink2"], elinewidth=1, capsize=0, zorder=3)
                if len(systems) <= 5:      # more bars than that: the report's table carries the values
                    ax.text(x, r["max"] + pad, f"{r['decode_tps']:.1f}", ha="center", va="bottom", fontsize=7.5,
                            color=th["ink2"])
        ax.set_xticks(range(len(CLASSES)), [c[1] for c in CLASSES])
        ax.set_title("greedy (temperature 0)" if mode == "greedy" else "sampling (T 0.6, top-p 0.95, top-k 20)",
                     fontsize=10, color=th["ink"], loc="left")
        ax.tick_params(axis="x", length=0)
    axes[0][0].set_ylabel("decode tokens/s")
    fig.text(0.99, 0.955, "median of sessions; whisker = min–max", ha="right", fontsize=8, color=th["ink2"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, -0.02), labelcolor=th["ink"])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=th["ink"])
    fig.tight_layout(rect=(0, 0.1 + 0.04 * (len(labels) > 4), 1, 0.95))
    p = os.path.join(out, f"{prefix}_2k_{theme}.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def lines(plt, rows, theme, out, title, prefix="decode"):
    th = THEMES[theme]
    setup(plt, th)
    sel = [r for r in rows if r["mode"] == "greedy" and r["max_tokens"] == 256 and not r["thinking"]]
    classes = [c for c in CLASSES if len({r["ctx"] for r in sel if r["cls"] == c[0]}) > 1]
    if not classes:
        return None
    fig, axes = plt.subplots(1, len(classes), figsize=(5.2 * len(classes) + 1.5, 4.2), sharey=True, squeeze=False)
    systems = sorted({r["system"] for r in sel}, key=sort_key)
    for ax, (cls, cname) in zip(axes[0], classes):
        ends = []
        for s in systems:
            pts = sorted((r["ctx"], r["decode_tps"]) for r in sel if r["system"] == s and r["cls"] == cls)
            if len(pts) < 2:
                continue
            color, hatch = style(s, theme, systems)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2,
                    linestyle="--" if hatch else "-", marker="o", markersize=6, markeredgecolor=th["bg"],
                    markeredgewidth=2, label=LABEL.get(base(s), s), zorder=3)
            ends.append(pts[-1])
        ymax = max((r["decode_tps"] for r in sel), default=1)
        last = None
        for pt in sorted(ends, key=lambda p: -p[1]):   # end labels, skipping ones that would collide
            if last is None or last - pt[1] > 0.045 * ymax:
                ax.annotate(f"{pt[1]:.1f}", pt, textcoords="offset points", xytext=(6, 0), va="center",
                            fontsize=7.5, color=th["ink2"])
                last = pt[1]
        ax.set_xscale("log", base=2)
        ticks = sorted({r["ctx"] for r in sel if r["cls"] == cls})
        ax.set_xticks(ticks, [f"{t // 1024}K" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("prompt tokens")
        ax.set_title(cname, fontsize=10, color=th["ink"], loc="left")
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("greedy decode tokens/s")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, -0.02), labelcolor=th["ink"])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=th["ink"])
    fig.tight_layout(rect=(0, 0.1 + 0.04 * (len(labels) > 4), 1, 0.95))
    p = os.path.join(out, f"{prefix}_ctx_{theme}.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--themes", default="light,dark")
    ap.add_argument("--title", default="", help="default: '<model of the systems> decode on <machine>'")
    ap.add_argument("--machine", default="M1 Max 32 GB")
    ap.add_argument("--prefix", default="decode", help="file names: <prefix>_2k_<theme>.png, <prefix>_ctx_<theme>.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(a.agg))["rows"]
    os.makedirs(a.out, exist_ok=True)
    title = a.title or model_title(rows, a.machine)
    for theme in a.themes.split(","):
        for p in (bars(plt, rows, theme, a.out, title + ", 2K context", a.prefix),
                  lines(plt, rows, theme, a.out, title + ", by context", a.prefix)):
            if p:
                print("wrote", p)


if __name__ == "__main__":
    main()
