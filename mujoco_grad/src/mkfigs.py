"""Render the report figures as vector PDFs (matplotlib), for \\includegraphics.

pgfplots was hanging on this many log-log axes; matplotlib renders the same content
in a couple of seconds and keeps the LaTeX compile fast.
"""

import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS

FIGS = os.path.join(RESULTS, "figs")
CAT = ["flight", "stance", "loaded", "onset", "release", "sliding", "impact"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.2, "axes.titlesize": 8,
    "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})


def load(n):
    p = os.path.join(RESULTS, n)
    return json.load(open(p)) if os.path.exists(p) else None


def jsonl(n):
    p = os.path.join(RESULTS, n)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def clean(xs, ys, floor):
    X, Y = [], []
    for x, y in zip(xs, ys):
        if y is None or y != y or not math.isfinite(y) or y <= 0:
            continue
        X.append(x); Y.append(max(y, floor))
    return X, Y


def panel(ax, title, series, xlim, ylim, xlabel, ylabel, floor):
    for lab, xs, ys, c, ls in series:
        X, Y = clean(xs, ys, floor)
        if X:
            ax.plot(X, Y, ls, color=c, marker="o", markersize=2.2, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, pad=3)
    ax.grid(True, which="major", color="0.85")
    ax.grid(True, which="minor", color="0.94", linewidth=0.3)
    ax.tick_params(length=2.2, pad=1.5)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=1)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=1)
    for s in ax.spines.values():
        s.set_color("0.5")


def grid_fig(names, build, ncol=3, figsize=(7.0, 6.2), legend_labels=None):
    from matplotlib.lines import Line2D
    n = len(names)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.ravel() if n > 1 else [axes]
    for i, nm in enumerate(names):
        build(axes[i], nm, i, nrow, ncol)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    if legend_labels:
        # explicit proxies: a series that is identically zero is filtered out of every
        # panel and would silently vanish from a handles-derived legend
        proxies = [Line2D([], [], color=c, ls=ls, marker="o", markersize=2.6, label=lab)
                   for lab, c, ls in legend_labels]
        if n < len(axes):
            axes[n].legend(handles=proxies, loc="center", frameon=False, fontsize=6.8)
            fig.tight_layout(pad=0.5, w_pad=0.9, h_pad=0.9)
        else:
            fig.tight_layout(pad=0.5, w_pad=0.9, h_pad=0.9)
            fig.legend(handles=proxies, loc="upper center", ncol=len(proxies),
                       bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=6.8)
        return fig
    fig.tight_layout(pad=0.5, w_pad=0.9, h_pad=0.9)
    return fig


def main():
    os.makedirs(FIGS, exist_ok=True)
    sw = load("phase34_sweep_S1.json")
    pyr = load("phase34_sweep_pyramidal.json")
    mjx = load("mjx_control_S1.json")
    tay = load("phase5_taylor_S1.json")
    scale = jsonl("bench_scale_S5.jsonl")

    # ---------------- figure 1: epsilon sweep (elliptic) --------------------
    def b1(ax, nm, i, nrow, ncol):
        rr = sw["results"][nm]["rows"]
        hs = [r["h"] for r in rr]
        s = [("MJWarp float32", hs, [r["E_gold_A"] for r in rr], ORANGE, "-"),
             ("CPU float64", hs, [r["E_cpu_A"] for r in rr], BLUE, "-"),
             ("MJWarp run-to-run noise", hs, [r["noise"] for r in rr], AQUA, "--")]
        panel(ax, nm, s, (5e-7, 2e-2), (1e-13, 1e3),
              "$h$" if i // ncol == nrow - 1 else "",
              "rel. error of $A$" if i % ncol == 0 else "", 1e-13)
    f = grid_fig(CAT, b1, legend_labels=[("MJWarp float32", ORANGE, "-"),
                                     ("CPU float64", BLUE, "-"),
                                     ("MJWarp run-to-run noise", AQUA, "--")])
    f.savefig(os.path.join(FIGS, "sweep.pdf")); plt.close(f)

    # ---------------- figure 2: precision control (pyramidal) ---------------
    shown = [c for c in CAT if c in mjx["results"] and c in pyr["results"]]

    def b2(ax, nm, i, nrow, ncol):
        mr = mjx["results"][nm]["rows"]; pr = pyr["results"][nm]["rows"]
        hs = [r["h"] for r in mr]
        s = [("MJWarp float32 (GPU)", [r["h"] for r in pr],
              [r["E_gold_A"] for r in pr], ORANGE, "-"),
             ("MJX float64 (GPU)", hs, [r["E_gold_A"] for r in mr], AQUA, "-"),
             ("CPU float64", hs, [r["E_cpu_A"] for r in mr], BLUE, "--")]
        panel(ax, nm, s, (5e-7, 2e-2), (1e-13, 1e3),
              "$h$" if i // ncol == nrow - 1 else "",
              "rel. error of $A$" if i % ncol == 0 else "", 1e-13)
        if all(r["E_gold_A"] != r["E_gold_A"] for r in pr):
            ax.text(0.5, 0.06, "MJWarp: NaN at every $h$", transform=ax.transAxes,
                    ha="center", fontsize=6.2, color=ORANGE)
    f = grid_fig(shown, b2, figsize=(7.0, 4.3),
             legend_labels=[("MJWarp float32 (GPU)", ORANGE, "-"),
                            ("MJX float64 (GPU)", AQUA, "-"),
                            ("CPU float64", BLUE, "--")])
    f.savefig(os.path.join(FIGS, "precision.pdf")); plt.close(f)

    # ---------------- figure 3: Taylor test ---------------------------------
    def b3(ax, nm, i, nrow, ncol):
        r = tay["results"][nm]; al = r["alphas"]; med = r["median"]
        s = [(r"CPU Jac $\to$ CPU sim", al, med["cpu_cpu"], BLUE, "-"),
             (r"MJWarp Jac $\to$ CPU sim", al, med["warp_cpu"], ORANGE, "-"),
             (r"MJWarp Jac $\to$ MJWarp sim", al, med["warp_warp"], AQUA, "--")]
        panel(ax, f"{nm}  (slope {r['slopes']['cpu_cpu']:.2f} / "
                  f"{r['slopes']['warp_cpu']:.2f})", s,
              (5e-5, 6e-1), (1e-11, 3e1),
              r"$\alpha$" if i // ncol == nrow - 1 else "",
              r"$r(\alpha)$" if i % ncol == 0 else "", 1e-11)
    f = grid_fig(CAT, b3, legend_labels=[(r"CPU Jac $\to$ CPU sim", BLUE, "-"),
                                     (r"MJWarp Jac $\to$ CPU sim", ORANGE, "-"),
                                     (r"MJWarp Jac $\to$ MJWarp sim", AQUA, "--")])
    f.savefig(os.path.join(FIGS, "taylor.pdf")); plt.close(f)

    # ---------------- figure 4: scaling -------------------------------------
    if scale:
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        ks = [r["K"] for r in scale]
        ax.plot(ks, [r["cpu_per"] * 1e3 for r in scale], "-o", color=BLUE,
                markersize=3, label="CPU rollout, 32 threads")
        gk = [(r["K"], r["gpu_per"] * 1e3) for r in scale
              if r["gpu_per"] == r["gpu_per"]]
        ax.plot([a for a, _ in gk], [b for _, b in gk], "-o", color=ORANGE,
                markersize=3, label="MJWarp GPU (CUDA graph)")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("shooting knots differentiated at once ($K$)", labelpad=1)
        ax.set_ylabel("ms per Jacobian", labelpad=1)
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.grid(True, which="major", color="0.85")
        ax.grid(True, which="minor", color="0.94", linewidth=0.3)
        ax.tick_params(length=2.2, pad=1.5)
        for s in ax.spines.values():
            s.set_color("0.5")
        ax.legend(frameon=False, fontsize=6.2, loc="upper left")
        fig.tight_layout(pad=0.4)
        fig.savefig(os.path.join(FIGS, "scale.pdf")); plt.close(fig)

    print("wrote", ", ".join(sorted(os.listdir(FIGS))))


if __name__ == "__main__":
    main()
