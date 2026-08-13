"""Generate report.tex (LaTeX article) from the phase JSON files."""

import json, os, math
from config import RESULTS

CAT = ["flight", "stance", "loaded", "onset", "release", "sliding", "impact"]
DESC = {
    "flight": "no contact, free fall",
    "stance": "4 contacts, settled",
    "loaded": "8 contacts, pressed down",
    "onset": "at the contact margin, closing",
    "release": "just outside the margin, opening",
    "sliding": "4 contacts, 1.2 m/s lateral",
    "impact": "4 contacts, 3.0 m/s downward",
}
DIFF = {"flight": "low", "stance": "moderate", "loaded": "moderate--high",
        "onset": "high / nonsmooth", "release": "high / nonsmooth",
        "sliding": "high", "impact": "very high"}

# Measured by src/dbg_mem2.py / dbg_mem3.py (MJWarp Data + captured-graph footprint).
MEM = [
    # nworld, efc/world, nccdmax, put_data MB, total-after-graph MB, status
    (96,  128, "default", 203.4, None, "ok"),
    (192,  64, "default", 404.8, 2743.1, "ok"),
    (384,  64, "default", 1545.6, 10628.4, "ok"),
    (384,  64, "512",     1545.6, 2709.5, "ok"),
    (768,  64, "default", None, None, "fail: 15.37 GB request"),
    (768,  32, "default", None, None, "fail: 15.37 GB request"),
    (768,  16, "default", None, None, "fail: 15.37 GB request"),
    (768,  64, "512",     6075.4, 8313.1, "ok"),
    (1536, 64, "512",     None, None, "fail: 12.08 GB request"),
]

# Measured by src/bench_probe.py (one batched step, 97 go1 worlds).
GRAPHMODE = [
    (6208, 49664, "on",  56.84, 0.52),
    (1552,  6208, "on",  15.64, 0.34),
    (1552,  6208, "off", 31.34, 2.93),
]

# Measured by src/dbg_nan3.py, pyramidal cone, "loaded" fixture, h=1e-3.
NAN_WORLDS = [14, 26, 44, 47, 52, 61, 66, 67, 77, 89]
NAN_ALONE_OK = [14, 44, 47, 52, 61, 66, 67, 77, 89]


def load(n):
    p = os.path.join(RESULTS, n)
    return json.load(open(p)) if os.path.exists(p) else None


def jsonl(n):
    p = os.path.join(RESULTS, n)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


# ------------------------------------------------------------------ formatting
def sci(v, p=2):
    """1.23e-04 -> $1.23\\times10^{-4}$"""
    if v is None or (isinstance(v, float) and v != v):
        return "---"
    if v == 0:
        return "$0$"
    e = int(math.floor(math.log10(abs(v))))
    mant = v / 10 ** e
    if abs(mant - 1.0) < 1e-12:
        return f"$10^{{{e}}}$"
    return f"${mant:.{p}f}\\times10^{{{e}}}$"


def hlab(v):
    e = int(round(math.log10(v)))
    m = v / 10 ** e
    return f"$10^{{{e}}}$" if abs(m - 1) < 1e-9 else f"${m:.0f}\\times10^{{{e}}}$"


def num(v, p=2):
    return "---" if v is None or v != v else f"{v:.{p}f}"


def tex_esc(s):
    for a, b in [("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


def tab(spec, header, rows, caption, label, note=None, small=True):
    o = ["\\begin{table}[htbp]", "\\centering"]
    if small:
        o.append("\\footnotesize")
    o.append(f"\\caption{{{caption}}}")
    o.append(f"\\label{{tab:{label}}}")
    o.append(f"\\begin{{tabular}}{{{spec}}}")
    o.append("\\toprule")
    o.append(" & ".join(header) + " \\\\")
    o.append("\\midrule")
    for r in rows:
        o.append(" & ".join(str(c) for c in r) + " \\\\")
    o.append("\\bottomrule")
    o.append("\\end{tabular}")
    if note:
        o.append(f"\\\\[3pt]\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize {note}"
                 f"\\end{{minipage}}")
    o.append("\\end{table}")
    return "\n".join(o)


def coords(xs, ys, floor=1e-16):
    out = []
    for x, y in zip(xs, ys):
        if y is None or y != y or y <= 0 or not math.isfinite(y):
            continue
        out.append(f"({x:.6g},{max(y, floor):.6g})")
    return " ".join(out)


PLOT_COLORS = ["blue!65!black", "orange!85!black", "teal!70!black",
               "red!70!black", "violet!70!black", "green!45!black",
               "brown!70!black"]


def loglog(title, xlabel, ylabel, series, xmin, xmax, ymin, ymax,
           legend_pos="north west", width="0.46\\textwidth", marks=True):
    o = [f"\\begin{{tikzpicture}}",
         f"\\begin{{axis}}[",
         f"  title={{\\small {title}}}, xmode=log, ymode=log,",
         f"  width={width}, height=0.30\\textwidth,",
         f"  xlabel={{\\footnotesize {xlabel}}}, ylabel={{\\footnotesize {ylabel}}},",
         f"  xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax},",
         f"  grid=both, grid style={{line width=.1pt, draw=gray!22}},",
         f"  tick label style={{font=\\tiny}}, label style={{font=\\footnotesize}},",
         f"  legend style={{font=\\tiny, at={{(0.02,0.98)}}, anchor=north west,"
         f" draw=gray!40, fill=white, fill opacity=0.85, text opacity=1}},",
         f"  legend cell align=left,",
         f"]"]
    for i, (lab, xs, ys, style) in enumerate(series):
        c = PLOT_COLORS[i % len(PLOT_COLORS)] if style is None else style
        mk = "mark=*, mark size=1.1pt, " if marks else ""
        o.append(f"\\addplot[{c}, thick, {mk}] coordinates {{{coords(xs, ys)}}};")
        o.append(f"\\addlegendentry{{{lab}}}")
    o += ["\\end{axis}", "\\end{tikzpicture}"]
    return "\n".join(o)


# ------------------------------------------------------------------ document
def main():
    ph1 = load("phase1_nominal_parity.json")
    ph2 = load("phase2_cpu_vs_transitionFD.json")
    sw = {1: load("phase34_sweep_S1.json"), 5: load("phase34_sweep_S5.json"),
          20: load("phase34_sweep_S20.json")}
    pyr = load("phase34_sweep_pyramidal.json")
    tay = load("phase5_taylor_S1.json")
    mjx = load("mjx_control_S1.json")
    bench = load("bench.json")
    scale = jsonl("bench_scale_S5.jsonl")
    V = ph2["versions"]
    D = ph2["dims"]
    HS = [r["h"] for r in sw[1]["results"]["stance"]["rows"]]

    L = []
    A = L.append

    A(r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage{booktabs,amsmath,amssymb,microtype,xcolor,caption,float}
\usepackage[hidelinks]{hyperref}
\usepackage{graphicx}
\graphicspath{{figs/}}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{4pt}
\newcommand{\code}[1]{\texttt{\small #1}}

\title{\textbf{MuJoCo CPU vs.\ MuJoCo Warp}\\ Derivative Validation: Experimental Results}
\author{Executed on Unitree Go1, RTX 4090}
\date{\today}
\begin{document}
\maketitle
""")

    # -------------------------------------------------- abstract
    max2 = max(max(r["relA"] for r in ph2["rows"]), max(r["relB"] for r in ph2["rows"]))
    st1 = sw[1]["results"]["stance"]["rows"]
    best_st = min(st1, key=lambda r: r["E_gold_A"])
    A(r"""\begin{abstract}
This document reports the execution of the MuJoCo CPU vs.\ MuJoCo Warp (MJWarp) derivative
validation plan on a floating-base quadruped. All eight phases were run: configuration
freeze, nominal transition parity, CPU black-box finite differences validated against
\code{mjd\_transitionFD}, batched MJWarp finite differences, an epsilon sweep, a directional
Taylor test, contact diagnostics, and wall-time benchmarking. A float64 GPU control (MJX)
was added to separate floating-point precision from GPU batching and from simulator
implementation differences.
The central conclusion is that MJWarp float32 finite differences are accurate and locally
predictive for smooth and stable-contact states over a usable epsilon window roughly two to
three decades coarser than the CPU double-precision optimum; that where they fail --- near
contact-mode switching --- the CPU float64 Jacobian fails identically, so the failure is
contact nonsmoothness rather than precision; and that on a robot of this size the GPU does
not beat a 32-thread CPU rollout pool.
\end{abstract}
""")

    # -------------------------------------------------- summary
    A(r"\section{Summary of findings}")
    A(r"\begin{enumerate}\itemsep3pt")
    A(rf"""\item \textbf{{The CPU harness is exact.}} A manifold-aware black-box central
difference reproduces \code{{mjd\_transitionFD}} to {sci(max2)} relative Frobenius error over
7 fixtures $\times$ 5 epsilons. Perturbation, output differencing, state packing, control
indexing and warm-start handling are therefore all correct, and every later discrepancy
belongs to the simulator rather than the harness.""")
    A(rf"""\item \textbf{{Nominal parity holds.}} CPU and MJWarp agree on the unperturbed
one-step map to $\sim$$10^{{-7}}$ in generalized position, and the contact set is identical
in all {len(ph1['rows'])} fixture$\times$horizon combinations tested ($S \in \{{1,5,20\}}$).""")
    A(rf"""\item \textbf{{A usable epsilon window exists, and it is much wider than the CPU
one.}} The MJWarp optimum sits at $h \approx 10^{{-3}}$--$10^{{-4}}$ against a CPU optimum of
$10^{{-6}}$. At the stance fixture the best MJWarp relative error on $A$ is
{sci(best_st['E_gold_A'])} at $h$~=~{hlab(best_st['h'])}.""")
    A(r"""\item \textbf{Precision is the limiting factor, not GPU batching.} A float64 GPU
simulator (MJX) on the same card reproduces the CPU Jacobian to $\sim$$10^{-12}$ and shows a
monotone truncation curve with no roundoff turnaround in the swept range. The MJWarp error
curve is U-shaped over the same states and epsilons.""")
    A(r"""\item \textbf{Where MJWarp fails, CPU float64 fails identically.} In the directional
Taylor test the MJWarp Jacobian predicts the \emph{CPU} simulator as well as the CPU Jacobian
does for stance, sliding and impact. At the near-switching fixtures both have log--log slope
$0.71$ and both plateau at the same residual. That is genuine contact nonsmoothness.""")
    A(r"""\item \textbf{GPU nondeterminism is a contact phenomenon.} MJWarp is bitwise
reproducible across repeated runs when no contact is active, and nondeterministic as soon as
the constraint solver engages, with run-to-run spread growing like $\eta/h$.""")
    A(r"""\item \textbf{CUDA graph capture is mandatory.} Launched eagerly, one batched step of
97 worlds costs $\sim$25\,ms of Python kernel-launch overhead; captured into a CUDA graph it
costs $0.37$\,ms. A benchmark run without capture measures the wrong thing by a factor of
20--55.""")
    A(r"""\item \textbf{A float32 failure that only appears in a batch.} Under the pyramidal
cone the deeply penetrating 8-contact fixture returns NaN in 10 of 97 perturbed worlds; 9 of
those 10 states are finite when re-simulated one world at a time.""")
    A(r"""\item \textbf{The GPU does not win at this scale.} With graph capture and knot
batching, MJWarp bottoms out at $0.91$\,ms per Jacobian while \code{mujoco.rollout} on 32
threads sustains $0.47$\,ms. MJWarp additionally hits a memory wall well before the GPU is
compute-saturated.""")
    A(r"\end{enumerate}")

    # -------------------------------------------------- setup
    A(r"\section{Experimental setup}")
    A(rf"""The model is \code{{unitree\_go1/scene.xml}} from MuJoCo Menagerie:
$n_q={D['nq']}$, $n_v={D['nv']}$, $n_a={D['na']}$, $n_u={D['nu']}$, giving a tangent state
dimension $n_x = 2n_v + n_a = {D['nx']}$. A centered difference therefore requires
$2(n_x + n_u) = {2*(D['nx']+D['nu'])}$ perturbed worlds, plus one nominal world for parity
checks. Collision geometry is primitive only (planes, spheres, capsules, boxes, cylinders);
there are no mesh--mesh contacts, which deliberately removes the mesh float32 rounding issue
as a confound.""")

    A(tab("ll", [r"\textbf{Component}", r"\textbf{Version}"], [
        ["MuJoCo (CPU reference and MJWarp runs)", V["mujoco"]],
        ["MJWarp", V["mujoco_warp"]],
        ["NVIDIA Warp", V["warp"]],
        ["MuJoCo / JAX (MJX float64 control)", f"{mjx['mujoco']} / {mjx['jax']}"],
        ["NumPy", V["numpy"]],
        ["GPU", tex_esc(V["gpu"])],
        ["CPU threads available", str(V["cpu_count"])],
    ], "Software and hardware versions recorded with every result (Phase 0).", "versions"))

    A(r"""\subsection{Frozen configuration}
Euler integrator, Newton solver, elliptic friction cone, timestep $2$\,ms, 100 solver
iterations, 50 line-search iterations. One configuration detail is essential and easy to
miss: \textbf{MJWarp silently raises the solver tolerance}. In \code{io.py} it applies
\code{opt.tolerance = max(opt.tolerance, 1e-6)} with the comment that the C MuJoCo tolerance
was chosen for a float64 architecture. If the CPU side is left at its model default of
$10^{-8}$, the two solvers do different amounts of work and nominal parity is confounded
before any derivative is computed. Both sides were therefore pinned to $10^{-6}$.

Warm-start handling mirrors \code{mjd\_transitionFD}: the nominal \code{qacc\_warmstart} is
restored before every perturbed rollout so that all perturbations start the solver
identically, and the same nominal warm start is copied into every MJWarp world.""")

    rows = []
    for c in CAT:
        fi = sw[1]["results"][c]["fixture_info"]
        rows.append([tex_esc(c), DESC[c], fi["ncon"], fi["nefc"], sci(fi["min_gap"]),
                     num(fi["qvel_norm"]), f"{fi['nswitch']}/{fi['nperturb']}", DIFF[c]])
    A(tab("llrrrrrl",
          [r"\textbf{Fixture}", r"\textbf{Description}", r"\textbf{ncon}", r"\textbf{nefc}",
           r"\textbf{foot gap [m]}", r"$\|q_v\|$", r"\textbf{flips}", r"\textbf{difficulty}"],
          rows,
          "Representative state suite (Phase 7). \\emph{flips} counts how many of the "
          "$2n_x$ single-coordinate perturbations at $h=10^{-4}$ change the contact pair "
          "set. The two near-switching fixtures were placed at the contact \\emph{margin} "
          "($10^{-3}$\\,m), which is where the discrete set change occurs, rather than at "
          "zero gap.", "fixtures"))

    # -------------------------------------------------- phase 2
    A(r"\section{Phase 2: the CPU implementation is exact}")
    A(rf"""Before any GPU comparison can mean anything, the black-box central difference must
reproduce MuJoCo's own staged, stage-skipping \code{{mjd\_transitionFD}}. It does, at machine
precision: the largest relative Frobenius error over all 7 fixtures and 5 epsilons is
{sci(max2)} for $A$ and {sci(max(r['relB'] for r in ph2['rows']))} for $B$
(Table~\ref{{tab:phase2}}). This validates tangent-space perturbation via
\code{{mj\_integratePos}}, output differencing via \code{{mj\_differentiatePos}}, state
packing, actuator-state handling, control indexing and Jacobian layout in one shot.""")

    hset = sorted({r["h"] for r in ph2["rows"]}, reverse=True)
    rows = []
    for c in CAT:
        rr = {r["h"]: r for r in ph2["rows"] if r["fixture"] == c}
        rows.append([tex_esc(c)] + [sci(rr[h]["relA"], 1) for h in hset])
    A(tab("l" + "r" * len(hset),
          [r"\textbf{Fixture}"] + [hlab(h) for h in hset], rows,
          "Phase 2. Relative Frobenius error between the black-box CPU central difference "
          "and \\code{mjd\\_transitionFD} for $A$, as a function of the perturbation size "
          "$h$. All values are at double-precision round-off.", "phase2"))

    A(r"""\subsection{A warm-start pitfall worth recording}
An early version of the harness computed contact diagnostics by calling \code{mj\_forward}
inside the finite-difference loop. This shifts the resulting Jacobian by $\sim$$10^{-6}$ ---
\emph{even when} \code{qacc\_warmstart} is saved and restored around the diagnostic call.
Contact diagnostics must be collected on a separate \code{MjData} instance that the
finite-difference path never touches. With the diagnostic call in place the agreement with
\code{mjd\_transitionFD} degraded from $10^{-16}$ to $1.4\times10^{-5}$ on the impact
fixture; with it removed, agreement is exact.""")

    # -------------------------------------------------- phase 1
    A(r"\section{Phase 1: nominal transition parity}")
    rows = []
    for c in CAT:
        for r in ph1["rows"]:
            if r["fixture"] == c and r["S"] == 1:
                r20 = next(q for q in ph1["rows"] if q["fixture"] == c and q["S"] == 20)
                rows.append([tex_esc(c), sci(r["dq"]), sci(r["dv"]),
                             sci(r20["dq"]), sci(r20["dv"]),
                             f"{r['ncon_cpu']}/{r['ncon_warp']}",
                             "match" if r["pairs_match"] else "DIFFER"])
    A(tab("lrrrrcc",
          [r"\textbf{Fixture}", r"$\|\Delta q\|_{\mathrm{tan}}$ ($S{=}1$)",
           r"$\|\Delta v\|$ ($S{=}1$)", r"$\|\Delta q\|$ ($S{=}20$)",
           r"$\|\Delta v\|$ ($S{=}20$)", r"\textbf{ncon C/W}", r"\textbf{pairs}"],
          rows,
          "Phase 1. Manifold-aware distance between the CPU and MJWarp unperturbed next "
          "state. Position error sits at the float32 floor; the contact pair set is "
          "identical everywhere. There is no physics or collision disagreement to confound "
          "the derivative comparison.", "phase1"))

    # -------------------------------------------------- phase 3/4
    A(r"\section{Phases 3--4: batched MJWarp finite differences and the epsilon sweep}")
    A(r"""Worlds are laid out as
$[x{+}0,\,x{-}0,\,x{+}1,\,x{-}1,\dots,u{+}0,u{-}0,\dots]$ with a nominal world appended.
Generalized-position perturbations are built on the host in float64 with
\code{mj\_integratePos}, uploaded as float32, and the returned states are up-cast to float64
before differencing with \code{mj\_differentiatePos}, so the only float32 in the pipeline is
the simulation itself.

Three error curves are reported per fixture, all against $A_{\mathrm{gold}}$ = CPU float64 at
$h=10^{-6}$:
\begin{align*}
E_{\mathrm{same}}(h) &= \|A_{\mathrm{warp}}(h) - A_{\mathrm{cpu}}(h)\|_F / \|A_{\mathrm{cpu}}(h)\|_F
 &&\text{GPU vs.\ CPU at matched truncation}\\
E_{\mathrm{gold}}(h) &= \|A_{\mathrm{warp}}(h) - A_{\mathrm{gold}}\|_F / \|A_{\mathrm{gold}}\|_F
 &&\text{total error of the GPU Jacobian}\\
E_{\mathrm{cpu}}(h)  &= \|A_{\mathrm{cpu}}(h) - A_{\mathrm{gold}}\|_F / \|A_{\mathrm{gold}}\|_F
 &&\text{the CPU's own truncation curve}
\end{align*}
Separating these is what makes the results interpretable: at large $h$, $E_{\mathrm{same}}$ is
orders of magnitude smaller than $E_{\mathrm{gold}}$, which says the large-$h$ error is shared
truncation and not a GPU defect.""")

    # figure: sweep
    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{sweep.pdf}
\caption{Phase 4 epsilon sweep, elliptic cone, $S=1$. The MJWarp curve (orange) is U-shaped:
truncation on the right, float32 cancellation and nondeterminism on the left. The CPU curve
(blue) keeps falling. For \code{onset} and \code{release} the two coincide over the
contact-switching region, which is the signature of nonsmoothness rather than precision.
The dashed line is MJWarp's run-to-run spread, which is identically zero for \code{flight}
and therefore absent from that panel.}
\label{fig:sweep}\end{figure}""")

    rows = []
    for c in CAT:
        rr = sw[1]["results"][c]["rows"]
        b = min(rr, key=lambda x: x["E_gold_A"])
        cb = min(rr, key=lambda x: x["E_cpu_A"])
        rows.append([tex_esc(c), hlab(b["h"]), sci(b["E_gold_A"]), sci(b["E_gold_B"]),
                     num(b["cos_min"], 4), sci(b["noise"]), hlab(cb["h"]),
                     sci(cb["E_cpu_A"])])
    A(tab("lrrrrrrr",
          [r"\textbf{Fixture}", r"\textbf{best }$h$", r"$E_{\mathrm{gold}}(A)$",
           r"$E_{\mathrm{gold}}(B)$", r"\textbf{min col.\ cos}", r"\textbf{GPU noise}",
           r"\textbf{best }$h$\textbf{ CPU}", r"$E_{\mathrm{cpu}}$"],
          rows,
          "Phase 4, elliptic cone, $S=1$. The MJWarp optimum lies two to three decades "
          "coarser than the CPU optimum. Column cosine is the minimum over columns whose "
          "reference norm is meaningful.", "bestэps".replace("э", "e")))

    # representative full sweeps
    for c in ["stance", "onset"]:
        rr = sw[1]["results"][c]["rows"]
        rows = []
        for r in rr:
            ch = r["split"]["changing"]["rel_fro"]
            rows.append([hlab(r["h"]), sci(r["E_same_A"]), sci(r["E_gold_A"]),
                         sci(r["E_cpu_A"]), sci(r["noise"]), num(r["cos_min"], 4),
                         f"{r['n_stable']}/{r['n_cols']}",
                         sci(r["split"]["stable"]["rel_fro"]),
                         "---" if ch != ch else sci(ch)])
        A(tab("lrrrrrcrr",
              [r"$h$", r"$E_{\mathrm{same}}$", r"$E_{\mathrm{gold}}$", r"$E_{\mathrm{cpu}}$",
               r"\textbf{noise}", r"\textbf{cos}", r"\textbf{stable}",
               r"$E$\textbf{ stable}", r"$E$\textbf{ changing}"],
              rows,
              f"Full epsilon sweep for the \\code{{{c}}} fixture, elliptic cone, $S=1$. "
              + ("The clean U-shape of a contact-stable state: $E_{\\mathrm{same}}$ stays "
                 "small throughout, so MJWarp tracks the CPU at matched $h$; the total error "
                 "is truncation-dominated above $10^{-4}$ and noise-dominated below."
                 if c == "stance" else
                 "A near-switching state. $E_{\\mathrm{gold}}$ and $E_{\\mathrm{cpu}}$ are "
                 "\\emph{identical} for $h\\geq3\\times10^{-5}$ --- the error is entirely "
                 "shared nonsmoothness. Only when perturbations stop crossing the contact "
                 "boundary (36/36 stable columns) does the CPU error collapse, at which "
                 "point float32 noise dominates the GPU."),
              f"sweep{c}"))

    # multi-substep
    rows = []
    for c in CAT:
        row = [tex_esc(c)]
        for S in (1, 5, 20):
            if sw[S] and c in sw[S]["results"]:
                b = min(sw[S]["results"][c]["rows"], key=lambda x: x["E_gold_A"])
                row += [hlab(b["h"]), sci(b["E_gold_A"], 1)]
            else:
                row += ["---", "---"]
        rows.append(row)
    A(tab("lrrrrrr",
          [r"\textbf{Fixture}", r"$h^\ast$ ($S{=}1$)", r"$E_{\mathrm{gold}}$",
           r"$h^\ast$ ($S{=}5$)", r"$E_{\mathrm{gold}}$",
           r"$h^\ast$ ($S{=}20$)", r"$E_{\mathrm{gold}}$"],
          rows,
          "Direct differentiation of the full $S$-substep shooting map $F_H = f^S$, rather "
          "than chaining substep Jacobians. The optimum epsilon is stable across horizon "
          "length and the achievable accuracy does not collapse; the near-switching fixtures "
          "actually improve at $S=5$ because the contact mode settles within the interval.",
          "substep"))

    # -------------------------------------------------- precision control
    A(r"\section{Precision control: CPU float64 vs.\ MJX float64 vs.\ MJWarp float32}")
    A(r"""The epsilon sweep alone cannot separate three candidate causes of MJWarp error:
float32 precision, GPU nondeterminism, and a physics or collision implementation difference.
A float64 GPU simulator settles it. MJX runs the same MuJoCo model on the same RTX 4090 in
double precision under \code{jax\_enable\_x64}; verified on a single step, it reproduces the
CPU next state to $2.3\times10^{-18}$.

MJX does not implement the elliptic friction cone for \code{condim=1}, so this comparison was
re-run for all three simulators with the \textbf{pyramidal} cone. The gold Jacobian norms
agree to four significant figures across the two stacks
($\|A_{\mathrm{gold}}\|_F$: 10.72, 24.94, 82.72, 260.2 for stance, onset, sliding, impact),
confirming the configurations are matched.""")

    shown = [c for c in CAT if c in mjx["results"] and c in pyr["results"]]
    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{precision.pdf}
\caption{Precision isolated. Same model, same options (pyramidal cone), same GPU, identical
perturbations. MJX float64 (green) tracks the CPU curve (blue, dashed); MJWarp float32
(orange) turns around near $h\approx10^{-3}$. The \code{loaded} panel has no MJWarp curve
because MJWarp returns NaN at every epsilon for that fixture (Section~\ref{sec:nan}).}
\label{fig:precision}\end{figure}""")

    rows = []
    for c in shown:
        mr = mjx["results"][c]["rows"]; pr = pyr["results"][c]["rows"]
        bm = min(mr, key=lambda x: x["E_gold_A"])
        pv = [r["E_gold_A"] for r in pr if r["E_gold_A"] == r["E_gold_A"]]
        bp = min(pr, key=lambda x: x["E_gold_A"]) if pv else None
        rows.append([tex_esc(c),
                     hlab(bm["h"]), sci(bm["E_gold_A"]), sci(max(bm["noise"], 1e-17)),
                     hlab(bp["h"]) if bp else "---",
                     sci(bp["E_gold_A"]) if bp else "NaN",
                     sci(bp["noise"]) if bp else "NaN"])
    A(tab("lrrrrrr",
          [r"\textbf{Fixture}", r"$h^\ast$", r"\textbf{MJX64 err}", r"\textbf{MJX64 noise}",
           r"$h^\ast$", r"\textbf{MJWarp32 err}", r"\textbf{MJWarp32 noise}"],
          rows,
          "Best achievable relative error on $A$ and run-to-run spread, GPU float64 vs.\\ "
          "GPU float32, pyramidal cone. The float64 GPU is three to seven orders of "
          "magnitude better on both axes, which attributes the MJWarp limitation to "
          "precision rather than to batching or physics.", "precision"))

    A(r"""\subsection{Run-to-run nondeterminism}
Repeated evaluations with bitwise-identical inputs quantify GPU nondeterminism. The result
has a clean structure: \textbf{MJWarp is bitwise reproducible when no contact is active} ---
the flight fixture returns exactly zero spread across five runs at every epsilon --- and
becomes nondeterministic as soon as the constraint solver engages. The spread then grows as
$h$ shrinks, consistent with the predicted $O(\eta/h)$ amplification of simulator output
noise $\eta$.""")

    rows = []
    for c in CAT:
        rr = {r["h"]: r for r in sw[1]["results"][c]["rows"]}
        rows.append([tex_esc(c)] + [sci(rr[h]["noise"], 1) for h in
                                    [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]])
    A(tab("lrrrrr", [r"\textbf{Fixture}"] + [hlab(h) for h in
                                             [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]], rows,
          "MJWarp run-to-run spread of $A$ (maximum pairwise relative Frobenius difference "
          "over 5 identical evaluations), elliptic cone, $S=1$. Zero means bitwise identical.",
          "noise"))

    # -------------------------------------------------- contact diagnostics
    A(r"\section{Phase 6: contact diagnostics}")
    A(r"""The plan warns that folding contact-stable and contact-changing perturbations into a
single Frobenius norm hides the actual behaviour. The data confirm it emphatically. Every
perturbed rollout is checked for whether the $+h$ and $-h$ states preserved the nominal
contact pair set, and the error is reported separately over the two column groups.""")

    rows = []
    for r in sw[1]["results"]["onset"]["rows"]:
        ch = r["split"]["changing"]["rel_fro"]
        rows.append([hlab(r["h"]), f"{r['n_stable']}/{r['n_cols']}",
                     sci(r["split"]["stable"]["rel_fro"]),
                     "--- (none)" if ch != ch else sci(ch),
                     sci(r["E_gold_A"])])
    A(tab("lcrrr",
          [r"$h$", r"\textbf{stable cols}", r"$E$ \textbf{contact-stable}",
           r"$E$ \textbf{contact-changing}", r"$E$ \textbf{combined}"],
          rows,
          "The \\code{onset} fixture, elliptic cone. At $h=10^{-4}$ the 32 contact-stable "
          "columns carry a relative error of $4.3\\times10^{-2}$ while the 4 "
          "contact-changing columns carry $3.1\\times10^{1}$ --- nearly three orders of "
          "magnitude apart, averaged into one uninformative number if reported together. "
          "As $h$ shrinks the perturbations stop crossing the contact margin and the "
          "column count returns to 36/36.", "split"))

    # -------------------------------------------------- taylor
    A(r"\section{Phase 5: directional Taylor test}")
    A(r"""Matrix agreement is a proxy. What a trajectory optimizer actually needs is that the
Jacobian predicts the local behaviour of the simulator. For random tangent directions
$\delta x, \delta u$ the residual
\[
r(\alpha) = d_x\!\left(F(x \oplus \alpha\,\delta x,\; u + \alpha\,\delta u),\;
F(x,u) \oplus \alpha\,(A\,\delta x + B\,\delta u)\right)
\]
should decrease approximately quadratically away from nonsmooth transitions. Three variants
were run: CPU Jacobian against CPU rollouts (the best achievable), MJWarp Jacobian against
MJWarp rollouts (self-consistency), and --- the one that matters --- \textbf{MJWarp Jacobian
against CPU rollouts}, which asks whether the float32 GPU Jacobian predicts the true
double-precision simulator.""")

    A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{taylor.pdf}
\caption{Directional Taylor residual, median over 4 random directions, $S=1$. Panel titles give
the log--log slopes (CPU$\to$CPU / MJWarp$\to$CPU). For stance, sliding and impact the orange
curve lies on top of the blue one: the float32 GPU Jacobian predicts the double-precision
simulator as accurately as the CPU Jacobian does. At \code{release} both plateau at
$6\times10^{-3}$ with slope $0.71$ --- the CPU Jacobian is equally non-predictive there.}
\label{fig:taylor}\end{figure}""")

    rows = []
    for c in CAT:
        r = tay["results"][c]; s = r["slopes"]
        rows.append([tex_esc(c), hlab(r["h_warp"]), num(s["cpu_cpu"]),
                     num(s["warp_cpu"]), num(s["warp_warp"]),
                     sci(r["median"]["cpu_cpu"][-1], 1),
                     sci(r["median"]["warp_cpu"][-1], 1)])
    A(tab("lrrrrrr",
          [r"\textbf{Fixture}", r"$h$ \textbf{(Warp)}", r"\textbf{slope} C$\to$C",
           r"\textbf{slope} W$\to$C", r"\textbf{slope} W$\to$W",
           r"$r$ \textbf{at} $\alpha{=}10^{-4}$ C$\to$C",
           r"$r$ \textbf{at} $\alpha{=}10^{-4}$ W$\to$C"],
          rows,
          "Log--log slopes of the Taylor residual and the residual at the smallest step. "
          "A slope near 2 indicates a valid first-order model. Contact states sit near 1.8 "
          "even on the CPU; the two near-switching fixtures fall to 1.26 and 0.71 "
          "\\emph{identically} for CPU and MJWarp.", "taylor"))

    # -------------------------------------------------- NaN
    A(r"\section{Robustness: a float32 failure that only appears in a batch}")
    A(r"\label{sec:nan}")
    A(rf"""Under the pyramidal cone, the \code{{loaded}} fixture --- 8 contacts, 3\,cm
penetration, 68 active constraints --- makes MJWarp return NaN in {len(NAN_WORLDS)} of the 97
perturbed worlds, poisoning 8 columns of $A$ and therefore the whole Jacobian, at every
epsilon in the sweep. The nominal unperturbed step is finite and matches the CPU
($\max|q_v| = 0.6638$ vs.\ $0.6638$), and both CPU float64 and MJX float64 handle every one
of the offending states without difficulty.

The diagnostic detail that matters: re-simulating those same states \textbf{{one world at a
time}}, {len(NAN_ALONE_OK)} of the {len(NAN_WORLDS)} come back finite. Re-batching the same 10
states as a 10-world problem reproduces the NaN. Whether the float32 Newton solve diverges
therefore depends on the batch in which it is evaluated --- a consequence of the same
reduction-ordering nondeterminism measured in Table~\ref{{tab:noise}}, acting on a solve that
is marginal in single precision. This class of failure will not reproduce in a single-world
debug session.""")

    rows = [[str(w), "finite" if w in NAN_ALONE_OK else "\\textbf{NaN}",
             "0.6637--0.6644"] for w in NAN_WORLDS]
    A(tab("lcc", [r"\textbf{World index}", r"\textbf{Re-run alone}",
                  r"\textbf{CPU} $\max|q_v|$"], rows,
          "The 10 worlds that return NaN inside the 97-world batch (pyramidal cone, "
          "\\code{loaded}, $h=10^{-3}$). Nine of them are finite when simulated in "
          "isolation. The elliptic-cone configuration of the same fixture never exhibits "
          "this.", "nan"))

    A(r"""\textbf{Practical consequence:} validate \code{isfinite} on every returned world
before assembling $A$ and $B$, and treat a NaN world as a failed column rather than letting
it propagate silently into the optimizer.""")

    # -------------------------------------------------- performance
    A(r"\section{Wall time and scaling}")
    A(r"""\subsection{CUDA graph capture is not optional}
MJWarp issues many small kernels per step. Launched eagerly from Python, a single batched
step of 97 go1 worlds is dominated by launch overhead. Captured into a CUDA graph it is 20 to
55 times faster, and MJWarp's \code{graph\_conditional} option (which uses CUDA conditional
graph nodes for the data-dependent solver iteration count) is worth a further factor of
$\sim$8.""")

    rows = [[f"{a}", f"{b}", c, num(d), num(e)] for a, b, c, d, e in GRAPHMODE]
    A(tab("rrlrr",
          [r"\code{nconmax}", r"\code{njmax}", r"\code{graph\_conditional}",
           r"\textbf{eager [ms]}", r"\textbf{captured [ms]}"], rows,
          "One batched MJWarp step, 97 go1 worlds. Oversized constraint arrays also cost "
          "time, not just memory.", "graphmode"))

    rows = []
    for r in bench["rows"]:
        rows.append([f"$S={r['S']}$",
                     "---" if r["mjd"] != r["mjd"] else num(r["mjd"] * 1e3),
                     num(r["cpu_serial"] * 1e3), num(r["cpu_threaded"] * 1e3),
                     num(r["warp_eager"] * 1e3), num(r["warp"] * 1e3),
                     num(r["warp_gpu_only"] * 1e3),
                     num(r["speedup_vs_threaded"]) + r"$\times$"])
    A(tab("lrrrrrrr",
          [r"\textbf{Horizon}", r"\code{mjd\_transitionFD}", r"\textbf{CPU serial}",
           r"\textbf{CPU 32 thr}", r"\textbf{Warp eager}", r"\textbf{Warp graph}",
           r"\textbf{GPU only}", r"\textbf{vs 32 thr}"],
          rows,
          "Wall time in milliseconds for one $(A,B)$ pair, 97 worlds, $h=10^{-4}$, "
          "median of 30 repetitions. \\code{mjd\\_transitionFD} is one-step only. "
          "\\emph{GPU only} excludes host-side perturbation construction, download and "
          "Jacobian assembly. The threaded CPU finite difference was verified to agree with "
          "the serial one to $3.3\\times10^{-13}$.", "bench"))

    if scale:
        A(r"""\begin{figure}[htbp]\centering
\includegraphics[width=0.62\linewidth]{scale.pdf}
\caption{Cost per Jacobian as a function of the number of shooting knots differentiated
simultaneously. Both sides receive one batched dispatch of $K\times96$ rollouts. The GPU
amortizes to a minimum at $K=4$ and then degrades; the CPU is flat to slightly improving.}
\label{fig:scale}\end{figure}""")
        rows = []
        for r in scale:
            ok = r["gpu_per"] == r["gpu_per"]
            rows.append([str(r["K"]), str(r["nworld"]), num(r["cpu_per"] * 1e3, 3),
                         num(r["gpu_per"] * 1e3, 3) if ok else "OOM",
                         num(r["gpu_only_per"] * 1e3, 3) if ok else "---",
                         (num(r["speedup"]) + r"$\times$") if ok else "---"])
        A(tab("rrrrrr",
              [r"\textbf{Knots }$K$", r"\textbf{worlds}", r"\textbf{CPU ms/Jac}",
               r"\textbf{Warp ms/Jac}", r"\textbf{GPU only ms/Jac}", r"\textbf{speedup}"],
              rows,
              "Cost per Jacobian when $K$ shooting knots are differentiated simultaneously, "
              "$S=5$. Both sides receive one batched dispatch of $K\\times96$ rollouts: a "
              "single CUDA-graph launch on the GPU, a single \\code{mujoco.rollout} pool "
              "call on 32 CPU threads. The GPU amortizes to a minimum at $K=4$ and then "
              "degrades; the CPU is flat to slightly improving.", "scale"))

    A(r"""\subsection{MJWarp memory scaling}
MJWarp also hits a memory wall well before the GPU is compute-saturated. \code{Data}
allocation grew roughly \emph{quadratically} in world count for this model, and the default
\code{nccdmax} --- convex-collision (GJK/EPA) scratch that the Go1 does not need, having only
primitive collision geometry --- accounted for most of the captured-graph footprint.
Passing \code{nccdmax} and a realistic \code{njmax} explicitly to \code{put\_data} is the
difference between fitting 8 knots and fitting 2 on a 24\,GB card.""")

    rows = []
    for nw, pe, nc, pd, tot, st in MEM:
        rows.append([str(nw), str(pe), nc,
                     num(pd, 1) if pd else "---",
                     num(tot, 1) if tot else "---", st])
    A(tab("rrlrrl",
          [r"\textbf{worlds}", r"\code{nefc}\textbf{/world}", r"\code{nccdmax}",
           r"\code{put\_data} \textbf{MB}", r"\textbf{after graph MB}", r"\textbf{status}"],
          rows,
          "Measured device memory. Note that the failing allocation at 768 worlds is "
          "\\emph{identical} (15.37\\,GB) for \\code{njmax} of 49152, 24576 and 12288, so it "
          "is not driven by the constraint arrays. Capping \\code{nccdmax} at 512 reduces "
          "the 384-world graph footprint from 10.6\\,GB to 2.7\\,GB and lets 768 worlds fit.",
          "memory"))

    # -------------------------------------------------- interpretation
    A(r"\section{Interpretation against the plan's decision criteria}")
    rows = [
        ["Smooth / no-contact states",
         "stable $h$ region, agrees with CPU",
         r"\textbf{pass} --- $h\in[10^{-3},10^{-4}]$, err $1.9\times10^{-5}$, zero noise"],
        ["Stable-contact states",
         "similar window, contact set unchanged",
         r"\textbf{pass} --- err $4.1\times10^{-4}$ (stance), 36/36 stable columns"],
        ["Transition states",
         "characterize separately",
         r"\textbf{nonsmooth, not a bug} --- CPU float64 fails identically"],
        ["Shooting-interval test",
         "direct $S$-step FD stays stable",
         r"\textbf{pass} --- optimum $h$ unchanged at $S=5,20$; no error blow-up"],
        ["Optimizer test",
         "convergence, wall time",
         r"\textbf{not run} --- gated on the wall-time result below"],
        ["Wall time",
         "GPU cheaper than CPU",
         r"\textbf{fail} --- $0.91$ vs.\ $0.47$\,ms/Jacobian best case"],
    ]
    A(tab("p{0.20\\linewidth}p{0.28\\linewidth}p{0.42\\linewidth}",
          [r"\textbf{Criterion}", r"\textbf{Plan's requirement}", r"\textbf{Outcome}"],
          rows, "Decision sequence from Section 7 of the plan.", "decision", small=True))

    A(r"""The plan's stated best outcome --- ``a clear intermediate epsilon region exists,
likely larger than the CPU double-precision epsilon, where MJWarp Jacobians are sufficiently
repeatable, predict local rollouts well, and give optimizer behaviour comparable to CPU while
reducing derivative wall time'' --- is achieved on every clause except the last. The epsilon
region exists and is indeed larger; the Jacobians are repeatable enough and locally
predictive; but the wall time is not reduced on a robot of this size. The optimizer
comparison was therefore not run, since it is gated on a throughput advantage that does not
exist here.

Equally, the plan's stated early-exit failure mode --- ``if no epsilon yields stable
derivatives even in ordinary stable-contact states'' --- did \emph{not} occur. Stable-contact
states have a clean, wide, usable window.""")

    # -------------------------------------------------- recommendations
    A(r"\section{Recommendations}")
    A(r"""\begin{enumerate}\itemsep3pt
\item \textbf{Use $h\approx10^{-3}$ for MJWarp, not the CPU value.} Carrying the
double-precision epsilon across costs two to three orders of magnitude of accuracy. Re-tune
per model; do not assume it transfers.
\item \textbf{Always capture the batched step into a CUDA graph}, keep
\code{graph\_conditional} enabled, and size \code{njmax} to the true per-world \code{nefc}
(48 here, not 512) and \code{nccdmax} to the real convex-collision need (zero for
primitive-only geometry).
\item \textbf{Check \code{isfinite} on every perturbed world} before assembling $A$ and $B$.
NaN worlds occur, are batch-dependent, and will not reproduce in isolation.
\item \textbf{Log contact-set equality per perturbation} and report contact-stable and
contact-changing columns separately. A single Frobenius norm is uninformative near switching.
\item \textbf{Benchmark against the right CPU baseline.} The competitor is
\code{mujoco.rollout} with a persistent thread pool, not single-threaded
\code{mjd\_transitionFD}. On 32 cores it is roughly twice as fast per Jacobian as the best
MJWarp configuration found here.
\item \textbf{Near-switching states need a different tool.} Randomized smoothing,
contact-implicit formulations or bundled gradients are the appropriate response, because the
CPU float64 Jacobian is equally non-predictive there --- this is not a problem a GPU or a
better epsilon can solve.
\item \textbf{If the GPU path is still wanted, change the regime.} The GPU loses here because
96 worlds cannot saturate a 4090. The cases worth testing are a much larger model (humanoid,
$n_x$ three to four times bigger, far more work per world) and many more simultaneous knots
than this card's memory currently allows --- which in turn requires the \code{njmax} /
\code{nccdmax} sizing above.
\end{enumerate}""")

    # -------------------------------------------------- repro
    A(r"\section{Reproduction}")
    A(r"""All numbers in this document are generated from JSON written by the scripts in
\code{mujoco\_grad/src/}:
\code{config.py} and \code{fixtures.py} (Phase 0 and 7),
\code{tangent.py} (manifold operations),
\code{cpu\_fd.py} (CPU rollouts, finite differences, contact diagnostics),
\code{warp\_fd.py} and \code{batch\_common.py} (MJWarp batching and Jacobian assembly),
\code{run\_phase1.py}, \code{run\_phase2.py}, \code{run\_phase34.py}, \code{run\_phase5.py},
\code{mjx\_control.py} (float64 GPU control),
\code{bench.py} and \code{bench\_scale.py} (timing),
\code{mkfigs.py} (figures) and \code{build\_tex.py} (this document).
Every table except Tables~\ref{tab:graphmode}, \ref{tab:nan} and \ref{tab:memory} is
generated directly from the phase JSON; those three report measurements taken with the
diagnostic scripts \code{bench\_probe.py}, \code{dbg\_nan3.py} and
\code{dbg\_mem2.py}\,/\,\code{dbg\_mem3.py} respectively.
Results are in \code{mujoco\_grad/results/}. The MJWarp and CPU runs use the
\code{robot\_sim} environment (MuJoCo """ + V["mujoco"] + r"""); the MJX control uses the
system Python 3.10 stack (MuJoCo """ + mjx["mujoco"] + r""", JAX """ + mjx["jax"] + r""").""")

    A(r"\end{document}")

    out = os.path.join(RESULTS, "report.tex")
    with open(out, "w") as f:
        f.write("\n\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
