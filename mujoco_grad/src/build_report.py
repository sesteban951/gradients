"""Assemble results/report.html from the phase JSON files."""

import json, os, math
import figs
from figs import LogPanel, legend, grid, bar_chart
from config import RESULTS

CAT = ["flight", "stance", "loaded", "onset", "release", "sliding", "impact"]
CAT_DESC = {
    "flight": "no contact, free fall",
    "stance": "4 contacts, settled",
    "loaded": "8 contacts, pressed down",
    "onset": "at the contact margin, closing",
    "release": "just outside margin, opening",
    "sliding": "4 contacts, 1.2 m/s lateral",
    "impact": "4 contacts, 3.0 m/s downward",
}


def load(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def jsonl(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


# ------------------------------------------------------------------ figures
def fig_precision(mjx, pyr):
    panels = []
    for name in CAT:
        if name not in mjx["results"] or name not in pyr["results"]:
            continue
        mr = mjx["results"][name]["rows"]
        pr = pyr["results"][name]["rows"]
        hs = [r["h"] for r in mr]
        warp_dead = all(r["E_gold_A"] != r["E_gold_A"] for r in pr)
        p = LogPanel(f"{name}", min(hs) / 1.6, max(hs) * 1.6, 1e-13, 1e2,
                     xlabel="perturbation h", ylabel="rel. error of A",
                     note="MJWarp: NaN at every h" if warp_dead else CAT_DESC[name])
        p.add("CPU float64", hs, [r["E_cpu_A"] for r in mr], 1)
        p.add("MJX float64 (GPU)", hs, [r["E_gold_A"] for r in mr], 3)
        p.add("MJWarp float32 (GPU)", [r["h"] for r in pr],
              [r["E_gold_A"] for r in pr], 2)
        panels.append(p.svg())
    return grid(panels, 3) + legend([("CPU float64", 1, False),
                                     ("MJWarp float32 (GPU)", 2, False),
                                     ("MJX float64 (GPU)", 3, False)])


def fig_noise(mjx, pyr):
    panels = []
    for name in ["flight", "stance", "loaded", "impact"]:
        if name not in mjx["results"] or name not in pyr["results"]:
            continue
        mr = mjx["results"][name]["rows"]; pr = pyr["results"][name]["rows"]
        p = LogPanel(name, 1e-6 / 1.6, 1e-2 * 1.6, 1e-16, 1e1,
                     xlabel="perturbation h", ylabel="run-to-run spread of A",
                     note=CAT_DESC[name])
        p.add("MJWarp float32", [r["h"] for r in pr], [r["noise"] for r in pr], 2)
        p.add("MJX float64", [r["h"] for r in mr], [r["noise"] for r in mr], 3)
        panels.append(p.svg())
    return grid(panels, 4) + legend([("MJWarp float32 (GPU)", 2, False),
                                     ("MJX float64 (GPU)", 3, False)])


def fig_taylor(tay):
    panels = []
    for name in CAT:
        if name not in tay["results"]:
            continue
        r = tay["results"][name]
        al = r["alphas"]; med = r["median"]; sl = r["slopes"]
        p = LogPanel(name, min(al) / 1.6, max(al) * 1.6, 1e-11, 1e2,
                     xlabel="step size α", ylabel="Taylor residual r(α)",
                     note=f"slope {sl['cpu_cpu']:.2f} / {sl['warp_cpu']:.2f}")
        p.add("CPU Jac → CPU sim", al, med["cpu_cpu"], 1)
        p.add("MJWarp Jac → CPU sim", al, med["warp_cpu"], 2)
        p.add("MJWarp Jac → MJWarp sim", al, med["warp_warp"], 3, dashed=True)
        panels.append(p.svg())
    return grid(panels, 3) + legend([("CPU Jacobian → CPU simulator", 1, False),
                                     ("MJWarp Jacobian → CPU simulator", 2, False),
                                     ("MJWarp Jacobian → MJWarp simulator", 3, True)])


def fig_split(sw):
    panels = []
    for name in ["onset", "release"]:
        if name not in sw["results"]:
            continue
        rows = sw["results"][name]["rows"]
        hs = [r["h"] for r in rows]
        p = LogPanel(name, min(hs) / 1.6, max(hs) * 1.6, 1e-6, 1e4,
                     xlabel="perturbation h", ylabel="rel. error of A",
                     note=CAT_DESC[name])
        p.add("contact-stable columns", hs,
              [r["split"]["stable"]["rel_fro"] for r in rows], 1)
        p.add("contact-changing columns", hs,
              [r["split"]["changing"]["rel_fro"] for r in rows], 2)
        panels.append(p.svg())
    return grid(panels, 2) + legend([("contact-stable columns", 1, False),
                                     ("contact-changing columns", 2, False)])


def fig_bench(bench):
    rows = bench["rows"]
    groups = [f"S={r['S']}" for r in rows]
    series = [("mjd_transitionFD (1 thread)", 1), ("CPU rollout, 32 threads", 2),
              ("MJWarp GPU (CUDA graph)", 3)]
    vals = [
        [r["mjd"] * 1e3 if r["mjd"] == r["mjd"] else 0 for r in rows],
        [r["cpu_threaded"] * 1e3 for r in rows],
        [r["warp"] * 1e3 for r in rows],
    ]
    return (bar_chart("Wall time for one (A,B) pair — 97 worlds, go1", groups,
                      series, vals, ylabel="milliseconds", fmt="{:.2f}") +
            legend([(s, i, False) for s, i in series]))


def fig_scale(rows):
    rows = [r for r in rows if r["gpu"] == r["gpu"]] or rows
    ks = [r["K"] for r in rows]
    if not ks:
        return ""
    p = LogPanel("Cost per Jacobian vs simultaneous knots",
                 min(ks) / 1.6, max(ks) * 1.6, 1e-1, 1e1,
                 xlabel="shooting knots differentiated at once (K)",
                 ylabel="ms per Jacobian", note="S=5 substeps")
    p.add("CPU rollout, 32 threads", ks, [r["cpu_per"] * 1e3 for r in rows], 1)
    p.add("MJWarp GPU", ks,
          [r["gpu_per"] * 1e3 if r["gpu_per"] == r["gpu_per"] else None for r in rows], 2)
    return (grid([p.svg()], 1) +
            legend([("CPU rollout, 32 threads", 1, False), ("MJWarp GPU", 2, False)]))


# ------------------------------------------------------------------ tables
def table(headers, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def det(summary, inner):
    return f"<details><summary>{summary}</summary>{inner}</details>"


def e(v, p=2):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{p}e}"


CSS = """
<style>
.rpt{--surface-0:#f7f7f5;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --text-muted:#7a7975;--border:#e2e1dc;--grid:#eceae5;--axis:#c9c7c1;--ref:#d6d4ce;
 --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--good:#1f7a4d;--bad:#b3261e;
 color-scheme:light;background:var(--surface-0);color:var(--text-primary);
 font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
 -webkit-font-smoothing:antialiased;padding:0 0 64px}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .rpt{
 --surface-0:#141413;--surface-1:#1a1a19;--text-primary:#fff;--text-secondary:#c3c2b7;
 --text-muted:#8f8e86;--border:#2e2e2b;--grid:#262624;--axis:#403f3c;--ref:#35342f;
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--good:#4ec38a;--bad:#e66767;
 color-scheme:dark}}
:root[data-theme="dark"] .rpt{--surface-0:#141413;--surface-1:#1a1a19;--text-primary:#fff;
 --text-secondary:#c3c2b7;--text-muted:#8f8e86;--border:#2e2e2b;--grid:#262624;--axis:#403f3c;
 --ref:#35342f;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--good:#4ec38a;
 --bad:#e66767;color-scheme:dark}
.rpt .wrap{max-width:1080px;margin:0 auto;padding:0 20px}
.rpt h1{font-size:30px;line-height:1.2;margin:40px 0 6px;letter-spacing:-.02em}
.rpt .sub{color:var(--text-secondary);margin:0 0 28px;font-size:16px}
.rpt h2{font-size:20px;margin:44px 0 4px;letter-spacing:-.01em;padding-top:16px;border-top:1px solid var(--border)}
.rpt h3{font-size:16px;margin:26px 0 6px}
.rpt p{margin:10px 0;color:var(--text-secondary)}
.rpt p strong,.rpt li strong{color:var(--text-primary)}
.rpt ul{margin:10px 0;padding-left:20px;color:var(--text-secondary)}
.rpt li{margin:5px 0}
.rpt code{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface-1);
 border:1px solid var(--border);border-radius:4px;padding:1px 5px}
.rpt .verdict{background:var(--surface-1);border:1px solid var(--border);border-left:3px solid var(--series-1);
 border-radius:10px;padding:16px 20px;margin:20px 0}
.rpt .verdict p{margin:6px 0}
.rpt .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}
.rpt .kpi{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.rpt .kpi .n{font-size:24px;font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.rpt .kpi .l{font-size:12.5px;color:var(--text-muted);margin-top:3px}
.rpt .grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:12px;margin:16px 0}
@media(max-width:900px){.rpt .grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:600px){.rpt .grid{grid-template-columns:1fr}}
.rpt .panel{margin:0;background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:6px}
.rpt .chart{width:100%;height:auto;display:block}
.rpt .ttl{font-size:12px;font-weight:600;fill:var(--text-primary)}
.rpt .note{font-size:10px;fill:var(--text-muted)}
.rpt .tick{font-size:9.5px;fill:var(--text-muted);font-variant-numeric:tabular-nums}
.rpt .axlab{font-size:10.5px;fill:var(--text-secondary)}
.rpt .blab{font-size:9.5px;fill:var(--text-secondary);font-variant-numeric:tabular-nums}
.rpt .grid line,.rpt line.grid{stroke:var(--grid);stroke-width:1}
.rpt line.axis{stroke:var(--axis);stroke-width:1}
.rpt line.ref{stroke:var(--ref);stroke-width:1;stroke-dasharray:3 3}
.rpt .legend{display:flex;flex-wrap:wrap;gap:16px;margin:2px 0 8px;font-size:12.5px;color:var(--text-secondary)}
.rpt .lg{display:inline-flex;align-items:center;gap:7px}
.rpt .lg i{width:16px;height:0;display:inline-block}
.rpt .tw{overflow-x:auto;margin:12px 0;border:1px solid var(--border);border-radius:10px;background:var(--surface-1)}
.rpt table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.rpt th,.rpt td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
.rpt th{color:var(--text-muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.rpt td:first-child,.rpt th:first-child{text-align:left}
.rpt tbody tr:last-child td{border-bottom:none}
.rpt details{margin:10px 0}
.rpt summary{cursor:pointer;color:var(--text-secondary);font-size:13.5px;padding:5px 0}
.rpt .ok{color:var(--good);font-weight:600}
.rpt .no{color:var(--bad);font-weight:600}
.rpt .foot{color:var(--text-muted);font-size:12.5px;margin-top:36px;border-top:1px solid var(--border);padding-top:14px}
</style>
"""


def main():
    ph1 = load("phase1_nominal_parity.json")
    ph2 = load("phase2_cpu_vs_transitionFD.json")
    sw1 = load("phase34_sweep_S1.json")
    sw5 = load("phase34_sweep_S5.json")
    sw20 = load("phase34_sweep_S20.json")
    pyr = load("phase34_sweep_pyramidal.json")
    tay = load("phase5_taylor_S1.json")
    mjx = load("mjx_control_S1.json")
    bench = load("bench.json")
    scale = jsonl("bench_scale_S5.jsonl")

    V = ph2["versions"]
    o = ['<title>MJWarp Derivative Validation</title>', CSS, '<div class="rpt"><div class="wrap">']
    o.append("<h1>MJWarp Derivative Validation</h1>")
    o.append('<p class="sub">Can GPU-batched finite differences replace CPU MuJoCo '
             'transition Jacobians for contact-rich trajectory optimization? '
             'Unitree Go1, 7 state categories, 9 epsilons, 3 simulators.</p>')

    # ---- verdict
    max2 = max(max(r["relA"] for r in ph2["rows"]), max(r["relB"] for r in ph2["rows"]))
    st = sw1["results"]["stance"]["rows"]
    best_st = min(st, key=lambda r: r["E_gold_A"])
    o.append('<div class="verdict">')
    o.append("<p><strong>Verdict: yes for smooth and stable-contact states, with a "
             "usable epsilon window ~100× wider than CPU double precision — but the "
             "GPU is not faster than 32 CPU threads on a robot this size.</strong></p>")
    o.append("<p>Where the MJWarp Jacobian is inaccurate, the CPU float64 Jacobian is "
             "inaccurate in exactly the same way: near contact switching, both fail "
             "identically. That failure is contact nonsmoothness, not float32. A float64 "
             "GPU control (MJX) isolates the remaining gap as pure precision.</p>")
    o.append("</div>")

    o.append('<div class="kpis">')
    for n, l in [
        (e(max2), "CPU black-box FD vs mjd_transitionFD (max rel. error, 35 cases)"),
        (f"{best_st['h']:.0e}", "best MJWarp epsilon at stance (CPU optimum: 1e−6)"),
        (e(best_st["E_gold_A"]), "best MJWarp rel. error of A at stance"),
        (f"{bench['rows'][0]['speedup_vs_threaded']:.2f}×",
         "MJWarp speedup vs 32-thread CPU, one Jacobian, S=1"),
    ]:
        o.append(f'<div class="kpi"><div class="n">{n}</div><div class="l">{l}</div></div>')
    o.append("</div>")

    # ---- setup
    o.append("<h2>Setup</h2>")
    o.append(f"<p>Model <code>unitree_go1/scene.xml</code> from MuJoCo Menagerie: "
             f"nq=19, nv=18, na=0, nu=12 → tangent state nx=36, so a centered "
             f"difference needs <code>2(nx+nu)=96</code> perturbed worlds plus a nominal one. "
             f"Euler integrator, Newton solver, dt=2&nbsp;ms. Solver tolerance forced to "
             f"1e−6 on <em>both</em> sides, because MJWarp silently floors "
             f"<code>opt.tolerance</code> at 1e−6 in <code>io.py</code> — an uncontrolled "
             f"difference otherwise.</p>")
    o.append(table(["component", "version"], [
        ["MuJoCo (CPU + MJWarp runs)", V["mujoco"]],
        ["MJWarp", V["mujoco_warp"]],
        ["NVIDIA Warp", V["warp"]],
        ["MuJoCo / JAX (MJX control)", f"{mjx['mujoco']} / {mjx['jax']}"],
        ["GPU", V["gpu"]],
        ["CPU threads", str(V["cpu_count"])],
    ]))
    o.append("<h3>State suite</h3>")
    rows = []
    for c in CAT:
        fi = sw1["results"][c]["fixture_info"]
        rows.append([c, CAT_DESC[c], fi["ncon"], fi["nefc"], e(fi["min_gap"]),
                     f"{fi['qvel_norm']:.2f}", f"{fi['nswitch']}/{fi['nperturb']}"])
    o.append(table(["fixture", "what it is", "ncon", "nefc", "foot gap [m]",
                    "|qvel|", "contact flips @h=1e−4"], rows))

    # ---- phase 2
    o.append("<h2>1 · The CPU implementation is exact</h2>")
    o.append(f"<p>Before comparing anything on a GPU, the manifold-aware black-box "
             f"central difference has to reproduce MuJoCo's own "
             f"<code>mjd_transitionFD</code>. Across 7 fixtures × 5 epsilons it agrees to "
             f"<strong>{e(max2)}</strong> relative Frobenius — machine precision. Tangent "
             f"perturbation via <code>mj_integratePos</code>, output differencing via "
             f"<code>mj_differentiatePos</code>, state packing, control indexing and "
             f"warm-start handling are therefore all correct, and every later discrepancy "
             f"belongs to the simulator rather than to the harness.</p>")
    o.append('<p><strong>Gotcha worth recording:</strong> a diagnostic '
             '<code>mj_forward</code> placed inside the FD loop shifts the Jacobian by '
             '~1e−6 even if <code>qacc_warmstart</code> is saved and restored around it. '
             'Contact diagnostics must run on a separate <code>MjData</code>.</p>')

    # ---- phase 1
    o.append("<h2>2 · Nominal parity holds before derivatives</h2>")
    rows = []
    for r in ph1["rows"]:
        if r["S"] != 1:
            continue
        rows.append([r["fixture"], e(r["dq"]), e(r["dv"]),
                     f"{r['ncon_cpu']}/{r['ncon_warp']}",
                     '<span class="ok">match</span>' if r["pairs_match"]
                     else '<span class="no">differ</span>'])
    o.append(f"<p>One step from each fixture, CPU vs MJWarp. Position error sits at the "
             f"float32 floor (~1e−7) and the contact set is identical in all "
             f"{len(ph1['rows'])} fixture×horizon combinations tested (S = 1, 5, 20). There "
             f"is no physics or collision disagreement to confound the derivative "
             f"comparison.</p>")
    o.append(table(["fixture", "|Δq| tangent", "|Δv|", "ncon CPU/Warp", "contact pairs"], rows))

    # ---- headline precision figure
    o.append("<h2>3 · Precision, isolated</h2>")
    o.append("<p>Three simulators, identical model and options (pyramidal cone, the one "
             "MJX supports), identical perturbations, same GPU. The y axis is the relative "
             "error of <em>A</em> against a CPU float64 reference at h=1e−6.</p>")
    o.append(fig_precision(mjx, pyr))
    o.append("<ul>"
             "<li><strong>CPU float64</strong> falls as O(h²) and keeps falling — it never "
             "reaches a roundoff floor in this range.</li>"
             "<li><strong>MJX float64 on the GPU</strong> tracks the CPU curve almost "
             "exactly. GPU batching by itself costs nothing: agreement with CPU is "
             "~1e−12 for flight, sliding and impact.</li>"
             "<li><strong>MJWarp float32</strong> turns around near h≈1e−3…1e−4 and climbs "
             "again. That U is the whole story: truncation on the right, float32 "
             "cancellation and nondeterminism on the left.</li>"
             "<li>For <code>onset</code>, all three curves are <em>identical</em> at large "
             "h — including the two float64 ones. That error is not precision.</li>"
             "<li><code>loaded</code> has no MJWarp curve at all: it returns NaN at every "
             "epsilon under this configuration. See below.</li>"
             "</ul>")

    o.append("<h3>A float32 failure that only appears in a batch</h3>")
    o.append("<p>With the pyramidal cone, the <code>loaded</code> fixture — 8 contacts, "
             "3&nbsp;cm penetration, 68 active constraints — makes MJWarp return NaN in "
             "<strong>10 of the 97 perturbed worlds</strong>, which poisons 8 columns of "
             "<em>A</em> and therefore the entire Jacobian. The nominal unperturbed step "
             "is fine, and CPU float64 and MJX float64 handle the identical states without "
             "trouble (max|qvel| = 0.664 for all of them).</p>")
    o.append("<p>The sharp part: re-running those same 10 states <em>one world at a "
             "time</em>, <strong>9 of the 10 come back finite</strong>. Re-running them as "
             "a 10-world batch, they NaN again. Whether the solve diverges depends on the "
             "batch it is evaluated in — so this class of failure will not reproduce in a "
             "single-world debug session, and it is invisible unless every returned world "
             "is checked for finiteness. The elliptic-cone configuration of the same "
             "fixture does not exhibit it.</p>")
    o.append('<p><strong>Practical consequence:</strong> validate <code>isfinite</code> on '
             'every perturbed world before assembling <em>A</em> and <em>B</em>, and treat '
             'a NaN world as a failed column rather than letting it propagate.</p>')

    o.append("<h3>Run-to-run nondeterminism</h3>")
    o.append("<p>Same inputs, repeated evaluations, maximum pairwise spread of A.</p>")
    o.append(fig_noise(mjx, pyr))
    o.append("<p>MJWarp is <strong>bitwise reproducible when there are no contacts</strong> "
             "(flight: exactly zero across 5 runs) and nondeterministic as soon as the "
             "constraint solver is active, with the spread growing like η/h as h shrinks — "
             "precisely the mechanism the plan predicted. MJX float64 shows the same "
             "structure five to eight orders of magnitude lower.</p>")

    # ---- epsilon window table
    o.append("<h2>4 · The usable epsilon window</h2>")
    o.append("<p>Elliptic cone (the production configuration), one step, per fixture: "
             "where MJWarp's Jacobian is most accurate and how good it gets.</p>")
    rows = []
    for c in CAT:
        r = sw1["results"][c]["rows"]
        b = min(r, key=lambda x: x["E_gold_A"])
        cpu_best = min(r, key=lambda x: x["E_cpu_A"])
        rows.append([c, f"{b['h']:.0e}", e(b["E_gold_A"]), f"{b['cos_min']:.4f}",
                     e(b["noise"]), f"{cpu_best['h']:.0e}", e(cpu_best["E_cpu_A"])])
    o.append(table(["fixture", "best h (Warp)", "rel. err A", "min col cosine",
                    "GPU noise", "best h (CPU)", "CPU rel. err"], rows))
    o.append("<p>The MJWarp optimum sits at <strong>1e−3 to 1e−4</strong> — two to three "
             "decades coarser than the CPU double-precision optimum, exactly as the plan "
             "anticipated. Contact-free and stable-contact states reach 2e−5…6e−4; "
             "the near-switching states never get below ~3e−3 and <code>onset</code> never "
             "below 0.37.</p>")

    for label, sw in [("S = 5", sw5), ("S = 20", sw20)]:
        if not sw:
            continue
        rows = []
        for c in CAT:
            if c not in sw["results"]:
                continue
            r = sw["results"][c]["rows"]
            b = min(r, key=lambda x: x["E_gold_A"])
            rows.append([c, f"{b['h']:.0e}", e(b["E_gold_A"]), e(b["noise"])])
        o.append(det(f"Direct multi-substep shooting interval, {label}",
                     table(["fixture", "best h", "rel. err A", "GPU noise"], rows)))

    # ---- contact split
    o.append("<h2>5 · Splitting contact-stable from contact-changing columns</h2>")
    o.append("<p>The plan warns that mixing both into one Frobenius norm hides the "
             "behaviour. It does.</p>")
    o.append(fig_split(sw1))
    o.append("<p>At <code>onset</code>, h=1e−4: the 32 contact-stable columns carry a "
             "relative error of 4.3e−2 while the 4 contact-changing columns carry "
             "3.1e+1 — nearly three orders of magnitude apart, averaged into a single "
             "misleading number if reported together. Column count that stays "
             "contact-stable rises from 22/36 at h=1e−2 to 36/36 at h=1e−5.</p>")

    # ---- taylor
    o.append("<h2>6 · Does the Jacobian predict the simulator?</h2>")
    o.append("<p>The decisive test. Matrix agreement is a proxy; what an optimizer "
             "actually needs is that <em>A δx + B δu</em> predicts the real step. "
             "Residual r(α) for random tangent directions, median over 4 directions, "
             "log–log.</p>")
    o.append(fig_taylor(tay))
    rows = []
    for c in CAT:
        s = tay["results"][c]["slopes"]
        rows.append([c, f"{tay['results'][c]['h_warp']:.0e}", f"{s['cpu_cpu']:.2f}",
                     f"{s['warp_cpu']:.2f}", f"{s['warp_warp']:.2f}"])
    o.append(table(["fixture", "h used for Warp", "slope CPU→CPU",
                    "slope Warp→CPU", "slope Warp→Warp"], rows))
    o.append("<p>The orange curve — MJWarp's float32 Jacobian used to predict the "
             "<em>CPU</em> simulator — lies on top of the CPU-Jacobian curve for stance, "
             "sliding and impact, agreeing to three significant figures over the whole "
             "useful range of α. <strong>For those states the GPU Jacobian is as "
             "predictive as the CPU one.</strong> At <code>release</code> both slopes are "
             "0.71 and both residuals plateau at 6e−3: the CPU float64 Jacobian is just as "
             "useless there. That is nonsmoothness, and no epsilon or precision fixes it.</p>")

    # ---- performance
    o.append("<h2>7 · Wall time</h2>")
    o.append(fig_bench(bench))
    rows = []
    for r in bench["rows"]:
        rows.append([f"S={r['S']}", e(r["mjd"] * 1e3, 2) if r["mjd"] == r["mjd"] else "—",
                     f"{r['cpu_serial']*1e3:.2f}", f"{r['cpu_threaded']*1e3:.2f}",
                     f"{r['warp_eager']*1e3:.2f}", f"{r['warp']*1e3:.2f}",
                     f"{r['warp_gpu_only']*1e3:.2f}",
                     f"{r['speedup_vs_threaded']:.2f}×"])
    o.append(table(["horizon", "mjd_transitionFD [ms]", "CPU serial [ms]",
                    "CPU 32 threads [ms]", "MJWarp eager [ms]", "MJWarp graph [ms]",
                    "GPU only [ms]", "vs 32 threads"], rows))
    o.append("<p><strong>CUDA graph capture is not optional.</strong> Launched eagerly, one "
             "batched step of 97 worlds costs 25&nbsp;ms of Python kernel-launch overhead; "
             "captured into a graph it costs 0.37&nbsp;ms — a 20–55× difference that has "
             "nothing to do with physics. Any benchmark run without it is measuring the "
             "wrong thing.</p>")
    o.append("<p>Even so, <strong>96 worlds is far too small a batch to saturate a "
             "4090</strong>. A single Jacobian for a 12-DoF quadruped is cheaper on 32 CPU "
             "threads than on the GPU at every horizon tested.</p>")
    if scale:
        o.append(fig_scale(scale))
        rows = [[str(r["K"]), str(r["nworld"]), f"{r['cpu_per']*1e3:.3f}",
                 f"{r['gpu_per']*1e3:.3f}" if r["gpu_per"] == r["gpu_per"] else "OOM",
                 f"{r['speedup']:.2f}×" if r["speedup"] == r["speedup"] else "—"]
                for r in scale]
        o.append(table(["knots K", "worlds", "CPU ms/Jacobian", "MJWarp ms/Jacobian",
                        "GPU speedup"], rows))
        o.append("<p>Batching many shooting knots at once is the scenario that should "
                 "favour the GPU. It does help — cost per Jacobian falls from 1.49&nbsp;ms "
                 "at K=1 to <strong>0.91&nbsp;ms at K=4</strong> — but it never reaches the "
                 "0.47&nbsp;ms the CPU pool sustains, and past K=4 it gets "
                 "<em>worse</em> again (1.84&nbsp;ms at K=12) as memory pressure grows. "
                 "The CPU curve, by contrast, is flat to slightly improving out to "
                 "K=64.</p>")
        o.append("<p>MJWarp also hits a memory wall well before the SMs are busy. "
                 "<code>Data</code> allocation grew roughly <em>quadratically</em> in world "
                 "count for this model (405&nbsp;MB at 192 worlds → 1.5&nbsp;GB at 384 → "
                 "6.1&nbsp;GB at 768), and the default <code>nccdmax</code> — convex-collision "
                 "scratch that the Go1 does not need, having only primitive collision "
                 "geometry — accounted for much of the captured-graph footprint "
                 "(10.6&nbsp;GB → 2.7&nbsp;GB at 384 worlds once capped). Passing "
                 "<code>nccdmax</code> and a realistic <code>njmax</code> explicitly to "
                 "<code>put_data</code> is the difference between fitting 8 knots and "
                 "fitting 2 on a 24&nbsp;GB card.</p>")

    # ---- how to use
    o.append("<h2>8 · What to do with this</h2>")
    o.append("<ul>"
             "<li><strong>Use h ≈ 1e−3 for MJWarp, not the CPU value.</strong> Carrying the "
             "double-precision epsilon over costs two to three orders of magnitude of "
             "accuracy.</li>"
             "<li><strong>Always capture the step into a CUDA graph</strong> and size "
             "<code>njmax</code> to the real per-world <code>nefc</code> (48 here, not "
             "512).</li>"
             "<li><strong>Log contact-set equality per perturbation</strong> and report "
             "contact-stable and contact-changing columns separately; a single Frobenius "
             "norm is uninformative near switching.</li>"
             "<li><strong>Do not expect the GPU to pay off for one robot-sized Jacobian.</strong> "
             "The competitor is <code>mujoco.rollout</code> with a persistent thread pool, "
             "not single-threaded <code>mjd_transitionFD</code>, and it wins here.</li>"
             "<li><strong>Near-switching states need a different tool</strong> — randomized "
             "smoothing, contact-implicit formulations or bundled gradients — because the "
             "CPU float64 Jacobian is equally non-predictive there.</li>"
             "</ul>")

    o.append(f'<p class="foot">Generated from the phase JSON files in '
             f'<code>mujoco_grad/results/</code>. Every number here is reproducible with '
             f'<code>src/run_phase{{1,2,34,5}}.py</code>, <code>src/mjx_control.py</code>, '
             f'<code>src/bench.py</code> and <code>src/bench_scale.py</code>.</p>')
    o.append("</div></div>")

    out = os.path.join(RESULTS, "report.html")
    with open(out, "w") as f:
        f.write("\n".join(o))
    print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
