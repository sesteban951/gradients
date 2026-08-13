"""Phase 3 + 4: MJWarp batched finite differences and the epsilon sweep.

Three error curves are recorded per fixture:
  E_same(h)  = ||A_warp(h) - A_cpu(h)|| / ||A_cpu(h)||     GPU-vs-CPU at matched truncation
  E_gold(h)  = ||A_warp(h) - A_gold||   / ||A_gold||       total error of the GPU Jacobian
  E_cpu(h)   = ||A_cpu(h)  - A_gold||   / ||A_gold||       CPU's own truncation curve
with A_gold = CPU float64 at H_GOLD.

Repeated MJWarp runs at identical inputs quantify GPU nondeterminism.
"""

import json, os, argparse
import numpy as np
import mujoco

from config import load_model, dims, versions, RESULTS
from fixtures import build_fixtures, CATEGORY_ORDER
from cpu_fd import blackbox_fd, contact_stability, prepare
from metrics import rel_fro, summary, split_summary
from warp_fd import WarpBatch, warp_fd

H_SWEEP = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
H_GOLD = 1e-6
REPEATS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1, help="substeps per shooting interval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    S = args.S

    m, fx, finfo = build_fixtures()
    d = mujoco.MjData(m)
    D = dims(m)
    nx, nu = D["nx"], D["nu"]
    nworld = 2 * (nx + nu) + 1
    batch = WarpBatch(m, nworld=nworld)

    print(f"model: unitree_go1  nv={D['nv']} nu={nu} nx={nx}  nworld={nworld}  S={S}")
    print(f"gold reference: CPU float64 central differences at h={H_GOLD:.0e}\n")

    results = {}
    for name in CATEGORY_ORDER:
        x, u = fx[name]
        ws = prepare(m, d, x, u)
        A_gold, B_gold = blackbox_fd(m, d, x, u, H_GOLD, S=S, centered=True)

        hdr = (f"[{name}]  ncon={finfo[name]['ncon']}  "
               f"|A_gold|_F={np.linalg.norm(A_gold):.4g}")
        print(hdr)
        print(f"  {'h':>8s} {'E_same(A)':>10s} {'E_gold(A)':>10s} {'E_cpu(A)':>10s} "
              f"{'E_gold(B)':>10s} {'noise':>9s} {'cos_min':>8s} {'stab':>7s} "
              f"{'E_stab':>9s} {'E_chg':>9s}")

        rows = []
        for h in H_SWEEP:
            A_cpu, B_cpu = blackbox_fd(m, d, x, u, h, S=S, centered=True)
            stable = contact_stability(m, x, u, h, S=S)

            reps_A, reps_B = [], []
            for _ in range(REPEATS):
                Aw, Bw, _sig = warp_fd(batch, m, x, u, h, S=S, warmstart=ws)
                reps_A.append(Aw); reps_B.append(Bw)
            A_w, B_w = reps_A[0], reps_B[0]

            # run-to-run nondeterminism: max pairwise relative Frobenius spread
            noise = max(rel_fro(reps_A[k], reps_A[0]) for k in range(1, REPEATS)) \
                if REPEATS > 1 else 0.0
            std_A = float(np.std(np.stack(reps_A), axis=0).max())

            s_same = summary(A_w, A_cpu)
            s_gold = summary(A_w, A_gold)
            sb_gold = summary(B_w, B_gold)
            spl = split_summary(A_w, A_gold, stable[:nx])

            e_cpu = rel_fro(A_cpu, A_gold)
            rows.append(dict(
                h=h,
                E_same_A=s_same["rel_fro"], E_gold_A=s_gold["rel_fro"], E_cpu_A=e_cpu,
                E_same_B=rel_fro(B_w, B_cpu), E_gold_B=sb_gold["rel_fro"],
                E_cpu_B=rel_fro(B_cpu, B_gold),
                noise=noise, std_max=std_A,
                cos_min=s_gold["cos_min"], cos_med=s_gold["cos_med"],
                max_abs=s_gold["max_abs"], p50=s_gold["p50"],
                p90=s_gold["p90"], p99=s_gold["p99"],
                n_stable=int(stable[:nx].sum()), n_cols=int(nx),
                split=spl,
                normA_cpu=float(np.linalg.norm(A_cpu)),
                normA_warp=float(np.linalg.norm(A_w)),
            ))
            r = rows[-1]
            print(f"  {h:8.0e} {r['E_same_A']:10.2e} {r['E_gold_A']:10.2e} "
                  f"{r['E_cpu_A']:10.2e} {r['E_gold_B']:10.2e} {noise:9.2e} "
                  f"{r['cos_min']:8.4f} {r['n_stable']:3d}/{nx:<3d} "
                  f"{spl['stable']['rel_fro']:9.2e} {spl['changing']['rel_fro']:9.2e}")
        results[name] = dict(rows=rows, fixture_info={
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in finfo[name].items() if k != "pairs"})
        print()

    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"phase34_sweep_S{S}.json")
    with open(out, "w") as f:
        json.dump(dict(versions=versions(), dims=D, S=S, H_GOLD=H_GOLD,
                       repeats=REPEATS, results=results), f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
