"""Phase 5: directional Taylor test.

    r(alpha) = d_x( F(x (+) alpha*dx, u + alpha*du),  F(x,u) (+) alpha*(A dx + B du) )

A valid first-order model gives r ~ O(alpha^2) away from nonsmooth transitions.
Run three ways:
    cpu_cpu   : CPU Jacobian     vs CPU rollout      (best achievable)
    warp_warp : MJWarp Jacobian  vs MJWarp rollout   (self-consistency on GPU)
    warp_cpu  : MJWarp Jacobian  vs CPU rollout      (does the GPU model predict truth?)
"""

import json, os, argparse
import numpy as np
import mujoco

from config import dims, versions, RESULTS
from fixtures import build_fixtures, CATEGORY_ORDER
from cpu_fd import blackbox_fd, prepare, cpu_rollout
from tangent import State, tangent_perturb, tangent_state_difference
from warp_fd import WarpBatch, warp_fd

ALPHAS = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
NDIR = 4
H_CPU = 1e-6
SEED = 0


def directions(nx, nu, ndir, rng):
    out = []
    for _ in range(ndir):
        v = rng.standard_normal(nx + nu)
        v /= np.linalg.norm(v)
        out.append((v[:nx].copy(), v[nx:].copy()))
    return out


def slope(alphas, resid):
    """Least-squares log-log slope over points that are above the noise floor."""
    a = np.asarray(alphas); r = np.asarray(resid)
    ok = (r > 0) & np.isfinite(r)
    if ok.sum() < 3:
        return float("nan")
    la, lr = np.log10(a[ok]), np.log10(r[ok])
    return float(np.polyfit(la, lr, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--h-warp", type=float, default=None,
                    help="fixed h for the MJWarp Jacobian (default: per-fixture best)")
    args = ap.parse_args()
    S = args.S

    m, fx, finfo = build_fixtures()
    d = mujoco.MjData(m)
    D = dims(m); nx, nu = D["nx"], D["nu"]
    rng = np.random.default_rng(SEED)
    dirs = directions(nx, nu, NDIR, rng)

    # per-fixture best MJWarp h from the phase-4 sweep, if available
    best_h = {}
    sweep_path = os.path.join(RESULTS, f"phase34_sweep_S{S}.json")
    if os.path.exists(sweep_path):
        sw = json.load(open(sweep_path))
        for k, v in sw["results"].items():
            best_h[k] = min(v["rows"], key=lambda r: r["E_gold_A"])["h"]

    fd_batch = WarpBatch(m, nworld=2 * (nx + nu) + 1)
    roll_batch = WarpBatch(m, nworld=len(ALPHAS) * NDIR + 1)

    results = {}
    for name in CATEGORY_ORDER:
        x, u = fx[name]
        ws = prepare(m, d, x, u)
        hw = args.h_warp or best_h.get(name, 1e-4)

        A_c, B_c = blackbox_fd(m, d, x, u, H_CPU, S=S, centered=True)
        A_w, B_w, _ = warp_fd(fd_batch, m, x, u, hw, S=S, warmstart=ws)

        y_cpu_nom = cpu_rollout(m, d, x, u, S=S, warmstart=ws)

        # --- batched MJWarp rollouts at every (direction, alpha) + nominal ------
        qpos = np.zeros((roll_batch.nworld, m.nq)); qvel = np.zeros((roll_batch.nworld, m.nv))
        act = np.zeros((roll_batch.nworld, m.na)); ctrl = np.zeros((roll_batch.nworld, nu))
        idx = {}
        w = 0
        for di, (dx, du) in enumerate(dirs):
            for ai, al in enumerate(ALPHAS):
                xp = tangent_perturb(m, x, al * dx)
                qpos[w], qvel[w] = xp.qpos, xp.qvel
                if m.na: act[w] = xp.act
                ctrl[w] = np.asarray(u) + al * du
                idx[(di, ai)] = w; w += 1
        qpos[-1], qvel[-1] = x.qpos, x.qvel
        if m.na: act[-1] = x.act
        ctrl[-1] = u
        roll_batch.set_batch(qpos, qvel, act, ctrl,
                             np.tile(ws, (roll_batch.nworld, 1)))
        roll_batch.step(S)
        wq, wv, wa = roll_batch.get_batch()
        y_warp_nom = State(wq[-1], wv[-1], wa[-1] if m.na else np.zeros(0))

        res = {k: np.zeros((NDIR, len(ALPHAS))) for k in ("cpu_cpu", "warp_warp", "warp_cpu")}
        for di, (dx, du) in enumerate(dirs):
            for ai, al in enumerate(ALPHAS):
                xp = tangent_perturb(m, x, al * dx)
                up = np.asarray(u) + al * du
                y_true_cpu = cpu_rollout(m, d, xp, up, S=S, warmstart=ws)
                wi = idx[(di, ai)]
                y_true_w = State(wq[wi], wv[wi], wa[wi] if m.na else np.zeros(0))

                pred_c = al * (A_c @ dx + B_c @ du)
                pred_w = al * (A_w @ dx + B_w @ du)
                yp_cc = tangent_perturb(m, y_cpu_nom, pred_c)
                yp_wc = tangent_perturb(m, y_cpu_nom, pred_w)
                yp_ww = tangent_perturb(m, y_warp_nom, pred_w)

                res["cpu_cpu"][di, ai] = np.linalg.norm(
                    tangent_state_difference(m, yp_cc, y_true_cpu))
                res["warp_cpu"][di, ai] = np.linalg.norm(
                    tangent_state_difference(m, yp_wc, y_true_cpu))
                res["warp_warp"][di, ai] = np.linalg.norm(
                    tangent_state_difference(m, yp_ww, y_true_w))

        med = {k: np.median(v, axis=0) for k, v in res.items()}
        slopes = {k: slope(ALPHAS, med[k]) for k in med}
        results[name] = dict(h_warp=hw, alphas=ALPHAS,
                             median={k: v.tolist() for k, v in med.items()},
                             slopes=slopes, ncon=finfo[name]["ncon"])

        print(f"[{name}]  h_warp={hw:.0e}  ncon={finfo[name]['ncon']}")
        print(f"  {'alpha':>8s} {'cpu_cpu':>11s} {'warp_warp':>11s} {'warp_cpu':>11s}")
        for ai, al in enumerate(ALPHAS):
            print(f"  {al:8.0e} {med['cpu_cpu'][ai]:11.3e} "
                  f"{med['warp_warp'][ai]:11.3e} {med['warp_cpu'][ai]:11.3e}")
        print(f"  log-log slope:  cpu_cpu={slopes['cpu_cpu']:.2f}  "
              f"warp_warp={slopes['warp_warp']:.2f}  warp_cpu={slopes['warp_cpu']:.2f}\n")

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"phase5_taylor_S{S}.json"), "w") as f:
        json.dump(dict(versions=versions(), S=S, results=results), f, indent=2)


if __name__ == "__main__":
    main()
