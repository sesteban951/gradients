"""Precision control: GPU-batched MuJoCo in float64 (MJX) vs CPU float64.

If a float64 GPU simulator reproduces the CPU Jacobian across the whole epsilon sweep
while MJWarp (float32) shows a U-shaped error curve, then the limiting factor for
MJWarp is precision -- not GPU batching, and not a physics/collision difference.

Runs under the python3.10 stack (mujoco 3.10.0 + jax 0.6.2), pyramidal cone, because
MJX does not implement the elliptic cone for condim=1.
"""

import os
os.environ.setdefault("GRAD_CONE", "pyramidal")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".35")

import json, argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from config import dims, versions, RESULTS, cone_name
from fixtures import build_fixtures, CATEGORY_ORDER
from cpu_fd import blackbox_fd, prepare
from metrics import rel_fro, summary
from batch_common import build_perturbation_batch, assemble_jacobians

H_SWEEP = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
H_GOLD = 1e-6
REPEATS = 3


def make_stepper(mx, dx0, S):
    def one(qpos, qvel, ctrl, ws):
        d = dx0.replace(qpos=qpos, qvel=qvel, ctrl=ctrl, qacc_warmstart=ws)
        for _ in range(S):
            d = mjx.step(mx, d)
        return d.qpos, d.qvel
    return jax.jit(jax.vmap(one))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--fixtures", default="flight,stance,loaded,onset,sliding,impact")
    args = ap.parse_args()
    S = args.S

    m, fx, finfo = build_fixtures()
    d = mujoco.MjData(m)
    D = dims(m); nx, nu = D["nx"], D["nu"]
    print(f"MJX control | mujoco {mujoco.__version__} | jax {jax.__version__} | "
          f"x64={jax.config.jax_enable_x64} | cone={cone_name(m)} | devices={jax.devices()}")

    mx = mjx.put_model(m, impl="jax")
    mujoco.mj_forward(m, d)
    dx0 = mjx.put_data(m, d, impl="jax")
    print("mjx data dtype:", dx0.qpos.dtype)
    step_batch = make_stepper(mx, dx0, S)

    names = [n for n in CATEGORY_ORDER if n in args.fixtures.split(",")]
    results = {}
    for name in names:
        x, u = fx[name]
        ws = prepare(m, d, x, u)
        A_gold, B_gold = blackbox_fd(m, d, x, u, H_GOLD, S=S, centered=True)
        print(f"\n[{name}]  ncon={finfo[name]['ncon']}  |A_gold|_F={np.linalg.norm(A_gold):.4g}")
        print(f"  {'h':>8s} {'E_same(A)':>10s} {'E_gold(A)':>10s} {'E_cpu(A)':>10s} "
              f"{'E_gold(B)':>10s} {'GPUnoise':>9s} {'cos_min':>8s}")
        rows = []
        for h in H_SWEEP:
            A_cpu, B_cpu = blackbox_fd(m, d, x, u, h, S=S, centered=True)
            qpos, qvel, act, ctrl = build_perturbation_batch(m, x, u, h)
            N = qpos.shape[0]
            wsb = np.tile(ws, (N, 1))
            reps = []
            for _ in range(REPEATS):
                fq, fv = step_batch(jnp.asarray(qpos), jnp.asarray(qvel),
                                    jnp.asarray(ctrl), jnp.asarray(wsb))
                fq = np.asarray(fq, dtype=np.float64); fv = np.asarray(fv, dtype=np.float64)
                reps.append(assemble_jacobians(m, fq, fv, np.zeros((N, 0)), h))
            A_x, B_x = reps[0]
            noise = max(rel_fro(reps[k][0], A_x) for k in range(1, REPEATS)) if REPEATS > 1 else 0.0
            s = summary(A_x, A_gold)
            row = dict(h=h, E_same_A=rel_fro(A_x, A_cpu), E_gold_A=s["rel_fro"],
                       E_cpu_A=rel_fro(A_cpu, A_gold), E_gold_B=rel_fro(B_x, B_gold),
                       noise=noise, cos_min=s["cos_min"])
            rows.append(row)
            print(f"  {h:8.0e} {row['E_same_A']:10.2e} {row['E_gold_A']:10.2e} "
                  f"{row['E_cpu_A']:10.2e} {row['E_gold_B']:10.2e} {noise:9.2e} "
                  f"{row['cos_min']:8.4f}")
        best = min(rows, key=lambda r: r["E_gold_A"])
        print(f"  -> best h = {best['h']:.0e}  E_gold(A) = {best['E_gold_A']:.3e}")
        results[name] = dict(rows=rows, ncon=finfo[name]["ncon"])

    out = os.path.join(RESULTS, f"mjx_control_S{S}.json")
    with open(out, "w") as f:
        json.dump(dict(mujoco=mujoco.__version__, jax=jax.__version__,
                       cone=cone_name(m), S=S, H_GOLD=H_GOLD,
                       repeats=REPEATS, results=results), f, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
