"""Phase 1: nominal transition parity, CPU vs MJWarp, before any derivatives."""

import json, os
import numpy as np
import mujoco

from config import load_model, dims, versions, RESULTS
from fixtures import build_fixtures, CATEGORY_ORDER
from cpu_fd import prepare, cpu_rollout, contact_signature
from tangent import State, tangent_state_difference
from warp_fd import WarpBatch

S_LIST = [1, 5, 20]


def main():
    m, fx, info = build_fixtures()
    d = mujoco.MjData(m)
    nv, na = m.nv, m.na
    batch = WarpBatch(m, nworld=1)

    rows = []
    hdr = (f"{'fixture':10s} {'S':>3s} {'|dq|_tan':>10s} {'|dv|':>10s} "
           f"{'rel_dq':>9s} {'rel_dv':>9s} {'ncon c/w':>10s} {'pairs':>6s}")
    print(hdr); print("-" * len(hdr))

    for name in CATEGORY_ORDER:
        x, u = fx[name]
        for S in S_LIST:
            ws = prepare(m, d, x, u)
            y_cpu = cpu_rollout(m, d, x, u, S=S, warmstart=ws)
            cpu_sig = contact_signature(m, d)

            batch.set_batch(x.qpos[None], x.qvel[None],
                            x.act[None] if na else np.zeros((1, 0)),
                            np.asarray(u)[None], ws[None])
            batch.step(S)
            qp, qv, ac = batch.get_batch()
            y_w = State(qp[0], qv[0], ac[0] if na else np.zeros(0))
            w_sig = batch.contact_signatures()[0]

            dd = tangent_state_difference(m, y_cpu, y_w)
            dq, dv = np.linalg.norm(dd[:nv]), np.linalg.norm(dd[nv:2 * nv])
            # relative to the motion actually produced over the interval
            move = tangent_state_difference(m, x, y_cpu)
            rel_dq = dq / max(np.linalg.norm(move[:nv]), 1e-12)
            rel_dv = dv / max(np.linalg.norm(move[nv:2 * nv]), 1e-12)
            same = cpu_sig["pairs"] == w_sig["pairs"]

            rows.append(dict(fixture=name, S=S, dq=float(dq), dv=float(dv),
                             rel_dq=float(rel_dq), rel_dv=float(rel_dv),
                             ncon_cpu=cpu_sig["ncon"], ncon_warp=w_sig["ncon"],
                             pairs_match=bool(same)))
            print(f"{name:10s} {S:3d} {dq:10.3e} {dv:10.3e} {rel_dq:9.2e} {rel_dv:9.2e} "
                  f"{cpu_sig['ncon']:4d}/{w_sig['ncon']:<5d} {'OK' if same else 'DIFF':>6s}")
        print()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "phase1_nominal_parity.json"), "w") as f:
        json.dump(dict(versions=versions(), rows=rows), f, indent=2)


if __name__ == "__main__":
    main()
