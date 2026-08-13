"""Phase 2: does my manifold-aware black-box CPU FD reproduce mjd_transitionFD?

This must pass before any GPU comparison is meaningful -- it isolates perturbation,
state-differencing, packing, indexing and layout bugs from simulator differences.
"""

import json, os
import numpy as np
import mujoco

from config import load_model, dims, versions, RESULTS
from fixtures import build_fixtures, CATEGORY_ORDER
from cpu_fd import blackbox_fd, transition_fd_reference
from metrics import rel_fro

H_LIST = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def main():
    m, fx, info = build_fixtures()
    d = mujoco.MjData(m)
    D = dims(m)
    print("versions:", json.dumps(versions(), indent=None))
    print("dims:", D)
    print()

    rows = []
    hdr = f"{'fixture':10s} {'h':>8s} {'relerr(A)':>11s} {'relerr(B)':>11s} {'|A|_F':>10s} {'|B|_F':>10s}"
    print(hdr); print("-" * len(hdr))
    for name in CATEGORY_ORDER:
        x, u = fx[name]
        for h in H_LIST:
            A_ref, B_ref = transition_fd_reference(m, d, x, u, h, centered=True)
            A_bb, B_bb = blackbox_fd(m, d, x, u, h, S=1, centered=True)
            eA, eB = rel_fro(A_bb, A_ref), rel_fro(B_bb, B_ref)
            rows.append(dict(fixture=name, h=h, relA=eA, relB=eB,
                             normA=float(np.linalg.norm(A_ref)),
                             normB=float(np.linalg.norm(B_ref))))
            print(f"{name:10s} {h:8.0e} {eA:11.3e} {eB:11.3e} "
                  f"{np.linalg.norm(A_ref):10.3e} {np.linalg.norm(B_ref):10.3e}")
        print()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "phase2_cpu_vs_transitionFD.json"), "w") as f:
        json.dump(dict(versions=versions(), dims=D, rows=rows), f, indent=2)


if __name__ == "__main__":
    main()
