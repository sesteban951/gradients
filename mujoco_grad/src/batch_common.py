"""Perturbation-batch construction and Jacobian assembly.

Shared by the MJWarp, MJX and threaded-CPU back ends, so it must not import warp.
"""

import numpy as np
import mujoco

from tangent import State, tangent_perturb, basis, tangent_state_difference


def build_perturbation_batch(m, x, u, h):
    """Return (qpos, qvel, act, ctrl) arrays for the 2(nx+nu)+1 worlds."""
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    nworld = 2 * (nx + nu) + 1

    qpos = np.zeros((nworld, m.nq))
    qvel = np.zeros((nworld, nv))
    act = np.zeros((nworld, na))
    ctrl = np.zeros((nworld, nu))

    for i in range(nx):
        for k, s in enumerate((+1.0, -1.0)):
            xp = tangent_perturb(m, x, basis(nx, i, s * h))
            w = 2 * i + k
            qpos[w], qvel[w] = xp.qpos, xp.qvel
            if na:
                act[w] = xp.act
            ctrl[w] = u

    base = 2 * nx
    for j in range(nu):
        for k, s in enumerate((+1.0, -1.0)):
            w = base + 2 * j + k
            qpos[w], qvel[w] = x.qpos, x.qvel
            if na:
                act[w] = x.act
            up = np.array(u, dtype=np.float64).copy()
            up[j] += s * h
            ctrl[w] = up

    # nominal world (last)
    qpos[-1], qvel[-1] = x.qpos, x.qvel
    if na:
        act[-1] = x.act
    ctrl[-1] = u
    return qpos, qvel, act, ctrl


def assemble_jacobians(m, qpos, qvel, act, h):
    """Manifold-aware assembly of A, B from the batched final states (float64)."""
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    def st(w):
        return State(qpos[w], qvel[w], act[w] if na else np.zeros(0))

    for i in range(nx):
        A[:, i] = tangent_state_difference(m, st(2 * i + 1), st(2 * i)) / (2 * h)
    base = 2 * nx
    for j in range(nu):
        p = base + 2 * j
        B[:, j] = tangent_state_difference(m, st(p + 1), st(p)) / (2 * h)
    return A, B
