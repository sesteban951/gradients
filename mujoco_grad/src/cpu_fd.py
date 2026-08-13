"""CPU-side rollouts, finite differences and contact diagnostics.

Warm-start handling mirrors mjd_transitionFD: the *nominal* qacc_warmstart is restored
before every perturbed rollout so that all perturbations start the solver identically.
"""

import numpy as np
import mujoco

from tangent import State, tangent_perturb, tangent_state_difference, basis


def prepare(m, d, x, u):
    """Load state+control into d, run mj_forward, return the nominal warm-start."""
    x.into_data(d)
    d.ctrl[:] = u
    mujoco.mj_forward(m, d)
    return d.qacc_warmstart.copy()


def cpu_rollout(m, d, x, u, S=1, warmstart=None):
    """Advance S steps from x with control u held fixed. Returns the final State."""
    x.into_data(d)
    d.ctrl[:] = u
    if warmstart is not None:
        d.qacc_warmstart[:] = warmstart
    for _ in range(S):
        mujoco.mj_step(m, d)
    return State.from_data(d)


def contact_signature(m, d):
    """Order-independent description of the active contact/constraint set."""
    n = d.ncon
    pairs = sorted(tuple(sorted((int(d.contact.geom[i][0]), int(d.contact.geom[i][1]))))
                   for i in range(n))
    return dict(
        ncon=int(n),
        nefc=int(d.nefc),
        pairs=tuple(pairs),
        dists=np.array([d.contact.dist[i] for i in range(n)], dtype=np.float64),
    )


def contact_probe(m, d, x, u):
    """Contact diagnostics at a state (after mj_forward, before stepping)."""
    x.into_data(d)
    d.ctrl[:] = u
    mujoco.mj_forward(m, d)
    sig = contact_signature(m, d)
    sig["solver_niter"] = int(np.sum(d.solver_niter))
    return sig


# ----------------------------------------------------------------------------- FD

def blackbox_fd(m, d, x, u, h, S=1, centered=True):
    """Black-box central-difference A, B in the tangent space.

    Deliberately does nothing but perturb, roll out and difference: any diagnostic
    mj_forward inside this loop perturbs the solver and shifts the result by ~1e-6.
    Use contact_stability() for diagnostics instead.
    """
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    warmstart = prepare(m, d, x, u)

    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))

    def roll(xs, us):
        return cpu_rollout(m, d, xs, us, S=S, warmstart=warmstart)

    y0 = None if centered else roll(x, u)

    for i in range(nx):
        yp = roll(tangent_perturb(m, x, basis(nx, i, h)), u)
        if centered:
            ym = roll(tangent_perturb(m, x, basis(nx, i, -h)), u)
            A[:, i] = tangent_state_difference(m, ym, yp) / (2 * h)
        else:
            A[:, i] = tangent_state_difference(m, y0, yp) / h

    for j in range(nu):
        up = np.array(u, dtype=np.float64).copy(); up[j] += h
        yp = roll(x, up)
        if centered:
            um = np.array(u, dtype=np.float64).copy(); um[j] -= h
            ym = roll(x, um)
            B[:, j] = tangent_state_difference(m, ym, yp) / (2 * h)
        else:
            B[:, j] = tangent_state_difference(m, y0, yp) / h

    return A, B


def contact_stability(m, x, u, h, S=1):
    """Per-column flag: did BOTH the +h and -h perturbations preserve the contact set?

    Uses its own MjData so it can never perturb an FD computation.
    Column order is [state 0..nx-1, control 0..nu-1].
    """
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    dd = mujoco.MjData(m)

    def sig_path(xs, us):
        """Contact sets seen at the start of each of the S substeps."""
        xs.into_data(dd)
        dd.ctrl[:] = us
        out = []
        for _ in range(S):
            mujoco.mj_forward(m, dd)
            out.append(contact_signature(m, dd)["pairs"])
            mujoco.mj_step(m, dd)
        return tuple(out)

    nom = sig_path(x, u)
    stable = np.ones(nx + nu, dtype=bool)
    for i in range(nx):
        p = sig_path(tangent_perturb(m, x, basis(nx, i, h)), u)
        mn = sig_path(tangent_perturb(m, x, basis(nx, i, -h)), u)
        stable[i] = (p == nom and mn == nom)
    for j in range(nu):
        up = np.array(u, dtype=np.float64).copy(); up[j] += h
        um = np.array(u, dtype=np.float64).copy(); um[j] -= h
        stable[nx + j] = (sig_path(x, up) == nom and sig_path(x, um) == nom)
    return stable


def transition_fd_reference(m, d, x, u, eps, centered=True):
    """MuJoCo's own mjd_transitionFD (one step, staged/skipped pipeline)."""
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    prepare(m, d, x, u)
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    mujoco.mjd_transitionFD(m, d, eps, centered, A, B, None, None)
    return A, B
