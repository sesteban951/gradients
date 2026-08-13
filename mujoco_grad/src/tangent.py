"""Manifold-aware state packing, perturbation and differencing.

State is x = (qpos in R^nq, qvel in R^nv, act in R^na).
Tangent state is dx = (dq_tan in R^nv, dv in R^nv, da in R^na), dim nx = 2nv+na.

This ordering matches MuJoCo's mjd_transitionFD convention.
"""

import numpy as np
import mujoco


class State:
    """A full simulator state (qpos on the manifold, qvel, act)."""

    __slots__ = ("qpos", "qvel", "act")

    def __init__(self, qpos, qvel, act):
        self.qpos = np.array(qpos, dtype=np.float64).copy()
        self.qvel = np.array(qvel, dtype=np.float64).copy()
        self.act = np.array(act, dtype=np.float64).copy()

    def copy(self):
        return State(self.qpos, self.qvel, self.act)

    @staticmethod
    def from_data(d):
        return State(d.qpos, d.qvel, d.act)

    def into_data(self, d):
        d.qpos[:] = self.qpos
        d.qvel[:] = self.qvel
        if self.act.size:
            d.act[:] = self.act


def tangent_perturb(m, x, dx):
    """x (+) dx.  Generalized positions move via mj_integratePos (quaternion-aware)."""
    nv, na = m.nv, m.na
    out = x.copy()
    dq = np.ascontiguousarray(dx[:nv], dtype=np.float64)
    mujoco.mj_integratePos(m, out.qpos, dq, 1.0)
    out.qvel += dx[nv:2 * nv]
    if na:
        out.act += dx[2 * nv:2 * nv + na]
    return out


def tangent_state_difference(m, x1, x2):
    """x2 (-) x1 in the tangent space.  Positions use mj_differentiatePos."""
    nv, na = m.nv, m.na
    dq = np.zeros(nv, dtype=np.float64)
    mujoco.mj_differentiatePos(m, dq, 1.0, x1.qpos, x2.qpos)
    parts = [dq, x2.qvel - x1.qvel]
    if na:
        parts.append(x2.act - x1.act)
    return np.concatenate(parts)


def basis(nx, i, h):
    e = np.zeros(nx)
    e[i] = h
    return e
