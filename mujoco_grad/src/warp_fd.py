"""Phase 3: batched finite differences in MuJoCo Warp.

World layout (Section 9 of the plan):
    [x+0, x-0, x+1, x-1, ..., u+0, u-0, u+1, u-1, ...]
plus one extra nominal world at the end for parity checks.

All differencing is done on the host in float64 using MuJoCo's manifold routines, so
the only float32 in the pipeline is the simulation itself.
"""

import numpy as np
import mujoco
import warp as wp
import mujoco_warp as mjw

from tangent import State, tangent_perturb, basis, tangent_state_difference
from batch_common import build_perturbation_batch, assemble_jacobians


class WarpBatch:
    """A fixed pool of MJWarp worlds that can be reloaded with new states."""

    def __init__(self, m, nworld, nconmax=None, njmax=None, per_world_con=64,
                 per_world_efc=512, nccdmax=None):
        self.m = m
        self.nworld = nworld
        self.mx = mjw.put_model(m)
        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        kw = dict(nworld=nworld,
                  nconmax=nconmax or nworld * per_world_con,
                  njmax=njmax or nworld * per_world_efc)
        # nccdmax defaults to a multiple of nconmax; the EPA/CCD scratch it sizes is
        # enormous and unnecessary for primitive-only collision geometry.
        if nccdmax is not None:
            kw["nccdmax"] = nccdmax
        self.dx = mjw.put_data(m, d0, **kw)
        self._nq, self._nv, self._na, self._nu = m.nq, m.nv, m.na, m.nu
        self._graphs = {}

    def capture(self, S):
        """Record S steps into a CUDA graph.

        Without this, one batched step is dominated by per-kernel Python launch
        overhead (~57 ms vs ~0.3 ms measured for 97 go1 worlds).
        """
        with wp.ScopedCapture() as cap:
            for _ in range(S):
                mjw.step(self.mx, self.dx)
        self._graphs[S] = cap.graph
        return self._graphs[S]

    # -- host <-> device -------------------------------------------------------
    def set_batch(self, qpos, qvel, act, ctrl, warmstart=None):
        """qpos (nworld,nq), qvel (nworld,nv), act (nworld,na), ctrl (nworld,nu)."""
        self.dx.qpos.assign(np.ascontiguousarray(qpos, dtype=np.float32))
        self.dx.qvel.assign(np.ascontiguousarray(qvel, dtype=np.float32))
        if self._na:
            self.dx.act.assign(np.ascontiguousarray(act, dtype=np.float32))
        self.dx.ctrl.assign(np.ascontiguousarray(ctrl, dtype=np.float32))
        if warmstart is not None:
            self.dx.qacc_warmstart.assign(np.ascontiguousarray(warmstart, dtype=np.float32))
        self.dx.time.assign(np.zeros(self.nworld, dtype=np.float32))

    def get_batch(self):
        """Returns float64-upcast (qpos, qvel, act) for every world."""
        qpos = self.dx.qpos.numpy().astype(np.float64)
        qvel = self.dx.qvel.numpy().astype(np.float64)
        act = (self.dx.act.numpy().astype(np.float64) if self._na
               else np.zeros((self.nworld, 0)))
        return qpos, qvel, act

    def step(self, S=1, graph=False):
        if graph:
            if S not in self._graphs:
                self.capture(S)
            wp.capture_launch(self._graphs[S])
        else:
            for _ in range(S):
                mjw.step(self.mx, self.dx)
        wp.synchronize()

    def max_nefc(self):
        return int(self.dx.nefc.numpy().max())

    # -- contact diagnostics ---------------------------------------------------
    def contact_signatures(self):
        """Per-world (ncon, sorted geom pairs) after the most recent forward pass."""
        nacon = int(self.dx.nacon.numpy()[0])
        if nacon == 0:
            return [dict(ncon=0, pairs=()) for _ in range(self.nworld)]
        wid = self.dx.contact.worldid.numpy()[:nacon]
        geom = self.dx.contact.geom.numpy()[:nacon]
        out = [[] for _ in range(self.nworld)]
        for k in range(nacon):
            w = int(wid[k])
            if 0 <= w < self.nworld:
                out[w].append(tuple(sorted((int(geom[k][0]), int(geom[k][1])))))
        return [dict(ncon=len(p), pairs=tuple(sorted(p))) for p in out]


def warp_fd(batch, m, x, u, h, S=1, warmstart=None, graph=False):
    """One full batched central-difference evaluation. Returns (A, B, signatures)."""
    qpos, qvel, act, ctrl = build_perturbation_batch(m, x, u, h)
    ws = None
    if warmstart is not None:
        ws = np.tile(np.asarray(warmstart, dtype=np.float64), (batch.nworld, 1))
    batch.set_batch(qpos, qvel, act, ctrl, ws)
    batch.step(S, graph=graph)
    qp, qv, ac = batch.get_batch()
    A, B = assemble_jacobians(m, qp, qv, ac, h)
    return A, B, batch.contact_signatures()
