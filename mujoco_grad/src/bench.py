"""Wall-time benchmark: how expensive is one (A, B) pair from each source?

  1. mjd_transitionFD          CPU, single thread, staged/skipped pipeline
  2. blackbox CPU serial       CPU, single thread, full rollouts
  3. threaded CPU rollout      CPU, mujoco.rollout with a persistent 32-thread pool
  4. MJWarp batched            GPU, 2(nx+nu)+1 worlds in one launch
"""

import json, os, time, argparse
import numpy as np
import mujoco
from mujoco import rollout

from config import dims, versions, RESULTS
from fixtures import build_fixtures
from cpu_fd import blackbox_fd, transition_fd_reference, prepare, cpu_rollout
from tangent import State, tangent_state_difference
from warp_fd import WarpBatch, warp_fd
from batch_common import build_perturbation_batch, assemble_jacobians


def timeit(fn, n, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


# --------------------------------------------------------------- threaded CPU
class ThreadedFD:
    def __init__(self, m, nthread):
        self.m = m
        self.nthread = nthread
        self.nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.pool = rollout.Rollout(nthread=nthread)
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]

    def close(self):
        self.pool.close()

    def __call__(self, x, u, h, S, warmstart):
        m = self.m
        nv, na, nu = m.nv, m.na, m.nu
        nx = 2 * nv + na
        qpos, qvel, act, ctrl = build_perturbation_batch(m, x, u, h)
        nroll = qpos.shape[0]

        init = np.zeros((nroll, self.nstate))
        d = self.datas[0]
        for w in range(nroll):
            d.qpos[:] = qpos[w]; d.qvel[:] = qvel[w]
            if na: d.act[:] = act[w]
            d.time = 0.0
            mujoco.mj_getState(m, d, init[w], mujoco.mjtState.mjSTATE_FULLPHYSICS)
        control = np.repeat(ctrl[:, None, :], S, axis=1)
        ws = np.tile(np.asarray(warmstart), (nroll, 1))

        state, _ = self.pool.rollout(m, self.datas, init, control,
                                     nstep=S, initial_warmstart=ws)
        final = state[:, -1, :]
        # FULLPHYSICS layout: [time, qpos, qvel, act, plugin]
        oq = 1
        ov = 1 + m.nq
        oa = 1 + m.nq + nv
        fq = final[:, oq:oq + m.nq]
        fv = final[:, ov:ov + nv]
        fa = final[:, oa:oa + na] if na else np.zeros((nroll, 0))
        return assemble_jacobians(m, fq, fv, fa, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--nthread", type=int, default=32)
    args = ap.parse_args()

    m, fx, _ = build_fixtures()
    d = mujoco.MjData(m)
    D = dims(m); nx, nu = D["nx"], D["nu"]
    nworld = 2 * (nx + nu) + 1
    x, u = fx["stance"]
    ws = prepare(m, d, x, u)
    h = 1e-4

    batch = WarpBatch(m, nworld=nworld, per_world_con=24, per_world_efc=128)
    thr = ThreadedFD(m, args.nthread)

    rows = []
    print(f"one (A,B) pair, go1: nx={nx} nu={nu} -> {nworld} worlds, h={h:.0e}, "
          f"{args.reps} reps, nthread={args.nthread}\n")
    hdr = (f"{'S':>3s} {'mjd_transFD':>12s} {'cpu_serial':>12s} "
           f"{'cpu_Nthr':>12s} {'warp_eager':>12s} {'warp_graph':>12s} "
           f"{'warp_gpu_only':>13s} {'vs_mjd':>8s} {'vs_thr':>8s}").replace("N", str(args.nthread))
    print(hdr); print("-" * len(hdr))

    for S in (1, 5, 20):
        batch.capture(S)
        if S == 1:
            t_mjd = timeit(lambda: transition_fd_reference(m, d, x, u, h, True), args.reps)
        else:
            t_mjd = float("nan")  # mjd_transitionFD is one-step only
        t_ser = timeit(lambda: blackbox_fd(m, d, x, u, h, S=S, centered=True),
                       max(3, args.reps // 4))
        t_thr = timeit(lambda: thr(x, u, h, S, ws), args.reps)
        t_eager = timeit(lambda: warp_fd(batch, m, x, u, h, S=S, warmstart=ws,
                                         graph=False), max(3, args.reps // 4))
        t_warp = timeit(lambda: warp_fd(batch, m, x, u, h, S=S, warmstart=ws,
                                        graph=True), args.reps)

        # GPU-only portion (exclude host perturbation build / gather / assembly)
        qp, qv, ac, ct = build_perturbation_batch(m, x, u, h)
        wsb = np.tile(ws, (nworld, 1))
        def gpu_only():
            batch.set_batch(qp, qv, ac, ct, wsb)
            batch.step(S, graph=True)
        t_gpu = timeit(gpu_only, args.reps)

        sp_mjd = t_mjd / t_warp if t_mjd == t_mjd else float("nan")
        sp_thr = t_thr / t_warp
        rows.append(dict(S=S, mjd=t_mjd, cpu_serial=t_ser, cpu_threaded=t_thr,
                         warp_eager=t_eager, warp=t_warp, warp_gpu_only=t_gpu,
                         speedup_vs_mjd=sp_mjd, speedup_vs_threaded=sp_thr))
        f = lambda v: "         n/a" if v != v else f"{v*1e3:9.2f} ms"
        g = lambda v: "     n/a" if v != v else f"{v:7.2f}x"
        print(f"{S:3d} {f(t_mjd)} {f(t_ser)} {f(t_thr)} {f(t_eager)} {f(t_warp)} "
              f"{f(t_gpu):>13s} {g(sp_mjd)} {g(sp_thr)}")

    # accuracy cross-check: threaded CPU FD must equal serial CPU FD exactly
    A_s, B_s = blackbox_fd(m, d, x, u, h, S=1, centered=True)
    A_t, B_t = thr(x, u, h, 1, ws)
    from metrics import rel_fro
    print(f"\nthreaded-vs-serial CPU FD agreement: relA={rel_fro(A_t, A_s):.3e} "
          f"relB={rel_fro(B_t, B_s):.3e}")

    thr.close()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "bench.json"), "w") as f:
        json.dump(dict(versions=versions(), dims=D, nworld=nworld, h=h,
                       nthread=args.nthread, rows=rows), f, indent=2)


if __name__ == "__main__":  # noqa
    main()
