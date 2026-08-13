"""Scaling: cost per Jacobian when K shooting knots are differentiated simultaneously.

One go1 Jacobian needs 2(nx+nu) = 96 worlds -- far too few to saturate a 4090. In
multiple shooting you need one Jacobian per knot, so the natural GPU batch is K*96
worlds. Both sides get ONE batched dispatch of K*96 rollouts:

  GPU : a single CUDA-graph launch over K*96 MJWarp worlds
  CPU : a single mujoco.rollout pool call over K*96 rollouts on 32 threads

Run one K per process (--single K); warp's allocations and captured graphs are not
released between iterations, so a single process OOMs well before the GPU is full.
"""

import json, os, time, argparse
import numpy as np
import mujoco
from mujoco import rollout

from config import dims, versions, RESULTS
from fixtures import build_fixtures
from cpu_fd import prepare
from batch_common import build_perturbation_batch
from warp_fd import WarpBatch


def timeit(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=int, required=True, help="number of knots K")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--nthread", type=int, default=32)
    ap.add_argument("--S", type=int, default=5)
    args = ap.parse_args()
    S, K = args.S, args.single

    m, fx, _ = build_fixtures()
    d = mujoco.MjData(m)
    D = dims(m); nx, nu = D["nx"], D["nu"]
    per_jac = 2 * (nx + nu)
    x, u = fx["stance"]
    ws = prepare(m, d, x, u)
    h = 1e-4
    nroll = K * per_jac

    qp1, qv1, ac1, ct1 = build_perturbation_batch(m, x, u, h)
    qp1, qv1, ct1 = qp1[:per_jac], qv1[:per_jac], ct1[:per_jac]
    qp = np.tile(qp1, (K, 1)); qv = np.tile(qv1, (K, 1)); ct = np.tile(ct1, (K, 1))
    ac = np.zeros((nroll, m.na)); wsb = np.tile(ws, (nroll, 1))

    # ---- CPU: one pool dispatch of all K*96 rollouts -------------------------
    nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    datas = [mujoco.MjData(m) for _ in range(args.nthread)]
    pool = rollout.Rollout(nthread=args.nthread)
    init = np.zeros((nroll, nstate))
    dd = datas[0]
    for w in range(nroll):
        dd.qpos[:] = qp[w]; dd.qvel[:] = qv[w]; dd.time = 0.0
        mujoco.mj_getState(m, dd, init[w], mujoco.mjtState.mjSTATE_FULLPHYSICS)
    control = np.repeat(ct[:, None, :], S, axis=1)

    def cpu():
        pool.rollout(m, datas, init, control, nstep=S, initial_warmstart=wsb)

    t_cpu = timeit(cpu, args.reps)

    # ---- GPU: one graph launch over all K*96 worlds ---------------------------
    t_gpu = float("nan"); t_gpu_only = float("nan"); err = None
    try:
        batch = WarpBatch(m, nworld=nroll, per_world_con=12, per_world_efc=64, nccdmax=512)
        batch.set_batch(qp, qv, ac, ct, wsb)
        batch.step(S, graph=False)          # warm up / load modules
        batch.set_batch(qp, qv, ac, ct, wsb)
        batch.capture(S)

        def gpu_full():
            batch.set_batch(qp, qv, ac, ct, wsb)
            batch.step(S, graph=True)
            batch.get_batch()

        def gpu_only():
            batch.step(S, graph=True)

        t_gpu = timeit(gpu_full, args.reps)
        t_gpu_only = timeit(gpu_only, args.reps)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:80]}"

    pool.close()
    row = dict(K=K, nworld=nroll, S=S, cpu=t_cpu, cpu_per=t_cpu / K,
               gpu=t_gpu, gpu_per=t_gpu / K, gpu_only=t_gpu_only,
               gpu_only_per=t_gpu_only / K,
               speedup=(t_cpu / t_gpu) if t_gpu == t_gpu else float("nan"), err=err)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"bench_scale_S{S}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    g = lambda v: "     n/a" if v != v else f"{v*1e3:8.3f}"
    print(f"{K:6d} {nroll:7d} {g(t_cpu)} {g(t_cpu/K)} {g(t_gpu)} {g(t_gpu/K)} "
          f"{g(t_gpu_only/K)} " +
          ("     n/a" if t_gpu != t_gpu else f"{t_cpu/t_gpu:7.2f}x") +
          (f"  [{err}]" if err else ""))


if __name__ == "__main__":
    main()
