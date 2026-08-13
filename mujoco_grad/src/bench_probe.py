import time, numpy as np, mujoco, warp as wp, mujoco_warp as mjw
from config import load_model, dims
from fixtures import build_fixtures
from cpu_fd import prepare
from batch_common import build_perturbation_batch

m, fx, _ = build_fixtures(); d = mujoco.MjData(m)
D = dims(m); nworld = 2*(D['nx']+D['nu'])+1
x,u = fx["stance"]; ws = prepare(m,d,x,u)
qp,qv,ac,ct = build_perturbation_batch(m,x,u,1e-4)

def timeit(fn,n=20,w=3):
    for _ in range(w): fn()
    wp.synchronize(); t=time.perf_counter()
    for _ in range(n): fn()
    wp.synchronize(); return (time.perf_counter()-t)/n

for nconmax, njmax, gcond in [(nworld*64, nworld*512, True),
                              (nworld*16, nworld*64,  True),
                              (nworld*16, nworld*64,  False)]:
    mx = mjw.put_model(m)
    mx.opt.graph_conditional = gcond
    d0 = mujoco.MjData(m); mujoco.mj_forward(m,d0)
    dx = mjw.put_data(m, d0, nworld=nworld, nconmax=nconmax, njmax=njmax)
    def setb():
        dx.qpos.assign(qp.astype(np.float32)); dx.qvel.assign(qv.astype(np.float32))
        dx.ctrl.assign(ct.astype(np.float32)); dx.qacc_warmstart.assign(np.tile(ws,(nworld,1)).astype(np.float32))
    setb()
    t_eager = timeit(lambda: (mjw.step(mx,dx), wp.synchronize()))
    # graph capture
    tg = float('nan')
    try:
        with wp.ScopedCapture() as cap:
            mjw.step(mx, dx)
        g = cap.graph
        t_g = timeit(lambda: (wp.capture_launch(g), wp.synchronize()))
        tg = t_g
    except Exception as e:
        tg_err = f"{type(e).__name__}: {str(e)[:90]}"
        print("   capture failed:", tg_err)
    print(f"nconmax={nconmax:6d} njmax={njmax:6d} gcond={gcond!s:5s} | eager {t_eager*1e3:8.2f} ms | graph {tg*1e3:8.2f} ms")
