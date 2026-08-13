import numpy as np, mujoco, warp as wp
from config import dims
from fixtures import build_fixtures
from cpu_fd import prepare
from batch_common import build_perturbation_batch
from warp_fd import WarpBatch
m, fx, _ = build_fixtures(); d = mujoco.MjData(m)
D = dims(m); per_jac = 2*(D['nx']+D['nu'])
x,u = fx["stance"]; ws = prepare(m,d,x,u)
qp1,qv1,ac1,ct1 = build_perturbation_batch(m,x,u,1e-4)
qp1,qv1,ct1 = qp1[:per_jac],qv1[:per_jac],ct1[:per_jac]
for tag, warm in [("no-warmup", False), ("with-warmup", True)]:
    K=1; nworld=K*per_jac
    b = WarpBatch(m, nworld=nworld, per_world_con=24, per_world_efc=128)
    qp=np.tile(qp1,(K,1)); qv=np.tile(qv1,(K,1)); ct=np.tile(ct1,(K,1))
    ac=np.zeros((nworld,m.na)); wsb=np.tile(ws,(nworld,1))
    b.set_batch(qp,qv,ac,ct,wsb)
    if warm:
        b.step(5, graph=False)
        b.set_batch(qp,qv,ac,ct,wsb)
    try:
        b.capture(5); b.step(5, graph=True)
        print(f"{tag}: OK  qpos0={b.get_batch()[0][0][:3]}")
    except Exception as e:
        print(f"{tag}: FAIL {type(e).__name__}: {str(e)[:120]}")
    del b
