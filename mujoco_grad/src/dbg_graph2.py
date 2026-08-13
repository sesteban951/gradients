import numpy as np, mujoco, warp as wp
from config import dims
from fixtures import build_fixtures
from cpu_fd import prepare
from batch_common import build_perturbation_batch
from warp_fd import WarpBatch
from bench import ThreadedFD, timeit
m, fx, _ = build_fixtures(); d = mujoco.MjData(m)
D = dims(m); per_jac = 2*(D['nx']+D['nu'])
x,u = fx["stance"]; ws = prepare(m,d,x,u)
qp1,qv1,ac1,ct1 = build_perturbation_batch(m,x,u,1e-4)
qp1,qv1,ct1 = qp1[:per_jac],qv1[:per_jac],ct1[:per_jac]
thr = ThreadedFD(m, 32)
K=1; nworld=K*per_jac; S=5
b = WarpBatch(m, nworld=nworld, per_world_con=24, per_world_efc=128)
qp=np.tile(qp1,(K,1)); qv=np.tile(qv1,(K,1)); ct=np.tile(ct1,(K,1))
ac=np.zeros((nworld,m.na)); wsb=np.tile(ws,(nworld,1))
b.set_batch(qp,qv,ac,ct,wsb)
b.capture(S)
print("capture ok")
def gpu():
    b.set_batch(qp,qv,ac,ct,wsb); b.step(S, graph=True); b.get_batch()
try:
    gpu(); print("set_batch-then-graph-launch OK")
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:150])
    print("retry without set_batch:", end=" ")
    try:
        b.step(S, graph=True); print("OK")
    except Exception as e2: print("FAIL", str(e2)[:100])
thr.close()
