import numpy as np, mujoco
from config import cone_name
from fixtures import build_fixtures
from cpu_fd import prepare, cpu_rollout
from tangent import State
from warp_fd import WarpBatch, warp_fd
from batch_common import build_perturbation_batch
m, fx, fi = build_fixtures(); d = mujoco.MjData(m)
x,u = fx["loaded"]; ws = prepare(m,d,x,u); h=1e-3
nx=2*m.nv+m.na; N=2*(nx+m.nu)+1
b = WarpBatch(m, nworld=N)
qp,qv,ac,ct = build_perturbation_batch(m,x,u,h)
b.set_batch(qp,qv,ac,ct,np.tile(ws,(N,1))); b.step(1)
_,v,_ = b.get_batch()
bad = sorted(set(np.argwhere(~np.isfinite(v))[:,0].tolist()))
print("cone:", cone_name(m), "| bad worlds in 97-batch:", bad)
print("   -> affected A columns (state):", [w//2 for w in bad if w < 2*nx])
# rerun the SAME bad states alone in a 1-world sim
b1 = WarpBatch(m, nworld=1)
res=[]
for w in bad:
    b1.set_batch(qp[w][None], qv[w][None], np.zeros((1,0)), ct[w][None], ws[None])
    b1.step(1); _,v1,_ = b1.get_batch()
    ycpu = cpu_rollout(m,d,State(qp[w],qv[w],np.zeros(0)),ct[w],S=1,warmstart=ws)
    res.append((w, bool(np.isfinite(v1).all()), float(np.abs(ycpu.qvel).max())))
print("   world | finite alone | cpu max|v|")
for w,f,c in res: print(f"   {w:5d} | {str(f):12s} | {c:.4f}")
# and a small batch of just those states
k=len(bad); bk = WarpBatch(m, nworld=k)
bk.set_batch(qp[bad], qv[bad], np.zeros((k,0)), ct[bad], np.tile(ws,(k,1))); bk.step(1)
_,vk,_ = bk.get_batch()
print(f"   same {k} states as a {k}-world batch: finite={np.isfinite(vk).all()}")
