import numpy as np, mujoco
from config import load_model, cone_name
from fixtures import build_fixtures
from cpu_fd import prepare, cpu_rollout
from warp_fd import WarpBatch, warp_fd
m, fx, fi = build_fixtures(); d = mujoco.MjData(m)
x,u = fx["loaded"]; ws = prepare(m,d,x,u)
print("cone:", cone_name(m), "| loaded nefc:", fi['loaded']['nefc'])
# 1) isolated single world, nothing else touched
b1 = WarpBatch(m, nworld=1)
b1.set_batch(x.qpos[None], x.qvel[None], np.zeros((1,0)), np.asarray(u)[None], ws[None])
b1.step(1); q,v,_ = b1.get_batch()
print("A) single world, fresh          : finite=%s max|v|=%.4g" % (np.isfinite(v).all(), np.nanmax(np.abs(v))))
# 2) full 97-world perturbation batch
nx=2*m.nv+m.na; b = WarpBatch(m, nworld=2*(nx+m.nu)+1)
A,B,_ = warp_fd(b,m,x,u,1e-3,S=1,warmstart=ws)
q2,v2,_ = b.get_batch()
nomw = b.nworld-1
print("B) 97-world batch, nominal world: finite=%s" % np.isfinite(v2[nomw]).all())
print("   nonfinite worlds: %d / %d" % (len(set(np.argwhere(~np.isfinite(v2))[:,0])), b.nworld))
# 3) same single world again AFTER the batch
b1.set_batch(x.qpos[None], x.qvel[None], np.zeros((1,0)), np.asarray(u)[None], ws[None])
b1.step(1); q3,v3,_ = b1.get_batch()
print("C) single world, after batch    : finite=%s max|v|=%.4g" % (np.isfinite(v3).all(), np.nanmax(np.abs(v3))))
print("cpu max|v| =", np.abs(cpu_rollout(m,d,x,u,S=1,warmstart=ws).qvel).max())
