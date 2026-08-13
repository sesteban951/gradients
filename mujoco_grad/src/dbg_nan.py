import os, numpy as np, mujoco
from config import load_model, cone_name
from fixtures import build_fixtures
from cpu_fd import prepare, cpu_rollout
from warp_fd import WarpBatch, warp_fd
from batch_common import build_perturbation_batch
m, fx, fi = build_fixtures(); d = mujoco.MjData(m)
print("cone:", cone_name(m))
nx=2*m.nv+m.na; nworld=2*(nx+m.nu)+1
b = WarpBatch(m, nworld=nworld)
x,u = fx["loaded"]; ws = prepare(m,d,x,u)
print("loaded: ncon=%d nefc=%d min_gap=%.3e" % (fi['loaded']['ncon'], fi['loaded']['nefc'], fi['loaded']['min_gap']))
for h in [1e-2, 1e-3]:
    A,B,_ = warp_fd(b,m,x,u,h,S=1,warmstart=ws)
    qp,qv,ac = b.get_batch()
    bad_q = np.argwhere(~np.isfinite(qp)); bad_v = np.argwhere(~np.isfinite(qv))
    print(f"h={h:.0e}: A finite={np.isfinite(A).all()} | nonfinite qpos entries={len(bad_q)} qvel={len(bad_v)}")
    if len(bad_q): print("   worlds with bad qpos:", sorted(set(bad_q[:,0]))[:10])
    if len(bad_v): print("   worlds with bad qvel:", sorted(set(bad_v[:,0]))[:10])
    print("   max|qvel| finite part:", np.nanmax(np.abs(qv[np.isfinite(qv)])))
# nominal single world
b1 = WarpBatch(m, nworld=1)
b1.set_batch(x.qpos[None], x.qvel[None], np.zeros((1,0)), np.asarray(u)[None], ws[None])
b1.step(1)
q1,v1,_ = b1.get_batch()
print("nominal warp qvel finite:", np.isfinite(v1).all(), "| max|v|:", np.abs(v1).max())
ycpu = cpu_rollout(m,d,x,u,S=1,warmstart=ws)
print("nominal cpu  max|v|:", np.abs(ycpu.qvel).max())
