import numpy as np, mujoco
from config import load_model
from fixtures import build_fixtures
from cpu_fd import prepare, cpu_rollout
from tangent import tangent_state_difference
from warp_fd import WarpBatch

m, fx, info = build_fixtures()
d = mujoco.MjData(m)
b = WarpBatch(m, nworld=4)
print("dx.nefc shape:", b.dx.nefc.numpy().shape, "| nacon shape:", b.dx.nacon.numpy().shape)
x, u = fx["stance"]
ws = prepare(m, d, x, u)
qpos = np.tile(x.qpos, (4,1)); qvel = np.tile(x.qvel,(4,1)); act = np.zeros((4,0)); ctrl=np.tile(u,(4,1))
b.set_batch(qpos, qvel, act, ctrl, np.tile(ws,(4,1)))
b.step(1)
qp, qv, ac = b.get_batch()
print("warp qpos[0][:7]:", np.round(qp[0][:7],6))
y_cpu = cpu_rollout(m, d, x, u, S=1, warmstart=ws)
print("cpu  qpos[:7]   :", np.round(y_cpu.qpos[:7],6))
from tangent import State
y_w = State(qp[0], qv[0], np.zeros(0))
dd = tangent_state_difference(m, y_cpu, y_w)
print("tangent err: |dq|=%.3e |dv|=%.3e" % (np.linalg.norm(dd[:m.nv]), np.linalg.norm(dd[m.nv:])))
print("warp contact sigs:", b.contact_signatures()[:2])
print("cpu ncon:", d.ncon)
