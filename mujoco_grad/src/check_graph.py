import numpy as np, mujoco
from config import dims
from fixtures import build_fixtures, CATEGORY_ORDER
from cpu_fd import prepare
from metrics import rel_fro
from warp_fd import WarpBatch, warp_fd
m, fx, _ = build_fixtures(); d = mujoco.MjData(m)
D = dims(m); nworld = 2*(D['nx']+D['nu'])+1
big  = WarpBatch(m, nworld=nworld)                                  # generous sizing, eager
lean = WarpBatch(m, nworld=nworld, per_world_con=24, per_world_efc=128)
print(f"{'fixture':10s} {'graph_vs_eager':>15s} {'lean_vs_big':>13s} {'max_nefc':>9s}")
for name in CATEGORY_ORDER:
    x,u = fx[name]; ws = prepare(m,d,x,u)
    Ae,_ ,_= warp_fd(big,  m,x,u,1e-4,S=1,warmstart=ws, graph=False)
    Ag,_ ,_= warp_fd(big,  m,x,u,1e-4,S=1,warmstart=ws, graph=True)
    Al,_ ,_= warp_fd(lean, m,x,u,1e-4,S=1,warmstart=ws, graph=True)
    print(f"{name:10s} {rel_fro(Ag,Ae):15.2e} {rel_fro(Al,Ae):13.2e} {lean.max_nefc():9d}")
