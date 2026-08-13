import sys, numpy as np, mujoco, warp as wp, mujoco_warp as mjw
from config import load_model
nworld = int(sys.argv[1]); nccd = sys.argv[2]
nccd = None if nccd=="auto" else int(nccd)
m = load_model(); d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
dev = wp.get_device("cuda:0"); f0 = dev.free_memory
kw = dict(nworld=nworld, nconmax=nworld*12, njmax=nworld*64)
if nccd is not None: kw["nccdmax"] = nccd
try:
    dx = mjw.put_data(m, d0, **kw)
    a=(f0-dev.free_memory)/1e6
    mx = mjw.put_model(m); mjw.step(mx,dx); wp.synchronize()
    with wp.ScopedCapture() as c: mjw.step(mx,dx)
    wp.capture_launch(c.graph); wp.synchronize()
    print(f"nworld={nworld:5d} nccdmax={str(nccd):>6s} | put_data {a:8.1f} MB | total {(f0-dev.free_memory)/1e6:8.1f} MB  OK")
except Exception as ex:
    print(f"nworld={nworld:5d} nccdmax={str(nccd):>6s} | FAIL {str(ex)[:70]}")
