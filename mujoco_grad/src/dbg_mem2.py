import sys, numpy as np, mujoco, warp as wp, mujoco_warp as mjw
from config import load_model
nworld, pe = int(sys.argv[1]), int(sys.argv[2])
m = load_model(); d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
dev = wp.get_device("cuda:0"); f0 = dev.free_memory
try:
    dx = mjw.put_data(m, d0, nworld=nworld, nconmax=nworld*12, njmax=nworld*pe)
    a = (f0-dev.free_memory)/1e6
    mx = mjw.put_model(m)
    mjw.step(mx, dx); wp.synchronize()
    b = (f0-dev.free_memory)/1e6
    with wp.ScopedCapture() as c: mjw.step(mx, dx)
    wp.capture_launch(c.graph); wp.synchronize()
    print(f"nworld={nworld:5d} efc/w={pe:3d} njmax={nworld*pe:7d} | put_data {a:8.1f} MB | +step {b:8.1f} MB | +graph {(f0-dev.free_memory)/1e6:8.1f} MB  OK")
except Exception as ex:
    print(f"nworld={nworld:5d} efc/w={pe:3d} njmax={nworld*pe:7d} | FAIL {str(ex)[:70]}")
