import numpy as np, mujoco, warp as wp, mujoco_warp as mjw, sys
from config import load_model, dims
m = load_model(); D = dims(m)
d0 = mujoco.MjData(m); mujoco.mj_forward(m, d0)
def mb(): return wp.get_device("cuda:0").free_memory/1e6
for nworld, pc, pe in [(96,24,128),(768,24,128),(768,12,64),(768,8,48),(3072,8,48),(6144,8,48)]:
    free0 = mb()
    try:
        dx = mjw.put_data(m, d0, nworld=nworld, nconmax=nworld*pc, njmax=nworld*pe)
        used = free0 - mb()
        print(f"nworld={nworld:5d} con/w={pc:3d} efc/w={pe:4d} -> nconmax={nworld*pc:7d} njmax={nworld*pe:8d} | Data ~{used:8.1f} MB")
        del dx
    except Exception as e:
        print(f"nworld={nworld:5d} con/w={pc:3d} efc/w={pe:4d} -> FAIL {str(e)[:90]}")
