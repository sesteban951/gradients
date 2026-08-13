import mujoco, mujoco_warp as mjw, numpy as np, warp as wp
from mujoco_warp._src import types as wt
print("Data fields (subset):")
import dataclasses
fields = [f.name for f in dataclasses.fields(wt.Data)]
for k in ["qpos","qvel","act","ctrl","qacc_warmstart","ncon","nefc","contact","efc","time","qacc"]:
    print(f"  {k}: {'YES' if k in fields else 'no'}")
print("\nContact fields:", [f.name for f in dataclasses.fields(wt.Contact)])
print("\nDisableBit:", [x.name for x in wt.DisableBit])
print("\nmjw public API:", [a for a in dir(mjw) if not a.startswith('_')])
