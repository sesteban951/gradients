import dataclasses, inspect, mujoco, mujoco_warp as mjw
from mujoco_warp._src import types as wt
f = [x.name for x in dataclasses.fields(wt.Data)]
print("ncon-like fields:", [x for x in f if 'con' in x.lower()][:20])
print("nefc-like:", [x for x in f if 'efc' in x.lower()][:10])
print("solver fields:", [x for x in f if 'solver' in x.lower() or 'niter' in x.lower()])
print()
print("put_data sig:", inspect.signature(mjw.put_data))
print("make_data sig:", inspect.signature(mjw.make_data))
print("get_data_into sig:", inspect.signature(mjw.get_data_into))
