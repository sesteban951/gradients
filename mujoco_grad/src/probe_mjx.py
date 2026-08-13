import os
os.environ.setdefault("JAX_PLATFORMS","cpu")   # feasibility only; GPU busy
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np, mujoco
from mujoco import mjx
M="/home/sergio/projects_third_party/mujoco_mpc/build/_deps/menagerie-src/unitree_go1"
m = mujoco.MjModel.from_xml_path(f"{M}/scene.xml")
m.opt.tolerance=1e-6; m.opt.iterations=100; m.opt.ls_iterations=50
print("mujoco", mujoco.__version__, "| jax", jax.__version__)
for cone,name in [(mujoco.mjtCone.mjCONE_ELLIPTIC,"elliptic"),(mujoco.mjtCone.mjCONE_PYRAMIDAL,"pyramidal")]:
    m.opt.cone = cone
    try:
        mx = mjx.put_model(m)
        print(f"  put_model cone={name}: OK")
    except Exception as e:
        print(f"  put_model cone={name}: FAIL -> {type(e).__name__}: {str(e)[:200]}")
