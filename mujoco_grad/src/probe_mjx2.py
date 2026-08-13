import os; os.environ.setdefault("JAX_PLATFORMS","cpu")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np, mujoco, inspect
from mujoco import mjx
print("mjx api:", [a for a in dir(mjx) if not a.startswith('_')][:40])
print("make_data sig:", inspect.signature(mjx.make_data))
print("put_data sig:", inspect.signature(mjx.put_data))
M="/home/sergio/projects_third_party/mujoco_mpc/build/_deps/menagerie-src/unitree_go1"
m = mujoco.MjModel.from_xml_path(f"{M}/scene.xml")
mx = mjx.put_model(m)
dx = mjx.make_data(mx)
print("Data fields:", [f for f in dx.__dataclass_fields__ if f in ('qpos','qvel','act','ctrl','qacc_warmstart','ncon','contact','efc_force','time')])
print("qpos dtype", dx.qpos.dtype, "shape", dx.qpos.shape)
