import os; os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ["GRAD_CONE"]="pyramidal"
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np, mujoco
from mujoco import mjx
from config import load_model
m = load_model()
print("cone:", mujoco.mjtCone(m.opt.cone).name)
mx = mjx.put_model(m)
dx = mjx.make_data(mx)
print("qpos dtype:", dx.qpos.dtype, "| qvel:", dx.qvel.dtype)
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d)
dx = mjx.put_data(m, d)
dx2 = jax.jit(mjx.step)(mx, dx)
mujoco.mj_step(m, d)
print("mjx  qpos[:7]:", np.round(np.asarray(dx2.qpos)[:7],9))
print("cpu  qpos[:7]:", np.round(d.qpos[:7],9))
print("max abs diff:", float(np.abs(np.asarray(dx2.qpos)-d.qpos).max()))
