import numpy as np, mujoco, warp as wp, mujoco_warp as mjw
M="/home/sergio/projects_third_party/mujoco_mpc/build/_deps/menagerie-src/unitree_go1"
m = mujoco.MjModel.from_xml_path(f"{M}/scene.xml")
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d)
print("warp", wp.__version__, "| devices:", wp.get_cuda_device_count())
try:
    mx = mjw.put_model(m)
    print("put_model OK")
    print("  opt.tolerance (warp) =", mx.opt.tolerance.numpy())
    print("  opt.iterations =", mx.opt.iterations)
    print("  cone =", mx.opt.cone, " solver =", mx.opt.solver, " integrator =", mx.opt.integrator)
    dx = mjw.put_data(m, d, nworld=4, nconmax=512, njmax=2048)
    print("put_data OK, nworld=4")
    mjw.step(mx, dx)
    wp.synchronize()
    print("step OK | qpos[0][:7] =", np.round(dx.qpos.numpy()[0][:7],6))
    print("  ncon:", dx.ncon.numpy())
except Exception as e:
    import traceback; traceback.print_exc()
