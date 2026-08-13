import numpy as np, mujoco, os
M="/home/sergio/projects_third_party/mujoco_mpc/build/_deps/menagerie-src/unitree_go1"
print("files:", sorted(os.listdir(M)))
m = mujoco.MjModel.from_xml_path(f"{M}/scene.xml")
print(f"nq={m.nq} nv={m.nv} na={m.na} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")
print(f"nx (tangent) = {2*m.nv+m.na}, nworld = {2*(2*m.nv+m.na+m.nu)}")
print("integrator:", mujoco.mjtIntegrator(m.opt.integrator).name, "solver:", mujoco.mjtSolver(m.opt.solver).name)
print("cone:", mujoco.mjtCone(m.opt.cone).name, "timestep:", m.opt.timestep, "tol:", m.opt.tolerance, "iters:", m.opt.iterations)
types = {}
for g in range(m.ngeom):
    if m.geom_contype[g] or m.geom_conaffinity[g]:
        t = mujoco.mjtGeom(m.geom_type[g]).name
        types[t] = types.get(t,0)+1
print("collidable geom types:", types)
print("jnt types:", [mujoco.mjtJoint(m.jnt_type[j]).name for j in range(m.njnt)][:3], "...")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m,d)
mujoco.mj_forward(m,d)
print("keyframes:", m.nkey, "| ncon at keyframe:", d.ncon, "| qpos0:", np.round(d.qpos[:7],3))
