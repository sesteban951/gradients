import sys, numpy as np, mujoco
sys.path.insert(0,'src')
from config import load_model
m = load_model()
print("actuator trntype/gaintype:", [mujoco.mjtGain(m.actuator_gaintype[i]).name for i in range(m.nu)][:3])
print("ctrlrange[0:3]:", m.actuator_ctrlrange[:3])
print("keyframe qpos:", np.round(m.key_qpos[0],3) if m.nkey else None)
print("keyframe ctrl:", np.round(m.key_ctrl[0],3) if m.nkey and m.key_ctrl.size else None)
d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0); mujoco.mj_forward(m,d)
# foot geoms = spheres
feet = [g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SPHERE and (m.geom_contype[g] or m.geom_conaffinity[g])]
print("foot sphere geoms:", feet, "radii:", [float(m.geom_size[g][0]) for g in feet])
print("foot world z:", np.round([d.geom_xpos[g][2]-m.geom_size[g][0] for g in feet],5))
print("ncon:", d.ncon, "nefc:", d.nefc)
for i in range(d.ncon):
    print("   con", i, "geoms", d.contact.geom[i], "dist %.3e"%d.contact.dist[i], "margin %.3e"%d.contact.includemargin[i])
print("geom margin (nonzero):", np.unique(m.geom_margin))
