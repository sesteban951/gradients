"""Phase 0: frozen, reproducible configuration shared by CPU and MJWarp."""

import os
import subprocess
import numpy as np
import mujoco

MENAGERIE = "/home/sergio/projects_third_party/mujoco_mpc/build/_deps/menagerie-src"
MODEL_PATH = f"{MENAGERIE}/unitree_go1/scene.xml"

# MJWarp floors opt.tolerance at 1e-6 in io.py (_put_model), so CPU must match or the
# two solvers do different amounts of work and nominal parity is confounded.
FROZEN = dict(
    tolerance=1e-6,
    ls_tolerance=0.01,
    iterations=100,
    ls_iterations=50,
    timestep=0.002,
    integrator=mujoco.mjtIntegrator.mjINT_EULER,
    solver=mujoco.mjtSolver.mjSOL_NEWTON,
    cone=mujoco.mjtCone.mjCONE_ELLIPTIC,
)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


CONES = {
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
}


def load_model(path=MODEL_PATH, cone=None):
    """Load the model and apply the frozen option set.

    cone may be overridden (env GRAD_CONE) so that CPU, MJWarp and MJX can be run on
    matched physics -- MJX does not implement the elliptic cone for condim=1.
    """
    m = mujoco.MjModel.from_xml_path(path)
    m.opt.tolerance = FROZEN["tolerance"]
    m.opt.ls_tolerance = FROZEN["ls_tolerance"]
    m.opt.iterations = FROZEN["iterations"]
    m.opt.ls_iterations = FROZEN["ls_iterations"]
    m.opt.timestep = FROZEN["timestep"]
    m.opt.integrator = FROZEN["integrator"]
    m.opt.solver = FROZEN["solver"]
    cone = cone or os.environ.get("GRAD_CONE")
    m.opt.cone = CONES[cone] if cone else FROZEN["cone"]
    return m


def cone_name(m):
    return mujoco.mjtCone(m.opt.cone).name


def dims(m):
    nv, na, nu = m.nv, m.na, m.nu
    nx = 2 * nv + na
    return dict(nq=m.nq, nv=nv, na=na, nu=nu, nx=nx, nworld=2 * (nx + nu))


def versions():
    import warp
    info = {
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "warp": warp.__version__,
    }
    try:
        import mujoco_warp
        from importlib.metadata import version
        info["mujoco_warp"] = version("mujoco_warp")
    except Exception as e:  # pragma: no cover
        info["mujoco_warp"] = f"unavailable ({e})"
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        info["gpu"] = gpu.stdout.strip()
    except Exception:
        info["gpu"] = "unknown"
    info["cpu_count"] = os.cpu_count()
    return info
