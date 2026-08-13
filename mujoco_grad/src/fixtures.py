"""Phase 7: representative state suite (Table 1 of the plan)."""

import numpy as np
import mujoco

from config import load_model, RESULTS
from tangent import State
from cpu_fd import contact_probe

FOOT_GEOMS = None  # filled at import from the model


def _foot_geoms(m):
    return [g for g in range(m.ngeom)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            and (m.geom_contype[g] or m.geom_conaffinity[g])]


def _min_foot_dist(m, d):
    """Signed distance of the lowest foot sphere to the ground plane (z=0)."""
    feet = _foot_geoms(m)
    return min(d.geom_xpos[g][2] - m.geom_size[g][0] for g in feet)


def settle(m, steps=4000):
    """Run the position servos at the home pose until the robot is at rest."""
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[2] = 0.35
    ctrl = m.key_ctrl[0].copy()
    d.ctrl[:] = ctrl
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return State.from_data(d), ctrl


def _set_base_z(m, x, z):
    y = x.copy()
    y.qpos[2] = z
    return y


def _base_z_for_gap(m, x, target_gap, lo=0.15, hi=0.60):
    """Bisect base height so the lowest foot sits at `target_gap` above the plane."""
    d = mujoco.MjData(m)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        y = _set_base_z(m, x, mid)
        y.into_data(d)
        mujoco.mj_forward(m, d)
        if _min_foot_dist(m, d) < target_gap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_fixtures():
    m = load_model()
    d = mujoco.MjData(m)
    stance, ctrl = settle(m)

    fx = {}

    # --- flight / no contact -------------------------------------------------
    # home joint angles + home ctrl => zero servo error, clean free-fall
    f = State(m.key_qpos[0], np.zeros(m.nv), np.zeros(m.na))
    f.qpos[2] = 0.80
    fx["flight"] = (f, ctrl.copy())

    # --- stable stance -------------------------------------------------------
    s = stance.copy()
    s.qvel[:] = 0.0
    fx["stance"] = (s, ctrl.copy())

    # --- loaded foot contact -------------------------------------------------
    # press the body down: deeper penetration => much larger normal forces
    z_loaded = _base_z_for_gap(m, stance, -0.030)
    l = _set_base_z(m, stance, z_loaded)
    l.qvel[:] = 0.0
    fx["loaded"] = (l, ctrl.copy())

    # --- near contact onset --------------------------------------------------
    # A contact enters d.contact when dist < margin, so the *discrete* switching
    # boundary is gap == margin, not gap == 0.  Sit just inside it, moving down:
    # a small upward perturbation drops the contacts entirely.
    margin = float(m.geom_margin.max())
    z_on = _base_z_for_gap(m, stance, margin - 2e-5)
    o = _set_base_z(m, stance, z_on)
    o.qvel[:] = 0.0
    o.qvel[2] = -0.30
    fx["onset"] = (o, ctrl.copy())

    # --- near contact release ------------------------------------------------
    # Just outside the margin, moving up: the same mode change in reverse.
    z_off = _base_z_for_gap(m, stance, margin + 2e-5)
    r = _set_base_z(m, stance, z_off)
    r.qvel[:] = 0.0
    r.qvel[2] = +0.60
    fx["release"] = (r, ctrl.copy())

    # --- sliding / friction transition ---------------------------------------
    sl = stance.copy()
    sl.qvel[:] = 0.0
    sl.qvel[0] = 1.20          # lateral base velocity -> feet slide
    fx["sliding"] = (sl, ctrl.copy())

    # --- highly dynamic impact ----------------------------------------------
    im = _set_base_z(m, stance, _base_z_for_gap(m, stance, -0.002))
    im.qvel[:] = 0.0
    im.qvel[2] = -3.0
    fx["impact"] = (im, ctrl.copy())

    # diagnostics
    from tangent import tangent_perturb, basis
    nx = 2 * m.nv + m.na
    info = {}
    for name, (x, u) in fx.items():
        sig = contact_probe(m, d, x, u)
        x.into_data(d); d.ctrl[:] = u; mujoco.mj_forward(m, d)
        # how many +-h state perturbations change the contact set? (h = 1e-4)
        nswitch = 0
        for i in range(nx):
            for s in (+1, -1):
                xp = tangent_perturb(m, x, basis(nx, i, s * 1e-4))
                if contact_probe(m, d, xp, u)["pairs"] != sig["pairs"]:
                    nswitch += 1
        info[name] = dict(
            ncon=sig["ncon"], nefc=sig["nefc"],
            min_gap=float(_min_foot_dist(m, d)),
            qvel_norm=float(np.linalg.norm(x.qvel)),
            qacc_norm=float(np.linalg.norm(d.qacc)),
            nswitch=nswitch, nperturb=2 * nx,
            pairs=sig["pairs"],
        )
    return m, fx, info


CATEGORY_ORDER = ["flight", "stance", "loaded", "onset", "release", "sliding", "impact"]


if __name__ == "__main__":
    import os
    m, fx, info = build_fixtures()
    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for name, (x, u) in fx.items():
        out[f"{name}__qpos"] = x.qpos
        out[f"{name}__qvel"] = x.qvel
        out[f"{name}__act"] = x.act
        out[f"{name}__ctrl"] = u
    np.savez(os.path.join(RESULTS, "fixtures.npz"), **out)

    hdr = f"{'fixture':10s} {'ncon':>5s} {'nefc':>5s} {'min_gap':>11s} {'|qvel|':>8s} {'|qacc|':>10s} {'switch':>10s}"
    print(hdr); print("-"*len(hdr))
    for name in CATEGORY_ORDER:
        i = info[name]
        sw = f"{i['nswitch']}/{i['nperturb']}"
        print(f"{name:10s} {i['ncon']:5d} {i['nefc']:5d} {i['min_gap']:11.2e} "
              f"{i['qvel_norm']:8.3f} {i['qacc_norm']:10.2f} {sw:>10s}")
