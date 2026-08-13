import numpy as np, mujoco
from fixtures import build_fixtures
from cpu_fd import blackbox_fd, transition_fd_reference
from metrics import rel_fro
m, fx, _ = build_fixtures(); d = mujoco.MjData(m)
for name in ["stance","onset","impact"]:
    x,u = fx[name]
    for h in [1e-3,1e-5]:
        Ar,Br = transition_fd_reference(m,d,x,u,h,True)
        A0,B0,_ = blackbox_fd(m,d,x,u,h,track_contacts=False)
        A1,B1,cs = blackbox_fd(m,d,x,u,h,track_contacts=True)
        print(f"{name:8s} h={h:.0e}  vs_ref={rel_fro(A0,Ar):.2e}  track_vs_notrack={rel_fro(A1,A0):.2e}  stable_cols={int(cs.sum())}/{cs.size}")
