"""Render the epsilon-sweep JSON as readable tables."""

import json, sys, os
from config import RESULTS
from fixtures import CATEGORY_ORDER


def show(path):
    R = json.load(open(path))
    S = R["S"]
    print(f"=== epsilon sweep, S={S}, gold=CPU64@h={R['H_GOLD']:.0e}, "
          f"repeats={R['repeats']} ===\n")
    for name in CATEGORY_ORDER:
        if name not in R["results"]:
            continue
        blk = R["results"][name]
        fi = blk["fixture_info"]
        print(f"[{name}]  ncon={fi['ncon']}  nefc={fi['nefc']}  "
              f"min_gap={fi['min_gap']:.2e}  |qvel|={fi['qvel_norm']:.2f}")
        print(f"  {'h':>8s} {'E_same(A)':>10s} {'E_gold(A)':>10s} {'E_cpu(A)':>10s} "
              f"{'E_gold(B)':>10s} {'GPUnoise':>9s} {'cos_min':>8s} {'stable':>8s} "
              f"{'E_stable':>9s} {'E_chang':>9s}")
        for r in blk["rows"]:
            sp = r["split"]
            st = f"{r['n_stable']}/{r['n_cols']}"
            chg = sp["changing"]["rel_fro"]
            chg_s = "     --  " if chg != chg else f"{chg:9.2e}"
            print(f"  {r['h']:8.0e} {r['E_same_A']:10.2e} {r['E_gold_A']:10.2e} "
                  f"{r['E_cpu_A']:10.2e} {r['E_gold_B']:10.2e} {r['noise']:9.2e} "
                  f"{r['cos_min']:8.4f} {st:>8s} {sp['stable']['rel_fro']:9.2e} {chg_s}")
        best = min(blk["rows"], key=lambda r: r["E_gold_A"])
        print(f"  -> best h = {best['h']:.0e}  E_gold(A) = {best['E_gold_A']:.3e}  "
              f"cos_min = {best['cos_min']:.4f}\n")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS, "phase34_sweep_S1.json")
    show(p)
