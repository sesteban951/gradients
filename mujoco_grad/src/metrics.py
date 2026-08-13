"""Error metrics for Jacobian comparison (Section 5.5 / 6 of the plan)."""

import numpy as np

EPS_DEN = 1e-9


def rel_fro(X, Xref, eps_den=EPS_DEN):
    return float(np.linalg.norm(X - Xref) / max(np.linalg.norm(Xref), eps_den))


def col_errors(X, Xref, eps_den=EPS_DEN):
    """Per-column relative error and cosine similarity."""
    dn = np.linalg.norm(X - Xref, axis=0)
    rn = np.linalg.norm(Xref, axis=0)
    rel = dn / np.maximum(rn, eps_den)
    xn = np.linalg.norm(X, axis=0)
    denom = np.maximum(xn * rn, eps_den)
    cos = np.sum(X * Xref, axis=0) / denom
    meaningful = rn > 1e-6 * max(np.max(rn), eps_den)
    return rel, cos, meaningful


def summary(X, Xref, eps_den=EPS_DEN):
    rel, cos, meaningful = col_errors(X, Xref, eps_den)
    absdiff = np.abs(X - Xref)
    return dict(
        rel_fro=rel_fro(X, Xref, eps_den),
        max_abs=float(absdiff.max()),
        p50=float(np.percentile(absdiff, 50)),
        p90=float(np.percentile(absdiff, 90)),
        p99=float(np.percentile(absdiff, 99)),
        col_rel_max=float(rel.max()),
        col_rel_med=float(np.median(rel)),
        cos_min=float(cos[meaningful].min()) if meaningful.any() else float("nan"),
        cos_med=float(np.median(cos[meaningful])) if meaningful.any() else float("nan"),
        ref_fro=float(np.linalg.norm(Xref)),
    )


def split_summary(X, Xref, stable_mask, eps_den=EPS_DEN):
    """Frobenius error restricted to contact-stable vs contact-changing columns."""
    out = {}
    for label, mask in (("stable", stable_mask), ("changing", ~stable_mask)):
        if mask.any():
            out[label] = dict(
                n=int(mask.sum()),
                rel_fro=rel_fro(X[:, mask], Xref[:, mask], eps_den),
            )
        else:
            out[label] = dict(n=0, rel_fro=float("nan"))
    return out
