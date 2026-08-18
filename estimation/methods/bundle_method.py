##
#
# Bundle Method
#
##

"""Derivative-free affine approximation, from Tracy et al. (2025), Sec. III.

The trajectory bundle method replaces the first-order Taylor series at the
heart of sequential convex programming with a *linear interpolation over
sampled points*.  Both produce an affine model of a nonlinear function near a
point; only the first needs a derivative.  This module implements the two
approximations and the gradient each one implies.

===============================  =============  ==================  =========
class                            equation       needs grad f?       evals
===============================  =============  ==================  =========
:class:`TaylorApproximation`     Eq. (1)        yes                 1
:class:`BundleApproximation`     Eqs. (5)-(8)   no                  ``m``
:class:`BundleGradient`          Eqs. (5)-(8)   no                  ``m``
===============================  =============  ==================  =========

The interpolant is built from ``m`` samples stacked into ``W_y`` (inputs,
Eq. 5) and ``W_p`` (outputs, Eq. 6).  The paper does not evaluate it at a
point directly; it reparametrises the problem in terms of an interpolation
weight ``alpha`` on the standard simplex, Eq. (3), and lets the convex solver
choose ``alpha``::

    y = W_y alpha            linear interpolation of inputs,   Eq. (7)
    p_hat = W_p alpha        linear interpolation of outputs,  Eq. (8)

so the surrogate is ``p(W_y alpha) ~= W_p alpha``, which is linear in
``alpha``.  To plot the approximation as a function of ``y`` -- which is what
the paper's own Fig. 2 does -- an ``alpha`` has to be picked for each query
point, and for ``m > n + 1`` the constraint ``W_y alpha = y`` does not pin one
down.  :class:`BundleApproximation` resolves that explicitly; see its
docstring.

Two facts are worth stating up front, because they connect this file to
``rand_smoothing.py``:

* **Inside the sample hull, ``m = n + 1`` makes the two readings agree.**  A
  simplex of samples admits exactly one ``alpha`` per query point, its
  barycentric coordinates, and ``W_p alpha`` is then the unique affine
  function through the ``n + 1`` sample values -- the same model the
  least-squares fit returns.  For larger ``m`` the interpolant is piecewise
  affine in ``y`` while the fitted model stays affine, and they differ.

  *Outside* the hull they part company for a different reason.  Barycentric
  coordinates would need a negative component there, which ``alpha >= 0``
  forbids, so the interpolant saturates on the boundary of the hull instead
  of extrapolating.  That is not a defect: it is the implicit trust region the
  paper points to in Sec. III-B, the reason a bundle step cannot run away from
  its samples the way a Taylor step can.

* **With coordinate sampling the bundle gradient is central differences.**
  Eqs. (26)-(27) place samples at ``z``, ``z +- mu e_i``.  Fitting an affine
  model to those by least squares recovers exactly
  ``(p(z + mu e_i) - p(z - mu e_i)) / (2 mu)`` per coordinate, which is
  :class:`~methods.rand_smoothing.CentralDifference`.  The bundle formulation
  is the more general object; that estimator is one corner of it.

What is *not* here is Sec. IV, the trajectory optimisation itself.  Eqs. (16)
and (22) need a dynamics model, knot points, slack variables and a conic
solver, none of which this repository has.
"""

# directory imports
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("GRAD_ROOT_DIR") or os.path.dirname(HERE)

# A stale override -- the tree moved but the variable did not -- would shadow
# the right path silently, so check it points at this package before trusting
# it and otherwise locate the package from __file__, which cannot go stale.
if not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(HERE)

sys.path.append(ROOT)

# standard imports
from abc import ABC, abstractmethod
import autograd.numpy as np
from numpy.random import default_rng
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

# custom imports
from src.function import FunctionAD
from src.function_examples import *
from methods.rand_smoothing import GradientEstimator, CentralDifference
from utils.plotting import (plot_landscape, plot_gradient_field, plot_surface,
                            sample_grid)


###########################################################
# SAMPLING -- Eq. (26), (27)
###########################################################

def coordinate_samples(z, delta):
    """The paper's deterministic coordinate perturbation, Eqs. (26)-(27).

    Returns ``2 n + 1`` points: ``z + delta_i e_i`` and ``z - delta_i e_i``
    for each coordinate, followed by ``z`` itself.  ``delta`` is the trust
    region, either a scalar or one width per coordinate.

    The paper reports that performance was largely independent of the sampling
    distribution and uses this scheme throughout, so it is the default here.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = z.size
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (n,))

    steps = np.diag(delta)                                 # (n, n)
    return np.vstack([z + steps, z - steps, z])            # (2n + 1, n)


def gaussian_samples(z, delta, num_samples, rng):
    """A Gaussian alternative to Eqs. (26)-(27), with ``z`` itself appended.

    Sec. IV-D notes that uniform and Gaussian sampling worked about as well as
    the coordinate scheme; this is here so that claim can be checked.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    draws = z + delta * rng.standard_normal((num_samples, z.size))
    return np.vstack([draws, z])                           # (K + 1, n)


###########################################################
# AFFINE MODEL FITTING
###########################################################

def affine_fit(Y, p, base):
    """Least-squares affine model of ``p`` over the samples ``Y``.

    ``Y`` is ``(m, n)`` and ``p`` is ``(m,)``.  Returns ``(d, C)`` for the
    model ``p(y) ~= d + C @ (y - base)``, so ``C`` is the gradient of the
    model and ``d`` its value at ``base``.  Expanding about ``base`` rather
    than the origin keeps the design matrix well conditioned when the samples
    sit far from it.

    With ``m = n + 1`` samples in general position the fit is exact and this
    is the unique affine interpolant through them.
    """
    A = np.column_stack([np.ones(len(Y)), np.asarray(Y) - base])
    coefficients, *_ = np.linalg.lstsq(A, np.asarray(p), rcond=None)
    return coefficients[0], coefficients[1:]


def simplex_weights(Wy, y, penalty=1e2):
    """``alpha`` on the standard simplex, Eq. (3), with ``W_y alpha ~= y``.

    Solves ``min ||W_y alpha - y||^2`` over ``alpha >= 0, sum(alpha) = 1``.
    The non-negativity is handled directly by the bounded least-squares
    solver; the sum constraint is appended as a weighted row of ones, scaled
    against the magnitude of the samples so that it binds regardless of the
    units ``y`` is measured in.  Both come out satisfied to machine precision,
    since a point inside the hull admits an exact representation.

    With ``m > n + 1`` the constraints leave a whole face of valid weights,
    and Eq. (8) returns a different value for each, so a tie-break is needed.
    The paper never faces this: there ``alpha`` is a decision variable the
    solver fixes by minimising cost.  Drawing the interpolant as a function of
    ``y``, or measuring its error, does have to choose.  BVLS resolves free
    variables through a least-squares solve, so what comes back is the
    minimum-norm ``alpha`` among the optimal ones -- the neutral choice, since
    it spreads weight across every sample able to represent ``y`` instead of
    favouring a subset.

    Points outside the convex hull of the samples cannot be represented at
    all.  There the residual is nonzero and this returns the ``alpha`` whose
    combination lands closest, which is the projection onto the hull.
    """
    Wy = np.asarray(Wy, dtype=float)
    m = Wy.shape[1]

    scale = penalty * max(1.0, float(np.abs(Wy).max()))
    A = np.vstack([Wy, scale * np.ones((1, m))])
    b = np.concatenate([np.asarray(y, dtype=float), [scale]])

    return lsq_linear(A, b, bounds=(0.0, np.inf), method="bvls").x


###########################################################
# AFFINE APPROXIMATIONS
###########################################################

class AffineApproximation(FunctionAD, ABC):
    """An affine model of ``f`` built around a point.

    Subclasses of :class:`FunctionAD`, so they can be handed to any helper in
    ``utils/plotting.py`` in place of the function they approximate, and the
    two can be drawn side by side.  :meth:`grad` is overridden rather than
    taken from autodiff -- the whole point of Sec. III is which models need a
    derivative to *build*, so differentiating the result would confuse the
    question.
    """

    @abstractmethod
    def gradient(self):
        """The model's gradient, constant over the domain by affineness."""

    def grad(self, x, order=1):
        if order != 1:
            raise ValueError("an affine model has no second derivative, "
                             f"got order={order!r}")
        return np.broadcast_to(self.gradient(), np.shape(x))


class TaylorApproximation(AffineApproximation):
    """First-order Taylor series about ``base``, Eq. (1).

    ``p(y) ~= p(base) + dp/dy (y - base)``.  Exact at ``base`` and degrading
    with distance from it.  This is the approximation SCP normally uses, and
    the one the bundle method exists to replace: it needs the derivative of
    ``f``, which is supplied here by autodiff.
    """

    def __init__(self, f, base):
        self.f = f
        self.base = np.atleast_1d(np.asarray(base, dtype=float))
        self.value = float(np.squeeze(f(self.base)))
        self.jacobian = np.atleast_1d(f.grad(self.base, 1))

    def evaluate(self, x):
        return self.value + self.jacobian @ (np.atleast_1d(x) - self.base)

    def gradient(self):
        return self.jacobian


class BundleApproximation(AffineApproximation):
    """Linear interpolation over sampled points, Eqs. (5)-(8).

    Builds ``W_y`` and ``W_p`` from ``samples`` and evaluates the interpolant
    at a query point ``y`` by solving Eq. (7) for ``alpha`` and returning
    Eq. (8), ``W_p alpha``.  See :func:`simplex_weights` for how ``alpha`` is
    pinned down when the samples over-determine it.

    No derivative of ``f`` is used anywhere: the model is built from ``m``
    function values alone.

    The interpolant does not extrapolate.  Beyond the convex hull of the
    samples no admissible ``alpha`` reproduces ``y``, and the value flattens
    onto the hull boundary -- the implicit trust region of Sec. III-B.  Query
    points are expected to lie within the hull, which for the paper's own
    Fig. 2 they do, the four corners spanning the whole domain.

    :meth:`gradient` returns something subtly different from
    :meth:`evaluate` -- the least-squares affine model through the same
    samples, since the interpolant itself is only piecewise affine in ``y``
    once ``m > n + 1`` and has no single gradient.  Inside the hull the two
    agree when ``m = n + 1``.
    """

    def __init__(self, f, samples):
        self.f = f
        self.Wy = np.asarray(samples, dtype=float).T       # (n, m), Eq. (5)
        self.Wp = np.array([float(np.squeeze(f(s))) for s in samples])  # Eq. (6)

        # The fitted affine model, used for the gradient and as the reference
        # the interpolant collapses onto when m = n + 1.
        base = self.Wy.mean(axis=1)
        self.offset, self.jacobian = affine_fit(self.Wy.T, self.Wp, base)
        self.base = base

    def weights(self, y):
        """The interpolation weights ``alpha`` at ``y``, Eqs. (3) and (7)."""
        return simplex_weights(self.Wy, np.atleast_1d(y))

    def evaluate(self, x):
        return self.Wp @ self.weights(x)                   # Eq. (8)

    def gradient(self):
        return self.jacobian


###########################################################
# THE GRADIENT THE INTERPOLANT IMPLIES
###########################################################

class BundleGradient(GradientEstimator):
    """Gradient of the linear interpolant, Eqs. (5)-(8).

    Samples around ``x``, fits an affine model to the values by least squares,
    and returns its linear coefficient.  Written as a
    :class:`~methods.rand_smoothing.GradientEstimator` so it drops into the
    same comparisons as the seven estimators in that file.

    ``sampling`` selects the scheme: ``"coordinate"`` is the paper's
    Eqs. (26)-(27), costing ``2 n + 1`` evaluations and reproducing central
    differences exactly; ``"gaussian"`` draws ``num_samples`` perturbations
    instead, which decouples the cost from the dimension and turns this into
    a regression estimate of the gradient.

    Unlike the single-direction estimators in ``rand_smoothing.py``, this uses
    every sample jointly rather than averaging independent one-sample
    estimates, so ``mu`` here is a trust-region width rather than a step size.
    """

    MU = 0.1

    def __init__(self, mu=None, num_samples=1, sigma=None, seed=None,
                 common_noise=False, sampling="coordinate"):
        super().__init__(mu=mu, num_samples=num_samples, sigma=sigma,
                         seed=seed, common_noise=common_noise)
        if sampling not in ("coordinate", "gaussian"):
            raise ValueError("sampling must be 'coordinate' or 'gaussian', "
                             f"got {sampling!r}")
        self.sampling = sampling

    def estimate(self, f, x):
        z = self._as_vector(x)

        if self.sampling == "coordinate":
            Y = coordinate_samples(z, self.mu)
        else:
            # Cached as offsets rather than points so that common_noise reuses
            # one perturbation pattern across every x, as it does elsewhere.
            Y = z + self._draw(z.size, self._gaussian_offsets)

        p = np.array([self._value(f, y) for y in Y])
        _, C = affine_fit(Y, p, z)
        return np.reshape(C, np.shape(x))

    def _gaussian_offsets(self, n):
        """``K`` Gaussian perturbations, plus a zero row for the base point."""
        return gaussian_samples(np.zeros(n), self.mu, self.num_samples,
                                self.rng)


###########################################################
# TESTING
###########################################################

if __name__ == "__main__":

    np.set_printoptions(precision=5, suppress=True)

    ###########################################################
    # THE TWO CLAIMS IN THE MODULE DOCSTRING
    ###########################################################

    f = SinExp()
    x = np.array([0.5, -0.25])

    # 1. With m = n + 1 the interpolant reproduces the samples exactly, and
    #    inside their hull it coincides with the fitted affine model.
    simplex = np.array([[0.4, -0.3], [0.7, -0.35], [0.55, -0.05]])
    bundle = BundleApproximation(f, simplex)
    interp = np.array([bundle(s) for s in simplex])
    print("m = n + 1: interpolant reproduces the samples")
    print(f"  f at samples      {bundle.Wp}")
    print(f"  interpolant       {interp}")
    print(f"  max difference    {np.abs(interp - bundle.Wp).max():.2e}")

    # Query points drawn as convex combinations, so they land inside the hull.
    rng = default_rng(0)
    weights = rng.dirichlet(np.ones(3), size=200)
    inside = weights @ simplex
    values = np.array([bundle(q) for q in inside])
    fitted = bundle.offset + (inside - bundle.base) @ bundle.jacobian
    print(f"  inside the hull, interpolant vs fitted affine model: "
          f"{np.abs(values - fitted).max():.2e}")

    # Outside it, the simplex constraint clamps instead of extrapolating.
    far = np.array([5.0, 5.0])
    print(f"  outside the hull, interpolant {bundle(far):+.4f} vs fitted "
          f"{bundle.offset + (far - bundle.base) @ bundle.jacobian:+.4f} "
          f"-- the implicit trust region\n")

    # 2. Coordinate sampling makes the bundle gradient central differences.
    bundle_g = BundleGradient(mu=1e-2).estimate(f, x)
    central = CentralDifference(mu=1e-2).estimate(f, x)
    print("coordinate sampling reduces to central differences")
    print(f"  bundle gradient   {bundle_g}")
    print(f"  central diff (12) {central}")
    print(f"  max difference    {np.abs(bundle_g - central).max():.2e}\n")

    ###########################################################
    # FIGURE 2 -- TAYLOR ERROR AGAINST INTERPOLATION ERROR
    ###########################################################

    # The paper's own comparison: f(x, y) = sin(x) e^y on [0, 2]^2, with the
    # Taylor series taken about (1, 1) and the interpolation built from the
    # four corners.  Both models are affine and cost about the same to form;
    # the point of the figure is that their errors are shaped differently.
    # The Taylor model is exact at one interior point and decays outwards; the
    # interpolant is exact at the four corners and worst between them.
    fig2_f = SinExp()
    fig2_bounds = (0.0, 2.0)
    corners = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])

    taylor = TaylorApproximation(fig2_f, np.array([1.0, 1.0]))
    interp = BundleApproximation(fig2_f, corners)

    print("Fig. 2: affine models of sin(x) e^y on [0, 2]^2")
    print(f"  Taylor about (1, 1), value {taylor.value:.4f}, "
          f"gradient {taylor.gradient()}")
    print(f"  interpolation over {len(corners)} corners, "
          f"fitted gradient {interp.gradient()}")

    GRID = 200
    XX, YY, Z = sample_grid(fig2_f, fig2_bounds, GRID)
    _, _, Z_taylor = sample_grid(taylor, fig2_bounds, GRID)
    _, _, Z_interp = sample_grid(interp, fig2_bounds, GRID)

    err_taylor = np.abs(Z_taylor - Z)
    err_interp = np.abs(Z_interp - Z)
    vmax = max(err_taylor.max(), err_interp.max())

    print(f"  Taylor        mean {err_taylor.mean():.4f}  "
          f"max {err_taylor.max():.4f}")
    print(f"  interpolation mean {err_interp.mean():.4f}  "
          f"max {err_interp.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")
    for ax, err, title in ((axes[0], err_taylor, "Taylor-series error"),
                           (axes[1], err_interp, "linear interpolation error")):
        mesh = ax.pcolormesh(XX, YY, err, cmap="coolwarm", vmin=0.0,
                             vmax=vmax, shading="auto")
        ax.contour(XX, YY, err, levels=12, colors="k", linewidths=0.3,
                   alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")
    axes[0].plot(*taylor.base, "k+", markersize=9)
    axes[1].plot(corners[:, 0], corners[:, 1], "k+", markersize=9,
                 linestyle="none")
    fig.colorbar(mesh, ax=axes, label="error")

    ###########################################################
    # THE MODELS THEMSELVES, AGAINST THE FUNCTION
    ###########################################################

    _, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    for ax, g, title in ((axes[0], fig2_f, r"$f = \sin(x)e^y$"),
                         (axes[1], taylor, "Taylor about (1, 1)"),
                         (axes[2], interp, "interpolation, 4 corners")):
        plot_landscape(g, fig2_bounds, ax=ax)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()

    ###########################################################
    # THE BUNDLE GRADIENT ON A NON-SMOOTH FUNCTION
    ###########################################################

    # The same plateau probe used in rand_smoothing.py.  Coordinate sampling
    # inherits the blindness of central differences when the trust region is
    # small, and sees the surrounding slope once it is wide enough.
    nasty = Nasty()
    x0 = np.array([0.9, 0.9])
    print(f"\nNasty at x = {x0}, a point on the clamped plateau")
    print(f"  autograd gradient        {nasty.grad(x0, 1)}")
    for mu in (0.01, 0.1, 0.3):
        g = BundleGradient(mu=mu).estimate(nasty, x0)
        print(f"  bundle, mu = {mu:<5}       {g}")
    for mu in (0.1, 0.3):
        g = BundleGradient(mu=mu, num_samples=64, seed=0,
                           sampling="gaussian").estimate(nasty, x0)
        print(f"  bundle, gaussian mu={mu}  {g}")

    plt.show()
