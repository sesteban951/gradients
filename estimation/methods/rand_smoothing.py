##
#
# Randomized Smoothing
#
##

"""Zero-order gradient estimates, from Jordana et al. (2025), Sec. II-B.

Each class here implements one of the paper's formulas: given a point ``x``
and the ability to *evaluate* a function, it returns a vector standing in for
the gradient there.  None of them touches :meth:`FunctionAD.grad`, which is
the point -- they are meant for functions whose derivative is unavailable,
unreliable, or (as with :class:`Nasty`) present but worthless.

The centrepiece is randomized smoothing, which replaces ``f`` with the
Gaussian-smoothed surrogate ``f_mu(x) = E[f(x + mu eps)]`` for
``eps ~ N(0, Sigma)``, Eq. (5).  Its gradient, Eq. (8), is an expectation of
evaluations of ``f`` alone, so it can be estimated without any derivative of
``f``.  The finite-difference and perturbation methods are here alongside it
as the baselines it is measured against.

===========================  ========  ==========================  =========
class                        equation  target                      evals
===========================  ========  ==========================  =========
:class:`ForwardDifference`   Eq. (2)   grad f(x)                   ``n + 1``
:class:`CentralDifference`   Sec. IIB  grad f(x)                   ``2 n``
:class:`RandomCoordinate`    Eq. (3)   grad f(x) / n  (see below)  ``K + 1``
:class:`SPSA`                Eq. (4)   grad f(x)                   ``2 K``
:class:`SmoothingVanilla`    Eq. (8)   grad f_mu(x)                ``K``
:class:`SmoothingForward`    Eq. (11)  grad f_mu(x)                ``K + 1``
:class:`SmoothingCentral`    Eq. (12)  grad f_mu(x)                ``2 K``
===========================  ========  ==========================  =========

The two groups estimate *different quantities*.  The first four approximate
the gradient of ``f`` itself, so their ``mu`` is a discretisation error to be
driven towards zero.  The last three approximate the gradient of the
surrogate, which is a genuinely different function -- one that stays
informative where ``f`` is flat or discontinuous.  There ``mu`` is the
smoothing width and the whole reason to bother, so it is left moderate.  The
two groups therefore carry different default ``mu``; see
:attr:`GradientEstimator.MU`.

Two properties worth keeping in mind when reading the results:

* **Eq. (3) is not an unbiased estimate of the gradient.**  Picking one
  coordinate uniformly gives ``E[g] = grad f(x) / n``, and the paper does not
  put the factor back -- coordinate descent folds it into the step size.  The
  formula is implemented as written; multiply by ``n`` to compare it against
  the others.  Every remaining estimator is unbiased for its stated target.

* **Eq. (8) has no baseline.**  Adding a constant to ``f`` changes the
  estimate even though it cannot change the gradient, so its variance grows
  with ``|f|``.  Eq. (11) subtracts ``f(x)`` and Eq. (12) uses
  ``f(x - mu eps)`` precisely to remove that dependence; both are translation
  invariant.
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
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt

# custom imports
from src.function_examples import *
from utils.plotting import (plot_landscape, plot_gradient_field, plot_surface,
                            plot_error_heatmaps)


###########################################################
# ESTIMATOR BASE CLASS
###########################################################

class GradientEstimator(ABC):
    """Base class for a zero-order gradient estimate.

    Hyperparameters are fixed at construction and the estimator is then
    applied to any function and point::

        est = SmoothingForward(mu=0.2, num_samples=500, seed=0)
        g = est.estimate(f, x)

    ``f`` is anything callable returning a scalar -- a :class:`FunctionAD`
    subclass, in practice.  ``x`` may be a vector of shape ``(n,)`` or a
    scalar; the estimate comes back shaped like ``x``.

    Parameters
    ----------
    mu
        Perturbation size.  Defaults to :attr:`MU`, which differs between the
        finite-difference and smoothing families (see the module docstring).
    num_samples
        ``K``, the number of random draws averaged into one estimate.  Ignored
        by the two deterministic estimators, which sweep all ``n`` coordinates.
    sigma
        Covariance ``Sigma`` of the Gaussian perturbation, shape ``(n, n)``.
        ``None`` means the identity.  Used only by the smoothing family; the
        other estimators define their own perturbation shape.
    seed
        Seed for this estimator's private generator, so runs are repeatable
        and two estimators never share a stream.
    common_noise
        When set, one draw of the noise is made and reused at *every* point.
        Independent draws at neighbouring points make a gradient field over a
        grid look like pure noise; a shared draw makes it coherent, at the
        cost of correlating the errors.  Off by default.
    """

    # Default perturbation size.  Small for the finite-difference family,
    # where mu is an error term; moderate for the smoothing family, where it
    # sets the width of the surrogate.
    MU = 1e-3

    def __init__(self, mu=None, num_samples=1, sigma=None, seed=None,
                 common_noise=False):
        self.mu = self.MU if mu is None else float(mu)
        self.num_samples = int(num_samples)
        self.sigma = None if sigma is None else np.asarray(sigma, dtype=float)
        self.rng = default_rng(seed)
        self.common_noise = common_noise

        # Caches, both filled lazily: the fixed noise draw when common_noise
        # is set (keyed by dimension), and the Cholesky factor of sigma.
        self._noise = {}
        self._chol = None

    @abstractmethod
    def estimate(self, f, x):
        """Gradient estimate of ``f`` at ``x``, shaped like ``x``."""

    def estimate_batch(self, f, X):
        """Estimate at many points.  ``X`` is a stack of inputs along axis 0.

        Mirrors :meth:`FunctionAD.grad_batch`, so the same array of points can
        be fed to either and the results compared directly.
        """
        return np.array([self.estimate(f, x) for x in X])

    ###########################################################
    # HELPERS
    ###########################################################

    @staticmethod
    def _as_vector(x):
        # Estimators work in one dimension internally so that a scalar input
        # and a length-1 vector follow the same code path.
        return np.atleast_1d(np.asarray(x, dtype=float))

    @staticmethod
    def _value(f, x):
        # Collapse whatever f returns -- python float, 0-d array, or a (1,)
        # array from a scalar-input function -- to a plain float.
        return float(np.squeeze(f(x)))

    def _draw(self, n, sampler):
        """Perturbations for one estimate, or the shared draw when
        ``common_noise`` is set.  ``sampler`` takes the dimension ``n``.
        """
        if not self.common_noise:
            return sampler(n)
        if n not in self._noise:
            self._noise[n] = sampler(n)
        return self._noise[n]

    def _cholesky(self, n):
        """Lower-triangular ``L`` with ``Sigma = L L'``, built once."""
        if self._chol is None:
            sigma = self.sigma
            if sigma is None or sigma.shape != (n, n):
                shape = "None" if sigma is None else sigma.shape
                raise ValueError(
                    f"sigma has shape {shape}, expected ({n}, {n})")
            # Raises if sigma is not symmetric positive definite, which is the
            # right moment to find out.
            self._chol = np.linalg.cholesky(sigma)
        return self._chol


###########################################################
# FINITE DIFFERENCES -- Eq. (2)
###########################################################

class ForwardDifference(GradientEstimator):
    """Forward finite differences, Eq. (2).

    Sweeps every canonical direction ``e_j``, so it costs ``n + 1``
    evaluations and returns the same answer every call.  This is the reference
    the sampled estimators are trying to match with fewer evaluations.
    """

    def estimate(self, f, x):
        z = self._as_vector(x)
        fz = self._value(f, z)
        g = np.array([(self._value(f, z + self.mu * e) - fz) / self.mu
                      for e in np.eye(z.size)])
        return np.reshape(g, np.shape(x))


class CentralDifference(GradientEstimator):
    """Central finite differences, the alternative noted below Eq. (2).

    Costs ``2 n`` evaluations rather than ``n + 1``, and in exchange the
    truncation error is ``O(mu^2)`` instead of ``O(mu)``.
    """

    def estimate(self, f, x):
        z = self._as_vector(x)
        g = np.array([(self._value(f, z + self.mu * e)
                       - self._value(f, z - self.mu * e)) / (2.0 * self.mu)
                      for e in np.eye(z.size)])
        return np.reshape(g, np.shape(x))


###########################################################
# RANDOM COORDINATE DESCENT -- Eq. (3)
###########################################################

class RandomCoordinate(GradientEstimator):
    """Random coordinate descent, Eq. (3).

    One canonical direction ``e_j`` drawn uniformly from ``{1, ..., n}``,
    differenced forward, averaged over ``K`` draws.  ``f(x)`` is evaluated
    once and shared, so the cost is ``K + 1``.

    As written this is *not* an unbiased estimate of the gradient: only one of
    ``n`` coordinates is touched per draw, so ``E[g] = grad f(x) / n``.  See
    the module docstring.
    """

    def estimate(self, f, x):
        z = self._as_vector(x)
        n = z.size
        fz = self._value(f, z)

        indices = self._draw(n, lambda d: self.rng.integers(0, d,
                                                            self.num_samples))
        g = np.zeros(n)
        for j in indices:
            e = np.zeros(n)
            e[j] = 1.0
            # Accumulate into coordinate j; a coordinate drawn twice gets two
            # contributions, which is what averaging over K draws means.
            g[j] += (self._value(f, z + self.mu * e) - fz) / self.mu

        return np.reshape(g / self.num_samples, np.shape(x))


###########################################################
# SPSA -- Eq. (4)
###########################################################

class SPSA(GradientEstimator):
    """Simultaneous perturbation stochastic approximation, Eq. (4).

    Perturbs *every* coordinate at once along a vector of independent ``+-1``
    Bernoulli entries, differences centrally, and projects back onto that same
    vector.  Two evaluations per draw regardless of ``n``, which is the whole
    appeal, and unbiased for the gradient up to ``O(mu^2)``.

    Spall divides componentwise by ``Delta_i`` rather than multiplying by
    ``Delta``; for ``+-1`` entries the two agree, since ``1 / Delta_i =
    Delta_i``.
    """

    def estimate(self, f, x):
        z = self._as_vector(x)
        deltas = self._draw(z.size, self._bernoulli)

        g = np.zeros(z.size)
        for d in deltas:
            diff = (self._value(f, z + self.mu * d)
                    - self._value(f, z - self.mu * d))
            g = g + (diff / (2.0 * self.mu)) * d

        return np.reshape(g / self.num_samples, np.shape(x))

    def _bernoulli(self, n):
        return self.rng.integers(0, 2, (self.num_samples, n)) * 2.0 - 1.0


###########################################################
# GAUSSIAN SMOOTHING -- Eq. (8), (11), (12)
###########################################################

class GaussianSmoothing(GradientEstimator, ABC):
    """Shared machinery for the three randomized-smoothing estimators.

    All three have the form ``g = mean_k w_k * Sigma^-1 eps_k`` with
    ``eps ~ N(0, Sigma)``, and differ only in the scalar weight ``w_k``.
    Subclasses supply that weight in :meth:`_weights`; it is exactly the
    numerator of the corresponding equation.

    Rather than form ``Sigma^-1``, the draw is made as ``eps = L z`` with
    ``z`` standard normal and ``Sigma = L L'``.  Then ``Sigma^-1 eps =
    L^-T L^-1 L z = L^-T z``, so one triangular solve on the *same* ``z``
    produces the second half of the pair.
    """

    # The smoothing width, not an error term: it is what makes f_mu smooth
    # where f is not, so it is deliberately far from zero.
    MU = 0.1

    @abstractmethod
    def _weights(self, f, z, eps):
        """Scalar coefficient for each of the ``K`` perturbations, shape
        ``(K,)``."""
        raise NotImplementedError

    def estimate(self, f, x):
        z = self._as_vector(x)
        eps, sigma_inv_eps = self._noise_pair(z.size)
        w = self._weights(f, z, eps)                       # (K,)
        g = w @ sigma_inv_eps / self.num_samples           # (n,)
        return np.reshape(g, np.shape(x))

    def _noise_pair(self, n):
        """``K`` draws of ``eps ~ N(0, Sigma)`` paired with ``Sigma^-1 eps``,
        each of shape ``(K, n)``."""
        Z = self._draw(n, lambda d: self.rng.standard_normal(
            (self.num_samples, d)))
        if self.sigma is None:
            return Z, Z                                    # Sigma = I
        L = self._cholesky(n)
        return Z @ L.T, solve_triangular(L.T, Z.T, lower=False).T


class SmoothingVanilla(GaussianSmoothing):
    """Randomized smoothing without a baseline, Eq. (8).

    ``w_k = f(x + mu eps_k) / mu``.  One evaluation per draw, the cheapest of
    the three, but the estimate moves when a constant is added to ``f`` -- so
    its variance scales with the *value* of ``f``, not just its variation.
    """

    def _weights(self, f, z, eps):
        return np.array([self._value(f, z + self.mu * e)
                         for e in eps]) / self.mu


class SmoothingForward(GaussianSmoothing):
    """Randomized smoothing with a forward difference, Eq. (11).

    ``w_k = (f(x + mu eps_k) - f(x)) / mu``.  Subtracting ``f(x)`` makes the
    estimate invariant to constant shifts of ``f`` and cuts the variance;
    ``f(x)`` is evaluated once and shared, so the cost is ``K + 1``.
    """

    def _weights(self, f, z, eps):
        fz = self._value(f, z)
        return np.array([self._value(f, z + self.mu * e) - fz
                         for e in eps]) / self.mu


class SmoothingCentral(GaussianSmoothing):
    """Randomized smoothing with a central difference, Eq. (12).

    ``w_k = (f(x + mu eps_k) - f(x - mu eps_k)) / (2 mu)``.  Also translation
    invariant, and symmetric about ``x``, at ``2 K`` evaluations.
    """

    def _weights(self, f, z, eps):
        return np.array([self._value(f, z + self.mu * e)
                         - self._value(f, z - self.mu * e)
                         for e in eps]) / (2.0 * self.mu)


###########################################################
# THE SURROGATE ITSELF -- Eq. (5)
###########################################################

class SmoothedFunction(FunctionAD):
    """The Gaussian-smoothed surrogate ``f_mu(x) = E[f(x + mu eps)]``, Eq. (5).

    The estimators above approximate the *gradient* of this surrogate without
    ever building it.  This class builds the surrogate itself, by Monte Carlo
    over the same expectation, so that it can be looked at::

        smooth = SmoothedFunction(Nasty(), mu=0.2, num_samples=100)
        plot_landscape(smooth, (-1.0, 1.0))

    It subclasses :class:`FunctionAD` purely to inherit the calling
    convention, so every helper in ``utils/plotting.py`` accepts it in place
    of the original.  :meth:`grad` is overridden: autodiff through the Monte
    Carlo sum would differentiate ``f`` itself, which is the one thing the
    whole exercise is trying to avoid.  The derivative comes from one of the
    estimators above instead.

    The perturbations are drawn **once** and reused at every ``x``.  That is
    not an optimisation: with a fresh draw per point the result would not be a
    function at all, just noise around ``f_mu``, and its contours would be
    speckle.  Holding the draw fixed makes this a genuine deterministic
    function -- the average of ``K`` shifted copies of ``f`` -- which is smooth
    for the same reason ``f_mu`` is, and converges to it as ``K`` grows.

    ``num_samples`` is the accuracy knob and the cost: evaluating this on an
    ``m``-point grid costs ``m * K`` evaluations of ``f``, so plotting it is
    a few orders of magnitude dearer than plotting ``f``.
    """

    def __init__(self, f, mu=0.1, num_samples=100, sigma=None, seed=0,
                 estimator=SmoothingCentral, batched=None):
        self.f = f
        self.mu = float(mu)
        self.num_samples = int(num_samples)

        # The gradient estimator doubles as the noise source, so the surface
        # and its gradient field are built from the same perturbations.
        self._est = estimator(mu=mu, num_samples=num_samples, sigma=sigma,
                              seed=seed, common_noise=True)
        if not isinstance(self._est, GaussianSmoothing):
            raise TypeError("estimator must be a GaussianSmoothing subclass, "
                            f"got {estimator.__name__}")

        # None means decide on first use; see _detect_batched.
        self._batched = batched

    def evaluate(self, x):
        eps, _ = self._est._noise_pair(np.size(x))
        points = x + self.mu * eps                     # (K, n)

        if self._batched is None:
            self._batched = self._detect_batched(points)
        if self._batched:
            return np.mean(self.f.evaluate(points.T))
        return np.mean([self._est._value(self.f, p) for p in points])

    def _detect_batched(self, points):
        """Whether ``f.evaluate`` handles a whole block of points at once.

        Functions written elementwise -- ``Nasty``, ``Sin``, ``SinExp`` --
        unpack their argument as ``x[0], x[1]`` and then use only
        broadcasting operations, so handing them the *transpose* of a stack of
        points evaluates every point in one call.  On ``Nasty`` that is around
        seventy times faster than looping, which is the difference between a
        surrogate plot taking a second and taking a minute.  Functions built
        on matrix products, such as ``Quadratic``, do not work this way.

        Rather than guess from the source, try it once and check the result
        against the loop it would replace.  A wrong shape, a raised exception,
        or a disagreement in the values all fall back to the slow path.
        """
        probe = points[:min(4, len(points))]
        try:
            batch = np.asarray(self.f.evaluate(probe.T))
        except Exception:
            return False
        if batch.shape != (len(probe),):
            return False
        loop = np.array([self._est._value(self.f, p) for p in probe])
        return bool(np.allclose(batch, loop))

    def grad(self, x, order=1):
        """Gradient of the surrogate, from the estimator it was built with.

        Only ``order=1`` exists: there is no second-order formula among the
        estimators, and autodiff is deliberately not an option here.
        """
        if order != 1:
            raise ValueError("only order=1 is available for a smoothed "
                             f"surrogate, got {order!r}")
        return self._est.estimate(self.f, x)


###########################################################
# REGISTRY
###########################################################

# Every estimator, in the order they appear in the paper.  Handy for sweeping
# all seven over the same function and point.
ESTIMATORS = {
    "forward diff  (2)": ForwardDifference,
    "central diff     ": CentralDifference,
    "rand coord    (3)": RandomCoordinate,
    "SPSA          (4)": SPSA,
    "smoothing     (8)": SmoothingVanilla,
    "smoothing    (11)": SmoothingForward,
    "smoothing    (12)": SmoothingCentral,
}

DETERMINISTIC = (ForwardDifference, CentralDifference)


###########################################################
# TESTING
###########################################################

if __name__ == "__main__":

    np.set_printoptions(precision=5, suppress=True)

    ###########################################################
    # CONFIGURATION -- everything below is computed on this f
    ###########################################################

    f = Nasty()
    # f = SinExp()          # smooth and harmonic, so f_mu == f exactly
    # f = Sin()             # smooth, strong curvature
    # f = Quadratic()

    bounds = (-1.0, 1.0)
    x0 = np.array([0.9, 0.9])          # probe point for the pointwise checks
    MU, K, REPEATS, GRID = 0.001, 50, 4, 40

    name_f = type(f).__name__

    # Every estimator gets the same mu and the same K, so comparisons are
    # between estimators rather than between smoothing widths or budgets.
    estimators = {name: cls(mu=MU, num_samples=K, seed=i)
                  for i, (name, cls) in enumerate(ESTIMATORS.items())}

    # -- how the reference is to be read ----------------------------------
    #
    # Autodiff supplies the "truth" everywhere below, which is only ground
    # truth where f is differentiable and only for estimators targeting
    # grad f.  On Nasty it is nothing of the sort: the clamp flattens wide
    # bands, so autograd reports zero across them and the two step
    # discontinuities are invisible to it.  There the error columns and the
    # heatmaps measure *disagreement with autograd*, not accuracy -- and the
    # smoothing family is charged most heavily exactly where it is the only
    # thing still seeing a descent direction.
    differentiable = not isinstance(f, Nasty)
    if not differentiable:
        print(f"NOTE: {name_f} is non-smooth.  The autograd reference is not\n"
              f"      ground truth here; read 'err' as disagreement with it.\n")

    ###########################################################
    # POINTWISE ESTIMATES
    ###########################################################

    # Every estimator should approach the autograd gradient as K grows, with
    # two documented exceptions: Eq. (3), which converges to grad f / n, and
    # the smoothing family, which converges to grad f_mu rather than grad f.
    #
    # Each cell below is a single seed, so the errors are themselves random
    # and neighbouring rows are not reliably ordered.  Eq. (8) can beat
    # Eq. (11) here by luck; averaged over seeds it does not, and the gap
    # widens without bound once a constant is added to f.
    truth = f.grad(x0, 1)

    print(f"{name_f} at x = {x0}")
    print(f"  autograd gradient   {truth}\n")
    print(f"  {'estimator':<18} {'K':>6}   {'estimate':<20} rel. err")

    scale = np.linalg.norm(truth)
    scale = scale if scale > 0.0 else 1.0     # Nasty's plateaus give ||g|| = 0

    for name, cls in ESTIMATORS.items():
        for n_samples in (1, 100, 10000):
            g = cls(mu=MU, num_samples=n_samples, seed=0).estimate(f, x0)
            err = np.linalg.norm(g - truth) / scale
            print(f"  {name:<18} {n_samples:>6}   "
                  f"[{g[0]:+9.5f} {g[1]:+9.5f}]   {err:7.4f}")
            if cls in DETERMINISTIC:
                break        # K has no effect; the sweep is over all n
        print()

    # Eq. (3) lands on grad f / n, so it only matches after rescaling by n.
    rescaled = len(x0) * RandomCoordinate(mu=MU, num_samples=20000,
                                          seed=0).estimate(f, x0)
    print(f"  rand coord (3) x n         {rescaled}")
    print(f"  autograd gradient          {truth}\n")

    ###########################################################
    # SQUARED ERROR OVER THE GRID
    ###########################################################

    # mu = 0.1 is chosen so that all seven estimators are legible on one
    # colour scale, which is a narrow window: the error of Eq. (8) grows as
    # 1/mu^2 while forward differences fall as mu^2 and central differences
    # as mu^4.  At mu = 1e-3 those two ends sit seventeen decades apart and a
    # shared scale shows nothing.  Move mu and the ranking moves with it.
    print(f"squared error over a {GRID}x{GRID} grid, "
          f"mu = {MU}, K = {K}, {REPEATS} repeats ...")

    fig, fields = plot_error_heatmaps(f, estimators, bounds, n=GRID,
                                      repeats=REPEATS)
    fig.suptitle(f"{name_f}: squared gradient error "
                 f"(mu={MU}, K={K}, {REPEATS} repeats)", fontsize=11)

    for name, E in fields.items():
        print(f"  {name}  median {np.median(E):.3e}   max {E.max():.3e}")

    ###########################################################
    # THE LANDSCAPE THE ESTIMATORS ARE RUN ON
    ###########################################################

    _, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    plot_landscape(f, bounds, ax=axes[0])
    plot_gradient_field(f, bounds, n=20, ax=axes[0])
    axes[0].set_title(f"{name_f}: gradient, true magnitude")

    plot_landscape(f, bounds, ax=axes[1])
    plot_gradient_field(f, bounds, n=20, ax=axes[1], normalize=True)
    axes[1].set_title(f"{name_f}: gradient, direction only")

    plt.tight_layout()

    # The same landscape as a 3-D surface.
    ax = plot_surface(f, bounds)
    ax.set_title(name_f)

    ###########################################################
    # THE SMOOTHED SURROGATE -- Eq. (5)
    ###########################################################

    # f_mu itself, rather than its gradient: the convolution that all three
    # smoothing estimators are differentiating without ever forming.  The
    # panels run from the widest smoothing down to none, so each step sharpens
    # the surrogate back towards f -- the kinks re-form, the plateaus flatten
    # out again and the two jumps steepen -- closing on the original as the
    # mu = 0 limit.  The colour ranges widen along the way, as progressively
    # less of the extremes is averaged off.
    #
    # Both figures are built from one list, so the surface panels correspond
    # one-to-one with the contour panels, and from the same SmoothedFunction
    # objects, so the two show the same draw of the noise rather than two
    # independent estimates of the same surrogate.
    MU_LADDER = (0.35, 0.15, 0.05)          # progressively lower
    K_SMOOTH = 400

    print(f"\nsmoothed surrogate f_mu at mu = {MU_LADDER}, K = {K_SMOOTH} ...")

    panels = [(SmoothedFunction(f, mu=mu, num_samples=K_SMOOTH),
               rf"$f_\mu$,  $\mu$ = {mu}") for mu in MU_LADDER]
    panels.append((f, rf"{name_f}: original,  $\mu$ = 0"))

    # As filled contours.
    _, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.0))
    for ax, (g, title) in zip(axes, panels):
        plot_landscape(g, bounds, ax=ax)
        ax.set_title(title, fontsize=10)

    plt.tight_layout()

    # The same ladder again as surfaces, panel for panel.
    fig = plt.figure(figsize=(4.8 * len(panels), 4.6))
    for i, (g, title) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        plot_surface(g, bounds, ax=ax)
        ax.set_title(title, fontsize=10)

    plt.tight_layout()

    plt.show()
