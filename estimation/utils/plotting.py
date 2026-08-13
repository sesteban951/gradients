##
#
# Plotting Helpers
#
##

"""Visualisation for any two-argument :class:`FunctionAD`.

Every helper takes the function object itself rather than an array, so the
same call works for the exact function and for a smoothed surrogate built on
top of it.  Each accepts an optional ``ax`` so panels can be composed.
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


###########################################################
# SAMPLING
###########################################################

def sample_grid(f, bounds=(-1.0, 1.0), n=200):
    """Evaluate ``f`` on an ``n x n`` grid spanning ``bounds`` in both
    arguments.  Returns ``(XX, YY, Z)``, each of shape ``(n, n)``.
    """
    axis = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(axis, axis)
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (n * n, 2)
    return XX, YY, f.evaluate_batch(points).reshape(XX.shape)


def gradient_grid(f, bounds=(-1.0, 1.0), n=20):
    """Gradient of ``f`` on an ``n x n`` grid spanning ``bounds`` in both
    arguments.  Returns ``(XX, YY, U, V)``, each of shape ``(n, n)``, where
    ``(U, V)`` are the two components of the gradient.
    """
    axis = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(axis, axis)
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (n * n, 2)
    G = f.grad_batch(points, 1)                           # (n * n, 2)
    return XX, YY, G[:, 0].reshape(XX.shape), G[:, 1].reshape(XX.shape)


def error_grid(f, estimator, bounds=(-1.0, 1.0), n=25, repeats=1,
               reference=None):
    """Squared gradient error of ``estimator`` on an ``n x n`` grid.

    Returns ``(XX, YY, E)``, each of shape ``(n, n)``, where ``E`` holds
    ``|| g_hat - g_true ||^2`` averaged over ``repeats`` independent estimates
    at that point.

    ``estimator`` is anything exposing ``estimate_batch(f, X)`` -- the classes
    in ``methods/rand_smoothing.py``, in practice.  Keeping this duck-typed
    means the plotting layer never imports the methods layer.

    A single repeat is one draw of a random quantity, so the field comes out
    speckled and neighbouring points are not comparable.  Averaging a handful
    of repeats turns it into an estimate of the mean squared error, which is
    what actually varies smoothly across the domain.

    ``reference`` supplies the ground truth as a callable mapping the
    ``(n * n, 2)`` stack of points to a matching stack of gradients.  The
    default is the autodiff gradient of ``f``.  That is the right reference
    only where ``f`` is differentiable and only for estimators targeting
    ``grad f``; the smoothing family targets ``grad f_mu``, so at large ``mu``
    part of what shows up here is that gap rather than sampling error.
    """
    axis = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(axis, axis)
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (n * n, 2)

    truth = f.grad_batch(points, 1) if reference is None else reference(points)

    total = np.zeros(len(points))
    for _ in range(repeats):
        G = estimator.estimate_batch(f, points)           # (n * n, 2)
        total = total + np.sum((G - truth) ** 2, axis=1)

    return XX, YY, (total / repeats).reshape(XX.shape)


###########################################################
# PLOTTING
###########################################################

def plot_landscape(f, bounds=(-1.0, 1.0), n=200, ax=None):
    """Filled contour of ``f`` over ``bounds`` in both arguments."""
    XX, YY, Z = sample_grid(f, bounds, n)

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    filled = ax.contourf(XX, YY, Z, levels=40, cmap="viridis")
    ax.contour(XX, YY, Z, levels=12, colors="k", linewidths=0.4, alpha=0.4)
    ax.figure.colorbar(filled, ax=ax, label="f(x, y)")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    return ax


def plot_gradient_field(f, bounds=(-1.0, 1.0), n=20, ax=None, normalize=False,
                        color="w", **kwargs):
    """Quiver plot of the gradient of ``f`` over ``bounds``.

    Arrow length is the true gradient magnitude unless ``normalize`` is set,
    in which case every arrow is unit length and only direction is shown.
    That matters for functions whose gradient varies over orders of magnitude,
    where true-length arrows collapse to invisible in the flat regions.
    """
    XX, YY, U, V = gradient_grid(f, bounds, n)

    if normalize:
        mag = np.sqrt(U ** 2 + V ** 2)
        mag = np.where(mag == 0.0, 1.0, mag)   # leave stationary points be
        U, V = U / mag, V / mag

    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.set_aspect("equal")
    ax.quiver(XX, YY, U, V, color=color, **kwargs)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_error_heatmaps(f, estimators, bounds=(-1.0, 1.0), n=25, repeats=1,
                        reference=None, ncols=4, cmap="magma",
                        max_decades=8, show_landscape=True):
    """One squared-error heatmap per estimator, on a shared colour scale.

    ``estimators`` maps a display name to an estimator object.  Every panel is
    drawn with the same :class:`~matplotlib.colors.LogNorm`, which is what
    makes them comparable: errors here routinely span several orders of
    magnitude, and a linear scale would collapse all but the worst panel into
    a single flat colour.

    ``max_decades`` bounds how far below the largest error the scale reaches.
    Deterministic finite differences can be accurate to machine precision, and
    without a floor those panels would stretch the scale over fifteen decades
    and leave no resolution for the rest.  Panels that bottom out are drawn in
    the under-colour, which is the honest reading: exact to within the range
    the figure can show.

    Returns ``(fig, fields)`` with ``fields`` the ``(n, n)`` error array for
    each estimator, so the numbers can be inspected as well as looked at.
    """
    names = list(estimators)
    grids = [error_grid(f, estimators[k], bounds, n, repeats, reference)
             for k in names]
    XX, YY = grids[0][0], grids[0][1]
    fields = {k: g[2] for k, g in zip(names, grids)}

    # A shared scale across every panel, floored so that a near-exact panel
    # cannot stretch it out of usefulness.
    stacked = np.array(list(fields.values()))
    positive = stacked[stacked > 0.0]
    if positive.size == 0:
        raise ValueError("every estimator was exact everywhere; nothing to plot")
    vmax = float(stacked.max())
    vmin = max(float(positive.min()), vmax * 10.0 ** (-max_decades))
    norm = LogNorm(vmin=vmin, vmax=vmax)

    total = len(names) + (1 if show_landscape else 0)
    nrows = int(np.ceil(total / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows),
                             squeeze=False, layout="constrained")
    flat = axes.ravel()

    for ax, name in zip(flat, names):
        mesh = ax.pcolormesh(XX, YY, np.clip(fields[name], vmin, vmax),
                             norm=norm, cmap=cmap, shading="auto")
        # The median is a better summary than the mean here: these fields are
        # heavy-tailed, and a handful of bad points would carry the mean.
        ax.set_title(f"{name.strip()}\nmedian {np.median(fields[name]):.1e}",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Attached to every panel so constrained_layout reserves its own column
    # rather than taking the space out of the last heatmap.
    fig.colorbar(mesh, ax=flat.tolist(), extend="min", shrink=0.85,
                 label=r"$\|\hat{g} - \nabla f\|^2$")

    if show_landscape:
        # The last panel shows f itself, so the error structure can be read
        # against the shape that produced it.
        ax = flat[len(names)]
        LX, LY, Z = sample_grid(f, bounds, 200)
        ax.contourf(LX, LY, Z, levels=40, cmap="viridis")
        ax.set_title("the function", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat[total:]:
        ax.axis("off")

    return fig, fields


def plot_surface(f, bounds=(-1.0, 1.0), n=80, ax=None, cmap="viridis",
                 contours=True, **kwargs):
    """3-D surface of ``f`` over ``bounds`` in both arguments.

    ``contours`` drops a filled contour of the same data onto the floor of the
    axes, which makes the shape easier to read than the surface alone.  Pass
    an ``ax`` built with ``projection="3d"`` to compose with other panels.
    """
    XX, YY, Z = sample_grid(f, bounds, n)

    if ax is None:
        ax = plt.figure(figsize=(6.0, 5.0)).add_subplot(projection="3d")
    elif not hasattr(ax, "plot_surface"):
        raise TypeError("ax must be a 3-D axes, built with projection='3d'")

    surface = ax.plot_surface(XX, YY, Z, cmap=cmap, linewidth=0,
                              antialiased=True, **kwargs)
    # pad keeps the bar clear of the z-axis label, which 3-D axes place wide.
    ax.figure.colorbar(surface, ax=ax, shrink=0.6, pad=0.12, label="f(x, y)")

    if contours:
        floor = Z.min() - 0.35 * (Z.max() - Z.min())
        ax.contourf(XX, YY, Z, levels=30, zdir="z", offset=floor, cmap=cmap,
                    alpha=0.6)
        ax.set_zlim(floor, Z.max())

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    return ax
