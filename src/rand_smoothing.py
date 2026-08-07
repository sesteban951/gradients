"""Randomized smoothing (Jordana et al., 2025, Sec. II-B3).

The method replaces ``f`` with the Gaussian-smoothed surrogate
``f_mu(x) = E[f(x + mu * eps)]`` for ``eps ~ N(0, Sigma)``, Eq. (5).  Its
gradient, Eq. (8), is an expectation of evaluations of ``f`` alone, so it can
be estimated without any derivative of ``f``.

For now this file only draws the unsmoothed landscape and its exact gradient
field; the surrogate and its sampled gradient estimators go on top of it next.
"""

import autograd.numpy as np
import matplotlib.pyplot as plt

from function_examples import *


###########################################################
# PLOTTING
###########################################################

def sample_grid(f, bounds=(-1.0, 1.0), n=200):
    """Evaluate ``f`` on an ``n x n`` grid spanning ``bounds`` in both
    arguments.  Returns ``(XX, YY, Z)``, each of shape ``(n, n)``.
    """
    axis = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(axis, axis)
    points = np.stack([XX.ravel(), YY.ravel()], axis=1)   # (n * n, 2)
    return XX, YY, f.evaluate_batch(points).reshape(XX.shape)


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


###########################################################
# TESTING
###########################################################

if __name__ == "__main__":

    # f = Quadratic()
    f = Sin()
    # f = SinExp()
    # f = Nasty()

    bounds = (-1.0, 1.0)

    _, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    plot_landscape(f, bounds, ax=axes[0])
    plot_gradient_field(f, bounds, n=20, ax=axes[0])
    axes[0].set_title("gradient, true magnitude")

    plot_landscape(f, bounds, ax=axes[1])
    plot_gradient_field(f, bounds, n=20, ax=axes[1], normalize=True)
    axes[1].set_title("gradient, direction only")

    plt.tight_layout()

    # The same landscape as a 3-D surface.
    ax = plot_surface(f, bounds)

    plt.show()
