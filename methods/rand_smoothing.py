##
#
# Randomized Smoothing
#
##

"""Randomized smoothing (Jordana et al., 2025, Sec. II-B3).

The method replaces ``f`` with the Gaussian-smoothed surrogate
``f_mu(x) = E[f(x + mu * eps)]`` for ``eps ~ N(0, Sigma)``, Eq. (5).  Its
gradient, Eq. (8), is an expectation of evaluations of ``f`` alone, so it can
be estimated without any derivative of ``f``.

For now this file only draws the unsmoothed landscape and its exact gradient
field; the surrogate and its sampled gradient estimators go on top of it next.
"""

# directory imports
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.getenv("GRAD_ROOT_DIR") or os.path.dirname(HERE)
sys.path.append(ROOT)

# standard imports
import matplotlib.pyplot as plt

# custom imports
from src.function_examples import *
from utils.plotting import plot_landscape, plot_gradient_field, plot_surface


###########################################################
# TESTING
###########################################################

if __name__ == "__main__":

    f = Nasty()
    # f = Quadratic()
    # f = Sin()
    # f = SinExp()
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
