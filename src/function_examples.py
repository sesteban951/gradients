##
#
# Example Functions
#
##

import autograd.numpy as np
from function import FunctionAD


###########################################################
# QUADRATIC
###########################################################

class Quadratic(FunctionAD):
    """f(x) = 0.5 * x' Q x + c' x, an example subclass with a known Hessian."""

    Q = np.array([[1.0, 1.0], [0.5, 3.0]])
    c = np.array([1.0, -1.0])

    def evaluate(self, x):
        return 0.5 * x @ self.Q @ x + self.c @ x


###########################################################
# SINE (1-D)
###########################################################

class Sine1D(FunctionAD):
    """f(x) = sin(x), a scalar-input function for 1-D sweeps."""

    def evaluate(self, x):
        return np.sin(x)


###########################################################
# SIN
###########################################################

class Sin(FunctionAD):
    """f(x) = sin(x1) + sin(x2), a smooth two-argument test function."""

    def evaluate(self, x):
        return np.sin(10.0 * x[0]) + np.sin(5.0 * x[1])


###########################################################
# SIN-EXP
###########################################################

class SinExp(FunctionAD):
    """f(x, y) = sin(x) * exp(y), a smooth two-argument test function."""

    def evaluate(self, z):
        return np.sin(z[0]) * np.exp(z[1])


###########################################################
# NASTY
###########################################################

class Nasty(FunctionAD):
    """A deliberately non-smooth landscape, built from three separate defects.

    * **Sharp corners** -- the L1 cone ``|x1| + |x2|`` kinks along both axes,
      so the gradient jumps between two values rather than varying
      continuously, and is undefined on the kink itself.
    * **Flat regions** -- the cone is clamped from below and above, leaving
      the gradient exactly zero across the central basin and the four outer
      corners.  A descent method landing in one gets no direction at all.
    * **Discontinuities** -- a half-plane step across ``x1 = 0`` and a
      circular one at radius ``0.75``.  Both are invisible to the derivative:
      the gradient of a jump is zero almost everywhere and undefined on it.

    This is the setting randomized smoothing is meant for.  ``grad`` still
    returns numbers here, but they are worthless: zero on the plateaus and
    blind to every cliff.  Convolving with a Gaussian removes all three
    defects at once, which is the point of the surrogate.
    """

    def evaluate(self, x):
        x1, x2 = x[0], x[1]

        bowl = 0.15 * (x1 ** 2 + x2 ** 2)         # smooth underlying trend
        cone = 1.0 * (np.abs(x1) + np.abs(x2))    # corners along both axes

        # Clamping top and bottom flattens the central basin and the four
        # outer corners into regions of identically zero gradient.
        clamped = np.minimum(np.maximum(bowl + cone, 0.45), 2.0)

        # Two jumps, one straight and one curved.
        half_plane = np.where(x1 > 0.0, 0.35, 0.0)
        disc = np.where(x1 ** 2 + x2 ** 2 < 0.75 ** 2, -0.30, 0.0)

        return clamped + half_plane + disc


###########################################################
# TESTING
###########################################################

if __name__ == "__main__":

    f = Quadratic()
    x = np.array([1.0, 2.0])

    print("f(x)        =", f(x))
    print("grad(x, 1)  =", f.grad(x, 1))     # Q x + c
    print("grad(x, 2)  =\n", f.grad(x, 2))   # Q

    # Batched evaluation: a cloud of points in R^2.
    X = np.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
    print("f(X)        =", f.evaluate_batch(X))      # (3,)
    print("grad(X, 1)  =\n", f.grad_batch(X, 1))     # (3, 2)

    # A scalar-input function, the shape used for 1-D sweeps.
    s = Sine1D()
    xs = np.linspace(0.0, np.pi, 5)
    print("sin(xs)     =", s.evaluate_batch(xs))     # (5,)
    print("cos(xs)     =", s.grad_batch(xs, 1))      # (5,)

    # The two-argument function the smoothing plots are drawn on.
    g = SinExp()
    z = np.array([0.5, -0.25])
    print("g(z)        =", g(z))                     # sin(0.5) * e^-0.25
    print("grad(z, 1)  =", g.grad(z, 1))             # (cos(x) e^y, sin(x) e^y)
