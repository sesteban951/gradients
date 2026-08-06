from abc import ABC, abstractmethod
import autograd.numpy as np
from autograd import grad, hessian

###########################################################
# FUNCTION BASE CLASS
###########################################################

class FunctionAD(ABC):
    """Scalar-valued function whose derivatives come from autodiff.

    Implementations only define :meth:`function`.  It must be written with
    ``autograd.numpy`` (imported above as ``np``) rather than plain NumPy, and
    must return a scalar; control flow, loops and helper methods are fine.

    The derivative functions are built once on first use and reused, so
    subclasses need no ``__init__``.
    """

    # Instance-level caches; assigned lazily on first call.
    _d1 = None
    _d2 = None

    @abstractmethod
    def function(self, x):
        """Evaluate the cost at ``x`` and return a scalar."""

    def grad_1(self, x):
        """First derivative at ``x``: the gradient, shaped like ``x``."""
        if self._d1 is None:
            self._d1 = grad(self.function)
        return self._d1(self._as_float(x))

    def grad_2(self, x):
        """Second derivative at ``x``: the Hessian, shaped ``x.shape * 2``."""
        if self._d2 is None:
            self._d2 = hessian(self.function)
        return self._d2(self._as_float(x))

    def __call__(self, x):
        return self.function(self._as_float(x))

    @staticmethod
    def _as_float(x):
        # autograd only differentiates floating-point input; an integer array
        # would silently produce zero derivatives instead of raising.
        return np.asarray(x, dtype=float)


###########################################################
# QUADRATIC
###########################################################

class Quadratic(FunctionAD):
    """f(x) = 0.5 * x' Q x + c' x, an example subclass with a known Hessian."""

    def __init__(self, Q, c):
        self.Q = np.asarray(Q, dtype=float)
        self.c = np.asarray(c, dtype=float)

    def function(self, x):
        return 0.5 * x @ self.Q @ x + self.c @ x


###########################################################
# TESTING
###########################################################

if __name__ == "__main__":
    
    Q = np.array([[2.0, 0.5], [0.5, 3.0]])
    c = np.array([1.0, -1.0])
    f = Quadratic(Q, c)
    x = np.array([1.0, 2.0])

    print("f(x)      =", f(x))
    print("grad_1(x) =", f.grad_1(x))       # Q x + c
    print("grad_2(x) =", f.grad_2(x))       # Q
