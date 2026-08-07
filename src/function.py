##
#
# Function Base Class
#
##

from abc import ABC, abstractmethod
import autograd.numpy as np
from autograd import grad as _grad, hessian as _hessian


###########################################################
# FUNCTION BASE CLASS
###########################################################

class FunctionAD(ABC):
    """Scalar-valued function whose derivatives come from autodiff.

    Implementations only define :meth:`evaluate`.  It must be written with
    ``autograd.numpy`` (imported above as ``np``) rather than plain NumPy, and
    must return a scalar; control flow, loops and helper methods are fine.
    It must *not* call :meth:`_as_float` on its own argument: that is applied
    once at the boundary below, and calling it again inside ``evaluate``
    severs the autograd tape.

    The derivative functions are built once on first use and reused, so
    subclasses need no ``__init__``.  Subclasses live in
    ``function_examples.py``.
    """

    # Maps derivative order to the autograd function that builds it.
    BUILDERS = {1: _grad, 2: _hessian}

    # Instance-level cache; assigned lazily on first call.
    _derivs = None

    @abstractmethod
    def evaluate(self, x):
        """Evaluate the cost at ``x`` and return a scalar."""

    def grad(self, x, order):
        """Derivative of ``order`` at ``x``.

        ``order=1`` gives the gradient, shaped like ``x``; ``order=2`` gives
        the Hessian, shaped ``x.shape * 2``.
        """
        if order not in self.BUILDERS:
            raise ValueError(f"order must be 1 or 2, got {order!r}")
        if self._derivs is None:
            # Assigning on the instance keeps the cache per-object; a mutable
            # class attribute would be shared by every subclass instance.
            self._derivs = {}
        if order not in self._derivs:
            self._derivs[order] = self.BUILDERS[order](self.evaluate)
        return self._derivs[order](self._as_float(x))

    def __call__(self, x):
        return self.evaluate(self._as_float(x))

    def evaluate_batch(self, X):
        """Evaluate at many points.  ``X`` is a stack of inputs along axis 0.

        Returns shape ``(m,)`` for ``m = len(X)``.  Iterating over axis 0 does
        the right thing for both vector inputs (``X`` of shape ``(m, n)``) and
        scalar ones (``X`` of shape ``(m,)``), so a ``linspace`` or a cloud of
        samples can be passed straight in.
        """
        return np.array([self(x) for x in X])

    def grad_batch(self, X, order=1):
        """Derivative of ``order`` at many points.

        Returns shape ``(m,) + x.shape`` for ``order=1`` and
        ``(m,) + x.shape * 2`` for ``order=2``.
        """
        return np.array([self.grad(x, order) for x in X])

    @staticmethod
    def _as_float(x):
        # autograd only differentiates floating-point input; an integer array
        # would silently produce zero derivatives instead of raising.
        return np.asarray(x, dtype=float)
