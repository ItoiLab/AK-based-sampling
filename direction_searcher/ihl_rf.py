import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Optional, Callable, Iterator

import numpy as np
from numpy.typing import ArrayLike
import jax.numpy as jnp
from jax import grad


__all__ = ["iHL_RF"]

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


@contextmanager
def _verbose_logging(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[iHL-RF] %(message)s"))
    old_level = _LOGGER.level
    old_propagate = _LOGGER.propagate

    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    try:
        yield
    finally:
        _LOGGER.removeHandler(handler)
        _LOGGER.setLevel(old_level)
        _LOGGER.propagate = old_propagate


class iHL_RF:
    def __init__(self, dim: int, LSF: Callable[[ArrayLike], float], LSF_grad: Optional[Callable[[ArrayLike], ArrayLike]] = None):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim should be a positive integer.")
        if not callable(LSF):
            raise TypeError("LSF should be a callable function.")
        if LSF_grad is not None and not callable(LSF_grad):
            raise TypeError("LSF_grad should be a callable function or None.")
        self._n_dim = dim
        self._LSF = LSF
        if LSF_grad is None:
            try:
                self._LSF_grad = grad(LSF)
            except Exception as e:
                raise ValueError("Failed to compute gradient of LSF.") from e
        else:
            self._LSF_grad = LSF_grad

        # params for initialization
        self._initial_point = np.zeros(self._n_dim)

        # params for learning rate
        self._eta = 1.5

        # params for stopping criteria
        self._max_iters = 20
        self._eps1 = 1e-3
        self._eps2 = 1e-3

        # result
        self._result = None

    @property
    def max_iters(self) -> int:
        return self._max_iters
    
    @max_iters.setter
    def max_iters(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("max_iters should be a positive integer.")
        self._max_iters = value

    @property
    def eps1(self) -> float:
        return self._eps1

    @eps1.setter
    def eps1(self, value: float):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("eps1 should be a positive number.")
        self._eps1 = float(value)

    @property
    def eps2(self) -> float:
        return self._eps2

    @eps2.setter
    def eps2(self, value: float):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("eps2 should be a positive number.")
        self._eps2 = float(value)

    @property
    def eta(self) -> float:
        return self._eta

    @eta.setter
    def eta(self, value: float):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("eta should be a positive number.")
        self._eta = float(value)

    @property
    def initial_point(self) -> np.ndarray:
        return self._initial_point.copy()

    @initial_point.setter
    def initial_point(self, value: ArrayLike):
        value = np.asarray(value, dtype=float)
        if value.shape != (self._n_dim,):
            raise ValueError("initial_point should have the same dimension as the problem.")
        if not np.all(np.isfinite(value)):
            raise ValueError("initial_point should contain only finite values.")
        self._initial_point = value

    @property
    def result(self):
        if self._result is None:
            return None
        return MappingProxyType(self._result)

    def set_parameters(
        self,
        max_iters: Optional[int] = None,
        eps1: Optional[float] = None,
        eps2: Optional[float] = None,
        eta: Optional[float] = None,
        initial_point: Optional[ArrayLike] = None,
    ):
        if max_iters is not None:
            self.max_iters = max_iters
        if eps1 is not None:
            self.eps1 = eps1
        if eps2 is not None:
            self.eps2 = eps2
        if eta is not None:
            self.eta = eta
        if initial_point is not None:
            self.initial_point = initial_point

    def _as_jax_point(self, point: ArrayLike) -> jnp.ndarray:
        point = jnp.asarray(point, dtype=float)
        if point.shape != (self._n_dim,):
            raise ValueError("point should have the same dimension as the problem.")
        return point

    def _evaluate_lsf(self, point: ArrayLike) -> float:
        point = self._as_jax_point(point)
        try:
            value = self._LSF(point)
        except Exception as e:
            raise ValueError("LSF failed to evaluate at the current point.") from e

        value_array = np.asarray(value, dtype=float)
        if value_array.shape != () or not np.isfinite(value_array).item():
            raise ValueError("LSF should return one finite scalar value.")
        return float(value_array)

    def _evaluate_grad(self, point: ArrayLike) -> jnp.ndarray:
        point = self._as_jax_point(point)
        try:
            gradient = self._LSF_grad(point)
        except Exception as e:
            raise ValueError(
                "LSF_grad failed to evaluate. If LSF_grad is None, LSF must be compatible with jax.grad."
            ) from e

        gradient_array = np.asarray(gradient, dtype=float)
        if gradient_array.shape != (self._n_dim,):
            raise ValueError("LSF_grad should return a gradient vector with the same dimension as the problem.")
        if not np.all(np.isfinite(gradient_array)):
            raise ValueError("LSF_grad should return only finite values.")

        gradient_norm = np.linalg.norm(gradient_array)
        if gradient_norm <= 0.0:
            raise ValueError("LSF_grad norm should be positive at the current point.")
        return jnp.asarray(gradient_array, dtype=float)

    def __call__(self, verbose: bool = False) -> np.ndarray:
        if not isinstance(verbose, bool):
            raise TypeError("verbose should be a boolean.")

        with _verbose_logging(verbose):
            ui = self._as_jax_point(self._initial_point)
            G0 = self._evaluate_lsf(ui)
            Gi = G0
            c = 1.0
            converged = False
            relative_scale = max(abs(G0), np.finfo(float).eps)
            history = [
                {
                    "iteration": 0,
                    "point": np.asarray(ui, dtype=float).copy(),
                    "G": Gi,
                    "beta": float(jnp.linalg.norm(ui)),
                }
            ]

            dG = self._evaluate_grad(ui)
            dG_norm = float(jnp.linalg.norm(dG))
            alpha = -dG / dG_norm

            _LOGGER.info(
                "Start iHL-RF: dim=%s, G0=%.6g, grad_norm=%.6g",
                self._n_dim,
                G0,
                dG_norm,
            )

            for i in range(self._max_iters):
                u_norm = float(jnp.linalg.norm(ui))
                c = max(c, u_norm / dG_norm) * self._eta
                m = 0.5 * u_norm**2 + c * abs(Gi)

                # search direction
                d = (Gi / dG_norm + alpha @ ui) * alpha - ui

                # step size
                lambdai = 1.0
                uu = ui
                GG = Gi
                accepted = False
                for _ in range(3):
                    uu = ui + lambdai * d
                    GG = self._evaluate_lsf(uu)
                    merit = 0.5 * float(jnp.linalg.norm(uu))**2 + c * abs(GG)
                    if merit < m:
                        accepted = True
                        break
                    lambdai /= 2.0

                if not accepted:
                    _LOGGER.info("Iteration %s: no merit decrease after step halving", i + 1)

                # update
                ui = uu
                Gi = GG
                dG = self._evaluate_grad(ui)
                dG_norm = float(jnp.linalg.norm(dG))
                alpha = -dG / dG_norm

                orthogonal_norm = float(jnp.linalg.norm(ui - (alpha @ ui) * alpha))
                relative_g = abs(Gi) / relative_scale
                history.append(
                    {
                        "iteration": i + 1,
                        "point": np.asarray(ui, dtype=float).copy(),
                        "G": Gi,
                        "beta": float(jnp.linalg.norm(ui)),
                    }
                )
                _LOGGER.info(
                    "Iteration %s/%s: G=%.6g, beta=%.6g, grad_norm=%.6g, step=%.6g, rel_G=%.6g, orth=%.6g",
                    i + 1,
                    self._max_iters,
                    Gi,
                    float(jnp.linalg.norm(ui)),
                    dG_norm,
                    lambdai,
                    relative_g,
                    orthogonal_norm,
                )

                # stopping criteria
                if relative_g < self._eps1 and orthogonal_norm < self._eps2:
                    converged = True
                    _LOGGER.info("Stopping criteria reached at iteration %s", i + 1)
                    break

            design_point = np.asarray(ui, dtype=float)
            beta = float(np.linalg.norm(design_point))
            self._result = {
                "design_point": design_point.copy(),
                "beta": beta,
                "G": Gi,
                "grad_norm": dG_norm,
                "n_iters": i + 1,
                "converged": converged,
                "history": tuple(history),
            }

            _LOGGER.info(
                "Finished iHL-RF: beta=%.6g, G=%.6g, n_iters=%s, converged=%s",
                beta,
                Gi,
                i + 1,
                converged,
            )

            return design_point
