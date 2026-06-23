"""Active learning reliability analysis with importance sampling (AK-IS)."""

import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Callable, Iterator, Literal, Optional, Tuple, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

from direction_searcher.ihl_rf import iHL_RF


__all__ = ["AK_IS"]

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


@contextmanager
def _verbose_logging(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[AK-IS] %(message)s"))
    old_level, old_propagate = _LOGGER.level, _LOGGER.propagate
    _LOGGER.addHandler(handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False
    try:
        yield
    finally:
        _LOGGER.removeHandler(handler)
        _LOGGER.setLevel(old_level)
        _LOGGER.propagate = old_propagate


class AK_IS:
    """Kriging-assisted importance sampling for a standard normal space.

    The importance proposal is ``N(sample_center, I)``.  By default its center
    is the design point found by :class:`direction_searcher.ihl_rf.iHL_RF` on
    the current GP mean.  A different direction-searcher factory can be set by
    :meth:`set_direction_searcher`, or a center can be fixed explicitly with
    :meth:`set_sample_center`.
    """

    def __init__(self, dim: int, LSF: Callable[[ArrayLike], ArrayLike]):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError("dim should be a positive integer.")
        if not callable(LSF):
            raise TypeError("LSF should be a callable function.")
        self._n_dim, self._LSF = dim, LSF

        self._n_initial_population = 10_000
        self._n_initial_DoEs = 20
        self._u_stop = 2.0
        self._cov_stop = 0.05
        self._max_iterations_u = 100
        self._kernel = C(1.0) * RBF(length_scale=np.ones(dim))
        self._gp_alpha, self._gp_normalize_y = 1e-10, True
        self._gp = GaussianProcessRegressor(
            kernel=self._kernel, alpha=self._gp_alpha,
            normalize_y=self._gp_normalize_y,
        )

        self._sample_distribution: Literal["uniform", "normal"] = "uniform"
        self._sample_range: Optional[Tuple[float, float]] = (-3.0, 3.0)
        self._initial_sampler: Callable[..., np.ndarray] = self._sample_LHS
        self._initial_sampler_is_builtin = True
        self._center_initial_DoEs = True

        # A factory is used instead of an instance because the GP changes while
        # AK-IS is running.  kwargs are forwarded to the factory on each run.
        self._direction_searcher: Callable[..., Any] = iHL_RF
        self._direction_searcher_kwargs: dict[str, Any] = {}
        self._fixed_sample_center: Optional[np.ndarray] = None
        self._result: Optional[dict[str, Any]] = None
        self._train_data: Optional[dict[str, Any]] = None

    @property
    def result(self):
        return None if self._result is None else MappingProxyType(self._result)

    @property
    def n_initial_population(self) -> int:
        return self._n_initial_population

    @n_initial_population.setter
    def n_initial_population(self, value: int) -> None:
        self._n_initial_population = self._positive_int("n_initial_population", value)

    @property
    def n_initial_DoEs(self) -> int:
        return self._n_initial_DoEs

    @n_initial_DoEs.setter
    def n_initial_DoEs(self, value: int) -> None:
        self._n_initial_DoEs = self._positive_int("n_initial_DoEs", value)

    @property
    def u_stop(self) -> float:
        return self._u_stop

    @u_stop.setter
    def u_stop(self, value: float) -> None:
        self._u_stop = self._positive_number("u_stop", value)

    @property
    def cov_stop(self) -> float:
        return self._cov_stop

    @cov_stop.setter
    def cov_stop(self, value: float) -> None:
        self._cov_stop = self._positive_number("cov_stop", value)

    @property
    def max_iterations_u(self) -> int:
        return self._max_iterations_u

    @max_iterations_u.setter
    def max_iterations_u(self, value: int) -> None:
        self._max_iterations_u = self._positive_int("max_iterations_u", value)

    @property
    def train_data(self):
        return None if self._train_data is None else MappingProxyType(self._train_data)

    @property
    def sample_center(self) -> Optional[np.ndarray]:
        """The user-fixed center, or ``None`` when it is searched automatically."""
        return None if self._fixed_sample_center is None else self._fixed_sample_center.copy()

    def _positive_int(self, name: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} should be a positive integer.")
        return value

    def _positive_number(self, name: str, value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} should be a positive number.")
        return float(value)

    def set_parameters(
        self, *, n_initial_population: Optional[int] = None,
        n_initial_DoEs: Optional[int] = None, u_stop: Optional[float] = None,
        cov_stop: Optional[float] = None, max_iterations_u: Optional[int] = None,
    ) -> None:
        if n_initial_population is not None:
            self.n_initial_population = n_initial_population
        if n_initial_DoEs is not None:
            self.n_initial_DoEs = n_initial_DoEs
        if u_stop is not None:
            self.u_stop = u_stop
        if cov_stop is not None:
            self.cov_stop = cov_stop
        if max_iterations_u is not None:
            self.max_iterations_u = max_iterations_u

    def set_surrogate_model(
        self, gp: Optional[GaussianProcessRegressor] = None, kernel: Optional[Any] = None,
        whitenoise: Optional[float] = None, alpha: Optional[float] = None,
        normalize_y: Optional[bool] = None,
    ) -> None:
        if gp is not None:
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp should be a GaussianProcessRegressor instance.")
            self._gp = gp
            return
        if kernel is not None:
            self._kernel = kernel
        elif whitenoise is not None:
            if not isinstance(whitenoise, (int, float)) or isinstance(whitenoise, bool) or whitenoise < 0:
                raise ValueError("whitenoise should be a non-negative number.")
            self._kernel = self._kernel + WhiteKernel(noise_level=whitenoise)
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
                raise ValueError("alpha should be a non-negative number.")
            self._gp_alpha = float(alpha)
        if normalize_y is not None:
            if not isinstance(normalize_y, bool):
                raise TypeError("normalize_y should be a boolean.")
            self._gp_normalize_y = normalize_y
        self._gp = GaussianProcessRegressor(kernel=self._kernel, alpha=self._gp_alpha, normalize_y=self._gp_normalize_y)

    def set_direction_searcher(self, searcher: Optional[Callable[..., Any]] = None, **kwargs: Any) -> None:
        """Set the design-point searcher factory (default: ``iHL_RF``).

        The factory should accept ``dim`` and ``LSF`` keyword arguments; if it
        accepts ``LSF_grad``, AK-IS also supplies a finite-difference gradient.
        Its returned object must be callable and return a length-``dim`` center.
        Passing ``None`` restores iHL-RF.  This also clears a fixed center.
        """
        if searcher is not None and not callable(searcher):
            raise TypeError("searcher should be callable or None.")
        self._direction_searcher = iHL_RF if searcher is None else searcher
        self._direction_searcher_kwargs = dict(kwargs)
        self._fixed_sample_center = None

    def set_sample_center(self, center: Optional[ArrayLike]) -> None:
        """Fix the importance-sampling center; pass ``None`` to re-enable FORM."""
        if center is None:
            self._fixed_sample_center = None
            return
        array = np.asarray(center, dtype=float)
        if array.shape != (self._n_dim,) or not np.all(np.isfinite(array)):
            raise ValueError("center should be a finite vector with shape (dim,).")
        self._fixed_sample_center = array.copy()

    def _sample_LHS(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        samples = qmc.LatinHypercube(d=self._n_dim, seed=rng).random(n_samples)
        return qmc.scale(samples, self._sample_range[0], self._sample_range[1]) if self._sample_distribution == "uniform" else norm.ppf(samples)

    def _sample_sobol(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        samples = qmc.Sobol(d=self._n_dim, scramble=True, seed=rng).random(n_samples)
        return qmc.scale(samples, self._sample_range[0], self._sample_range[1]) if self._sample_distribution == "uniform" else norm.ppf(samples)

    def set_initial_sampler(
        self, sampler: Optional[Callable[[int], np.ndarray]] = None,
        method: Optional[Literal["LHS", "Sobol"]] = None,
        distribution: Optional[Literal["uniform", "normal"]] = None,
        dist_range: Optional[Tuple[float, float]] = None,
        center_on_is_center: bool = True,
    ) -> None:
        if not isinstance(center_on_is_center, bool):
            raise TypeError("center_on_is_center should be a boolean.")
        if sampler is not None:
            if not callable(sampler):
                raise TypeError("sampler should be callable.")
            self._initial_sampler = sampler
            self._initial_sampler_is_builtin = False
            self._center_initial_DoEs = center_on_is_center
            return
        method, distribution = method or "LHS", distribution or self._sample_distribution
        if method not in ("LHS", "Sobol") or distribution not in ("uniform", "normal"):
            raise ValueError("method must be 'LHS'/'Sobol' and distribution must be 'uniform'/'normal'.")
        if distribution == "uniform":
            dist_range = dist_range or self._sample_range
            if not isinstance(dist_range, tuple) or len(dist_range) != 2 or dist_range[0] >= dist_range[1]:
                raise ValueError("uniform dist_range should be an increasing pair.")
        self._sample_distribution, self._sample_range = distribution, dist_range if distribution == "uniform" else None
        self._initial_sampler = self._sample_sobol if method == "Sobol" else self._sample_LHS
        self._initial_sampler_is_builtin = True
        self._center_initial_DoEs = center_on_is_center

    def _sample_initial(self, n_samples: int, rng: np.random.Generator, center: np.ndarray) -> np.ndarray:
        """Generate the initial DoE around ``center`` for built-in samplers.

        A custom sampler can optionally accept ``center=``.  If it does not,
        its coordinates are used unchanged so custom sampling remains fully
        under the caller's control.
        """
        try:
            values = self._initial_sampler(n_samples, rng=rng, center=center)
        except TypeError:
            try:
                values = self._initial_sampler(n_samples, rng=rng)
            except TypeError:
                values = self._initial_sampler(n_samples)
        values = np.asarray(values, dtype=float)
        if values.shape != (n_samples, self._n_dim) or not np.all(np.isfinite(values)):
            raise ValueError(f"sampler should return finite values with shape ({n_samples}, {self._n_dim}).")
        if self._initial_sampler_is_builtin and self._center_initial_DoEs:
            values = values + center
        return values

    def _evaluate_lsf(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray([self._LSF(point) for point in points], dtype=float)
        if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("LSF should return one finite scalar value per sample.")
        return values

    def _estimate_population(self, population: np.ndarray, train: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distance, index = KDTree(train).query(population)
        unknown = distance >= 1e-8
        estimate, sigma, u_value = np.empty(len(population)), np.zeros(len(population)), np.full(len(population), np.inf)
        if np.any(unknown):
            estimate[unknown], sigma[unknown] = self._gp.predict(population[unknown], return_std=True)
            u_value[unknown] = np.divide(np.abs(estimate[unknown]), sigma[unknown], out=np.full(np.sum(unknown), np.inf), where=sigma[unknown] > 1e-12)
        estimate[~unknown] = response[index[~unknown]]
        return estimate, sigma, u_value

    def _find_center(self, response_function: Callable[[ArrayLike], float]) -> np.ndarray:
        if self._fixed_sample_center is not None:
            return self._fixed_sample_center.copy()

        def gradient(point: ArrayLike) -> np.ndarray:
            point = np.asarray(point, dtype=float)
            step = 1e-5 * np.maximum(1.0, np.abs(point))
            return np.array([
                (response_function(point + np.eye(self._n_dim)[i] * step[i])
                 - response_function(point - np.eye(self._n_dim)[i] * step[i])) / (2 * step[i])
                for i in range(self._n_dim)
            ])

        try:
            finder = self._direction_searcher(dim=self._n_dim, LSF=response_function, LSF_grad=gradient, **self._direction_searcher_kwargs)
        except TypeError:
            finder = self._direction_searcher(dim=self._n_dim, LSF=response_function, **self._direction_searcher_kwargs)
        if not callable(finder):
            raise TypeError("direction searcher factory should return a callable object.")
        center = np.asarray(finder(), dtype=float)
        if center.shape != (self._n_dim,) or not np.all(np.isfinite(center)):
            raise ValueError("direction searcher should return a finite vector with shape (dim,).")
        return center

    def _find_sample_center(self) -> np.ndarray:
        """Find the IS center from the true limit-state function."""
        def lsf(point: ArrayLike) -> float:
            value = np.asarray(self._LSF(np.asarray(point, dtype=float)), dtype=float)
            if value.shape != () or not np.isfinite(value):
                raise ValueError("LSF should return one finite scalar value per sample.")
            return float(value)

        return self._find_center(lsf)

    @staticmethod
    def _weights(population: np.ndarray, center: np.ndarray) -> np.ndarray:
        # phi(u) / phi(u; center, I), evaluated in log space for stability.
        log_weights = -population @ center + 0.5 * np.dot(center, center)
        return np.exp(np.clip(log_weights, np.log(np.finfo(float).tiny), np.log(np.finfo(float).max)))

    @overload
    def __call__(self, return_gp: Literal[False] = False, seed: Optional[int] = None, verbose: bool = False) -> float: ...
    @overload
    def __call__(self, return_gp: Literal[True], seed: Optional[int] = None, verbose: bool = False) -> tuple[float, GaussianProcessRegressor]: ...

    def __call__(self, return_gp: bool = False, seed: Optional[int] = None, verbose: bool = False):
        if not isinstance(return_gp, bool) or not isinstance(verbose, bool):
            raise TypeError("return_gp and verbose should be booleans.")
        with _verbose_logging(verbose):
            rng = np.random.default_rng(seed)
            center = self._find_sample_center()
            train = self._sample_initial(self._n_initial_DoEs, rng, center)
            response = self._evaluate_lsf(train)
            self._gp.fit(train, response)
            population = rng.normal(loc=center, size=(self._n_initial_population, self._n_dim))
            weights = self._weights(population, center)
            _LOGGER.info("Start AK-IS: dim=%s, center=%s, population=%s", self._n_dim, center, len(population))

            min_u = np.inf
            for iteration in range(self._max_iterations_u):
                self._gp.fit(train, response)
                estimate, sigma, u_values = self._estimate_population(population, train, response)
                nxt = int(np.argmin(u_values))
                min_u = float(u_values[nxt])
                _LOGGER.info("U iteration %s/%s: min U=%.6g", iteration + 1, self._max_iterations_u, min_u)
                if min_u > self._u_stop:
                    break
                train = np.vstack((train, population[nxt]))
                response = np.hstack((response, self._evaluate_lsf(population[nxt:nxt + 1])))

            self._gp.fit(train, response)
            estimate, sigma, _ = self._estimate_population(population, train, response)
            contributions = weights * (estimate <= 0.0)
            pf = float(np.mean(contributions))
            cov = np.inf if pf <= 0.0 else float(np.std(contributions, ddof=1) / (np.sqrt(len(population)) * pf))
            self._result = {"pf": pf, "n_called": len(train), "n_population": len(population), "min_u_value": min_u, "cov": cov, "sample_center": center.copy()}
            self._train_data = {"initial_DoE": (train[:self._n_initial_DoEs], response[:self._n_initial_DoEs]), "DoE": (train[self._n_initial_DoEs:], response[self._n_initial_DoEs:]), "population": (population, estimate, sigma, weights)}
            _LOGGER.info("Finished AK-IS: pf=%.6g, cov=%.6g, n_called=%s", pf, cov, len(train))
            return (pf, self._gp) if return_gp else pf

    def wipe(self) -> None:
        self.set_surrogate_model()
        self._result = self._train_data = None
