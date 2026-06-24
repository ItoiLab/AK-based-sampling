"""AK-SYS: active-learning reliability analysis for systems."""

import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Tuple, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from scipy.stats import norm, qmc
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

__all__ = ["AK_SYS"]
_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


@contextmanager
def _verbose_logging(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[AK-SYS] %(message)s"))
    old_level, old_propagate = _LOGGER.level, _LOGGER.propagate
    _LOGGER.addHandler(handler); _LOGGER.setLevel(logging.INFO); _LOGGER.propagate = False
    try:
        yield
    finally:
        _LOGGER.removeHandler(handler); _LOGGER.setLevel(old_level); _LOGGER.propagate = old_propagate


class AK_SYS:
    """AK-MCS extension for systems with several limit-state functions.

    ``gate='AND'`` estimates a series system (all component failures are
    required), while ``gate='OR'`` estimates a parallel system (any component
    failure is sufficient).  A separate GP is fitted for every component.
    """
    def __init__(self, dim: int, LSFs: Sequence[Callable[[ArrayLike], ArrayLike]], gate: Literal["AND", "OR"] = "AND"):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0: raise ValueError("dim should be a positive integer.")
        if isinstance(LSFs, (str, bytes)) or not isinstance(LSFs, Sequence) or not LSFs or not all(callable(f) for f in LSFs):
            raise TypeError("LSFs should be a non-empty sequence of callables.")
        if gate not in ("AND", "OR"): raise ValueError("gate should be either 'AND' or 'OR'.")
        self._n_dim, self._LSFs, self._gate = dim, tuple(LSFs), gate
        self._n_components = len(LSFs)
        self._n_initial_population, self._n_initial_DoEs = 10_000, 20
        self._u_stop, self._cov_stop = 2.0, 0.05
        self._max_iterations_u, self._max_iterations_cov = 100, 20
        self._kernel = C(1.0) * RBF(length_scale=np.ones(dim))
        self._gp_alpha, self._gp_normalize_y = 1e-10, True
        self._gps = self._new_gps()
        self._sample_distribution: Literal["uniform", "normal"] = "uniform"
        self._sample_range: Optional[Tuple[float, float]] = (-3., 3.)
        self._initial_sampler: Callable[..., np.ndarray] = self._sample_LHS
        self._result: Optional[dict[str, Any]] = None
        self._train_data: Optional[dict[str, Any]] = None

    def _new_gps(self) -> list[GaussianProcessRegressor]:
        return [GaussianProcessRegressor(kernel=self._kernel, alpha=self._gp_alpha, normalize_y=self._gp_normalize_y) for _ in range(self._n_components)]

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} should be a positive integer.")
        return value
    @staticmethod
    def _positive_number(name: str, value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0: raise ValueError(f"{name} should be a positive number.")
        return float(value)

    @property
    def result(self): return None if self._result is None else MappingProxyType(self._result)
    @property
    def train_data(self): return None if self._train_data is None else MappingProxyType(self._train_data)
    @property
    def gate(self) -> Literal["AND", "OR"]: return self._gate
    @property
    def n_initial_population(self) -> int: return self._n_initial_population
    @n_initial_population.setter
    def n_initial_population(self, v: int) -> None: self._n_initial_population = self._positive_int("n_initial_population", v)
    @property
    def n_initial_DoEs(self) -> int: return self._n_initial_DoEs
    @n_initial_DoEs.setter
    def n_initial_DoEs(self, v: int) -> None: self._n_initial_DoEs = self._positive_int("n_initial_DoEs", v)
    @property
    def u_stop(self) -> float: return self._u_stop
    @u_stop.setter
    def u_stop(self, v: float) -> None: self._u_stop = self._positive_number("u_stop", v)
    @property
    def cov_stop(self) -> float: return self._cov_stop
    @cov_stop.setter
    def cov_stop(self, v: float) -> None: self._cov_stop = self._positive_number("cov_stop", v)
    @property
    def max_iterations_u(self) -> int: return self._max_iterations_u
    @max_iterations_u.setter
    def max_iterations_u(self, v: int) -> None: self._max_iterations_u = self._positive_int("max_iterations_u", v)
    @property
    def max_iterations_cov(self) -> int: return self._max_iterations_cov
    @max_iterations_cov.setter
    def max_iterations_cov(self, v: int) -> None: self._max_iterations_cov = self._positive_int("max_iterations_cov", v)

    def set_parameters(self, *, n_initial_population: Optional[int] = None, n_initial_DoEs: Optional[int] = None, u_stop: Optional[float] = None, cov_stop: Optional[float] = None, max_iterations_u: Optional[int] = None, max_iterations_cov: Optional[int] = None) -> None:
        if n_initial_population is not None: self.n_initial_population = n_initial_population
        if n_initial_DoEs is not None: self.n_initial_DoEs = n_initial_DoEs
        if u_stop is not None: self.u_stop = u_stop
        if cov_stop is not None: self.cov_stop = cov_stop
        if max_iterations_u is not None: self.max_iterations_u = max_iterations_u
        if max_iterations_cov is not None: self.max_iterations_cov = max_iterations_cov

    def set_surrogate_model(self, gp: Optional[GaussianProcessRegressor] = None, kernel: Optional[Any] = None, whitenoise: Optional[float] = None, alpha: Optional[float] = None, normalize_y: Optional[bool] = None) -> None:
        if gp is not None:
            if not isinstance(gp, GaussianProcessRegressor): raise TypeError("gp should be a GaussianProcessRegressor instance.")
            self._gps = [clone(gp) for _ in self._LSFs]; return
        if kernel is not None: self._kernel = kernel
        elif whitenoise is not None:
            if not isinstance(whitenoise, (int, float)) or isinstance(whitenoise, bool) or whitenoise < 0: raise ValueError("whitenoise should be non-negative.")
            self._kernel += WhiteKernel(noise_level=whitenoise)
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0: raise ValueError("alpha should be non-negative.")
            self._gp_alpha = float(alpha)
        if normalize_y is not None:
            if not isinstance(normalize_y, bool): raise TypeError("normalize_y should be a boolean.")
            self._gp_normalize_y = normalize_y
        self._gps = self._new_gps()

    def _sample_LHS(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        values = qmc.LatinHypercube(d=self._n_dim, seed=rng).random(n)
        return qmc.scale(values, self._sample_range[0], self._sample_range[1]) if self._sample_distribution == "uniform" else norm.ppf(values)
    def _sample_sobol(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        values = qmc.Sobol(d=self._n_dim, scramble=True, seed=rng).random(n)
        return qmc.scale(values, self._sample_range[0], self._sample_range[1]) if self._sample_distribution == "uniform" else norm.ppf(values)
    def set_initial_sampler(self, sampler: Optional[Callable[[int], np.ndarray]] = None, method: Optional[Literal["LHS", "Sobol"]] = None, distribution: Optional[Literal["uniform", "normal"]] = None, dist_range: Optional[Tuple[float, float]] = None) -> None:
        if sampler is not None:
            if not callable(sampler): raise TypeError("sampler should be callable.")
            self._initial_sampler = sampler; return
        method, distribution = method or "LHS", distribution or self._sample_distribution
        if method not in ("LHS", "Sobol") or distribution not in ("uniform", "normal"): raise ValueError("invalid method or distribution.")
        if distribution == "uniform":
            dist_range = dist_range or self._sample_range
            if not isinstance(dist_range, tuple) or len(dist_range) != 2 or dist_range[0] >= dist_range[1]: raise ValueError("uniform dist_range should be an increasing pair.")
        self._sample_distribution, self._sample_range = distribution, dist_range if distribution == "uniform" else None
        self._initial_sampler = self._sample_sobol if method == "Sobol" else self._sample_LHS
    def _initial(self, n: int, rng: np.random.Generator) -> np.ndarray:
        try: values = self._initial_sampler(n, rng=rng)
        except TypeError: values = self._initial_sampler(n)
        values = np.asarray(values, dtype=float)
        if values.shape != (n, self._n_dim) or not np.all(np.isfinite(values)): raise ValueError("sampler returned invalid samples.")
        return values

    def _evaluate(self, component: int, points: np.ndarray) -> np.ndarray:
        values = np.asarray([self._LSFs[component](p) for p in points], dtype=float)
        if values.shape != (len(points),) or not np.all(np.isfinite(values)): raise ValueError("each LSF should return one finite scalar per sample.")
        return values
    def _predict(self, population: np.ndarray, trains: list[np.ndarray], values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        means, stds = np.empty((self._n_components, len(population))), np.zeros((self._n_components, len(population)))
        for i, gp in enumerate(self._gps):
            distance, nearest = KDTree(trains[i]).query(population); unknown = distance >= 1e-8
            if np.any(unknown): means[i, unknown], stds[i, unknown] = gp.predict(population[unknown], return_std=True)
            means[i, ~unknown] = values[i][nearest[~unknown]]
        return means, stds
    def _failure(self, means: np.ndarray) -> np.ndarray: return np.all(means <= 0., axis=0) if self._gate == "AND" else np.any(means <= 0., axis=0)
    def _response(self, means: np.ndarray) -> np.ndarray: return np.max(means, axis=0) if self._gate == "AND" else np.min(means, axis=0)

    @overload
    def __call__(self, return_gp: Literal[False] = False, seed: Optional[int] = None, verbose: bool = False) -> float: ...
    @overload
    def __call__(self, return_gp: Literal[True], seed: Optional[int] = None, verbose: bool = False) -> tuple[float, tuple[GaussianProcessRegressor, ...]]: ...
    def __call__(self, return_gp: bool = False, seed: Optional[int] = None, verbose: bool = False):
        if not isinstance(return_gp, bool) or not isinstance(verbose, bool): raise TypeError("return_gp and verbose should be booleans.")
        with _verbose_logging(verbose):
            rng = np.random.default_rng(seed); population = rng.normal(size=(self._n_initial_population, self._n_dim))
            initial = self._initial(self._n_initial_DoEs, rng)
            trains = [initial.copy() for _ in self._LSFs]; values = [self._evaluate(i, initial) for i in range(self._n_components)]
            pf, cov, min_u = 1., np.inf, np.inf
            for ci in range(self._max_iterations_cov):
                for ui in range(self._max_iterations_u):
                    for i, gp in enumerate(self._gps): gp.fit(trains[i], values[i])
                    means, stds = self._predict(population, trains, values)
                    control = np.argmax(means, axis=0) if self._gate == "AND" else np.argmin(means, axis=0)
                    columns = np.arange(len(population)); mu, sigma = means[control, columns], stds[control, columns]
                    u = np.divide(np.abs(mu), sigma, out=np.full(len(population), np.inf), where=sigma > 1e-12)
                    nxt = int(np.argmin(u)); min_u, pf = float(u[nxt]), float(np.mean(self._failure(means)))
                    _LOGGER.info("CoV %s/%s, U %s/%s: min U=%.6g, pf=%.6g", ci + 1, self._max_iterations_cov, ui + 1, self._max_iterations_u, min_u, pf)
                    if min_u >= self._u_stop: break
                    comp, point = int(control[nxt]), population[nxt:nxt + 1]
                    trains[comp] = np.vstack((trains[comp], point)); values[comp] = np.hstack((values[comp], self._evaluate(comp, point)))
                if pf <= 0.: cov = np.inf; break
                cov = float(np.sqrt((1. - pf) / (pf * len(population))))
                if cov <= self._cov_stop: break
                population = np.vstack((population, rng.normal(size=(self._n_initial_population, self._n_dim))))
            for i, gp in enumerate(self._gps): gp.fit(trains[i], values[i])
            means, stds = self._predict(population, trains, values); pf = float(np.mean(self._failure(means)))
            cov = np.inf if pf <= 0. else float(np.sqrt((1. - pf) / (pf * len(population))))
            self._result = {"pf": pf, "n_called": tuple(len(v) for v in values), "n_population": len(population), "min_u_value": min_u, "cov": cov, "gate": self._gate}
            self._train_data = {"initial_DoE": tuple((trains[i][:self._n_initial_DoEs], values[i][:self._n_initial_DoEs]) for i in range(self._n_components)), "DoE": tuple((trains[i][self._n_initial_DoEs:], values[i][self._n_initial_DoEs:]) for i in range(self._n_components)), "population": (population, means, stds, self._response(means))}
            _LOGGER.info("Finished AK-SYS: pf=%.6g, cov=%.6g", pf, cov)
            return (pf, tuple(self._gps)) if return_gp else pf
    def wipe(self) -> None:
        self._gps = self._new_gps(); self._result = self._train_data = None
