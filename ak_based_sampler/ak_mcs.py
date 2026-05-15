import logging
from contextlib import contextmanager
from types import MappingProxyType
from typing import Tuple, Optional, Callable, Literal, Any, Iterator, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm, qmc
from scipy.spatial import KDTree

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


__all__ = ["AK_MCS"]

_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


@contextmanager
def _verbose_logging(enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[AK_MCS] %(message)s"))
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


class AK_MCS:
    def __init__(self, dim: int, LSF: Callable[[ArrayLike], ArrayLike]):
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim should be a positive integer.")
        if not callable(LSF):
            raise TypeError("LSF should be a callable function.")
        self._n_dim = dim
        self._LSF = LSF

        # params of the initial setting
        self._n_initial_population = 1_000_000
        self._n_initial_DoEs = 20

        # params of the stopping criteria
        self._u_stop = 2.0
        self._cov_stop = 0.05
        self._max_iterations_u = 100
        self._max_iterations_cov = 100

        # params of the surrogate model
        self._kernel = C(1.0) * RBF(length_scale=np.ones(dim)) 
        self._gp_alpha = 1e-10
        self._gp_normalize_y = True
        self._gp = GaussianProcessRegressor(kernel=self._kernel, alpha=self._gp_alpha, normalize_y=self._gp_normalize_y)

        # initial sampler
        self._sample_distribution = 'uniform'
        self._sample_range = (-3.0, 3.0)
        self._initial_sampler = self._sample_LHS

        # result
        self._result = None
        self._train_data = None

    @property
    def n_initial_population(self) -> int:
        return self._n_initial_population
    
    @n_initial_population.setter
    def n_initial_population(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("the number of initial population should be a positive integer.")
        self._n_initial_population = value

    @property
    def n_initial_DoEs(self) -> int:
        return self._n_initial_DoEs 
    
    @n_initial_DoEs.setter
    def n_initial_DoEs(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("the number of initial DoEs should be a positive integer.")
        self._n_initial_DoEs = value

    @property
    def u_stop(self) -> float:
        return self._u_stop

    @u_stop.setter
    def u_stop(self, value: float):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("stopping criteria on U value should be a positive number.")
        self._u_stop = float(value)

    @property
    def cov_stop(self) -> float:
        return self._cov_stop
    
    @cov_stop.setter
    def cov_stop(self, value: float):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError("stopping criteria on CoV should be a positive number.")
        self._cov_stop = float(value)

    @property
    def max_iterations_u(self) -> int:
        return self._max_iterations_u
    
    @max_iterations_u.setter
    def max_iterations_u(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("the maximum number of iterations for U value should be a positive integer.")
        self._max_iterations_u = value

    @property
    def max_iterations_cov(self) -> int:
        return self._max_iterations_cov

    @max_iterations_cov.setter
    def max_iterations_cov(self, value: int):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("the maximum number of iterations for CoV should be a positive integer.")
        self._max_iterations_cov = value

    @property
    def result(self):
        if self._result is None:
            return None
        return MappingProxyType(self._result)

    @property
    def train_data(self):
        if self._train_data is None:
            return None
        return MappingProxyType(self._train_data)

    def set_parameters(
        self,
        n_initial_population: Optional[int] = None,
        n_initial_DoEs: Optional[int] = None,
        u_stop: Optional[float] = None,
        cov_stop: Optional[float] = None,
        max_iterations_u: Optional[int] = None,
        max_iterations_cov: Optional[int] = None,
    ):
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
        if max_iterations_cov is not None:
            self.max_iterations_cov = max_iterations_cov
            
    def set_surrogate_model(self, gp: Optional[GaussianProcessRegressor] = None, kernel: Optional[Any] = None, whitenoise: Optional[float] = None, alpha: Optional[float] = None, normalize_y: Optional[bool] = None):
        if gp is not None:
            if not isinstance(gp, GaussianProcessRegressor):
                raise TypeError("gp should be a GaussianProcessRegressor instance.")
            self._gp = gp
        else:
            if kernel is not None:
                self._kernel = kernel
            elif whitenoise is not None:
                if not isinstance(whitenoise, (int, float)) or isinstance(whitenoise, bool) or whitenoise < 0:
                    raise ValueError("whitenoise should be a non-negative number.")
                self._kernel = self._kernel + WhiteKernel(noise_level=whitenoise)
            if alpha is not None:
                if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or alpha < 0:
                    raise ValueError("alpha should be a non-negative number.")
                self._gp_alpha = alpha
            if normalize_y is not None:
                if not isinstance(normalize_y, bool):
                    raise TypeError("normalize_y should be a boolean.")
                self._gp_normalize_y = normalize_y

            self._gp = GaussianProcessRegressor(kernel=self._kernel, alpha=self._gp_alpha, normalize_y=self._gp_normalize_y)

    def _sample_LHS(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=self._n_dim, seed=rng)
        samples = sampler.random(n=n_samples)
        if self._sample_distribution == 'uniform':
            samples = qmc.scale(samples, l_bounds=self._sample_range[0], u_bounds=self._sample_range[1])
        elif self._sample_distribution == 'normal':
            samples = norm.ppf(samples, loc=0.0, scale=1.0)
        else:
            raise ValueError("Invalid sample distribution. Supported distributions are 'uniform' and 'normal'.")
        return samples
            
    def _sample_sobol(self, n_samples: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        sampler = qmc.Sobol(d=self._n_dim, scramble=True, seed=rng)
        samples = sampler.random(n=n_samples)
        if self._sample_distribution == 'uniform':
            samples = qmc.scale(samples, l_bounds=self._sample_range[0], u_bounds=self._sample_range[1])
        elif self._sample_distribution == 'normal':
            samples = norm.ppf(samples, loc=0.0, scale=1.0)
        else:
            raise ValueError("Invalid sample distribution. Supported distributions are 'uniform' and 'normal'.")
        return samples
    
    def set_initial_sampler(self, sampler: Optional[Callable[[int], np.ndarray]] = None, method: Optional[Literal['LHS', 'Sobol']] = None, distribution: Optional[Literal['uniform', 'normal']] = None, dist_range: Optional[Tuple[float, float]] = None):
        if sampler is not None:
            if not callable(sampler):
                raise TypeError("sampler should be callable.")
            self._initial_sampler = sampler
        else:
            method = method or 'LHS'
            distribution = distribution or self._sample_distribution
            if method not in ['LHS', 'Sobol']:
                raise ValueError("method should be either 'LHS' or 'Sobol'.")
            if distribution not in ['uniform', 'normal']:
                raise ValueError("distribution should be either 'uniform' or 'normal'.")
            if distribution == 'uniform':
                dist_range = dist_range or self._sample_range
                if (
                    not isinstance(dist_range, tuple)
                    or len(dist_range) != 2
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in dist_range)
                    or dist_range[0] >= dist_range[1]
                ):
                    raise ValueError("for uniform distribution, range should be a tuple of two numbers (lower bound, upper bound).")
            else:
                dist_range = None
            self._sample_distribution = distribution
            self._sample_range = dist_range
            self._initial_sampler = self._sample_sobol if method == 'Sobol' else self._sample_LHS

    def _sample_initial(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        try:
            samples = self._initial_sampler(n_samples, rng=rng)
        except TypeError:
            samples = self._initial_sampler(n_samples)
        samples = np.asarray(samples, dtype=float)
        if samples.shape != (n_samples, self._n_dim):
            raise ValueError(f"sampler should return an array with shape ({n_samples}, {self._n_dim}).")
        if not np.all(np.isfinite(samples)):
            raise ValueError("sampler should return only finite values.")
        return samples

    def _evaluate_lsf(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray([self._LSF(point) for point in points], dtype=float)
        if values.shape != (points.shape[0],) or not np.all(np.isfinite(values)):
            raise ValueError("LSF should return one finite scalar value per sample.")
        return values

    def _estimate_population(
        self,
        u_population: np.ndarray,
        u_train: np.ndarray,
        G_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_population = u_population.shape[0]
        tree = KDTree(u_train)
        distances, nearest_train_indices = tree.query(u_population)
        unknown_mask = distances >= 1e-8

        G_estimate = np.empty(n_population, dtype=float)
        sigma_estimate = np.zeros(n_population, dtype=float)
        u_values = np.full(n_population, np.inf)

        if np.any(unknown_mask):
            G_pred, sigma = self._gp.predict(u_population[unknown_mask], return_std=True)
            G_estimate[unknown_mask] = G_pred
            sigma_estimate[unknown_mask] = sigma
            u_values[unknown_mask] = np.where(sigma > 1e-12, np.abs(G_pred) / sigma, np.inf)

        if np.any(~unknown_mask):
            G_estimate[~unknown_mask] = G_train[nearest_train_indices[~unknown_mask]]

        return G_estimate, sigma_estimate, u_values

    @overload
    def __call__(self, return_gp: Literal[False] = False, seed: Optional[int] = None, verbose: bool = False) -> float: ...

    @overload
    def __call__(self, return_gp: Literal[True], seed: Optional[int] = None, verbose: bool = False) -> Tuple[float, GaussianProcessRegressor]: ...
    
    def __call__(self, return_gp: bool = False, seed: Optional[int] = None, verbose: bool = False):
        if not isinstance(return_gp, bool):
            raise TypeError("return_gp should be a boolean.")
        if not isinstance(verbose, bool):
            raise TypeError("verbose should be a boolean.")

        with _verbose_logging(verbose):
            rng = np.random.default_rng(seed)
            _LOGGER.info("Start AK-MCS: dim=%s, seed=%s", self._n_dim, seed)

            # Step 1: Generate a Monte Carlo population
            n_population = self._n_initial_population
            _LOGGER.info("Step 1: generating Monte Carlo population (n=%s)", n_population)
            u_population = rng.normal(size=(n_population, self._n_dim))

            # Step 2: Generate initial DoEs and evaluate the LSF
            _LOGGER.info("Step 2: generating and evaluating initial DoEs (n=%s)", self._n_initial_DoEs)
            u_train = self._sample_initial(self._n_initial_DoEs, rng)
            G_train = self._evaluate_lsf(u_train)

            pf_pred = 1.0
            iteration_u = 0
            iteration_cov = 0
            min_u_value = np.inf
            cov_estimate = np.inf
            G_estimate = np.full(n_population, np.nan)
            sigma_estimate = np.full(n_population, np.nan)

            while iteration_cov < self._max_iterations_cov:
                _LOGGER.info(
                    "CoV iteration %s/%s: population=%s, training_points=%s",
                    iteration_cov + 1,
                    self._max_iterations_cov,
                    n_population,
                    u_train.shape[0],
                )
                while iteration_u < self._max_iterations_u:

                    # Step 3: Fit the surrogate model
                    _LOGGER.info("Step 3: fitting surrogate model (U iteration %s/%s)", iteration_u + 1, self._max_iterations_u)
                    self._gp.fit(u_train, G_train)

                    # Step4: Predict LSF values for the Monte Carlo population
                    _LOGGER.info("Step 4: predicting population response")
                    G_estimate, sigma_estimate, u_values = self._estimate_population(u_population, u_train, G_train)

                    # Step 5: Identify the next training point
                    _LOGGER.info("Step 5: selecting next training point")
                    nxt_idx = np.argmin(u_values)

                    # Estimate failure probability
                    pf_pred = np.mean(G_estimate <= 0.0)

                    # Step 6: Check stopping criteria on U value
                    min_u_value = u_values[nxt_idx]
                    _LOGGER.info("Step 6: min U=%.6g, pf=%.6g", min_u_value, pf_pred)
                    if min_u_value > self._u_stop:
                        _LOGGER.info("U stopping criterion reached: %.6g > %.6g", min_u_value, self._u_stop)
                        break

                    # Step 7: Update training data
                    _LOGGER.info("Step 7: evaluating LSF at selected point")
                    u_train = np.vstack((u_train, u_population[nxt_idx]))
                    G_train = np.hstack((G_train, self._evaluate_lsf(u_population[nxt_idx: nxt_idx + 1])))

                    iteration_u += 1

                    # Record train data

                # Step 8: Estimate CoV and check stopping criteria
                if pf_pred <= 0.0:
                    cov_estimate = np.inf
                    _LOGGER.info("Step 8: pf is zero; CoV is infinite")
                    break
                cov_estimate = np.sqrt((1 - pf_pred) / (pf_pred * n_population))
                _LOGGER.info("Step 8: estimated CoV=%.6g", cov_estimate)
                if cov_estimate <= self._cov_stop:
                    _LOGGER.info("CoV stopping criterion reached: %.6g <= %.6g", cov_estimate, self._cov_stop)
                    break

                # Step 9: Update the Monte Carlo population for the next iteration
                n_population = max(
                    u_train.shape[0],
                    int(np.ceil((1 - pf_pred) / (pf_pred * self._cov_stop**2))),
                )
                n_append = n_population - u_train.shape[0]
                _LOGGER.info("Step 9: updating Monte Carlo population (new n=%s, appended=%s)", n_population, n_append)
                u_population_append = rng.normal(size=(n_append, self._n_dim)) if n_append > 0 else np.empty((0, self._n_dim))
                u_population = np.vstack((u_train, u_population_append))

                iteration_cov += 1

            _LOGGER.info("Finalizing population estimates")
            self._gp.fit(u_train, G_train)
            G_estimate, sigma_estimate, _ = self._estimate_population(u_population, u_train, G_train)
            pf_pred = np.mean(G_estimate <= 0.0)
            if pf_pred <= 0.0:
                cov_estimate = np.inf
            else:
                cov_estimate = np.sqrt((1 - pf_pred) / (pf_pred * n_population))

            self._result = {
                "pf": pf_pred,
                "n_called": u_train.shape[0],
                "n_population": n_population,
                "min_u_value": min_u_value,
                "cov": cov_estimate
            }

            self._train_data = {
                "initial_DoE": (u_train[:self._n_initial_DoEs], G_train[:self._n_initial_DoEs]),
                "DoE": (u_train[self._n_initial_DoEs:], G_train[self._n_initial_DoEs:]),
                "population": (u_population, G_estimate, sigma_estimate)
            }

            _LOGGER.info(
                "Finished AK-MCS: pf=%.6g, n_called=%s, n_population=%s, min_u=%.6g, cov=%.6g",
                pf_pred,
                u_train.shape[0],
                n_population,
                min_u_value,
                cov_estimate,
            )

            if return_gp:
                return pf_pred, self._gp
            return pf_pred
    
    def wipe(self) -> None:
        self.set_surrogate_model()
        self._result = None
        self._train_data = None
