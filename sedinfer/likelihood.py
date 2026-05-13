from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Sequence

import numpy as np

from sedinfer.data import SEDDataset
from sedinfer.parameters import ParameterSpace
from sedinfer.units import MassNormalization


class PhotometricSimulationError(RuntimeError):
    """Controlled error raised when photometric simulation cannot produce finite fluxes."""


@dataclass
class GaussianPhotometricLikelihood:
    """Backend-agnostic Gaussian photometric log posterior for one SED.

    The returned value is ``log_prior(theta) + log_likelihood(data | theta)``.
    The likelihood aligns model and observed photometry by band name, applies an
    optional uncertainty floor in quadrature, and applies ``10**log10_mass`` only
    when the backend explicitly declares ``MassNormalization.PER_SOLAR_MASS``.
    Backend numerical failures and non-finite model fluxes return ``-inf``;
    configuration errors such as missing mass parameters or shape mismatches
    raise clear exceptions.
    """

    backend: object
    dataset: SEDDataset
    parameter_space: ParameterSpace
    filters: object | None = None
    sigma_floor: float | None = None

    def log_prob(self, theta: Sequence[float]) -> float:
        return self._log_prob_checked(theta)

    def _log_prob_checked(self, theta: Sequence[float]) -> float:
        theta = np.asarray(theta, dtype=float)
        log_prior = self.parameter_space.log_prior(theta)
        if not np.isfinite(log_prior):
            return -np.inf

        f_obs, sigma, idx, active_bands = self.dataset.active_arrays()
        if self.sigma_floor is not None:
            floor = float(self.sigma_floor)
            if floor < 0.0:
                raise ValueError("sigma_floor must be non-negative.")
            sigma = np.sqrt(sigma**2 + floor**2)

        try:
            model_flux = self._predict_active_model_flux(theta, idx=idx, active_bands=active_bands)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return -np.inf

        if model_flux.shape != f_obs.shape:
            raise ValueError(f"Model flux shape {model_flux.shape} does not match data shape {f_obs.shape}.")
        if not np.all(np.isfinite(model_flux)):
            return -np.inf

        residual = (f_obs - model_flux) / sigma
        logdet = np.sum(np.log(2.0 * np.pi * sigma**2))
        return float(log_prior - 0.5 * (np.sum(residual**2) + logdet))

    def simulate(self, theta: Sequence[float], noise_fn, rng: np.random.Generator | None = None) -> np.ndarray:
        """Simulate flux-like observations for active/masked bands.

        ``theta`` may be a single vector with shape ``(dim,)`` or a batch with
        shape ``(n, dim)``. The simulator returns the same active-band vector
        convention consumed by ``log_prob``. ``noise_fn`` is called with the
        noiseless active flux and must return Gaussian sigma values with the
        same shape. Extended signatures accepting ``theta=`` and/or ``rng=`` are
        also supported.
        """

        if rng is None:
            rng = np.random.default_rng()
        theta_arr = np.asarray(theta, dtype=float)
        single = theta_arr.ndim == 1
        if single:
            theta_batch = theta_arr[None, :]
        elif theta_arr.ndim == 2:
            theta_batch = theta_arr
        else:
            raise ValueError("theta must have shape (dim,) or (n, dim).")

        _, _, idx, active_bands = self.dataset.active_arrays()
        draws = []
        for row in theta_batch:
            try:
                flux = self._predict_active_model_flux(row, idx=idx, active_bands=active_bands)
            except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                raise PhotometricSimulationError(f"Backend numerical failure during simulation: {exc}") from exc
            if not np.all(np.isfinite(flux)):
                raise PhotometricSimulationError("Backend produced non-finite noiseless flux.")
            sigma = np.asarray(_call_noise_fn(noise_fn, flux, theta=row, rng=rng), dtype=float)
            if sigma.shape != flux.shape:
                raise ValueError(f"noise_fn returned sigma shape {sigma.shape}; expected {flux.shape}.")
            if not np.all(np.isfinite(sigma)) or np.any(sigma < 0.0):
                raise ValueError("noise_fn must return finite non-negative sigma values.")
            draws.append(flux + rng.normal(loc=0.0, scale=sigma, size=flux.shape))

        out = np.stack(draws, axis=0)
        return out[0] if single else out

    rvs = simulate

    def _predict_active_model_flux(
        self, theta: Sequence[float], idx: np.ndarray | None = None, active_bands: Sequence[str] | None = None
    ) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        params = self.parameter_space.to_dict(theta)
        backend_params = dict(params)
        log10_mass = None

        mass_norm = getattr(self.backend, "mass_normalization", None)
        if mass_norm is None:
            raise ValueError("Backend must declare mass_normalization.")
        mass_norm = MassNormalization(mass_norm)

        if mass_norm == MassNormalization.PER_SOLAR_MASS:
            if "log10_mass" not in params:
                raise ValueError(
                    "Backend returns PER_SOLAR_MASS photometry, but ParameterSpace is missing 'log10_mass'."
                )
            log10_mass = float(params["log10_mass"])
            backend_params.pop("log10_mass", None)

        if idx is None or active_bands is None:
            _, _, idx, active_bands = self.dataset.active_arrays()

        filters = self.filters
        if filters is None:
            filters = self.dataset.metadata.get("filters")

        model = self.backend.predict_photometry(backend_params, filters)
        model_flux = self._align_model_flux(model, idx, active_bands)
        if log10_mass is not None:
            model_flux = (10.0**log10_mass) * model_flux
        return np.asarray(model_flux, dtype=float)

    @staticmethod
    def _align_model_flux(model, active_indices: np.ndarray, active_bands: Sequence[str]) -> np.ndarray:
        del active_indices
        flux = np.asarray(model.flux, dtype=float)
        names = tuple(str(name) for name in getattr(model, "band_names", ()))
        if len(names) != flux.size:
            raise ValueError("ModelPhotometry band_names length must match flux length.")
        if len(set(names)) != len(names):
            raise ValueError("ModelPhotometry band_names must be unique.")
        active_bands = tuple(str(name) for name in active_bands)
        missing = [name for name in active_bands if name not in names]
        if missing:
            raise ValueError(f"Model photometry is missing active band(s): {', '.join(missing)}")
        lookup = {name: i for i, name in enumerate(names)}
        return np.asarray([flux[lookup[name]] for name in active_bands], dtype=float)


def _call_noise_fn(noise_fn, flux: np.ndarray, *, theta: np.ndarray, rng: np.random.Generator):
    try:
        params = inspect.signature(noise_fn).parameters
    except (TypeError, ValueError):
        return noise_fn(flux)
    kwargs = {}
    if "theta" in params:
        kwargs["theta"] = theta
    if "rng" in params:
        kwargs["rng"] = rng
    return noise_fn(flux, **kwargs)
