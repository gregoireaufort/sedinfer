from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Prior:
    """Scalar prior interface used by ``ParameterSpace``."""

    def logpdf(self, x: float) -> float:
        raise NotImplementedError

    def sample(self, rng: np.random.Generator, size=None):
        raise NotImplementedError


@dataclass(frozen=True)
class UniformPrior(Prior):
    """Uniform prior on the closed interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError("UniformPrior requires high > low.")

    def logpdf(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return -np.inf
        if x < self.low or x > self.high:
            return -np.inf
        return float(-np.log(self.high - self.low))

    def sample(self, rng: np.random.Generator, size=None):
        return rng.uniform(self.low, self.high, size=size)


@dataclass(frozen=True)
class NormalPrior(Prior):
    """Gaussian prior with mean ``mu`` and standard deviation ``sigma``."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if not self.sigma > 0:
            raise ValueError("NormalPrior requires sigma > 0.")

    def logpdf(self, x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        z = (float(x) - self.mu) / self.sigma
        return float(-0.5 * z**2 - np.log(self.sigma) - 0.5 * np.log(2.0 * np.pi))

    def sample(self, rng: np.random.Generator, size=None):
        return rng.normal(self.mu, self.sigma, size=size)


@dataclass(frozen=True)
class LogUniformPrior(Prior):
    """Log-uniform prior on the closed positive interval ``[low, high]``."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.low > 0 or not self.high > self.low:
            raise ValueError("LogUniformPrior requires 0 < low < high.")

    def logpdf(self, x: float) -> float:
        x = float(x)
        if not np.isfinite(x):
            return -np.inf
        if x <= 0 or x < self.low or x > self.high:
            return -np.inf
        return float(-np.log(np.log(self.high) - np.log(self.low)) - np.log(x))

    def sample(self, rng: np.random.Generator, size=None):
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size=size))


@dataclass(frozen=True)
class DeltaPrior(Prior):
    """Point-mass prior at ``value`` with configurable numerical tolerance."""

    value: float
    rtol: float = 1e-12
    atol: float = 1e-12

    def logpdf(self, x: float) -> float:
        if not np.isfinite(x):
            return -np.inf
        return 0.0 if np.isclose(float(x), self.value, rtol=self.rtol, atol=self.atol) else -np.inf

    def sample(self, rng: np.random.Generator, size=None):
        del rng
        if size is None:
            return float(self.value)
        return np.full(size, self.value, dtype=float)
