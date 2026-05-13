from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass
class SEDDataset:
    """Observed photometry for one SED.

    Parameters
    ----------
    band_names
        Names of the photometric bands. These names define the data order and
        are used by likelihoods to align model photometry.
    flux, sigma
        One-dimensional observed fluxes and Gaussian 1-sigma uncertainties in
        the same linear flux units. The package does not reinterpret units.
    mask
        Optional boolean array where ``True`` means the band is usable. Invalid
        or non-positive uncertainty values are also excluded from active arrays.
    metadata
        Extra caller-owned context, such as a ``FilterSet`` for backends.
    """

    band_names: Sequence[str]
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.band_names = tuple(str(name) for name in self.band_names)
        self.flux = np.asarray(self.flux, dtype=float)
        self.sigma = np.asarray(self.sigma, dtype=float)
        if self.flux.shape != self.sigma.shape:
            raise ValueError("flux and sigma must have the same shape.")
        if self.flux.ndim != 1:
            raise ValueError("flux and sigma must be one-dimensional arrays.")
        if len(self.band_names) != self.flux.size:
            raise ValueError("band_names length must match flux length.")
        if self.mask is not None:
            self.mask = np.asarray(self.mask, dtype=bool)
            if self.mask.shape != self.flux.shape:
                raise ValueError("mask must have the same shape as flux.")
        self.metadata = dict(self.metadata)

    @property
    def active_mask(self) -> np.ndarray:
        finite = np.isfinite(self.flux) & np.isfinite(self.sigma) & (self.sigma > 0.0)
        if self.mask is None:
            return finite
        return np.asarray(self.mask, dtype=bool) & finite

    @property
    def active_indices(self) -> np.ndarray:
        return np.where(self.active_mask)[0]

    @property
    def active_flux(self) -> np.ndarray:
        return self.flux[self.active_mask]

    @property
    def active_sigma(self) -> np.ndarray:
        return self.sigma[self.active_mask]

    @property
    def active_band_names(self) -> tuple[str, ...]:
        idx = self.active_indices
        return tuple(self.band_names[i] for i in idx)

    def active_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        idx = self.active_indices
        return self.flux[idx], self.sigma[idx], idx, tuple(self.band_names[i] for i in idx)
