from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sedinfer.units import MassNormalization


@dataclass(frozen=True)
class ModelPhotometry:
    """Predicted photometry from a backend.

    ``flux`` must be a one-dimensional vector in the backend's declared flux
    normalization. ``band_names`` names each element so likelihoods can align
    model and data without assuming positional agreement.
    """

    band_names: Sequence[str]
    flux: np.ndarray
    metadata: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "band_names", tuple(str(name) for name in self.band_names))
        flux = np.asarray(self.flux, dtype=float)
        if flux.ndim != 1:
            raise ValueError("ModelPhotometry.flux must be one-dimensional.")
        if len(self.band_names) != flux.size:
            raise ValueError("band_names length must match flux length.")
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "metadata", {} if self.metadata is None else dict(self.metadata))


class SEDBackend:
    """Common backend interface.

    Backends must declare ``mass_normalization`` and implement
    ``predict_photometry(params, filters)``. They should not apply any
    likelihood-specific mass scaling; that is handled centrally by
    ``GaussianPhotometricLikelihood`` when the normalization is
    ``PER_SOLAR_MASS``.
    """

    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params: Mapping[str, float], filters) -> ModelPhotometry:
        raise NotImplementedError
