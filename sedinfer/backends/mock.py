from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sedinfer.backends.base import ModelPhotometry, SEDBackend
from sedinfer.units import MassNormalization


@dataclass
class MockBackend(SEDBackend):
    """Deterministic backend for tests and examples."""

    flux: Sequence[float]
    band_names: Sequence[str] | None = None
    mass_normalization: MassNormalization = MassNormalization.ABSOLUTE
    fail_on_call: bool = False

    def predict_photometry(self, params: Mapping[str, float], filters) -> ModelPhotometry:
        del params
        if self.fail_on_call:
            raise FloatingPointError("Mock backend numerical failure.")
        flux = np.asarray(self.flux, dtype=float)
        if self.band_names is not None:
            names = tuple(str(name) for name in self.band_names)
        elif hasattr(filters, "names"):
            names = tuple(filters.names)
        else:
            names = tuple(str(i) for i in range(flux.size))
        return ModelPhotometry(band_names=names, flux=flux)
