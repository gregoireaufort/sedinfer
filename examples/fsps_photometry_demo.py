"""Minimal FSPS photometry demo.

Requires python-fsps, sedpy, and a configured SPS_HOME pointing at FSPS grids.
"""

from __future__ import annotations

import numpy as np

from sedinfer.backends.fsps import FSPSBackend
from sedinfer.filters import FilterSet


def main() -> None:
    from sedpy.observate import load_filters

    filters = FilterSet(load_filters(["sdss_g0", "sdss_r0", "sdss_i0"]), names=["sdss_g0", "sdss_r0", "sdss_i0"])
    backend = FSPSBackend()
    params = {
        "zred": 0.1,
        "logzsol": -0.3,
        "dust2": 0.2,
        "tabular_time_gyr": np.array([0.01, 1.0, 5.0]),
        "tabular_sfr_msun_per_yr": np.array([1.0, 1.0, 0.2]),
    }
    phot = backend.predict_photometry(params, filters)
    for band, flux in zip(phot.band_names, phot.flux):
        print(f"{band}: {flux:.8e} maggies")


if __name__ == "__main__":
    main()
