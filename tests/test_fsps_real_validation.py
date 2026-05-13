import importlib.util
import os
from pathlib import Path

import pytest


@pytest.mark.fsps
def test_validate_fsps_backend_against_direct_calculation():
    for package in ("fsps", "sedpy", "astropy"):
        if importlib.util.find_spec(package) is None:
            pytest.skip(f"{package} is not importable.")
    sps_home = os.environ.get("SPS_HOME")
    if not sps_home or not Path(sps_home).exists():
        pytest.skip("SPS_HOME is not configured or does not exist.")

    from examples.validate_fsps_backend import DEFAULT_FILTER_NAMES, load_filter_set, run_validation

    try:
        load_filter_set(DEFAULT_FILTER_NAMES)
    except Exception as exc:
        pytest.skip(f"Requested sedpy filters could not be loaded: {exc}")

    result = run_validation(DEFAULT_FILTER_NAMES)
    assert result["max_relative_flux_difference"] <= 1e-10
    assert result["max_ab_magnitude_difference"] <= 1e-8
