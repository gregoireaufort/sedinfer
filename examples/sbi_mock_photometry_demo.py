"""Tiny MAF posterior demo with mock photometry.

Requires optional dependencies:

    pip install torch nflows
"""

from __future__ import annotations

import numpy as np

from inftools.sbi import simulate_training_set, train_maf_posterior
from sedinfer.backends.base import ModelPhotometry, SEDBackend
from sedinfer.data import SEDDataset
from sedinfer.likelihood import GaussianPhotometricLikelihood
from sedinfer.parameters import ParameterSpace
from sedinfer.priors import UniformPrior
from sedinfer.units import MassNormalization


class LinearColorBackend(SEDBackend):
    mass_normalization = MassNormalization.ABSOLUTE

    def predict_photometry(self, params, filters):
        del filters
        z = float(params["z"])
        dust = float(params["dust2"])
        flux = np.array([1.0 + z - 0.2 * dust, 0.8 + 0.5 * z + dust], dtype=float)
        return ModelPhotometry(band_names=["g", "r"], flux=flux)


def main() -> None:
    rng = np.random.default_rng(123)
    parameter_space = ParameterSpace(
        names=["z", "dust2"],
        priors={"z": UniformPrior(0.0, 2.0), "dust2": UniformPrior(0.0, 1.0)},
    )
    dataset = SEDDataset(["g", "r"], flux=np.zeros(2), sigma=np.ones(2))
    likelihood = GaussianPhotometricLikelihood(LinearColorBackend(), dataset, parameter_space)

    def noise_fn(flux):
        sigma_floor = 0.02
        frac_error = 0.05
        return sigma_floor + frac_error * np.abs(flux)

    theta_train, x_train = simulate_training_set(parameter_space, likelihood, n=512, noise_fn=noise_fn, rng=rng)
    estimator = train_maf_posterior(
        theta_train,
        x_train,
        hidden_features=32,
        num_transforms=2,
        num_blocks=1,
        epochs=20,
        batch_size=128,
        device="cpu",
        seed=123,
    )

    theta_true = np.array([0.8, 0.3])
    x_obs = likelihood.simulate(theta_true, noise_fn=noise_fn, rng=rng)
    samples = estimator.sample(x_obs, num_samples=2000)
    print("true theta:", theta_true)
    print("observed flux:", x_obs)
    print("posterior mean:", samples.mean(axis=0))
    print("posterior std:", samples.std(axis=0))


if __name__ == "__main__":
    main()
