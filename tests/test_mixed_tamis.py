import numpy as np

from inftools.core import Posterior
from inftools.mixed_tamis import run_mixed_tamis
from sedinfer.parameters import ParameterSpace
from sedinfer.priors import ChoicePrior, UniformPrior


def test_mixed_tamis_returns_weighted_samples_on_discrete_support():
    space = ParameterSpace(
        names=("x", "template"),
        priors={
            "x": UniformPrior(-5.0, 5.0),
            "template": ChoicePrior([-1.0, 1.0]),
        },
    )

    def log_prob(theta):
        theta = np.asarray(theta, dtype=float)
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        x, template = theta
        template_bonus = 0.0 if template == 1.0 else -4.0
        return prior + template_bonus - 0.5 * ((x - template) / 0.35) ** 2

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_tamis(
        posterior,
        space,
        x0=np.asarray([0.0, -1.0]),
        n_comp=2,
        T_max=4,
        n_per_iter=80,
        alpha=30,
        seed=10,
    )

    assert result.samples.shape == (320, 2)
    assert set(np.unique(result.samples[:, 1])).issubset({-1.0, 1.0})
    assert np.all(np.isfinite(result.meta["weights_norm"]))
    assert np.isclose(np.sum(result.meta["weights_norm"]), 1.0)
    assert result.map_estimate[1] == 1.0
    assert np.all(np.isfinite(result.meta["betas"]))


def test_mixed_tamis_delegates_all_discrete_case_to_grid_sampler():
    space = ParameterSpace(
        names=("template",),
        priors={"template": ChoicePrior([0.0, 1.0])},
    )

    def log_prob(theta):
        prior = space.log_prior(theta)
        if not np.isfinite(prior):
            return -np.inf
        return prior if theta[0] == 1.0 else prior - 3.0

    posterior = Posterior(log_prob, dim=space.ndim, theta_names=space.names)
    result = run_mixed_tamis(posterior, space, seed=1)

    assert result.samples.shape == (2, 1)
    assert result.map_estimate[0] == 1.0
    assert np.isclose(np.sum(result.meta["weights_norm"]), 1.0)
