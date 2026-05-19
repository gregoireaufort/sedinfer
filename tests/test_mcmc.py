import numpy as np
import pytest

from inftools.core import Posterior


def test_run_emcee_seed_reproducible_when_emcee_installed():
    pytest.importorskip("emcee")
    from inftools.mcmc import run_emcee

    posterior = Posterior(
        log_prob_fn=lambda theta: -0.5 * float(theta[0] ** 2),
        dim=1,
        theta_names=["x"],
    )

    result_a = run_emcee(posterior, x0=np.array([0.1]), nwalkers=8, nsteps=12, seed=123, progress=False)
    result_b = run_emcee(posterior, x0=np.array([0.1]), nwalkers=8, nsteps=12, seed=123, progress=False)

    assert np.allclose(result_a.samples, result_b.samples)
    assert np.allclose(result_a.logp, result_b.logp)
    assert np.allclose(result_a.meta["raw_chain"], result_b.meta["raw_chain"])


def test_run_emcee_rejects_rng_and_seed_together_when_emcee_installed():
    pytest.importorskip("emcee")
    from inftools.mcmc import run_emcee

    posterior = Posterior(log_prob_fn=lambda theta: -0.5 * float(theta[0] ** 2), dim=1)
    with pytest.raises(ValueError, match="either rng or seed"):
        run_emcee(
            posterior,
            x0=np.array([0.0]),
            nwalkers=8,
            nsteps=2,
            rng=np.random.default_rng(1),
            seed=1,
            progress=False,
        )
