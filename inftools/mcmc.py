# inftools/mcmc.py

from __future__ import annotations
from typing import Optional
import numpy as np
import emcee

from .core import Posterior, SamplingResult, Array


def run_rw_metropolis(
    posterior: Posterior,
    x0: Array,
    nsteps: int = 5000,
    proposal_cov: Optional[Array] = None,
    burnin: int = 0,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> SamplingResult:
    """
    Simple random-walk Metropolis-Hastings with Gaussian proposal.

    proposal_cov:
        Proposal covariance matrix. If None, uses 0.1 * I.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.asarray(x0, dtype=float)
    dim = posterior.dim
    assert x.shape[0] == dim

    if proposal_cov is None:
        proposal_cov = 0.1 * np.eye(dim)
    proposal_cov = np.asarray(proposal_cov, dtype=float)

    chol = np.linalg.cholesky(proposal_cov)

    samples = np.zeros((nsteps, dim))
    logp = np.zeros(nsteps)

    current_lp = posterior.log_prob_fn(x)
    accept = 0

    for t in range(nsteps):
        step = chol @ rng.normal(size=dim)
        x_prop = x + step
        lp_prop = posterior.log_prob_fn(x_prop)

        if np.isfinite(lp_prop):
            alpha = np.exp(lp_prop - current_lp)
        else:
            alpha = 0.0

        if rng.uniform() < alpha:
            x = x_prop
            current_lp = lp_prop
            accept += 1

        samples[t] = x
        logp[t] = current_lp

    acc_rate = accept / nsteps

    # Burn-in / thinning
    idx = np.arange(nsteps)
    mask = idx >= burnin
    idx = idx[mask][::thin]

    samples_thin = samples[idx]
    logp_thin = logp[idx]

    cov = np.cov(samples_thin.T) if samples_thin.shape[0] > 1 else None

    map_est = samples_thin[np.argmax(logp_thin)] if samples_thin.size > 0 else None

    return SamplingResult(
        samples=samples_thin,
        logp=logp_thin,
        map_estimate=map_est,
        cov=cov,
        meta={"accept_rate": acc_rate, "proposal_cov": proposal_cov},
    )


def run_emcee(
    posterior: Posterior,
    x0: Array,
    nwalkers: int = 32,
    nsteps: int = 1000,
    pool=None,
    burnin: int = 0,
    thin: int = 1,
) -> SamplingResult:
    """
    Run emcee ensemble sampler given a Posterior.

    x0:
        Initial point (used to initialize all walkers in a small Gaussian ball).
    """
    x0 = np.asarray(x0, dtype=float)
    assert x0.shape[0] == posterior.dim

    pos = x0 + 1e-4 * np.random.randn(nwalkers, posterior.dim)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        posterior.dim,
        posterior.log_prob_fn,
        pool=pool,
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    chain = sampler.get_chain()
    logp_chain = sampler.get_log_prob()

    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    logp = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)

    cov = np.cov(samples.T) if samples.shape[0] > 1 else None
    map_est = samples[np.argmax(logp)] if samples.size > 0 else None

    return SamplingResult(
        samples=samples,
        logp=logp,
        map_estimate=map_est,
        cov=cov,
        meta={
            "raw_chain": chain,
            "raw_logp": logp_chain,
            "nwalkers": nwalkers,
            "nsteps": nsteps,
        },
    )