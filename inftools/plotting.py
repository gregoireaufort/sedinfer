# inftools/plotting.py

from __future__ import annotations
from typing import Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .core import SamplingResult


def plot_trace(
    result: SamplingResult,
    param_names: Optional[Sequence[str]] = None,
    chain: Optional[np.ndarray] = None,
):
    """
    Simple trace plots for each parameter.

    If you used emcee and have the raw chain in result.meta["raw_chain"],
    pass that as `chain`; otherwise it just plots the flattened samples.
    """
    if chain is not None:
        # chain shape (nsteps, nwalkers, ndim)
        nsteps, nwalkers, ndim = chain.shape
        xs = np.arange(nsteps)
        fig, axes = plt.subplots(ndim, 1, figsize=(8, 2.5 * ndim), sharex=True)
        if ndim == 1:
            axes = [axes]
        for d in range(ndim):
            ax = axes[d]
            for w in range(nwalkers):
                ax.plot(xs, chain[:, w, d], alpha=0.3)
            pname = param_names[d] if param_names else f"θ[{d}]"
            ax.set_ylabel(pname)
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        return fig, axes

    # Fallback: single chain (flattened samples)
    samples = result.samples
    ndim = samples.shape[1]
    xs = np.arange(samples.shape[0])
    fig, axes = plt.subplots(ndim, 1, figsize=(8, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    for d in range(ndim):
        ax = axes[d]
        ax.plot(xs, samples[:, d], alpha=0.7)
        pname = param_names[d] if param_names else f"θ[{d}]"
        ax.set_ylabel(pname)
    axes[-1].set_xlabel("iteration")
    fig.tight_layout()
    return fig, axes


def _gaussian_ellipse(mu, cov, nsig=1.0, **kwargs):
    """Draw nsig-σ ellipse for 2D Gaussian."""
    mu = np.asarray(mu)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    t = np.linspace(0, 2 * np.pi, 200)
    r = nsig * np.sqrt(vals)
    ellipse = (vecs @ (r[:, None] * np.array([np.cos(t), np.sin(t)]))).T
    x = mu[0] + ellipse[:, 0]
    y = mu[1] + ellipse[:, 1]
    plt.plot(x, y, **kwargs)


def plot_mcmc_vs_gaussian(
    samples: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    true_theta: Optional[Sequence[float]] = None,
    title: str = "",
):
    """
    Contour plot: MCMC KDE + Gaussian ellipse, with optional true point.

    samples: (Nsamples, 2)
    mu: MAP (2,)
    cov: (2,2)
    """
    samples = np.asarray(samples)
    x = samples[:, 0]
    y = samples[:, 1]

    kde = gaussian_kde(np.vstack([x, y]))

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= 0.2 * dx
    xmax += 0.2 * dx
    ymin -= 0.2 * dy
    ymax += 0.2 * dy

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, 100),
        np.linspace(ymin, ymax, 100),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    levels = np.linspace(0.1 * zz.max(), 0.9 * zz.max(), 5)
    plt.contour(xx, yy, zz, levels=levels, alpha=0.7, linewidths=1.0)

    _gaussian_ellipse(mu, cov, nsig=1.0, color="C1", lw=2, label="Gaussian 1σ")
    _gaussian_ellipse(mu, cov, nsig=2.0, color="C1", lw=1, ls="--", label="Gaussian 2σ")

    if true_theta is not None:
        plt.scatter(true_theta[0], true_theta[1], color="red", marker="*", s=100, label="True")

    plt.xlabel("θ[0]")
    plt.ylabel("θ[1]")
    plt.title(title)
    plt.legend()
    plt.tight_layout()