from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Callable

import numpy as np


def _require_sbi_dependencies():
    try:
        torch = importlib.import_module("torch")
        nflows = importlib.import_module("nflows")
        return torch, nflows
    except ImportError as exc:
        raise ImportError(
            "inftools.sbi requires optional dependencies torch and nflows. "
            "Install them with, for example: pip install torch nflows"
        ) from exc


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-8) -> "Standardizer":
        values = np.asarray(values, dtype=float)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.std + self.mean

    @property
    def log_abs_det_inverse(self) -> float:
        return float(np.sum(np.log(self.std)))


def build_maf(theta_dim: int, x_dim: int, hidden_features: int = 128, num_transforms: int = 5, num_blocks: int = 2):
    """Build a conditional MAF q(theta | x) using nflows."""

    torch, _ = _require_sbi_dependencies()
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import ReversePermutation

    transforms = []
    for _ in range(int(num_transforms)):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=int(theta_dim),
                hidden_features=int(hidden_features),
                context_features=int(x_dim),
                num_blocks=int(num_blocks),
                use_residual_blocks=False,
                random_mask=False,
                activation=torch.nn.functional.relu,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
        )
        transforms.append(ReversePermutation(features=int(theta_dim)))
    return Flow(CompositeTransform(transforms), StandardNormal([int(theta_dim)]))


class MAFPosteriorEstimator:
    """NumPy-facing conditional MAF posterior estimator q(theta | x)."""

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        hidden_features: int = 128,
        num_transforms: int = 5,
        num_blocks: int = 2,
        learning_rate: float = 1e-3,
        device: str | None = None,
        standardize: bool = True,
    ) -> None:
        torch, _ = _require_sbi_dependencies()
        self.torch = torch
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.learning_rate = float(learning_rate)
        self.standardize = bool(standardize)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        flow = build_maf(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_blocks=num_blocks,
        )
        self.flow = _prepare_flow_for_device(flow, torch, self.device)
        self.theta_standardizer: Standardizer | None = None
        self.x_standardizer: Standardizer | None = None
        self.history: dict[str, list[float]] = {"train_loss": []}

    def fit(
        self,
        theta_train: np.ndarray,
        x_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
        validation_split: float = 0.0,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        del validation_split
        torch = self.torch
        theta_train = _as_2d(theta_train, self.theta_dim, "theta_train")
        x_train = _as_2d(x_train, self.x_dim, "x_train")
        if theta_train.shape[0] != x_train.shape[0]:
            raise ValueError("theta_train and x_train must have the same number of rows.")
        if self.standardize:
            self.theta_standardizer = Standardizer.fit(theta_train)
            self.x_standardizer = Standardizer.fit(x_train)
            theta_fit = self.theta_standardizer.transform(theta_train)
            x_fit = self.x_standardizer.transform(x_train)
        else:
            self.theta_standardizer = Standardizer(np.zeros(self.theta_dim), np.ones(self.theta_dim))
            self.x_standardizer = Standardizer(np.zeros(self.x_dim), np.ones(self.x_dim))
            theta_fit = theta_train
            x_fit = x_train

        if seed is not None:
            torch.manual_seed(int(seed))
        theta_t = torch.as_tensor(theta_fit, dtype=torch.float32, device=self.device)
        x_t = torch.as_tensor(x_fit, dtype=torch.float32, device=self.device)
        dataset = torch.utils.data.TensorDataset(theta_t, x_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
        opt = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        self.history = {"train_loss": []}
        self.flow.train()
        for epoch in range(int(epochs)):
            losses = []
            for theta_b, x_b in loader:
                loss = -self.flow.log_prob(inputs=theta_b, context=x_b).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            mean_loss = float(np.mean(losses)) if losses else np.nan
            self.history["train_loss"].append(mean_loss)
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}: loss={mean_loss:.6g}")
        return self.history

    def sample(self, x_obs: np.ndarray, num_samples: int = 10000) -> np.ndarray:
        self._check_fitted()
        torch = self.torch
        x = _as_context_batch(x_obs, self.x_dim)
        x_std = self.x_standardizer.transform(x)
        context = torch.as_tensor(x_std, dtype=torch.float32, device=self.device)
        self.flow.eval()
        with torch.no_grad():
            samples_std = self.flow.sample(int(num_samples), context=context)
        samples_np = samples_std.detach().cpu().numpy()
        if samples_np.ndim == 3 and samples_np.shape[0] == 1:
            samples_np = samples_np[0]
        elif samples_np.ndim == 2:
            pass
        return self.theta_standardizer.inverse_transform(samples_np)

    def log_prob(self, theta: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        self._check_fitted()
        torch = self.torch
        theta_arr = _as_2d(theta, self.theta_dim, "theta")
        x_arr = _as_context_batch(x_obs, self.x_dim)
        if x_arr.shape[0] == 1 and theta_arr.shape[0] > 1:
            x_arr = np.repeat(x_arr, theta_arr.shape[0], axis=0)
        if x_arr.shape[0] != theta_arr.shape[0]:
            raise ValueError("x_obs must have one row or the same number of rows as theta.")
        theta_std = self.theta_standardizer.transform(theta_arr)
        x_std = self.x_standardizer.transform(x_arr)
        self.flow.eval()
        with torch.no_grad():
            lp_std = self.flow.log_prob(
                inputs=torch.as_tensor(theta_std, dtype=torch.float32, device=self.device),
                context=torch.as_tensor(x_std, dtype=torch.float32, device=self.device),
            )
        lp = lp_std.detach().cpu().numpy() - self.theta_standardizer.log_abs_det_inverse
        return lp[0] if np.asarray(theta).ndim == 1 else lp

    def _check_fitted(self) -> None:
        if self.theta_standardizer is None or self.x_standardizer is None:
            raise RuntimeError("Estimator must be fit before calling sample or log_prob.")


def simulate_training_set(
    parameter_space,
    simulator,
    n: int,
    noise_fn: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator | None = None,
    max_retries: int = 100,
    return_metadata: bool = False,
):
    """Sample theta from priors and simulate flux-like observations."""

    if rng is None:
        rng = np.random.default_rng()
    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative.")
    failures = []
    theta_rows = []
    x_rows = []
    attempts = 0
    while len(theta_rows) < n:
        if attempts - len(theta_rows) > int(max_retries):
            raise RuntimeError(
                f"Too many failed simulations: {len(failures)} failures while collecting {len(theta_rows)}/{n}."
            )
        attempts += 1
        theta = parameter_space.sample_prior(1, rng=rng)[0]
        try:
            x = _simulate_one(simulator, theta, noise_fn, rng)
            x = np.asarray(x, dtype=float)
            if x.ndim != 1 or not np.all(np.isfinite(x)):
                raise ValueError(f"Simulator returned invalid observation shape/content: shape={x.shape}.")
        except Exception as exc:
            failures.append({"theta": theta, "error": repr(exc)})
            continue
        theta_rows.append(theta)
        x_rows.append(x)
    theta_out = np.asarray(theta_rows, dtype=float)
    x_out = np.asarray(x_rows, dtype=float)
    if return_metadata:
        return theta_out, x_out, {"attempts": attempts, "failures": failures}
    return theta_out, x_out


def train_maf_posterior(theta_train: np.ndarray, x_train: np.ndarray, **kwargs) -> MAFPosteriorEstimator:
    theta_train = np.asarray(theta_train, dtype=float)
    x_train = np.asarray(x_train, dtype=float)
    estimator_kwargs = {
        key: kwargs.pop(key)
        for key in list(kwargs)
        if key
        in {
            "hidden_features",
            "num_transforms",
            "num_blocks",
            "learning_rate",
            "device",
            "standardize",
        }
    }
    estimator = MAFPosteriorEstimator(theta_dim=theta_train.shape[1], x_dim=x_train.shape[1], **estimator_kwargs)
    estimator.fit(theta_train, x_train, **kwargs)
    return estimator


def sample_posterior(estimator: MAFPosteriorEstimator, x_obs: np.ndarray, num_samples: int = 10000) -> np.ndarray:
    return estimator.sample(x_obs, num_samples=num_samples)


def _prepare_flow_for_device(flow, torch, device):
    """Force nflows modules to float32 before moving to accelerators.

    Some nflows distributions/register buffers as float64 depending on the
    process default dtype. Apple MPS does not support float64 tensors, so moving
    the raw flow directly to MPS can fail even though all training arrays are
    float32. Converting on CPU first keeps construction robust across CPU, CUDA,
    and MPS.
    """

    flow = flow.to(dtype=torch.float32)
    return flow.to(device=device)


def _simulate_one(simulator, theta: np.ndarray, noise_fn, rng: np.random.Generator) -> np.ndarray:
    if hasattr(simulator, "simulate"):
        return simulator.simulate(theta, noise_fn=noise_fn, rng=rng)
    if hasattr(simulator, "rvs"):
        return simulator.rvs(theta, noise_fn=noise_fn, rng=rng)
    return simulator(theta, noise_fn=noise_fn, rng=rng)


def _as_2d(values: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != int(dim):
        raise ValueError(f"{name} must have shape ({dim},) or (n, {dim}); got {arr.shape}.")
    return arr


def _as_context_batch(values: np.ndarray, dim: int) -> np.ndarray:
    return _as_2d(values, dim, "x_obs")
