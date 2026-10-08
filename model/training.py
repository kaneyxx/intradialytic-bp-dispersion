from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.linear_model import Ridge

from .config import ModelConfig
from .model import GaussianAdditiveNAMLSS


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def bootstrap_indices_by_group(group_keys: np.ndarray, *, seed: int) -> np.ndarray:
    groups = np.asarray(group_keys).astype(str)
    unique = np.unique(groups)
    if unique.size == 0:
        raise ValueError("Cannot bootstrap empty group array.")
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(unique, size=unique.size, replace=True)
    blocks = [np.flatnonzero(groups == key) for key in sampled]
    return np.concatenate(blocks).astype(int, copy=False)


def ridge_initial_residual_scale(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    pred = ridge.predict(X)
    residual_sd = np.nanstd(y - pred, axis=0)
    residual_sd = np.where(np.isfinite(residual_sd) & (residual_sd > 1e-6), residual_sd, np.nanstd(y, axis=0))
    residual_sd = np.where(np.isfinite(residual_sd) & (residual_sd > 1e-6), residual_sd, np.array([10.0, 6.0]))
    target_mean = np.nanmean(y, axis=0)
    target_mean = np.where(np.isfinite(target_mean), target_mean, np.array([130.0, 70.0]))
    return target_mean.astype(float), residual_sd.astype(float)


def _tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=device)


@torch.no_grad()
def predict_model(
    model: GaussianAdditiveNAMLSS,
    X: np.ndarray,
    cfg: ModelConfig,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    dev = torch.device(device) if isinstance(device, str) else device
    model.to(dev)
    model.eval()
    xt = _tensor(X, dev)
    mu_chunks: list[np.ndarray] = []
    sigma_chunks: list[np.ndarray] = []
    for start in range(0, xt.shape[0], int(cfg.batch_size)):
        out = model(
            xt[start:start + int(cfg.batch_size)],
            sigma_floor=(cfg.sigma_floor_sbp, cfg.sigma_floor_dbp),
            sigma_cap=cfg.sigma_cap,
        )
        mu_chunks.append(out["mu"].cpu().numpy())
        sigma_chunks.append(out["sigma"].cpu().numpy())
    return {
        "mu": np.concatenate(mu_chunks, axis=0) if mu_chunks else np.empty((0, 2)),
        "sigma": np.concatenate(sigma_chunks, axis=0) if sigma_chunks else np.empty((0, 2)),
    }


@torch.no_grad()
def predict_ensemble(
    models: list[GaussianAdditiveNAMLSS],
    X: np.ndarray,
    cfg: ModelConfig,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    if not models:
        raise ValueError("At least one model is required.")
    mus = []
    sigmas = []
    for model in models:
        out = predict_model(model, X, cfg, device=device)
        mus.append(out["mu"])
        sigmas.append(out["sigma"])
    mu_stack = np.stack(mus, axis=0)
    sigma_stack = np.stack(sigmas, axis=0)
    mu_mean = mu_stack.mean(axis=0)
    epistemic_var = mu_stack.var(axis=0)
    aleatoric_var = (sigma_stack ** 2).mean(axis=0)
    predictive_var = aleatoric_var + epistemic_var
    return {
        "mu": mu_mean,
        "sigma": np.sqrt(np.clip(aleatoric_var, 1e-12, np.inf)),
        "aleatoric_var": aleatoric_var,
        "epistemic_var": epistemic_var,
        "predictive_var": predictive_var,
        "member_mu": mu_stack,
    }


