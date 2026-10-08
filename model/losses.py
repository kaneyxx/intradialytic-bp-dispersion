from __future__ import annotations

import math

import torch


def clamp_sigma(
    log_sigma: torch.Tensor,
    *,
    sigma_floor: torch.Tensor | float,
    sigma_cap: float,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    sigma = torch.exp(torch.clamp(log_sigma, -20.0, 20.0))
    floor = torch.as_tensor(sigma_floor, dtype=sigma.dtype, device=sigma.device)
    return torch.clamp(sigma, min=float(epsilon)).maximum(floor).clamp(max=float(sigma_cap))


def gaussian_nll_with_clamped_sigma(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    *,
    sigma_floor: tuple[float, float] = (2.0, 1.5),
    sigma_cap: float = 60.0,
    epsilon: float = 1e-6,
    loss_weights: tuple[float, float] = (1.0, 1.0),
    reduction: str = "mean",
) -> torch.Tensor:
    """Two-output Gaussian NLL with clinically bounded sigma.

    y/mu/log_sigma all have shape (B, 2), ordered as SBP then DBP.
    """
    if y_true.shape != mu.shape or mu.shape != log_sigma.shape:
        raise ValueError(
            f"y_true, mu, log_sigma must have same shape. "
            f"Got {y_true.shape}, {mu.shape}, {log_sigma.shape}."
        )
    if y_true.ndim != 2 or y_true.shape[1] != 2:
        raise ValueError("Gaussian BP NLL expects shape (batch, 2).")

    floor = torch.as_tensor(sigma_floor, dtype=mu.dtype, device=mu.device).view(1, 2)
    sigma = clamp_sigma(log_sigma, sigma_floor=floor, sigma_cap=sigma_cap, epsilon=epsilon)
    z = (y_true - mu) / sigma
    nll = torch.log(sigma + float(epsilon)) + 0.5 * z.square() + 0.5 * math.log(2.0 * math.pi)
    weights = torch.as_tensor(loss_weights, dtype=mu.dtype, device=mu.device).view(1, 2)
    nll = nll * weights
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'.")
    return nll.mean()


def mse_mu_loss(y_true: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    if y_true.shape != mu.shape:
        raise ValueError("y_true and mu must have same shape.")
    return torch.mean((y_true - mu).square())
