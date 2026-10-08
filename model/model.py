from __future__ import annotations

import math

import torch
import torch.nn as nn

from .losses import clamp_sigma


class AdditiveFeatureHead(nn.Module):
    """One additive RBF head: intercept + sum_j f_j(x_j)."""

    def __init__(self, num_features: int, num_bases: int = 24, input_range: tuple[float, float] = (-3.0, 3.0)) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_bases = int(num_bases)
        if self.num_features < 1:
            raise ValueError("num_features must be >= 1")
        if self.num_bases < 4:
            raise ValueError("num_bases must be >= 4")

        centers = torch.linspace(float(input_range[0]), float(input_range[1]), self.num_bases)
        self.centers = nn.Parameter(centers.repeat(self.num_features, 1))
        width = (float(input_range[1]) - float(input_range[0])) / max(1, self.num_bases - 1)
        self.log_widths = nn.Parameter(torch.full((self.num_features, self.num_bases), math.log(width)))
        self.coefficients = nn.Parameter(torch.empty(self.num_features, self.num_bases))
        self.feature_bias = nn.Parameter(torch.zeros(self.num_features))
        self.intercept = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, mean=0.0, std=0.01)
        nn.init.zeros_(self.feature_bias)
        nn.init.zeros_(self.intercept)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -10.0, 10.0)
        widths = torch.exp(torch.clamp(self.log_widths, -5.0, 5.0)) + 1e-3
        z = (x.unsqueeze(-1) - self.centers.unsqueeze(0)) / widths.unsqueeze(0)
        return torch.exp(-0.5 * torch.clamp(z, -10.0, 10.0).square())

    def contributions(self, x: torch.Tensor) -> torch.Tensor:
        b = self.basis(x)
        contrib = (b * self.coefficients.unsqueeze(0)).sum(dim=-1) + self.feature_bias.unsqueeze(0)
        return torch.nan_to_num(contrib, nan=0.0, posinf=20.0, neginf=-20.0)

    def forward(self, x: torch.Tensor, return_contrib: bool = False) -> dict[str, torch.Tensor]:
        contrib = self.contributions(x)
        value = self.intercept + contrib.sum(dim=1)
        out = {"value": value}
        if return_contrib:
            out["contrib"] = contrib
        return out


class GaussianAdditiveNAMLSS(nn.Module):
    """Independent Gaussian NAMLSS heads for SBP and DBP."""

    def __init__(self, num_features: int, num_bases: int = 24) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_bases = int(num_bases)
        self.mu_sbp = AdditiveFeatureHead(num_features, num_bases)
        self.mu_dbp = AdditiveFeatureHead(num_features, num_bases)
        self.log_sigma_sbp = AdditiveFeatureHead(num_features, num_bases)
        self.log_sigma_dbp = AdditiveFeatureHead(num_features, num_bases)

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_contrib: bool = False,
        sigma_floor: tuple[float, float] = (2.0, 1.5),
        sigma_cap: float = 60.0,
    ) -> dict[str, torch.Tensor]:
        mu_sbp = self.mu_sbp(x, return_contrib=return_contrib)
        mu_dbp = self.mu_dbp(x, return_contrib=return_contrib)
        ls_sbp = self.log_sigma_sbp(x, return_contrib=return_contrib)
        ls_dbp = self.log_sigma_dbp(x, return_contrib=return_contrib)
        mu = torch.stack([mu_sbp["value"], mu_dbp["value"]], dim=1)
        log_sigma = torch.stack([ls_sbp["value"], ls_dbp["value"]], dim=1)
        sigma = clamp_sigma(
            log_sigma,
            sigma_floor=torch.as_tensor(sigma_floor, dtype=mu.dtype, device=mu.device).view(1, 2),
            sigma_cap=sigma_cap,
        )
        out = {"mu": mu, "log_sigma": log_sigma, "sigma": sigma}
        if return_contrib:
            out["contrib_mu_sbp"] = mu_sbp["contrib"]
            out["contrib_mu_dbp"] = mu_dbp["contrib"]
            out["contrib_log_sigma_sbp"] = ls_sbp["contrib"]
            out["contrib_log_sigma_dbp"] = ls_dbp["contrib"]
        return out

    def set_sigma_trainable(self, trainable: bool) -> None:
        for head in (self.log_sigma_sbp, self.log_sigma_dbp):
            for param in head.parameters():
                param.requires_grad = bool(trainable)

    @torch.no_grad()
    def reset_log_sigma_intercepts(self, sigma_sbp: float, sigma_dbp: float) -> None:
        self.log_sigma_sbp.intercept.fill_(math.log(max(float(sigma_sbp), 1e-6)))
        self.log_sigma_dbp.intercept.fill_(math.log(max(float(sigma_dbp), 1e-6)))

    @torch.no_grad()
    def set_mu_intercepts(self, mean_sbp: float, mean_dbp: float) -> None:
        self.mu_sbp.intercept.fill_(float(mean_sbp))
        self.mu_dbp.intercept.fill_(float(mean_dbp))
