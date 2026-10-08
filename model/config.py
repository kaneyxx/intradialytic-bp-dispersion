"""Training configuration of the NAMLSS forecasting models."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Values used for Models A and B are in spec_model_A.json and spec_model_B.json (training_config)."""
    num_bases: int = 24
    batch_size: int = 16384
    n_epochs: int = 1500
    warmup_epochs: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 5e-4
    gradient_clip: float = 1.0
    sigma_floor_sbp: float = 2.0
    sigma_floor_dbp: float = 1.5
    sigma_cap: float = 60.0
    epsilon: float = 1e-6
    loss_weight_sbp: float = 1.0
    loss_weight_dbp: float = 1.0
    ensemble_members: int = 5
    bootstrap_replicates: int = 1
    patience: int = 100
    min_delta: float = 1e-4
    seed: int = 0
