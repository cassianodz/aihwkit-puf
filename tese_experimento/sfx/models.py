"""NIDS model and local training, identical to fl_sentinel_puf.py."""
from __future__ import annotations

import torch
import torch.nn as nn


def create_model(input_dim: int, hidden: int = 64) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1), nn.Sigmoid())


def train_local(model: nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int = 3,
                lr: float = 0.005) -> nn.Module:
    """Full-batch Adam steps with a fresh optimizer, as in the original script
    (its LOCAL_EPOCHS = 3 are three full-batch steps)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return model


def get_flat(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.state_dict().values()]).clone()


def set_flat(model: nn.Module, vec: torch.Tensor) -> nn.Module:
    sd = model.state_dict()
    i = 0
    for k, v in sd.items():
        n = v.numel()
        sd[k] = vec[i:i + n].view_as(v).clone()
        i += n
    model.load_state_dict(sd)
    return model


def n_params(model: nn.Module) -> int:
    return sum(v.numel() for v in model.state_dict().values())
