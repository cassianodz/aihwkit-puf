"""Aggregation rules used as baselines.

All rules take the client updates U (n x d, i.e. local weights minus the
current global weights) and return the aggregated update plus the weight that
each client received (0 = discarded), so acceptance statistics can be compared
across defenses.

References
----------
FedAvg        McMahan et al., AISTATS 2017.
Median, Trimmed mean   Yin et al., ICML 2018 (coordinate-wise).
Multi-Krum    Blanchard et al., NeurIPS 2017.
FoolsGold     Fung et al., RAID 2020 (Algorithm 1, with pardoning and logit).
FLTrust       Cao et al., NDSS 2021 (ReLU-clipped cosine trust, norm alignment).
"""
from __future__ import annotations

import numpy as np
import torch


def fedavg(U: torch.Tensor, sizes: np.ndarray | None = None):
    w = torch.ones(len(U)) if sizes is None else torch.tensor(sizes, dtype=torch.float32)
    w = w / w.sum()
    return (w[:, None] * U).sum(0), w.numpy()


def median(U: torch.Tensor):
    return torch.quantile(U, 0.5, dim=0), np.full(len(U), np.nan)


def trimmed_mean(U: torch.Tensor, beta: float = 0.2):
    n = len(U)
    k = int(np.floor(beta * n))
    if n - 2 * k < 1:
        k = (n - 1) // 2
    s, _ = torch.sort(U, dim=0)
    return s[k:n - k].mean(0), np.full(n, np.nan)


def multi_krum(U: torch.Tensor, f: int, m: int | None = None):
    n = len(U)
    f = int(min(max(f, 0), n - 1))
    m = n - f if m is None else m
    m = int(min(max(m, 1), n))
    d2 = torch.cdist(U, U) ** 2
    k = max(n - f - 2, 1)
    # score = sum of squared distances to the k nearest neighbours (self excluded)
    scores = torch.sort(d2, dim=1).values[:, 1:k + 1].sum(1)
    sel = torch.argsort(scores)[:m]
    w = np.zeros(n)
    w[sel.numpy()] = 1.0 / m
    return U[sel].mean(0), w


class FoolsGold:
    """Stateful FoolsGold: similarity of each client's *historical* update."""

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa
        self.history: dict[int, torch.Tensor] = {}

    def __call__(self, U: torch.Tensor, ids: list[int]):
        for u, i in zip(U, ids):
            self.history[i] = self.history.get(i, torch.zeros_like(u)) + u
        H = torch.stack([self.history[i] for i in ids]).numpy().astype(np.float64)
        n = len(ids)
        if n == 1:
            return U[0], np.ones(1)
        Hn = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
        cs = Hn @ Hn.T - np.eye(n)
        maxcs = cs.max(axis=1)
        for i in range(n):              # pardoning
            for j in range(n):
                if i != j and maxcs[i] < maxcs[j]:
                    cs[i, j] *= maxcs[i] / maxcs[j]
        wv = 1.0 - cs.max(axis=1)
        wv = np.clip(wv, 0.0, 1.0)
        if wv.max() <= 0:
            return torch.zeros_like(U[0]), np.zeros(n)
        wv = wv / wv.max()
        wv[wv == 1.0] = 0.99
        with np.errstate(divide="ignore"):
            wv = self.kappa * (np.log(wv / (1.0 - wv)) + 0.5)
        wv[np.isinf(wv) | (wv > 1)] = 1.0
        wv[np.isnan(wv) | (wv < 0)] = 0.0
        if wv.sum() == 0:
            return torch.zeros_like(U[0]), wv
        w = torch.tensor(wv / wv.sum(), dtype=torch.float32)
        return (w[:, None] * U).sum(0), wv


def fltrust(U: torch.Tensor, g0: torch.Tensor):
    """g0 = update computed by the server on its clean root dataset."""
    n0 = g0.norm().clamp(min=1e-12)
    cos = (U @ g0) / (U.norm(dim=1).clamp(min=1e-12) * n0)
    ts = torch.relu(cos)
    if ts.sum() <= 0:
        return torch.zeros_like(g0), ts.numpy()
    Un = U * (n0 / U.norm(dim=1).clamp(min=1e-12))[:, None]
    return (ts[:, None] * Un).sum(0) / ts.sum(), (ts / ts.sum()).numpy()
