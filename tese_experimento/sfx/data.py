"""NSL-KDD loading, splits, controller canary pool and client partitions.

Preprocessing is identical to fl_sentinel_puf.py (binary label = not 'normal',
one-hot of the 41 features -> 122 columns, train/test split with
random_state=42, StandardScaler fitted on the training split), so client
shards in the 'contiguous' partition are exactly the ones of the original
experiment.

What changes, and why
---------------------
* The controller's canary pool is carved out of the test split and removed
  from it, so canary flows never count in the reported metrics (the original
  script drew the 100 canaries from X_test and kept them in the test set).
* Attack families (DoS, Probe, R2L, U2R) are kept for non-IID partitions,
  stratified canaries and family-targeted attacks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# NSL-KDD attack type -> family (Tavallaee et al., 2009). Types that only occur
# in KDDTest+ are listed for completeness.
FAMILY_OF = {
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "mailbomb": "DoS", "processtable": "DoS",
    "udpstorm": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "named": "R2L", "sendmail": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "worm": "R2L",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "httptunnel": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
FAMILIES = ["normal", "DoS", "Probe", "R2L", "U2R"]
ATTACK_FAMILIES = FAMILIES[1:]


@dataclass
class Dataset:
    X_train: torch.Tensor
    y_train: torch.Tensor          # float, shape (n, 1), 1 = attack
    f_train: np.ndarray            # family index (0 = normal)
    X_test: torch.Tensor           # evaluation split (canary pool removed)
    y_test: torch.Tensor
    f_test: np.ndarray
    X_pool: torch.Tensor           # controller pool (canaries, FLTrust root data)
    y_pool: torch.Tensor
    f_pool: np.ndarray
    canary_source: str
    n_features: int

    def describe(self) -> dict:
        def fam_counts(f):
            return {FAMILIES[i]: int((f == i).sum()) for i in range(len(FAMILIES))}
        return {
            "n_train": int(len(self.y_train)), "n_test": int(len(self.y_test)),
            "n_pool": int(len(self.y_pool)), "n_features": int(self.n_features),
            "test_benign_share": float((self.y_test == 0).float().mean()),
            "train_families": fam_counts(self.f_train),
            "test_families": fam_counts(self.f_test),
            "pool_families": fam_counts(self.f_pool),
            "canary_source": self.canary_source,
        }


def load_nslkdd(path: str, canary_source: str = "pool", pool_frac: float = 0.08,
                pool_min_per_family: int = 5, split_seed: int = 42) -> Dataset:
    """Load KDDTrain+ (dataset.txt) with the original preprocessing.

    canary_source='test' reproduces the original protocol: the pool is the
    whole test split and the canaries are its first attack flows (they remain
    in the evaluation set). canary_source='pool' removes a stratified pool from
    the test split.
    """
    data = pd.read_csv(path, header=None)
    labels = data.iloc[:, -2].astype(str).str.strip().values
    y = (labels != "normal").astype(int)
    fam = np.array([0 if lab == "normal" else FAMILIES.index(FAMILY_OF.get(lab, "DoS"))
                    for lab in labels], dtype=np.int64)
    X = pd.get_dummies(data.iloc[:, :-2]).astype(float).values

    idx = np.arange(len(y))
    # Same permutation as train_test_split(X.values, y, test_size=0.2, random_state=42)
    tr, te = train_test_split(idx, test_size=0.2, random_state=split_seed)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr])
    X_te = scaler.transform(X[te])

    if canary_source == "test":
        pool_mask = np.ones(len(te), dtype=bool)          # pool = whole test split
        eval_mask = np.ones(len(te), dtype=bool)          # and it stays in the test set
    elif canary_source == "pool":
        rng = np.random.default_rng(12345)
        pool_mask = np.zeros(len(te), dtype=bool)
        f_te = fam[te]
        for c in range(len(FAMILIES)):
            members = np.flatnonzero(f_te == c)
            if len(members) == 0:
                continue
            n_take = int(round(pool_frac * len(members)))
            if c > 0:  # keep at least a few flows of every attack family
                n_take = max(n_take, min(pool_min_per_family, len(members) // 2))
            pool_mask[rng.choice(members, size=n_take, replace=False)] = True
        eval_mask = ~pool_mask
    else:
        raise ValueError(f"unknown canary_source {canary_source!r}")

    def T(a):
        return torch.tensor(a, dtype=torch.float32)

    return Dataset(
        X_train=T(X_tr), y_train=T(y[tr]).unsqueeze(1), f_train=fam[tr],
        X_test=T(X_te[eval_mask]), y_test=T(y[te][eval_mask]).unsqueeze(1), f_test=fam[te][eval_mask],
        X_pool=T(X_te[pool_mask]), y_pool=T(y[te][pool_mask]).unsqueeze(1), f_pool=fam[te][pool_mask],
        canary_source=canary_source, n_features=X_tr.shape[1],
    )


# ---------------------------------------------------------------------------
# Client partitions
# ---------------------------------------------------------------------------
def partition_contiguous(n_samples: int, n_shards: int) -> list[np.ndarray]:
    """Original partition: equal contiguous shards of the (already shuffled) split."""
    size = n_samples // n_shards
    return [np.arange(i * size, (i + 1) * size) for i in range(n_shards)]


def partition_dirichlet(families: np.ndarray, n_shards: int, beta: float,
                        rng: np.random.Generator, floor: int = 50) -> list[np.ndarray]:
    """Label-skewed partition over the five NSL-KDD classes (Hsu et al., 2019).

    Small beta -> each client sees few traffic classes; large beta -> close to IID.
    Every client first receives `floor` flows drawn at random (1% of an IID
    shard), so that no client is left empty at very small beta; the remaining
    flows are split class by class with Dirichlet(beta) proportions.
    """
    all_idx = rng.permutation(len(families))
    base, rest = all_idx[: floor * n_shards], all_idx[floor * n_shards:]
    shards: list[list[int]] = [base[i * floor:(i + 1) * floor].tolist() for i in range(n_shards)]
    for c in np.unique(families[rest]):
        members = rng.permutation(rest[families[rest] == c])
        props = rng.dirichlet(np.full(n_shards, beta))
        cuts = (np.cumsum(props) * len(members)).astype(int)[:-1]
        for s, part in enumerate(np.split(members, cuts)):
            shards[s].extend(part.tolist())
    return [np.sort(np.array(s, dtype=np.int64)) for s in shards]


# ---------------------------------------------------------------------------
# Canary sampling (controller side)
# ---------------------------------------------------------------------------
class CanarySampler:
    """Draws the canary set used by Sentinel-Flow in each round.

    mode='fixed'      : the same K attack flows every round (original: first K).
    mode='resample'   : K attack flows drawn from the pool every round.
    mode='stratified' : K flows split equally across attack families
                        (with replacement for rare families).
    benign_k > 0      : also draw benign flows (two-sided canaries).
    """

    def __init__(self, data: Dataset, k: int, mode: str, rng: np.random.Generator, benign_k: int = 0):
        self.data, self.k, self.mode, self.rng, self.benign_k = data, k, mode, rng, benign_k
        self.attack_idx = np.flatnonzero(data.y_pool.squeeze(1).numpy() == 1)
        self.benign_idx = np.flatnonzero(data.y_pool.squeeze(1).numpy() == 0)
        if len(self.attack_idx) == 0:
            raise ValueError("canary pool has no attack flows")

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        d = self.data
        if self.mode == "fixed":
            sel = self.attack_idx[: self.k]
        elif self.mode == "resample":
            sel = self.rng.choice(self.attack_idx, size=min(self.k, len(self.attack_idx)), replace=False)
        elif self.mode == "stratified":
            fams = d.f_pool[self.attack_idx]
            present = [c for c in range(1, len(FAMILIES)) if (fams == c).any()]
            per = np.full(len(present), self.k // len(present))
            per[: self.k - per.sum()] += 1
            parts = []
            for c, n in zip(present, per):
                members = self.attack_idx[fams == c]
                parts.append(self.rng.choice(members, size=n, replace=n > len(members)))
            sel = np.concatenate(parts)
        else:
            raise ValueError(f"unknown canary mode {self.mode!r}")
        if self.benign_k > 0:   # two-sided canaries: benign flows check false alarms
            sel = np.concatenate([sel, self.rng.choice(self.benign_idx, size=self.benign_k, replace=False)])
        return d.X_pool[sel], d.y_pool[sel], d.f_pool[sel]
