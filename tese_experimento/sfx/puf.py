"""Device-level PCM PUF ("PUF v2") for the dual-attestation protocol.

Why a new model
---------------
In puf/enrollment.py the response is the sign of programming noise drawn with
torch.manual_seed(device_id * 100_000): every round returns the same bits, the
challenge has no effect, and without the seed a fresh read agrees with the
enrollment only by chance (similarity ~0.50). The statistics in
puf/population_params/pcm_ibm_calibrated.json are labelled "inter_device";
in the PCM literature a "device" is one memory cell, so here they are applied
cell by cell, which is what gives a crossbar a fingerprint.

Model (all conductances in microsiemens)
----------------------------------------
Static, drawn once per cell at "fabrication":
    o_j  ~ N(0, sigma_static)          single-shot programming offset (0.5 uS)
    dnu_j ~ N(0, nu_static_sigma)      drift-exponent offset (0.012)
    q_j  ~ N(1, qs_static_sigma)       read-noise scale (0.07), clipped to [0.5, 2]
Fresh, drawn at every programming / read (aihwkit PCMLikeNoiseModel formulas):
    g_prog = g_T + o_j + sigma_prog(g_T) * N(0, 1)          (aihwkit polynomial,
             coefficients in aihwkit order [0.26348, 1.9650, -1.1731])
    nu     = |mu_nu(g_T) + sd_nu(g_T) * N(0, 1) + dnu_j|
    g(t)   = g_prog * ((t + t0) / t0) ** -nu  +  1/f read noise scaled by q_j

Challenge = a target level g_T and L cell pairs (j, k); both cells are
programmed single-shot to g_T and read after t_read_delay; response bit =
[g_j > g_k]. A fresh challenge is drawn every round, so a recorded response
does not answer the next one.

Enrollment ("soft", in a trusted facility): every cell is programmed E times
and its offset is estimated as the mean of g - g_T. The verifier keeps these
estimates, predicts the expected bit of any pair, and issues only pairs whose
estimated offsets differ by more than a threshold that keeps a fraction
`keep` of random pairs (dark-bit masking). Similarity = fraction of matching
bits; the identity is accepted if similarity >= tau_puf.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class PUFParams:
    n_cells: int = 7808                 # 64 x 122 crossbar of the first NIDS layer
    sigma_static: float = 0.5           # uS, calibration file: inter_device.g_prog_offset
    nu_static_sigma: float = 0.012      # calibration file: inter_device.drift_nu_offset
    qs_static_sigma: float = 0.07       # calibration file: inter_device.Qs_scale_offset
    g_low: float = 1.0                  # challenge range (uS)
    g_high: float = 3.0
    g_max: float = 25.0
    prog_coeff: tuple = (0.26348, 1.9650, -1.1731)
    prog_noise_scale: float = 1.0
    read_noise_scale: float = 1.0
    t_read_delay: float = 1.0           # s between programming and read-out
    t0: float = 20.0                    # aihwkit PCMLikeNoiseModel defaults
    t_read: float = 250e-9
    enroll_reps: int = 8
    reps: int = 3                       # programming/read repetitions averaged per response
    keep: float = 0.25                  # fraction of random pairs that pass the masking threshold
    n_bits: int = 256                   # L
    tau: float = 0.90


def sigma_prog(g_t: np.ndarray | float, p: PUFParams) -> np.ndarray:
    x = np.asarray(g_t) / p.g_max
    c0, c1, c2 = p.prog_coeff
    return (c0 + c1 * x + c2 * x ** 2) * (p.g_max / 25.0)


class PCMChip:
    """The PUF region of one physical chip."""

    def __init__(self, chip_id: int, p: PUFParams, seed_base: int = 2026):
        self.chip_id, self.p = chip_id, p
        rng = np.random.default_rng([seed_base, chip_id])
        self.offset = rng.normal(0.0, p.sigma_static, p.n_cells)
        self.nu_off = rng.normal(0.0, p.nu_static_sigma, p.n_cells)
        self.qs = np.clip(rng.normal(1.0, p.qs_static_sigma, p.n_cells), 0.5, 2.0)

    def read(self, cells: np.ndarray, g_t: float, rng: np.random.Generator, reps: int = 1,
             t_read_delay: float | None = None) -> np.ndarray:
        """Program `cells` single-shot to g_t and read them; returns (reps, n)."""
        p = self.p
        t_delay = p.t_read_delay if t_read_delay is None else t_read_delay
        shape = (reps, len(cells))
        g_prog = g_t + self.offset[cells] + p.prog_noise_scale * sigma_prog(g_t, p) * rng.standard_normal(shape)
        g_prog = np.clip(g_prog, 0.0, None)
        g_rel = max(g_t / p.g_max, 1e-6)
        mu = np.clip(-0.0155 * np.log(g_rel) + 0.0244, 0.049, 0.1)
        sd = np.clip(-0.0125 * np.log(g_rel) - 0.0059, 0.008, 0.045)
        nu = np.abs(mu + sd * rng.standard_normal(shape) + self.nu_off[cells])
        t = t_delay + p.t0
        g_drift = g_prog * (t / p.t0) ** (-nu)
        q_s = np.clip(0.0088 / np.clip((np.abs(g_prog) / p.g_max) ** 0.65, 1e-3, None), None, 0.2)
        sig_noise = q_s * np.sqrt(np.log((t + p.t_read) / (2 * p.t_read)))
        g = g_drift + np.abs(g_drift) * p.read_noise_scale * self.qs[cells] * sig_noise * rng.standard_normal(shape)
        return np.clip(g, 0.0, None)

    def respond(self, challenge: "Challenge", rng: np.random.Generator, reps: int | None = None) -> np.ndarray:
        reps = self.p.reps if reps is None else reps
        j, k = challenge.pairs[:, 0], challenge.pairs[:, 1]
        g = self.read(np.concatenate([j, k]), challenge.g_t, rng, reps=reps).mean(0)
        n = len(j)
        return g[:n] > g[n:]


@dataclass
class Challenge:
    g_t: float
    pairs: np.ndarray                   # (L, 2) cell indices


@dataclass
class Enrollment:
    offset_hat: np.ndarray
    threshold: float


@dataclass
class PUFVerifier:
    p: PUFParams
    db: dict = field(default_factory=dict)

    def enroll(self, identity: int, chip: PCMChip, rng: np.random.Generator) -> Enrollment:
        p = self.p
        g_ref = 0.5 * (p.g_low + p.g_high)
        cells = np.arange(p.n_cells)
        g = chip.read(cells, g_ref, rng, reps=p.enroll_reps)
        off_hat = (g - g_ref).mean(0)
        a, b = rng.integers(0, p.n_cells, size=(2, 20000))
        diffs = np.abs(off_hat[a] - off_hat[b])
        thr = float(np.quantile(diffs, 1.0 - p.keep)) if p.keep < 1.0 else 0.0
        e = Enrollment(off_hat, thr)
        self.db[identity] = e
        return e

    def challenge(self, identity: int, rng: np.random.Generator) -> Challenge:
        p, e = self.p, self.db[identity]
        g_t = float(rng.uniform(p.g_low, p.g_high))
        chosen = []
        need = p.n_bits
        while need > 0:
            m = int(np.ceil(need / max(p.keep, 1e-3) * 1.5)) + 8
            a, b = rng.integers(0, p.n_cells, size=(2, m))
            ok = (a != b) & (np.abs(e.offset_hat[a] - e.offset_hat[b]) >= e.threshold)
            cand = np.stack([a[ok], b[ok]], axis=1)[:need]
            chosen.append(cand)
            need -= len(cand)
        return Challenge(g_t, np.concatenate(chosen)[: p.n_bits])

    def expected(self, identity: int, ch: Challenge) -> np.ndarray:
        o = self.db[identity].offset_hat
        return o[ch.pairs[:, 0]] > o[ch.pairs[:, 1]]

    def similarity(self, identity: int, ch: Challenge, response: np.ndarray) -> float:
        return float(np.mean(response == self.expected(identity, ch)))

    def verify(self, identity: int, ch: Challenge, response: np.ndarray) -> tuple[bool, float]:
        if identity not in self.db:
            return False, 0.0
        s = self.similarity(identity, ch, response)
        return s >= self.p.tau, s


# ---------------------------------------------------------------------------
# Attacks and analytics
# ---------------------------------------------------------------------------
def bradley_terry_scores(pairs: np.ndarray, bits: np.ndarray, n_cells: int, C: float = 1.0) -> np.ndarray:
    """Modeling attack: logistic regression on pair indicators (Bradley-Terry MLE)
    of eavesdropped challenge-response pairs; returns one latent score per cell."""
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    if len(pairs) == 0 or len(np.unique(bits)) < 2:
        return np.zeros(n_cells)
    m = len(pairs)
    X = csr_matrix((np.tile([1.0, -1.0], m), (np.repeat(np.arange(m), 2), pairs.ravel())), shape=(m, n_cells))
    lr = LogisticRegression(fit_intercept=False, C=C, max_iter=300)
    lr.fit(X, bits.astype(int))
    return lr.coef_.ravel()


def predict_with_scores(scores: np.ndarray, ch: Challenge, rng: np.random.Generator) -> np.ndarray:
    a, b = scores[ch.pairs[:, 0]], scores[ch.pairs[:, 1]]
    out = a > b
    ties = a == b
    out[ties] = rng.random(int(ties.sum())) < 0.5
    return out


def ber_gaussian(sigma_static: float, sigma_fresh: float, enroll_reps: int) -> float:
    """Closed-form bit-error rate without masking (Gaussian approximation):
    P(sign(dO + dF) != sign(dO + dE)) = arccos(rho) / pi."""
    se2 = sigma_fresh ** 2 / enroll_reps
    rho = sigma_static ** 2 / np.sqrt((sigma_static ** 2 + sigma_fresh ** 2) * (sigma_static ** 2 + se2))
    return float(np.arccos(np.clip(rho, -1, 1)) / np.pi)


def fresh_sigma(g_t: float, p: PUFParams) -> float:
    """Per-cell fresh noise (programming + read) at level g_t, in uS."""
    t = p.t_read_delay + p.t0
    q_s = min(0.0088 / max((g_t / p.g_max) ** 0.65, 1e-3), 0.2)
    s_read = g_t * p.read_noise_scale * q_s * np.sqrt(np.log((t + p.t_read) / (2 * p.t_read)))
    return float(np.sqrt((p.prog_noise_scale * sigma_prog(g_t, p)) ** 2 + s_read ** 2))


def frr_binomial(ber: float, n_bits: int, tau: float) -> float:
    """P(similarity < tau) for a genuine device with independent bit errors."""
    max_err = int(np.floor(n_bits * (1 - tau) + 1e-9))
    return float(stats.binom.sf(max_err, n_bits, ber))


def far_binomial(p_match: float, n_bits: int, tau: float) -> float:
    """P(similarity >= tau) for an impostor whose bits match with prob p_match."""
    need = int(np.ceil(n_bits * tau - 1e-9))
    return float(stats.binom.sf(need - 1, n_bits, p_match))
