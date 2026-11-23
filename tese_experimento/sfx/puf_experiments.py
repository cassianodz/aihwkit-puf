"""E7: characterization of the device-level PCM PUF (NumPy only, no aihwkit needed)."""
from __future__ import annotations

import time
from dataclasses import asdict, replace

import numpy as np

from .puf import (PCMChip, PUFParams, PUFVerifier, bradley_terry_scores, ber_gaussian,
                  far_binomial, frr_binomial, fresh_sigma, predict_with_scores)


def _population(p: PUFParams, n_chips: int, seed: int):
    rng = np.random.default_rng([seed, 7])
    chips = [PCMChip(i, p, seed_base=1000 + seed) for i in range(n_chips)]
    ver = PUFVerifier(p)
    for i, c in enumerate(chips):
        ver.enroll(i, c, rng)
    return chips, ver, rng


def genuine_impostor(p: PUFParams, n_chips: int = 20, n_rounds: int = 50, seed: int = 0) -> dict:
    """Genuine and impostor (another chip answers) similarities with fresh challenges."""
    chips, ver, rng = _population(p, n_chips, seed)
    gen, imp, uni = [], [], []
    for _ in range(n_rounds):
        for i in range(n_chips):
            ch = ver.challenge(i, rng)
            gen.append(ver.similarity(i, ch, chips[i].respond(ch, rng)))
            other = (i + 1 + rng.integers(n_chips - 1)) % n_chips
            imp.append(ver.similarity(i, ch, chips[other].respond(ch, rng)))
    # uniqueness / uniformity on common random challenges (no masking, all chips)
    for _ in range(20):
        pairs = rng.integers(0, p.n_cells, size=(p.n_bits, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        from .puf import Challenge
        ch = Challenge(float(rng.uniform(p.g_low, p.g_high)), pairs)
        R = np.array([c.respond(ch, rng) for c in chips])
        iu = np.triu_indices(n_chips, 1)
        hd = (R[:, None, :] != R[None, :, :]).mean(-1)[iu]
        uni.append({"inter_hd": float(hd.mean()), "uniformity": float(R.mean()),
                    "bit_aliasing_sd": float(R.mean(0).std())})
    gen, imp = np.array(gen), np.array(imp)
    ber = 1 - gen.mean()
    return {
        "params": asdict(p), "genuine": gen.round(4).tolist(), "impostor": imp.round(4).tolist(),
        "ber": float(ber), "genuine_min": float(gen.min()), "impostor_max": float(imp.max()),
        "frr_empirical": float((gen < p.tau).mean()), "far_empirical": float((imp >= p.tau).mean()),
        "frr_binomial": frr_binomial(ber, p.n_bits, p.tau),
        "far_binomial_impostor": far_binomial(float(imp.mean()), p.n_bits, p.tau),
        "uniqueness": float(np.mean([u["inter_hd"] for u in uni])),
        "uniformity": float(np.mean([u["uniformity"] for u in uni])),
        "bit_aliasing_sd": float(np.mean([u["bit_aliasing_sd"] for u in uni])),
    }


def reliability_grid(sigmas, keeps, reps_list, base: PUFParams, n_chips=10, n_rounds=20, seed=0) -> list[dict]:
    out = []
    for s in sigmas:
        for k in keeps:
            for r in reps_list:
                p = replace(base, sigma_static=s, keep=k, reps=r)
                chips, ver, rng = _population(p, n_chips, seed)
                sims = []
                for _ in range(n_rounds):
                    for i in range(n_chips):
                        ch = ver.challenge(i, rng)
                        sims.append(ver.similarity(i, ch, chips[i].respond(ch, rng)))
                ber = 1 - float(np.mean(sims))
                g_mid = 0.5 * (p.g_low + p.g_high)
                out.append({"sigma_static": s, "keep": k, "reps": r, "ber": ber,
                            "ber_gauss_nomask": ber_gaussian(s, fresh_sigma(g_mid, p) / np.sqrt(r), p.enroll_reps),
                            "frr_binomial": frr_binomial(ber, p.n_bits, p.tau),
                            "frr_empirical": float(np.mean(np.array(sims) < p.tau))})
    return out


def level_time_grid(levels, delays, base: PUFParams, n_chips=10, n_rounds=20, seed=0) -> list[dict]:
    """BER versus challenge conductance and programming-to-read delay."""
    out = []
    for g in levels:
        for t in delays:
            p = replace(base, g_low=g, g_high=g, t_read_delay=t)
            chips, ver, rng = _population(p, n_chips, seed)
            sims = []
            for _ in range(n_rounds):
                for i in range(n_chips):
                    ch = ver.challenge(i, rng)
                    sims.append(ver.similarity(i, ch, chips[i].respond(ch, rng)))
            out.append({"g_t": g, "t_read_delay": t, "ber": 1 - float(np.mean(sims)),
                        "fresh_sigma": fresh_sigma(g, p)})
    return out


def modeling_attack(budgets, base: PUFParams, keeps=(1.0, 0.25), n_targets=5, n_test=40, seed=0) -> list[dict]:
    """An eavesdropper records M challenge-response pairs of a target identity
    (responses sent in clear) and fits a Bradley-Terry model to answer new
    challenges. Reports the attacker's similarity and acceptance rate."""
    out = []
    for keep in keeps:
        p = replace(base, keep=keep)
        chips, ver, rng = _population(p, n_targets, seed)
        for m in budgets:
            sims, t_fit = [], 0.0
            for tgt in range(n_targets):
                pairs, bits = [], []
                got = 0
                while got < m:                      # eavesdropped rounds
                    ch = ver.challenge(tgt, rng)
                    pairs.append(ch.pairs); bits.append(chips[tgt].respond(ch, rng)); got += len(ch.pairs)
                if m > 0:
                    P = np.concatenate(pairs)[:m]; B = np.concatenate(bits)[:m]
                    t0 = time.time(); scores = bradley_terry_scores(P, B, p.n_cells); t_fit += time.time() - t0
                else:
                    scores = np.zeros(p.n_cells)
                for _ in range(n_test):
                    ch = ver.challenge(tgt, rng)
                    sims.append(ver.similarity(tgt, ch, predict_with_scores(scores, ch, rng)))
            sims = np.array(sims)
            out.append({"keep": keep, "crps": int(m), "rounds_equivalent": m / p.n_bits,
                        "attacker_similarity": float(sims.mean()), "attacker_similarity_sd": float(sims.std()),
                        "acceptance_rate": float((sims >= p.tau).mean()), "fit_seconds": t_fit / n_targets})
    return out


def legacy_puf_check(n_devices=20, n_rounds=50) -> dict:
    """Reproduces the behaviour of puf/enrollment.py (seeded noise) in NumPy, for comparison."""
    import json
    from pathlib import Path
    params = json.loads((Path(__file__).resolve().parents[2] / "puf" / "population_params" / "pcm_ibm_calibrated.json").read_text())
    gmax = float(params["conductance_range"]["g_max_uS"]); sig_off = params["inter_device"]["g_prog_offset"]["sigma_uS"]

    def signature(dev, rnd, fresh=False):
        off = np.random.default_rng(dev).normal(0, sig_off)
        coeffs = [-1.1731, 1.9650, 0.2635 + off]          # order used in identity.py
        g = np.random.default_rng(rnd).uniform(0.3, 1.7, 256)
        sig = coeffs[0] + coeffs[1] * g / gmax + coeffs[2] * (g / gmax) ** 2
        rng = np.random.default_rng() if fresh else np.random.default_rng(dev * 100_000)
        return (sig * rng.standard_normal(256)) > 0

    sigs = {d: [signature(d, r) for r in range(1, n_rounds + 1)] for d in range(n_devices)}
    identical = all((s == sigs[d][0]).all() for d in range(n_devices) for s in sigs[d])
    fresh = [float(np.mean(signature(d, 1, True) == sigs[d][0])) for d in range(n_devices)]
    hd = [float(np.mean(sigs[a][0] != sigs[b][0])) for a in range(n_devices) for b in range(a + 1, n_devices)]
    return {"identical_across_rounds": bool(identical), "uniqueness": float(np.mean(hd)),
            "fresh_read_similarity_mean": float(np.mean(fresh)), "fresh_read_similarity_min": float(np.min(fresh))}
