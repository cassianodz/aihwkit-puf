"""Federated NIDS training with attacks, robust aggregation and dual attestation.

One function, run_federation(cfg, data), covers every scenario of
fl_sentinel_puf.py and the new experiments. With the 'original' settings
(canary_source='test', canary_mode='fixed', puf='assumed', backend='preset',
alpha=4) it reproduces the six original scenarios.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import aggregators as agg
from .analog import AnalogBackend
from .data import ATTACK_FAMILIES, FAMILIES, CanarySampler, Dataset, partition_contiguous, partition_dirichlet
from .models import create_model, get_flat, set_flat, train_local
from .puf import PCMChip, PUFParams, PUFVerifier


@dataclass
class RunConfig:
    exp: str = "adhoc"
    tag: str = ""
    seed: int = 0
    # federation -------------------------------------------------------------
    n_clients: int = 20              # identities that hold a data shard
    rounds: int = 50
    local_steps: int = 3
    lr: float = 0.005
    hidden: int = 64
    partition: str = "contiguous"    # contiguous (original, IID) | dirichlet
    beta: float = 0.5                # Dirichlet concentration
    weighted_avg: bool = True        # FedAvg weights = shard sizes
    # attack -----------------------------------------------------------------
    attack: str = "none"             # none | flip | targeted | flood | smart_sybil
    malicious: list = field(default_factory=list)   # empty -> last n_malicious ids
    n_malicious: int = 0
    alpha: float | None = None       # boost; None -> identities / attacker identities
    target_family: str = "DoS"
    sybil_ids: list = field(default_factory=list)   # own_shard mode (original: 10..19)
    sybil_phys: int | None = None    # identity backed by the adversary's chip (original: 15)
    n_virtual: int = 0               # copies mode: data-less identities
    sybil_mode: str = "own_shard"    # own_shard (original) | copies
    sybil_jitter: float = 0.01       # copies: relative Gaussian jitter of each copy
    sybil_trust_oracle: bool = True  # adversary forges the behavioral attestation
    sybil_puf: str = "own_chip"      # own_chip | random
    # defense ----------------------------------------------------------------
    defense: str = "fedavg"          # fedavg | median | trimmed | krum | foolsgold | fltrust | sentinel | dual
    tau_trust: float = 0.60
    trust_rule: str = "overall"      # overall (original) | per_family | delta (see below)
    delta_max: float = 0.20          # 'delta': max per-class drop w.r.t. the median of probed clients,
                                     # applied on top of the original test (natural-mix recall >= tau_trust)
    canary_benign_k: int = 0         # benign canaries (two-sided attestation); 0 = original
    trim_beta: float = 0.2
    krum_f: int | None = None        # None -> number of attacker identities
    fltrust_root: int = 200
    # canaries ---------------------------------------------------------------
    canary_source: str = "pool"      # test (original) | pool
    canary_k: int = 100
    canary_mode: str = "resample"    # fixed (original) | resample | stratified
    # PUF --------------------------------------------------------------------
    puf: str = "none"                # none | assumed (original) | v2
    puf_tau: float = 0.90
    puf_bits: int = 256
    puf_sigma_static: float = 0.5
    puf_keep: float = 0.25
    puf_reps: int = 3
    # analog backend ---------------------------------------------------------
    backend: str = "preset"          # preset (original) | inference | digital
    t_inference: float = 0.0
    inp_bits: int = 7
    out_bits: int = 9
    eval_every: int = 1

    def run_id(self) -> str:
        return f"{self.exp}__{self.tag}__s{self.seed}" if self.tag else f"{self.exp}__s{self.seed}"


@dataclass
class Identity:
    id: int
    shard: np.ndarray | None
    role: str        # honest | malicious | sybil_phys | sybil_virtual
    chip: int        # chip that physically answers PUF challenges

    @property
    def adversarial(self) -> bool:
        return self.role != "honest"


def build_population(cfg: RunConfig, data: Dataset, rng: np.random.Generator) -> list[Identity]:
    n = cfg.n_clients
    if cfg.partition == "contiguous":
        shards = partition_contiguous(len(data.y_train), n)
    elif cfg.partition == "dirichlet":
        shards = partition_dirichlet(data.f_train, n, cfg.beta, rng)
    else:
        raise ValueError(cfg.partition)
    pop = [Identity(i, shards[i], "honest", i) for i in range(n)]
    if cfg.attack in ("flip", "targeted", "flood"):
        mal = cfg.malicious or list(range(n - cfg.n_malicious, n))
        for i in mal:
            pop[i].role = "malicious"
    elif cfg.attack == "smart_sybil":
        if cfg.sybil_mode == "own_shard":
            ids = cfg.sybil_ids or list(range(n // 2, n))
            phys = cfg.sybil_phys if cfg.sybil_phys is not None else ids[len(ids) // 2]
            for i in ids:
                pop[i].role = "sybil_phys" if i == phys else "sybil_virtual"
                pop[i].chip = phys
        elif cfg.sybil_mode == "copies":
            phys = cfg.sybil_phys if cfg.sybil_phys is not None else n - 1
            pop[phys].role = "sybil_phys"
            for v in range(cfg.n_virtual):
                pop.append(Identity(n + v, None, "sybil_virtual", phys))
        else:
            raise ValueError(cfg.sybil_mode)
    return pop


def poisoned_labels(cfg: RunConfig, data: Dataset, ident: Identity) -> torch.Tensor:
    y = data.y_train[torch.as_tensor(ident.shard)]
    if ident.role == "honest":
        return y
    if cfg.attack == "targeted":
        fam = FAMILIES.index(cfg.target_family)
        y = y.clone()
        y[torch.from_numpy(data.f_train[ident.shard] == fam)] = 0.0
        return y
    if cfg.attack == "flood":
        return torch.ones_like(y)     # false-alarm flooding: everything labelled attack
    return torch.zeros_like(y)        # flip / smart_sybil: everything labelled benign (original)


@torch.no_grad()
def evaluate(backend: AnalogBackend, model, data: Dataset, families: bool = False) -> dict:
    probs = backend.predict(model, data.X_test).squeeze(1)
    pred = (probs > 0.5).float()
    y = data.y_test.squeeze(1)
    tp = float(((pred == 1) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    fp = float(((pred == 1) & (y == 0)).sum()); fn = float(((pred == 0) & (y == 1)).sum())
    out = {"acc": 100 * (tp + tn) / len(y), "asr": 100 * fn / max(tp + fn, 1)}
    if families:
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        out.update({"precision": 100 * prec, "recall": 100 * rec,
                    "f1": 100 * 2 * prec * rec / max(prec + rec, 1e-12),
                    "fpr": 100 * fp / max(fp + tn, 1), "tp": tp, "tn": tn, "fp": fp, "fn": fn})
        miss = {}
        for c, name in enumerate(FAMILIES):
            if c == 0:
                continue
            m = torch.from_numpy(data.f_test == c)
            if m.any():
                miss[name] = 100 * float((pred[m] == 0).float().mean())
        out["family_miss"] = miss
    return out


def run_federation(cfg: RunConfig, data: Dataset, verbose: bool = False, return_model: bool = False) -> dict:
    t_start = time.time()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    rng_part, rng_can, rng_puf, rng_syb = (np.random.default_rng([cfg.seed, k]) for k in (11, 12, 13, 14))

    backend = AnalogBackend(kind=cfg.backend, t_inference=cfg.t_inference,
                            inp_bits=cfg.inp_bits, out_bits=cfg.out_bits)
    pop = build_population(cfg, data, rng_part)
    n_ids = len(pop)
    attackers = [p for p in pop if p.adversarial]
    alpha = cfg.alpha if cfg.alpha is not None else (n_ids / max(len(attackers), 1))
    labels = {p.id: poisoned_labels(cfg, data, p) for p in pop if p.shard is not None}
    Xs = {p.id: data.X_train[torch.as_tensor(p.shard)] for p in pop if p.shard is not None}
    sizes = {p.id: (len(p.shard) if p.shard is not None else None) for p in pop}

    sampler = CanarySampler(data, cfg.canary_k, cfg.canary_mode, rng_can, benign_k=cfg.canary_benign_k)
    pool_fams = data.f_pool[sampler.attack_idx]
    nat_weight = {c: float((pool_fams == c).mean()) for c in np.unique(pool_fams)}   # natural attack mix

    # PUF infrastructure: every identity is enrolled with its own chip
    verifier = chips = None
    if cfg.defense == "dual" and cfg.puf == "v2":
        pp = PUFParams(sigma_static=cfg.puf_sigma_static, keep=cfg.puf_keep, reps=cfg.puf_reps,
                       n_bits=cfg.puf_bits, tau=cfg.puf_tau)
        chips = {p.id: PCMChip(p.id, pp) for p in pop}
        verifier = PUFVerifier(pp)
        for p in pop:
            verifier.enroll(p.id, chips[p.id], rng_puf)

    fools = agg.FoolsGold() if cfg.defense == "foolsgold" else None
    root_idx = None
    if cfg.defense == "fltrust":
        yp = data.y_pool.squeeze(1).numpy()
        half = cfg.fltrust_root // 2
        root_idx = np.concatenate([rng_can.choice(np.flatnonzero(yp == c), size=half, replace=False) for c in (0, 1)])

    model = create_model(data.n_features, cfg.hidden)
    hist = {"round": [], "acc": [], "asr": []}
    acc_log = {"honest_accepted": [], "honest_total": [], "adv_accepted": [], "adv_total": [],
               "trust_honest": [], "trust_adv": [], "puf_honest_min": [], "puf_adv_max": []}
    timing = {"train": 0.0, "canary": 0.0, "puf": 0.0, "aggregate": 0.0, "evaluate": 0.0}

    for r in range(cfg.rounds):
        g = get_flat(model)
        canX, canY, canF = sampler.sample()
        fam_ids = list(np.unique(canF))
        fam_masks = [torch.from_numpy(canF == c) for c in fam_ids]
        W, ids, wsize, accepted_flags = [], [], [], []
        trust_h, trust_a, puf_h, puf_a = [], [], [], []
        phys_w = None
        # ---- local training (and model poisoning) --------------------------
        t0 = time.time()
        local = {}
        for p in pop:
            if p.shard is None:
                continue
            m = copy.deepcopy(model)
            train_local(m, Xs[p.id], labels[p.id], cfg.local_steps, cfg.lr)
            w = get_flat(m)
            if p.adversarial:
                w = g + alpha * (w - g)
            local[p.id] = w
            if p.role == "sybil_phys":
                phys_w = w
        for p in pop:
            if p.shard is None:   # virtual identity: jittered copy of the adversary's update
                base = phys_w if phys_w is not None else g
                d = base - g
                noise = torch.randn_like(d) * cfg.sybil_jitter * d.norm() / np.sqrt(d.numel())
                local[p.id] = g + d + noise
        timing["train"] += time.time() - t0

        # ---- attestation ---------------------------------------------------
        # pass 1: physical check, then canary probing of the identities that passed it
        ok_puf_d, rates_d, trust_d, nat_d = {}, {}, {}, {}
        for p in pop:
            w = local[p.id]
            ok_puf = True
            if cfg.defense == "dual":
                t1 = time.time()
                if cfg.puf == "assumed":
                    ok_puf = p.role != "sybil_virtual"
                elif cfg.puf == "v2":
                    ch = verifier.challenge(p.id, rng_puf)
                    if p.role == "sybil_virtual" and cfg.sybil_puf == "random":
                        resp = rng_syb.random(cfg.puf_bits) < 0.5
                    else:
                        resp = chips[p.chip].respond(ch, rng_puf)
                    ok_puf, sim = verifier.verify(p.id, ch, resp)
                    (puf_a if p.adversarial else puf_h).append(sim)
                else:
                    raise ValueError("defense 'dual' needs puf='assumed' or 'v2'")
                timing["puf"] += time.time() - t1
            ok_puf_d[p.id] = ok_puf
            if cfg.defense in ("sentinel", "dual") and ok_puf:
                t1 = time.time()
                if p.role.startswith("sybil") and cfg.sybil_trust_oracle:
                    trust_d[p.id] = 1.0          # forged attestation (Smart Sybil)
                else:
                    mm = set_flat(copy.deepcopy(model), w)
                    hit = ((backend.predict(mm, canX) > 0.5).float() == canY).squeeze(1)
                    rates_d[p.id] = np.array([float(hit[fm].float().mean()) for fm in fam_masks])
                    att = [(nat_weight.get(c, 0.0), r) for c, r in zip(fam_ids, rates_d[p.id]) if c != 0]
                    wsum = sum(wt for wt, _ in att)
                    nat_d[p.id] = sum(wt * r for wt, r in att) / wsum if wsum > 0 else 1.0
                    if cfg.trust_rule == "per_family":
                        trust_d[p.id] = float(rates_d[p.id].min())
                    elif cfg.trust_rule == "overall":
                        trust_d[p.id] = float(hit.float().mean())
                timing["canary"] += time.time() - t1
        # pass 2 ('delta'): per-class drop with respect to the median of the probed clients
        if cfg.defense in ("sentinel", "dual") and cfg.trust_rule == "delta" and rates_d:
            med = np.median(np.stack(list(rates_d.values())), axis=0)
            for pid, rt in rates_d.items():
                trust_d[pid] = 1.0 - max(0.0, float(np.max(med - rt)))
        thr = (1.0 - cfg.delta_max) if cfg.trust_rule == "delta" else cfg.tau_trust
        for p in pop:
            w = local[p.id]
            ok_beh = True
            if cfg.defense in ("sentinel", "dual") and ok_puf_d[p.id]:
                trust = trust_d[p.id]
                ok_beh = trust >= thr
                if cfg.trust_rule == "delta" and p.id in nat_d:   # gate 1: original test on the natural attack mix
                    ok_beh = ok_beh and nat_d[p.id] >= cfg.tau_trust
                (trust_a if p.adversarial else trust_h).append(round(trust, 4))
            accepted = ok_puf_d[p.id] and ok_beh
            accepted_flags.append(accepted)
            if accepted or cfg.defense not in ("sentinel", "dual"):
                W.append(w); ids.append(p.id)
                wsize.append(sizes[p.id] if sizes[p.id] is not None else sizes.get(p.chip) or 1)

        # ---- aggregation ---------------------------------------------------
        t2 = time.time()
        weights = None
        if W:
            U = torch.stack([w - g for w in W])
            d = cfg.defense
            if d in ("fedavg", "sentinel", "dual"):
                delta, weights = agg.fedavg(U, np.array(wsize) if cfg.weighted_avg else None)
            elif d == "median":
                delta, weights = agg.median(U)
            elif d == "trimmed":
                delta, weights = agg.trimmed_mean(U, cfg.trim_beta)
            elif d == "krum":
                f = cfg.krum_f if cfg.krum_f is not None else len(attackers)
                delta, weights = agg.multi_krum(U, f)
            elif d == "foolsgold":
                delta, weights = fools(U, ids)
            elif d == "fltrust":
                ms = copy.deepcopy(model)
                train_local(ms, data.X_pool[root_idx], data.y_pool[root_idx], cfg.local_steps, cfg.lr)
                delta, weights = agg.fltrust(U, get_flat(ms) - g)
            else:
                raise ValueError(d)
            set_flat(model, g + delta)
        timing["aggregate"] += time.time() - t2

        # ---- bookkeeping ---------------------------------------------------
        if cfg.defense in ("sentinel", "dual"):
            got = dict(zip([p.id for p in pop], accepted_flags))
        else:   # robust aggregators: "accepted" = received a positive weight
            wmap = dict(zip(ids, weights if weights is not None else []))
            got = {p.id: (np.isnan(wmap.get(p.id, np.nan)) or wmap.get(p.id, 0) > 0) for p in pop}
        acc_log["honest_accepted"].append(sum(got[p.id] for p in pop if not p.adversarial))
        acc_log["honest_total"].append(sum(1 for p in pop if not p.adversarial))
        acc_log["adv_accepted"].append(sum(got[p.id] for p in pop if p.adversarial))
        acc_log["adv_total"].append(len(attackers))
        acc_log["trust_honest"].append(trust_h)
        acc_log["trust_adv"].append(trust_a)
        acc_log["puf_honest_min"].append(min(puf_h) if puf_h else None)
        acc_log["puf_adv_max"].append(max(puf_a) if puf_a else None)

        last = r == cfg.rounds - 1
        if (r + 1) % cfg.eval_every == 0 or last:
            t3 = time.time()
            ev = evaluate(backend, model, data, families=last)
            timing["evaluate"] += time.time() - t3
            hist["round"].append(r + 1); hist["acc"].append(ev["acc"]); hist["asr"].append(ev["asr"])
            if verbose and ((r + 1) % 10 == 0 or last):
                print(f"  [{cfg.run_id()}] round {r + 1:3d}  acc {ev['acc']:6.2f}%  asr {ev['asr']:6.2f}%", flush=True)
            if last:
                final = ev

    ha, ht = np.array(acc_log["honest_accepted"]), np.array(acc_log["honest_total"])
    aa, at = np.array(acc_log["adv_accepted"]), np.array(acc_log["adv_total"])
    out = {
        "config": asdict(cfg), "run_id": cfg.run_id(), "alpha_used": alpha, "n_identities": n_ids,
        "final": final, "history": hist, "acceptance": acc_log,
        "honest_acceptance_rate": float(ha.sum() / max(ht.sum(), 1)),
        "adversary_acceptance_rate": float(aa.sum() / max(at.sum(), 1)) if at.sum() else None,
        "timing": timing, "wall_time": time.time() - t_start,
        "versions": _versions(),
    }
    if return_model:
        out["model"] = model
    return out


def _versions() -> dict:
    import platform
    out = {"python": platform.python_version(), "torch": torch.__version__}
    try:
        import aihwkit
        out["aihwkit"] = aihwkit.__version__
    except Exception:  # noqa: BLE001
        out["aihwkit"] = None
    return out
