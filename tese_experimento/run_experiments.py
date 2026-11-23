#!/usr/bin/env python3
"""Extended evaluation of Sentinel-Flow + PCM PUF (experiments E1-E9).

Usage (from tese_experimento/):
    python run_experiments.py e1 --workers 4            # one experiment
    python run_experiments.py all --workers 4           # everything
    python run_experiments.py e2 --quick                # smoke test (3 rounds, tiny grids)
    python run_experiments.py e2 --seeds 5              # more repetitions

Each run is saved as results/<exp>/<run_id>.json and skipped if it already
exists, so an interrupted sweep resumes where it stopped. Figures:
    python make_figures.py

Experiments
    e1  replication of the six scenarios over seeds, original vs corrected protocol
    e2  attack landscape: malicious fraction x boosting factor, six defenses
    e3  Sybil scaling: honest clients x virtual identities, five defenses
    e4  heterogeneity: Dirichlet non-IID partitions, trust threshold sweep
    e5  adaptive attackers (family-targeted evasion, false-alarm flooding) vs canary design
    e6  analog non-idealities: PCM drift x ADC resolution (aihwkit inference model)
    e7  device-level PUF characterization (NumPy)
    e8  dual attestation with the device-level PUF in the loop: PUF quality x tau_puf
    e9  overhead and scalability
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROUNDS = 50
SYBIL_IDS = list(range(10, 20))   # original Smart Sybil identities
PHYS = 15                         # original physical adversary


def base_cfg(exp, **kw):
    from sfx.fl import RunConfig
    kw.setdefault("rounds", ROUNDS)
    return RunConfig(exp=exp, **kw)


# ---------------------------------------------------------------------------
# Experiment grids
# ---------------------------------------------------------------------------
def grid_e1(seeds, quick):
    scen = {
        "S1_baseline": dict(attack="none", defense="fedavg"),
        "S2_attack": dict(attack="flip", n_malicious=5, alpha=4.0, defense="fedavg"),
        "S3_sentinel": dict(attack="flip", n_malicious=5, alpha=4.0, defense="sentinel"),
        "S4_dual": dict(attack="flip", n_malicious=5, alpha=4.0, defense="dual"),
        "S5A_sybil_sentinel": dict(attack="smart_sybil", sybil_ids=SYBIL_IDS, sybil_phys=PHYS, alpha=4.0, defense="sentinel"),
        "S5B_sybil_dual": dict(attack="smart_sybil", sybil_ids=SYBIL_IDS, sybil_phys=PHYS, alpha=4.0, defense="dual"),
    }
    protocols = {
        "orig": dict(canary_source="test", canary_mode="fixed", puf="assumed"),
        "fixed": dict(canary_source="pool", canary_mode="resample", puf="v2"),
    }
    jobs = []
    for (pn, pk), (sn, sk), s in itertools.product(protocols.items(), scen.items(), seeds):
        jobs.append(base_cfg("e1", tag=f"{pn}_{sn}", seed=s, **pk, **sk))
    return jobs


def grid_e2(seeds, quick):
    n_mal = [1, 2, 4, 6, 8, 10] if not quick else [2, 8]
    alphas = [1.0, 2.0, 4.0, 8.0, 16.0] if not quick else [1.0, 8.0]
    defenses = ["fedavg", "median", "trimmed", "krum", "fltrust", "sentinel"]
    jobs = []
    for d, s in itertools.product(defenses, seeds):
        jobs.append(base_cfg("e2", tag=f"{d}_m0_a0", seed=s, attack="none", defense=d, eval_every=ROUNDS))
        for m, a in itertools.product(n_mal, alphas):
            jobs.append(base_cfg("e2", tag=f"{d}_m{m}_a{a:g}", seed=s, attack="flip", n_malicious=m,
                                 alpha=a, defense=d, krum_f=m, eval_every=ROUNDS))
    return jobs


def grid_e3(seeds, quick):
    honest = [10, 20, 40] if not quick else [10, 20]
    virtual = [0, 1, 3, 5, 10, 20, 40] if not quick else [0, 10]
    defenses = ["fedavg", "median", "krum", "fltrust", "foolsgold", "sentinel", "dual"]
    jobs = []
    for h, v, d, s in itertools.product(honest, virtual, defenses, seeds):
        jobs.append(base_cfg("e3", tag=f"{d}_h{h}_v{v}", seed=s, n_clients=h + 1, attack="smart_sybil",
                             sybil_mode="copies", sybil_phys=h, n_virtual=v, alpha=4.0, defense=d,
                             puf="v2" if d == "dual" else "none", eval_every=ROUNDS))
    # Sybils that add noise to their copies to look dissimilar (evading similarity-based defenses)
    for j, d, s in itertools.product([0.5, 1.0, 2.0] if not quick else [1.0], ["foolsgold", "krum", "dual"], seeds):
        jobs.append(base_cfg("e3", tag=f"{d}_h20_v10_j{j:g}", seed=s, n_clients=21, attack="smart_sybil",
                             sybil_mode="copies", sybil_phys=20, n_virtual=10, sybil_jitter=j, alpha=4.0,
                             defense=d, puf="v2" if d == "dual" else "none", eval_every=ROUNDS))
    return jobs


def grid_e4(seeds, quick):
    parts = [("iid", dict(partition="contiguous"))] + [
        (f"b{b:g}", dict(partition="dirichlet", beta=b)) for b in ([0.05, 0.1, 0.3, 1.0, 10.0] if not quick else [0.1])]
    taus = [0.4, 0.5, 0.6, 0.7, 0.8] if not quick else [0.6]
    deltas = [0.1, 0.2, 0.3] if not quick else [0.2]
    atk = dict(attack="flip", n_malicious=5, alpha=4.0)
    jobs = []
    for (pn, pk), s in itertools.product(parts, seeds):
        jobs.append(base_cfg("e4", tag=f"{pn}_fedavg_noattack", seed=s, **pk, attack="none", defense="fedavg", eval_every=ROUNDS))
        for d in ["fedavg", "median", "trimmed", "krum", "fltrust", "foolsgold"]:
            jobs.append(base_cfg("e4", tag=f"{pn}_{d}", seed=s, **pk, **atk, defense=d, eval_every=ROUNDS))
        for t in taus:
            jobs.append(base_cfg("e4", tag=f"{pn}_sentinel_tau{t:g}", seed=s, **pk, **atk, defense="sentinel",
                                 tau_trust=t, eval_every=ROUNDS))
        for dl in deltas:
            jobs.append(base_cfg("e4", tag=f"{pn}_delta{dl:g}", seed=s, **pk, **atk, defense="sentinel",
                                 trust_rule="delta", delta_max=dl, canary_mode="stratified", canary_benign_k=50,
                                 eval_every=ROUNDS))
    return jobs


DESIGNS = {
    "orig": dict(defense="sentinel", trust_rule="overall", canary_mode="resample"),
    "strat": dict(defense="sentinel", trust_rule="overall", canary_mode="stratified"),
    "delta": dict(defense="sentinel", trust_rule="delta", delta_max=0.2, canary_mode="stratified", canary_benign_k=50),
    "fedavg": dict(defense="fedavg"),
}


def grid_e5(seeds, quick):
    attackers = {"DoS": dict(attack="targeted", target_family="DoS"),
                 "Probe": dict(attack="targeted", target_family="Probe"),
                 "R2L": dict(attack="targeted", target_family="R2L"),
                 "flood": dict(attack="flood")}
    if quick:
        attackers = {k: attackers[k] for k in ("Probe", "flood")}
    jobs = []
    for (an, ak), (dn, dk), s in itertools.product(attackers.items(), DESIGNS.items(), seeds):
        jobs.append(base_cfg("e5", tag=f"{an}_{dn}", seed=s, n_malicious=5, alpha=4.0, **ak, **dk, eval_every=ROUNDS))
    for (dn, dk), s in itertools.product(DESIGNS.items(), seeds):
        if dn != "fedavg":
            jobs.append(base_cfg("e5", tag=f"none_{dn}", seed=s, attack="none", **dk, eval_every=ROUNDS))
    for k, dn, s in itertools.product([25, 50, 100, 200] if not quick else [25], ["orig", "delta"], seeds):
        jobs.append(base_cfg("e5", tag=f"ksweep_Probe_{dn}_k{k}", seed=s, n_malicious=5, alpha=4.0,
                             attack="targeted", target_family="Probe", canary_k=k, **DESIGNS[dn], eval_every=ROUNDS))
    return jobs


def grid_e8(seeds, quick):
    sig = [0.1, 0.2, 0.3, 0.5] if not quick else [0.2]
    taus = [0.8, 0.9, 0.95] if not quick else [0.9]
    sy = dict(attack="smart_sybil", sybil_ids=SYBIL_IDS, sybil_phys=PHYS, alpha=4.0, defense="dual", puf="v2",
              eval_every=ROUNDS)
    jobs = [base_cfg("e8", tag=f"sig{sg:g}_tau{t:g}", seed=s, puf_sigma_static=sg, puf_tau=t, **sy)
            for sg, t, s in itertools.product(sig, taus, seeds)]
    jobs += [base_cfg("e8", tag="random_guess", seed=s, sybil_puf="random", **sy) for s in seeds]
    jobs += [base_cfg("e8", tag="nomask_reps1", seed=s, puf_keep=1.0, puf_reps=1, **sy) for s in seeds]
    return jobs


def grid_e9(seeds, quick):
    ns = [10, 20, 40, 80] if not quick else [10, 20]
    jobs = []
    for n, d in itertools.product(ns, ["fedavg", "sentinel", "dual"]):
        jobs.append(base_cfg("e9", tag=f"{d}_n{n}", seed=0, n_clients=n, rounds=10, defense=d,
                             puf="v2" if d == "dual" else "none", eval_every=10))
    return jobs


FL_GRIDS = {"e1": grid_e1, "e2": grid_e2, "e3": grid_e3, "e4": grid_e4, "e5": grid_e5, "e8": grid_e8, "e9": grid_e9}
DEFAULT_SEEDS = {"e1": 5, "e2": 2, "e3": 2, "e4": 2, "e5": 3, "e6": 3, "e8": 2, "e9": 1}


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
_DATA: dict = {}
_DATA_PATH = None


def _init(data_path):
    global _DATA_PATH
    _DATA_PATH = data_path
    import torch
    torch.set_num_threads(1)


def _data(source):
    from sfx.data import load_nslkdd
    if source not in _DATA:
        _DATA[source] = load_nslkdd(_DATA_PATH, canary_source=source)
    return _DATA[source]


def _save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, default=float))
    tmp.replace(path)


def run_fl_job(args):
    cfg_dict, out_dir = args
    from sfx.fl import RunConfig, run_federation
    cfg = RunConfig(**cfg_dict)
    path = Path(out_dir) / cfg.exp / f"{cfg.run_id()}.json"
    if path.exists():
        return cfg.run_id(), "skip", 0.0
    t = time.time()
    try:
        res = run_federation(cfg, _data(cfg.canary_source))
        _save(path, res)
        f = res["final"]
        return cfg.run_id(), f"acc {f['acc']:.2f} asr {f['asr']:.2f}", time.time() - t
    except Exception:  # noqa: BLE001
        return cfg.run_id(), "ERROR\n" + traceback.format_exc(), time.time() - t


def run_e6_job(args):
    seed, out_dir, quick, rounds = args
    import numpy as np
    import torch
    from sfx.analog import AnalogBackend
    from sfx.fl import RunConfig, evaluate, run_federation
    path = Path(out_dir) / "e6" / f"e6__s{seed}.json"
    if path.exists():
        return f"e6 s{seed}", "skip", 0.0
    t = time.time()
    data = _data("pool")
    cfg = RunConfig(exp="e6", seed=seed, rounds=rounds, backend="digital", eval_every=rounds)
    res = run_federation(cfg, data, return_model=True)
    model = res.pop("model")
    times = [1.0, 60.0, 3600.0, 86400.0, 604800.0, 2592000.0, 31536000.0] if not quick else [1.0, 86400.0]
    bits = [4, 5, 6, 7, 8, 9] if not quick else [5, 9]
    reps = 5 if not quick else 1
    rows = []
    for gdc, ti, b in itertools.product([True, False], times, bits):
        accs, asrs = [], []
        for _ in range(reps):
            be = AnalogBackend(kind="inference", t_inference=ti, drift_compensation=gdc, out_bits=b, inp_bits=7)
            ev = evaluate(be, model, data)
            accs.append(ev["acc"]); asrs.append(ev["asr"])
        rows.append({"gdc": gdc, "t_inference": ti, "out_bits": b, "acc_mean": float(np.mean(accs)),
                     "acc_sd": float(np.std(accs)), "asr_mean": float(np.mean(asrs)), "asr_sd": float(np.std(asrs))})
    refs = {}
    for kind in ["digital", "preset"]:
        evs = [evaluate(AnalogBackend(kind=kind), model, data) for _ in range(1 if kind == "digital" else reps)]
        refs[kind] = {"acc_mean": float(np.mean([e["acc"] for e in evs])), "asr_mean": float(np.mean([e["asr"] for e in evs]))}
    _save(path, {"seed": seed, "grid": rows, "reference": refs, "train_final": res["final"]})
    return f"e6 s{seed}", "done", time.time() - t


def run_e7_job(args):
    part, out_dir, quick = args
    from sfx import puf_experiments as pe
    from sfx.puf import PUFParams
    path = Path(out_dir) / "e7" / f"e7__{part}.json"
    if path.exists():
        return f"e7 {part}", "skip", 0.0
    t = time.time()
    base = PUFParams()
    if part == "genuine_impostor":
        out = [pe.genuine_impostor(base, seed=s, n_rounds=10 if quick else 50) for s in range(1 if quick else 3)]
        out_nomask = pe.genuine_impostor(PUFParams(keep=1.0, reps=1), seed=0, n_rounds=10 if quick else 50)
        obj = {"default": out, "nomask_reps1": out_nomask}
    elif part == "reliability_grid":
        obj = pe.reliability_grid([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0] if not quick else [0.2, 0.5],
                                  [1.0, 0.5, 0.25, 0.1, 0.05] if not quick else [1.0, 0.25],
                                  [1, 3, 5] if not quick else [3], base,
                                  n_rounds=5 if quick else 20)
    elif part == "level_time":
        obj = pe.level_time_grid([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0] if not quick else [1.0, 5.0],
                                 [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5] if not quick else [1.0, 1e4],
                                 base, n_rounds=5 if quick else 20)
    elif part == "modeling_attack":
        obj = pe.modeling_attack([0, 256, 1024, 2560, 5120, 12800, 25600, 51200, 102400] if not quick else [0, 2560],
                                 base, n_targets=2 if quick else 5, n_test=10 if quick else 40)
    elif part == "legacy":
        obj = pe.legacy_puf_check()
    else:
        raise ValueError(part)
    _save(path, obj)
    return f"e7 {part}", "done", time.time() - t


# ---------------------------------------------------------------------------
def build_jobs(exp, seeds_n, quick, out_dir, rounds):
    global ROUNDS
    if rounds:
        ROUNDS = rounds
    if quick:
        ROUNDS = 3
    seeds = list(range(seeds_n if seeds_n else (1 if quick else DEFAULT_SEEDS.get(exp, 1))))
    if exp in FL_GRIDS:
        cfgs = FL_GRIDS[exp](seeds, quick)
        if quick:
            cfgs = [replace(c, rounds=min(c.rounds, 3), eval_every=1) for c in cfgs]
        return [(run_fl_job, (asdict(c), out_dir)) for c in cfgs]
    if exp == "e6":
        return [(run_e6_job, (s, out_dir, quick, ROUNDS)) for s in seeds]
    if exp == "e7":
        return [(run_e7_job, (p, out_dir, quick)) for p in
                ["modeling_attack", "reliability_grid", "level_time", "genuine_impostor", "legacy"]]
    raise ValueError(exp)


def _dispatch(job):
    fn, args = job
    return fn(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", choices=["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "all"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds (default per experiment)")
    ap.add_argument("--rounds", type=int, default=None, help=f"communication rounds (default {ROUNDS})")
    ap.add_argument("--quick", action="store_true", help="smoke test: 3 rounds, 1 seed, tiny grids")
    ap.add_argument("--data", default=str(HERE / "dataset.txt"))
    ap.add_argument("--out", default=None, help="results directory (default results/ or results_quick/)")
    a = ap.parse_args()
    out_dir = a.out or str(HERE / ("results_quick" if a.quick else "results"))
    exps = ["e7", "e6", "e1", "e9", "e8", "e5", "e4", "e3", "e2"] if a.exp == "all" else [a.exp]
    jobs = []
    for e in exps:
        jobs += build_jobs(e, a.seeds, a.quick, out_dir, a.rounds)
    print(f"{len(jobs)} jobs, {a.workers} workers, results in {out_dir}", flush=True)
    t0 = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.data,), maxtasksperchild=200) as pool:
        for rid, msg, dt in pool.imap_unordered(_dispatch, jobs):
            done += 1
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            print(f"[{done:4d}/{len(jobs)}] {rid:45s} {msg}  ({dt:5.1f}s, elapsed {el/60:5.1f} min, eta {eta/60:5.1f} min)", flush=True)
    print(f"finished in {(time.time() - t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
