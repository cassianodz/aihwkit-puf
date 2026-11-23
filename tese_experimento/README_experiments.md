# Extended evaluation: Sentinel-Flow + PCM PUF (experiments E1–E9)

This folder adds a reproducible experiment suite on top of `fl_sentinel_puf.py`.
The original script is untouched. With the *original protocol* settings the
suite reproduces its six scenarios (same data, split, model, local training,
attack, thresholds and analog preset).

```
tese_experimento/
├── fl_sentinel_puf.py        original experiment (unchanged)
├── dataset.txt               NSL-KDD KDDTrain+ (125,973 flows, 122 features after one-hot)
├── sfx/                      the suite
│   ├── data.py               loading, controller canary pool, IID / Dirichlet partitions, canary sampling
│   ├── models.py             MLP 122-64-1 and local training (3 full-batch Adam steps, lr 0.005)
│   ├── analog.py             aihwkit backends: PCMPreset (original), InferenceRPUConfig + PCM noise/drift
│   ├── aggregators.py        FedAvg, Median, Trimmed mean, Multi-Krum, FoolsGold, FLTrust
│   ├── puf.py                device-level PCM PUF (cell-level variation, pairwise challenges)
│   ├── puf_experiments.py    PUF characterization (NumPy only)
│   └── fl.py                 federation loop: attacks, defenses, Sentinel-Flow, dual attestation
├── run_experiments.py        runs E1–E9, one JSON per run in results/<exp>/
├── make_figures.py           figures (PDF + PNG) and LaTeX tables in figures/
└── README_experiments.md
```

## Running

```bash
conda activate aihwkit_puf          # torch + aihwkit, as for fl_sentinel_puf.py
cd ~/aihwkit-puf/tese_experimento

python run_experiments.py all --quick --workers 4   # smoke test, about 1 minute
python run_experiments.py all --workers 4           # everything (resumable)
python make_figures.py                               # figures/ and figures/tables/
```

Useful options: `--seeds N` (more repetitions), `--rounds R`, `--out DIR`.
Runs that already have a JSON file are skipped, so an interrupted sweep
resumes where it stopped. Delete `results/<exp>/` to rerun an experiment.

Reference run (2 CPU cores, aihwkit 1.2.0, torch 2.15, Python 3.13): about
2.5 hours for all 1,035 runs with the default seeds. Time grows linearly with
seeds; with 8 cores and `--seeds 5` everywhere expect roughly 3–4 hours.

## What each experiment answers

| Exp. | Question | Design | Main output |
|---|---|---|---|
| E1 | Are the Table 4 results stable, and do they survive the corrected protocol? | 6 scenarios × 5 seeds × {original protocol, corrected protocol} | `e1_final_ci`, `e1_asr_rounds`, `tables/e1_table.tex` |
| E2 | How does Sentinel-Flow compare with standard robust aggregation across attack strength? | malicious fraction {5–50%} × boosting α {1–16} × {FedAvg, Median, Trimmed mean, Multi-Krum, FLTrust, Sentinel-Flow} | `e2_landscape_heatmaps`, `e2_landscape_3d` |
| E3 | How many emulated identities break each defense? | honest clients {10, 20, 40} × virtual identities {0–40} × {FedAvg, Median, Multi-Krum, FLTrust, FoolsGold, Sentinel-Flow, dual} | `e3_sybil_scaling`, `e3_sybil_heatmaps`, `e3_sybil_3d` |
| E4 | Does behavioral attestation reject honest clients when traffic is heterogeneous? | Dirichlet β {0.05–10, IID} × trust threshold τ / differential rule δ, plus baselines | `e4_noniid_heatmaps`, `e4_noniid_defenses`, `e4_false_rejection_3d` |
| E5 | Can an adversary that knows the canary design evade it? | family-targeted evasion (DoS, Probe, R2L) and false-alarm flooding × canary design | `e5_adaptive_heatmaps`, `e5_canary_budget`, `tables/e5_honest_table.tex` |
| E6 | Does the NIDS stay accurate on PCM over time, and how much ADC precision is needed? | time since programming {1 s–1 year} × ADC bits {4–9} × drift compensation | `e6_drift_adc_heatmaps`, `e6_drift_adc_3d`, `e6_drift_lines` |
| E7 | Does a realistic PCM PUF meet τ_puf = 0.90, and how fast can it be modelled? | static variation × masking × repetitions; level × read delay; modeling attack | `e7_*`, `tables/e7_puf_table.tex` |
| E8 | With the real PUF in the loop, what does PUF noise cost honest clients? | Smart Sybil + dual attestation, σ_s × τ_puf | `e8_puf_in_loop`, `tables/e8_table.tex` |
| E9 | What does attestation cost per round? | {10, 20, 40, 80} clients × {FedAvg, Sentinel-Flow, dual} | `e9_overhead`, `tables/e9_bytes.tex` |

## Protocol corrections (used by default, "corrected protocol")

* **Canaries are not test data.** A stratified pool of 2,020 flows (8% of the
  test split, at least 5 per attack family) is removed from the test split and
  given to the controller; canaries are re-drawn from it every round. The
  original protocol (first 100 attack flows of the test split, kept in the test
  set) is still available with `canary_source="test", canary_mode="fixed"`.
* **The PUF decision is computed.** `puf="v2"` challenges every identity with a
  fresh challenge and compares the chip's response with the enrolled one. The
  original `puf_ok = (i == phys_hacker_id)` is `puf="assumed"`.
* **Two-sided differential canary test** (`trust_rule="delta"`, used in E4/E5):
  100 attack canaries stratified across families plus 50 benign canaries. A
  client must (1) pass the original test with its per-family detection
  re-weighted to the natural attack mix of the pool (≥ τ = 0.6) and (2) stay
  within δ = 0.2 of the median of its peers on every class, benign included.
  Gate (2) stops family-targeted evasion and false-alarm flooding; gate (1)
  stops poisoning that enters in the first rounds and would otherwise drag
  the peer median down.
* **The Smart Sybil still forges the behavioral check** (`trust = 1.0`), which
  is the threat model of Scenario 5; the difference is that its virtual
  identities must now answer PUF challenges with the only chip the adversary
  owns.

## The device-level PUF (sfx/puf.py)

`puf/enrollment.py` derives the response from noise drawn with
`torch.manual_seed(device_id * 100_000)`, so the response never changes and
does not depend on the challenge; with a fresh draw it matches the enrollment
only by chance (≈0.50). The coefficients in `identity.py` are also in the
reverse of aihwkit's order, which makes σ_prog negative for every challenge.

The new model applies the calibrated "inter_device" statistics of
`pcm_ibm_calibrated.json` cell by cell (in PCM papers a *device* is one cell):
a static single-shot programming offset (σ_s = 0.5 µS), a static drift-exponent
offset and a static read-noise scale per cell, plus fresh programming noise,
drift and 1/f read noise from aihwkit's `PCMLikeNoiseModel`. A challenge is a
target level and 256 cell pairs; the response bit says which cell of each pair
ended up with the higher conductance. Without masking the bit-error rate is
≈29%, so the default verifier keeps only the 25% most separated pairs
(dark-bit masking) and averages 3 programming cycles per response.

## Figures: 2-D or 3-D?

Every three-variable result has a heatmap with the values printed in the
cells and a 3-D surface (`*_3d.pdf`). For the article, the heatmaps are the
safer choice: a reader can read every value, nothing hides behind a peak, and
they survive grayscale printing. The 3-D surfaces work well in slides and
posters; if one goes in the paper, `e6_drift_adc_3d` (a smooth surface) reads
best.
