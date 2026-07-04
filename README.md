# aihwkit-puf — IBM Analog Hardware Acceleration Kit: PUF & Hardware Security Edition

## Description

**aihwkit-puf** expands the capabilities of in-memory computing devices beyond artificial intelligence, transforming physical hardware imperfections into robust **Hardware-Intrinsic Security** primitives[cite: 1]. It serves as an extension of the **Sentinel-Flow** framework (Zago et al. 2025), introducing a hardware authentication layer based on analog Phase-Change Memory (PCM) and ReRAM chips to guarantee the resilience of Federated Network Intrusion Detection Systems (NIDS) in Software-Defined Networks (SDN) via Behavioral Attestation with Active Semantic Probing and an Analog PUF Hardware Root-of-Trust[cite: 2].

By harnessing the inherent "chaos" of analog crossbar arrays—such as device-to-device variability, programming noise, and read fluctuations—this framework enables the precise simulation, extraction, and evaluation of **Analog Physical Unclonable Functions (PUFs)**[cite: 1].

> :lock: **Security Branch:** This experimental extension shifts the focus from deep learning to cryptographic entropy, utilizing the stochastic properties of Phase-Change Memory (PCM) arrays to generate unclonable silicon fingerprints[cite: 1].

---

## Key Features

The toolkit has been supercharged with a dedicated `puf` module alongside an interactive dynamic experimentation interface, introducing:

### 🛡️ Analog Hardware Fingerprinting
A suite of specialized primitives to model and extract unique cryptographic identities directly from analog conductance states[cite: 1]:
* **Device Identity Emulation:** `DeviceIdentity` constructs inject explicit programming noise and physical variations, ensuring that every simulated PCM array is born with a unique, unclonable structural fingerprint[cite: 1].
* **Analog PUF Core:** The `AnalogPUFModel` seamlessly translates digital challenges into analog responses, accurately simulating hardware realities like bounded conductance ranges, read noise, and ADC quantization limits[cite: 1].

### 🧬 Drift-Resilient Signature Enrollment
State-of-the-art signature extraction designed to survive the physical realities of analog memory[cite: 1]:
* **Differential Paired Comparison:** The `PUFEnrollment` engine employs an advanced challenge-pairing strategy[cite: 1]. By evaluating bits based on relative analog states (`analog(x_a) > analog(x_b)`), the system naturally cancels out the systematic conductance drift of PCM devices, ensuring long-term signature stability[cite: 1].
* **Bias Mitigation:** Automated hardware-aware scaling and offset management preserve entropy, prevent saturation near hardware limits, and overcome quantization loss[cite: 1].

### 📊 Cryptographic Quality Metrics
A built-in evaluation suite (`PUFMetrics`) rigorously tests extracted signatures against gold-standard security benchmarks based on Maiti & Schaumont 2011 (IEEE Trans. VLSI)[cite: 1, 2]:
* **Uniqueness (~50%):** Validates the inter-device Hamming distance to guarantee that pairs of devices are maximamente distinct and no two simulated chips share the same identity[cite: 1, 2].
* **Reliability (>90%):** Proves the robustness of the device fingerprint against cycle-to-cycle noise and multiple read operations over time[cite: 1, 2].
* **Uniformity (~50%):** Ensures an optimal balance of balanced bits (maximum entropy) across the population to eliminate 0/1 bias[cite: 1, 2].

### 🌐 Dynamic Experimentation & Simulation
* **Interactive Live Simulation Dashboard:** A dedicated, standalone interactive webpage (`sentinel_puf_presentation.html`) provides a dynamic, visual execution of the experiment[cite: 2]. It includes a drag-and-drop network topology interface, live simulation charts, and an interactive demonstration of Sybil attack detections[cite: 2].

---

## Repository Structure

```
aihwkit-puf/
├── puf/                          ← PUF module (analog hardware simulation)[cite: 2]
│   ├── __init__.py[cite: 2]
│   ├── identity.py               ← DeviceIdentity: unique physical parameters per chip[cite: 2]
│   ├── analog_model.py           ← AnalogPUFModel: MLP with hardware noise[cite: 2]
│   ├── enrollment.py             ← PUFEnrollment + PUFVerifier[cite: 2]
│   ├── metrics.py                ← Uniqueness, Reliability, Uniformity (Maiti 2011)[cite: 2]
│   └── population_params/[cite: 2]
│       ├── pcm_ibm_calibrated.json   ← IBM PCM parameters (Joshi et al. 2020)[cite: 2]
│       └── reram_hafnium.json        ← CMO/HfOx parameters (Falcone et al. 2025)[cite: 2]
│
├── integration/                  ← Sentinel-Flow + PUF Protocol[cite: 2]
│   ├── __init__.py[cite: 2]
│   └── sentinel_flow_puf.py      ← SentinelFlowPUF: protocol orchestrator[cite: 2]
│
├── fl_sentinel_puf.py            ← Complete FL experiment script (4 scenarios)[cite: 2]
├── sentinel_puf_results.json     ← Generated experiment raw data[cite: 2]
├── sentinel_puf_presentation.html ← Interactive dynamic experiment viewer page[cite: 2]
├── sentinel_puf_results.png      ← Evaluation charts generated by the experiment[cite: 2]
└── dataset.txt                   ← KDD/NSL-KDD dataset[cite: 2]
```

---

## Installation

```bash
# 1. Clone the repository
git clone [https://github.com/cassianodz/aihwkit-puf.git](https://github.com/cassianodz/aihwkit-puf.git)
cd aihwkit-puf

# 2. Create and activate the conda environment
conda create -n aihwkit_puf python=3.10 -y
conda activate aihwkit_puf

# 3. Install dependencies
pip install -v -e .                       # Installs aihwkit from fork source
pip install torch pandas scikit-learn matplotlib
```
*(Note: Alongside new security features, the toolkit retains its core high-performant C++ simulator and seamless PyTorch integration[cite: 1].)*

---

## Usage

### 1. Run the Full Experiment
Execute the full federated learning pipeline across 4 automated evaluation environments (Baseline, Weight Scaling Attack, Sentinel-Flow, and Sentinel-Flow+PUF)[cite: 2]:

```bash
# Ensure your environment is active and dataset is in place
conda activate aihwkit_puf
cp /path/to/kdd.txt dataset.txt

# Run the 4 evaluation scenarios
python fl_sentinel_puf.py
```

**Expected Output:**
```
CENÁRIO 1: FedAvg Honesto     → Acurácia: 86.60%  ASR:  7.20%
CENÁRIO 2: Weight Scaling     → Acurácia: 53.27%  ASR: 100.00%
CENÁRIO 3: Sentinel-Flow      → Acurácia: 94.29%  ASR:  6.69%
CENÁRIO 4: Sentinel-Flow+PUF  → Acurácia: 94.29%  ASR:  6.69%
Exportado: sentinel_puf_results.json
```

### 2. Validate the PUF Simulator
To run a fast validation check on the analog population parameters and quality metrics execution[cite: 2]:

```bash
python -c "
from puf.identity   import DeviceIdentity
from puf.analog_model import AnalogPUFModel
from puf.enrollment import PUFEnrollment
from puf.metrics    import PUFMetrics

devices = [DeviceIdentity(i, 'pcm') for i in range(20)]
models  = [AnalogPUFModel(d) for d in devices]
enroll  = PUFEnrollment(n_challenges=512)
sigs    = [[enroll.extract_signature(m) for _ in range(3)] for m in models]
PUFMetrics.full_report(sigs)
"
```

**Expected Output:**
```
PUF QUALITY REPORT
Uniqueness  : 0.501 +/- 0.022  (ideal ~0.50)  [OK]
Reliability : 1.000 (min 1.000)  (ideal >0.90)  [OK]
Overall     : PASSED
```

### 3. Launch the Dynamic Experiment Page
Open `sentinel_puf_presentation.html` directly in any modern web browser[cite: 2]. This standalone browser page utilizes React and Recharts via CDN to provide an interactive playground[cite: 2]:
* **Live Simulation:** Interact dynamically with the network topology layout, perform chip drag-and-drops, and watch execution charts update in real time[cite: 2].
* **Sybil Attack Sandbox:** View interactive demonstrations of hardware-rooted identity validation blocking rogue nodes[cite: 2].
* **Data Import Engine:** Upload your real experiment file (`sentinel_puf_results.json`) to populate the interactive dashboards with your specific simulation results[cite: 2].

---

## Protocol Architecture

For each Federated Learning (FL) round, the orchestrator executes the following pipeline[cite: 2]:
1. **[PUF Challenge]:** SDN controller sends a challenge sequence to the client chip (target conductances within the $[0.3, 1.7]\ \mu\text{S}$ range)[cite: 2].
2. **[PUF Response]:** The analog chip returns a response derived from the hardware PCM programming error signature[cite: 2].
3. **[PUF Verification]:** SDN controller checks identity validity: $\text{Hamming}(\text{response}, \text{enrolled}) < 10\%$[cite: 2].
4. **[Behavioral Probing]:** SDN controller injects canary flows (e.g., 100 sample attack vectors) to test node outputs[cite: 2].
5. **[Behavioral Evaluation]:** SDN controller computes the local model's Trust Score: $\text{Trust Score} = \text{correct detections} / \text{total probes}$[cite: 2].
6. **[Decision Filter]:** The updates are accepted if and only if both the PUF hardware validation passes AND the Trust Score is $\ge 60\%$[cite: 2].
7. **[Aggregation]:** Weighted FedAvg execution is performed strictly over the approved client updates[cite: 2].
8. **[Rollback Safeguard]:** If zero clients pass verification, the round is aborted and rolls back to the global model state of the prior round[cite: 2].

---

## Simulation Parameters

| Parameter | Value | Source / Context |
|-----------|-------|------------------|
| Device Type | IBM PCM 90nm ($\text{Ge}_2\text{Sb}_2\text{Te}_5$) | Joshi et al. 2020[cite: 2] |
| $g_{\max}$ | $25\ \mu\text{S}$ | Joshi et al. 2020[cite: 2] |
| Challenge Range | $[0.3, 1.7]\ \mu\text{S}$ | Non-zero noise floor regime[cite: 2] |
| $\sigma_{\text{prog}}$ (inter-device) | $0.5\ \mu\text{S}$ | Single-shot programming variation[cite: 2] |
| $\sigma_{\text{prog}}$ (intra-device) | $\sim 0.75\ \mu\text{S} \ @ \ g_T=1.0\ \mu\text{S}$ | Joshi 2020, polynomial noise scaling[cite: 2] |
| PUF Threshold | $90\%$ similarity | Standard security baseline literature[cite: 2] |
| $n_{\text{challenges}}$ | 512 bits | Validated target: uniqueness $\sim 50\%$[cite: 2] |

> 📌 **Methodological Note on Variance ($\sigma = 0.5\ \mu\text{S}$):** While Joshi et al. 2020 achieves a tight error tolerance of $\sigma = 0.038\ \mu\text{S}$ by utilizing multi-cycle iterative write-verify loops, this security application purposefully deploys a **single-shot programming methodology**[cite: 2]. In hardware security, systemic physical variability is an asset rather than a defect; a variance of $0.5\ \mu\text{S}$ accurately represents initial hardware state distributions prior to iterative feedback adjustments[cite: 2].

---

## Fork Contributions vs. Original Framework

| Component | original `aihwkit` | This `aihwkit-puf` Fork |
|------------|-----------------|-----------|
| `PCMLikeNoiseModel` |  Hardcoded inside engine[cite: 2] |  Reused & exposed modularly[cite: 2] |
| Inter-device Variation |  Not modeled[cite: 2] |  Supported natively via `DeviceIdentity`[cite: 2] |
| PUF Enrollment / Verification |  Missing[cite: 2] |  Supported natively via `PUFEnrollment`[cite: 2] |
| Security Metrics Engine |  Missing[cite: 2] |  Supported natively via `PUFMetrics`[cite: 2] |
| FL / NIDS System Integration |  Missing[cite: 2] |  Supported natively via `SentinelFlowPUF`[cite: 2] |
| Sybil Attack Mitigation Layers |  Missing[cite: 2] |  Supported natively via PUF Root-of-Trust verification[cite: 2] |

---

## References

* Zago et al. 2025 — *Sentinel-Flow* (Core Behavioral Attestation Framework)[cite: 2]
* Joshi et al. 2020 — PCM inference noise model. *Nature Communications* 11, 2473[cite: 2]
* Le Gallo et al. 2018 — PCM drift characterization. *Advanced Electronic Materials* 4[cite: 2]
* Falcone et al. 2025 — CMO/HfOx ReRAM modeling. arXiv:2502.04524[cite: 2]
* Maiti & Schaumont 2011 — PUF quality metrics. *IEEE Trans. VLSI*[cite: 2]
* Gao et al. 2020 — Physical Unclonable Functions. *Nature Electronics*[cite: 2]
