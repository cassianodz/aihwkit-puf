## Description

_IBM Analog Hardware Acceleration Kit: PUF & Hardware Security Edition_ expands the cutting-edge capabilities of in-memory computing devices beyond artificial intelligence, transforming physical hardware imperfections into robust **Hardware-Intrinsic Security** primitives.

By harnessing the inherent "chaos" of analog crossbar arrays—such as device-to-device variability, programming noise, and read fluctuations—this framework enables the precise simulation, extraction, and evaluation of **Analog Physical Unclonable Functions (PUFs)**.

> :lock: **Security Branch:** This experimental extension shifts the focus from deep learning to cryptographic entropy, utilizing the stochastic properties of Phase-Change Memory (PCM) arrays to generate unclonable silicon fingerprints.

The toolkit has been supercharged with a dedicated `puf` module, introducing:

### 🛡️ Analog Hardware Fingerprinting
A suite of specialized primitives to model and extract unique cryptographic identities directly from analog conductance states:
* **Device Identity Emulation:** `DeviceIdentity` constructs that inject explicit programming noise and physical variations, ensuring that every simulated PCM array is born with a unique, unclonable structural fingerprint.
* **Analog PUF Core:** The `AnalogPUFModel` seamlessly translates digital challenges into analog responses, accurately simulating hardware realities like bounded conductance ranges, read noise, and ADC quantization limits.

### 🧬 Drift-Resilient Signature Enrollment
State-of-the-art signature extraction designed to survive the physical realities of analog memory:
* **Differential Paired Comparison:** The `PUFEnrollment` engine employs an advanced challenge-pairing strategy. By evaluating bits based on relative analog states (`analog(x_a) > analog(x_b)`), the system naturally cancels out the systematic conductance drift of PCM devices, ensuring long-term signature stability.
* **Bias Mitigation:** Automated hardware-aware scaling and offset management to preserve entropy, prevent saturation near hardware limits, and overcome quantization loss.

### 📊 Cryptographic Quality Metrics
A built-in evaluation suite (`PUFMetrics`) to rigorously test the extracted signatures against gold-standard security benchmarks:
* **Uniqueness:** Validates the inter-device Hamming distance to guarantee that no two simulated chips share the same identity.
* **Reliability:** Proves the robustness of the device fingerprint against cycle-to-cycle noise and multiple read operations.
* **Uniformity:** Ensures an optimal balance of the cryptographic keys to maximize systemic entropy and eliminate 0/1 bias.

_Alongside these new security features, the toolkit retains its core high-performant (CUDA-capable) C++ simulator and seamless PyTorch integration, providing the ultimate playground for analog AI and hardware security research._
