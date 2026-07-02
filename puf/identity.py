import json
import numpy as np
from pathlib import Path

PARAMS_DIR = Path(__file__).parent / "population_params"

class DeviceIdentity:
    SUPPORTED_TYPES = ["pcm", "reram"]

    def __init__(self, device_id: int, device_type: str = "pcm"):
        if device_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"device_type deve ser um de {self.SUPPORTED_TYPES}")
        self.device_id = device_id
        self.device_type = device_type
        self.rng = np.random.default_rng(seed=device_id)
        fname = "pcm_ibm_calibrated.json" if device_type == "pcm" else "reram_hafnium.json"
        with open(PARAMS_DIR / fname) as f:
            self.params = json.load(f)
        self.g_max = float(self.params["conductance_range"]["g_max_uS"])
        self._sample_identity()

    def _sample_identity(self):
        inter = self.params["inter_device"]
        if self.device_type == "pcm":
            self.g_prog_offset  = self.rng.normal(0.0, inter["g_prog_offset"]["sigma_uS"])
            self.drift_nu_offset = self.rng.normal(0.0, inter["drift_nu_offset"]["sigma"])
            self.qs_scale = float(np.clip(
                self.rng.normal(inter["Qs_scale_offset"]["mu_scale"],
                                inter["Qs_scale_offset"]["sigma_scale"]), 0.5, 2.0))

    def to_noise_model(self):
        from aihwkit.inference import PCMLikeNoiseModel
        coeffs = [-1.1731, 1.9650, 0.2635 + float(self.g_prog_offset)]
        return PCMLikeNoiseModel(
            prog_coeff=coeffs, g_max=self.g_max,
            drift_scale=float(1.0 + self.drift_nu_offset * 10),
            read_noise_scale=float(self.qs_scale))

    def summary(self) -> dict:
        return {"device_id": self.device_id, "device_type": self.device_type,
                "g_max_uS": self.g_max,
                "g_prog_offset_uS": round(float(self.g_prog_offset), 6),
                "drift_nu_offset": round(float(self.drift_nu_offset), 6),
                "qs_scale": round(float(self.qs_scale), 6)}

    def __repr__(self):
        return f"DeviceIdentity(id={self.device_id}, type={self.device_type})"
