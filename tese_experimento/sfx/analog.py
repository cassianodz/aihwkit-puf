"""Analog inference backends built on IBM aihwkit.

kind='preset'    : convert_to_analog(model, PCMPreset()) exactly as in
                   fl_sentinel_puf.py (IO noise and 7-bit DAC / 9-bit ADC,
                   no programming noise and no drift).
kind='inference' : InferenceRPUConfig + PCMLikeNoiseModel (programming noise,
                   conductance drift and 1/f read noise measured on IBM PCM)
                   with optional global drift compensation; converter
                   resolutions are configurable.
kind='digital'   : the floating-point model itself (reference).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

try:
    from aihwkit.inference import GlobalDriftCompensation, PCMLikeNoiseModel
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.presets import PCMPreset
    AIHWKIT = True
except ImportError:  # pragma: no cover - lets the PUF-only experiments run anywhere
    AIHWKIT = False


def bits_to_res(bits: int) -> float:
    """aihwkit resolution for a converter with the given number of bits."""
    return 1.0 / (2 ** bits - 2)


@dataclass
class AnalogBackend:
    kind: str = "preset"
    t_inference: float = 0.0       # seconds after programming
    drift_compensation: bool = True
    inp_bits: int = 7
    out_bits: int = 9
    out_noise: float | None = None  # None -> aihwkit default of the config
    g_max: float = 25.0
    prog_noise_scale: float = 1.0
    read_noise_scale: float = 1.0
    drift_scale: float = 1.0

    def __post_init__(self):
        if self.kind != "digital" and not AIHWKIT:
            raise RuntimeError("aihwkit is required for analog backends (kind='digital' works without it)")

    def rpu_config(self):
        if self.kind == "preset":
            return PCMPreset()
        cfg = InferenceRPUConfig()
        cfg.forward.inp_res = bits_to_res(self.inp_bits)
        cfg.forward.out_res = bits_to_res(self.out_bits)
        if self.out_noise is not None:
            cfg.forward.out_noise = self.out_noise
        cfg.noise_model = PCMLikeNoiseModel(g_max=self.g_max, prog_noise_scale=self.prog_noise_scale,
                                            read_noise_scale=self.read_noise_scale,
                                            drift_scale=self.drift_scale)
        if self.drift_compensation:
            cfg.drift_compensation = GlobalDriftCompensation()
        return cfg

    def analog_copy(self, model: torch.nn.Module) -> torch.nn.Module:
        """A programmed analog copy of the model (one programming event)."""
        if self.kind == "digital":
            return copy.deepcopy(model).eval()
        am = convert_to_analog(copy.deepcopy(model), rpu_config=self.rpu_config())
        am.eval()
        if self.kind == "inference":
            am.drift_analog_weights(self.t_inference)   # programs, drifts and calibrates GDC
        return am

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, X: torch.Tensor) -> torch.Tensor:
        return self.analog_copy(model)(X)
