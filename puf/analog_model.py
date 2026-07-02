import torch
import torch.nn as nn
import torch.nn.functional as F

class AnalogPUFModel(nn.Module):
    """
    MLP com simulação de hardware analógico PCM/ReRAM usando offset scheme.

    CORREÇÃO FÍSICA: PCM não suporta condutâncias negativas.
    Pesos são mapeados para [0, g_max] via offset:
        g_target = w * (g_max/2 / w_max) + g_max/2

    Referência: Joshi et al. 2020, Nature Communications.
    """
    def __init__(self, device_identity, input_dim=78, hidden_dim=64, t_inference=25.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.device_identity = device_identity
        self.noise_model = device_identity.to_noise_model()
        self.g_max    = device_identity.g_max
        self.g_offset = self.g_max / 2.0
        self.t_inference = t_inference

    def _apply_analog_noise(self, weight: torch.Tensor) -> torch.Tensor:
        w_max = weight.abs().max().clamp(min=1e-8)
        scale = self.g_offset / w_max
        g_target = weight * scale + self.g_offset          # [0, g_max]
        g_prog   = self.noise_model.apply_programming_noise_to_conductance(g_target.clone())
        g_noisy  = self.noise_model.apply_drift_noise_to_conductance(
                       g_prog, g_target.clone(), self.t_inference)
        return (g_noisy - self.g_offset) / scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1  = self._apply_analog_noise(self.fc1.weight)
        out = F.relu(F.linear(x, w1, self.fc1.bias))
        w2  = self._apply_analog_noise(self.fc2.weight)
        return torch.sigmoid(F.linear(out, w2, self.fc2.bias))

    def forward_digital(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
