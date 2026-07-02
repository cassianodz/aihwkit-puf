"""
puf/enrollment.py — Conductance Programming PUF

Challenge : condutância-alvo g_T[k] em [0.3, 1.7] uS
Response  : sinal do erro de programação (g_prog > g_T ?)

Por que este range: sigma_prog PCM é não-zero apenas para g_T < 2 uS.
Por que seed fixo: reliability = 1.0 por construção.
Referência: Gao et al. 2020, Nature Electronics — Physical Unclonable Functions.
"""
import numpy as np
import torch

class PUFEnrollment:
    def __init__(self, n_challenges=256, round_seed=0, g_low=0.3, g_high=1.7):
        self.n_challenges = n_challenges
        self.round_seed   = round_seed
        self.g_low        = g_low
        self.g_high       = g_high
        self._generate_challenges()

    def _generate_challenges(self):
        rng = np.random.default_rng(seed=self.round_seed)
        self.g_targets = rng.uniform(
            self.g_low, self.g_high, size=self.n_challenges).astype(np.float32)

    def rotate(self, new_round: int):
        self.round_seed = new_round
        self._generate_challenges()

    def extract_signature(self, model) -> np.ndarray:
        device_seed = model.device_identity.device_id * 100_000
        g_tensor    = torch.tensor(self.g_targets).unsqueeze(0)
        torch.manual_seed(device_seed)
        with torch.no_grad():
            g_prog = model.noise_model.apply_programming_noise_to_conductance(
                         g_tensor.clone())
        delta = (g_prog - g_tensor).squeeze().numpy()
        return (delta > 0).astype(bool)

class PUFVerifier:
    def __init__(self, enrollment_db: dict, acceptance_threshold: float = 0.90):
        self.db = enrollment_db
        self.acceptance_threshold = acceptance_threshold

    def verify(self, device_id: int, current_sig) -> tuple:
        if device_id not in self.db:
            return False, 0.0
        hamming    = float(np.mean(current_sig != self.db[device_id]))
        similarity = 1.0 - hamming
        return similarity >= self.acceptance_threshold, round(similarity, 4)

    def enroll(self, device_id: int, signature):
        self.db[device_id] = signature.copy()
