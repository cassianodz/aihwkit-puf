"""
fl_sentinel_puf.py
==================
Experimento completo: Sentinel-Flow + PUF via In-Memory Computing (Analog AI).

Nesta versão, a matriz do NIDS não é digital. O Aprendizado Federado roda 
dentro de matrizes analógicas PCM usando o IBM Analog Hardware Acceleration Kit.
O chip acelera a inferência e usa suas próprias não-idealidades físicas como PUF.

Cenários:
    1. FedAvg Honesto       — baseline analógico sem ataque
    2. Ataque Agressivo      — clean-label poisoning + weight scaling (4.0x)
    3. Sentinel-Flow         — atestação comportamental via canary flows
    4. Sentinel-Flow + PUF   — comportamental + autenticação analógica
    5. Smart Sybil           — Sinergia: fraude de software barrada pelo silício

Uso:
    conda activate aihwkit_puf
    python fl_sentinel_puf.py
"""

import os
import sys
import copy
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── IBM AIHWKIT Imports (A Mágica Analógica) ──────────────────────────────────
# Importando as ferramentas para rodar a rede neural nas matrizes de resistores
try:
    from aihwkit.nn import AnalogLinear, AnalogSequential
    from aihwkit.optim import AnalogSGD
    from aihwkit.simulator.configs import SingleRPUConfig
    from aihwkit.simulator.presets import PCMPreset
    ANALOG_TRAINING_AVAILABLE = True
    print("[AIHWKIT] Módulos de treinamento analógico carregados com sucesso.")
except ImportError as e:
    ANALOG_TRAINING_AVAILABLE = False
    print(f"[AIHWKIT] ERRO FATAL: {e}. Certifique-se de que o aihwkit está instalado.")
    sys.exit(1)

# ── PUF Imports (Segurança de Hardware) ───────────────────────────────────────
PUF_ROOT = os.path.expanduser("~/aihwkit-puf")
sys.path.insert(0, PUF_ROOT)

try:
    from puf.identity import DeviceIdentity
    from puf.enrollment import PUFEnrollment
    PUF_AVAILABLE = True
    print("[PUF] Módulo de segurança em hardware carregado com sucesso.")
except ImportError as e:
    PUF_AVAILABLE = False
    print(f"[PUF] AVISO: módulo não encontrado ({e}). Cenários de hardware desabilitados.")

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS
# =============================================================================
torch.manual_seed(42)
np.random.seed(42)

print("\n" + "="*60)
print("CARREGANDO DATASET...")
print("="*60)

# Carrega o dataset CIC-IDS2017 processado
data = pd.read_csv('dataset.txt', header=None)
y = (data.iloc[:, -2] != 'normal').astype(int).values
X = pd.get_dummies(data.iloc[:, :-2]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

INPUT_DIM = X_train.shape[1]

# Canary set: 100 amostras de ataque reais do conjunto de teste
attack_idx = (y_test == 1).nonzero(as_tuple=True)[0]
canary_X   = X_test[attack_idx[:100]]
canary_y   = y_test[attack_idx[:100]]

print(f"  Treino   : {X_train.shape}")
print(f"  Teste    : {X_test.shape}")
print(f"  Features : {INPUT_DIM}")

# =============================================================================
# 2. CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================
NUM_CLIENTS          = 20
# ATENÇÃO: Simulação analógica é lenta. Reduzido para 50 rounds para testes rápidos.
# Para o resultado final da tese, volte para 200.
COMMUNICATION_ROUNDS = 50 
LOCAL_EPOCHS         = 1  
POISONED_CLIENTS     = [15, 16, 17, 18, 19]  # 5 atacantes originais
TRUST_THRESHOLD      = 0.60
PUF_THRESHOLD        = 0.90
PUF_CHALLENGES       = 256

# Partição Não-IID dos dados
client_data_size = len(X_train) // NUM_CLIENTS
client_X = [X_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]
client_y = [y_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]

# Rótulos envenenados padrão (Clean-label poisoning para 5 atacantes)
client_y_poisoned = copy.deepcopy(client_y)
for pc in POISONED_CLIENTS:
    lbl = client_y_poisoned[pc].clone()
    lbl[lbl == 1] = 0
    client_y_poisoned[pc] = lbl

# =============================================================================
# 3. UTILITÁRIOS DE APRENDIZADO ANALÓGICO
# =============================================================================
def create_model():
    """ 
    Cria uma rede neural cujos pesos são armazenados em condutância de resistores PCM.
    Isso simula o paradigma de In-Memory Computing para os nós IoT.
    """
    # Define a configuração física do chip (Memória PCM)
    rpu_config = SingleRPUConfig(device=PCMPreset())
    
    return AnalogSequential(
        AnalogLinear(INPUT_DIM, 64, rpu_config=rpu_config),
        nn.ReLU(),
        AnalogLinear(64, 1, rpu_config=rpu_config),
        nn.Sigmoid()
    )

def evaluate(model, verbose=False):
    model.eval()
    with torch.no_grad():
        preds = (model(X_test) > 0.5).float()
        acc   = (preds == y_test).float().mean().item() * 100
        true_atk = (y_test == 1)
        asr   = ((preds[true_atk] == 0).sum().item() / true_atk.sum().item()) * 100
    if verbose:
        print(f"    Acuracia (Analógica): {acc:.2f}%  ASR: {asr:.2f}%")
    return acc, asr

def fedavg(global_model, local_weights_list):
    """ Média federada clássica executada pelo controlador SDN """
    gw = copy.deepcopy(global_model.state_dict())
    for key in gw.keys():
        gw[key] = torch.stack([w[key] for w in local_weights_list], dim=0).mean(dim=0)
    global_model.load_state_dict(gw)

def apply_weight_scaling(local_model, global_model, scale_factor):
    sd_local  = local_model.state_dict()
    sd_global = global_model.state_dict()
    for key in sd_local.keys():
        diff = sd_local[key] - sd_global[key]
        sd_local[key].copy_(sd_global[key] + scale_factor * diff)

def train_local(local_model, X, y, epochs=LOCAL_EPOCHS, lr=0.05):
    """
    Treinamento utilizando pulsos elétricos (AnalogSGD).
    A taxa de aprendizado costuma ser ligeiramente maior no meio analógico.
    """
    opt = AnalogSGD(local_model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(local_model(X), y).backward()
        opt.step() # Aplica os pulsos nas células PCM

def compute_trust_score(local_model, canary_X, canary_y):
    local_model.eval()
    with torch.no_grad():
        preds = (local_model(canary_X) > 0.5).float()
        return (preds == canary_y).float().mean().item()

# =============================================================================
# 4. PUF: HARDWARE DE AUTENTICAÇÃO
# =============================================================================
class ClientHardware:
    def __init__(self, device_id: int):
        self.device_identity = DeviceIdentity(device_id, "pcm")
        self.noise_model     = self.device_identity.to_noise_model()

def puf_enroll(hardware_list, enrollment):
    db = {}
    for hw in hardware_list:
        db[hw.device_identity.device_id] = enrollment.extract_signature(hw)
    return db

def puf_verify(hw, enrollment, enrolled_db, threshold=PUF_THRESHOLD):
    did = hw.device_identity.device_id
    if did not in enrolled_db:
        return False, 0.0
    sim = 1.0 - float(np.mean(enrollment.extract_signature(hw) != enrolled_db[did]))
    return sim >= threshold, round(sim, 4)

# =============================================================================
# EXECUÇÃO DOS CENÁRIOS
# =============================================================================

# ---------------------------------------------------------
# 1. BASELINE HONESTO (Analógico)
# ---------------------------------------------------------
print("\n" + "="*60 + "\nCENÁRIO 1: FedAvg Honesto (Baseline Analógico)\n" + "="*60)
model_base = create_model()
history_base_asr = []
for r in range(COMMUNICATION_ROUNDS):
    lw = []
    for i in range(NUM_CLIENTS):
        lm = create_model()
        lm.load_state_dict(model_base.state_dict())
        train_local(lm, client_X[i], client_y[i])
        lw.append(lm.state_dict())
    fedavg(model_base, lw)
    _, asr = evaluate(model_base)
    history_base_asr.append(asr)
    if (r + 1) % 10 == 0:
        print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
        evaluate(model_base, verbose=True)
acc_base, asr_base = evaluate(model_base)


# ---------------------------------------------------------
# 2. ATAQUE DE ESCALONAMENTO AGRESSIVO (Matriz de PCM Satura)
# ---------------------------------------------------------
print("\n" + "="*60 + "\nCENÁRIO 2: Weight Scaling Attack (4.0x)\n" + "="*60)
model_atk = create_model()
history_atk_asr = []
sf_atk = NUM_CLIENTS / len(POISONED_CLIENTS)
for r in range(COMMUNICATION_ROUNDS):
    lw = []
    for i in range(NUM_CLIENTS):
        lm = create_model()
        lm.load_state_dict(model_atk.state_dict())
        train_local(lm, client_X[i], client_y_poisoned[i])
        if i in POISONED_CLIENTS:
            apply_weight_scaling(lm, model_atk, sf_atk)
        lw.append(lm.state_dict())
    fedavg(model_atk, lw)
    _, asr = evaluate(model_atk)
    history_atk_asr.append(asr)
    if (r + 1) % 10 == 0:
        print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
        evaluate(model_atk, verbose=True)
acc_atk, asr_atk = evaluate(model_atk)


# ---------------------------------------------------------
# 3. SENTINEL-FLOW (COMPORTAMENTAL)
# ---------------------------------------------------------
print("\n" + "="*60 + "\nCENÁRIO 3: Sentinel-Flow (Comportamental)\n" + "="*60)
model_sf = create_model()
history_sf_asr = []
for r in range(COMMUNICATION_ROUNDS):
    lw = []
    for i in range(NUM_CLIENTS):
        lm = create_model()
        lm.load_state_dict(model_sf.state_dict())
        train_local(lm, client_X[i], client_y_poisoned[i])
        if i in POISONED_CLIENTS:
            apply_weight_scaling(lm, model_sf, sf_atk)
        if compute_trust_score(lm, canary_X, canary_y) >= TRUST_THRESHOLD:
            lw.append(lm.state_dict())
    if lw: fedavg(model_sf, lw)
    _, asr = evaluate(model_sf)
    history_sf_asr.append(asr)
    if (r + 1) % 10 == 0:
        print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
        evaluate(model_sf, verbose=True)
acc_sf, asr_sf = evaluate(model_sf)


# Preparação para cenários com PUF e Smart Sybil
history_puf_asr = []
history_smsf_asr = []
history_smpuf_asr = []
acc_puf = asr_puf = acc_smsf = asr_smsf = acc_smpuf = asr_smpuf = None

if PUF_AVAILABLE:
    print("\nInicializando infraestrutura de hardware (PUF Analógico)...")
    client_hardware = [ClientHardware(i) for i in range(NUM_CLIENTS)]
    enrollment = PUFEnrollment(n_challenges=PUF_CHALLENGES)
    enrolled_db = puf_enroll(client_hardware, enrollment)

    # ---------------------------------------------------------
    # 4. SENTINEL-FLOW + PUF
    # ---------------------------------------------------------
    print("\n" + "="*60 + "\nCENÁRIO 4: Sentinel-Flow + PUF (Analógico)\n" + "="*60)
    model_puf = create_model()
    for r in range(COMMUNICATION_ROUNDS):
        lw = []
        enrollment.rotate(r + 1)
        for i in range(NUM_CLIENTS):
            lm = create_model()
            lm.load_state_dict(model_puf.state_dict())
            train_local(lm, client_X[i], client_y_poisoned[i])
            if i in POISONED_CLIENTS:
                apply_weight_scaling(lm, model_puf, sf_atk)
            
            puf_ok, _ = puf_verify(client_hardware[i], enrollment, enrolled_db)
            if puf_ok and compute_trust_score(lm, canary_X, canary_y) >= TRUST_THRESHOLD:
                lw.append(lm.state_dict())
        if lw: fedavg(model_puf, lw)
        _, asr = evaluate(model_puf)
        history_puf_asr.append(asr)
        if (r + 1) % 10 == 0:
            print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
            evaluate(model_puf, verbose=True)
    acc_puf, asr_puf = evaluate(model_puf)

    # =========================================================================
    # CENÁRIO 5: SMART SYBIL (BYPASS DE SOFTWARE + SINERGIA PUF)
    # =========================================================================
    SMART_POISONED = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # 10 atacantes
    physical_attacker_id = 15 # O único chip PCM real que o hacker possui
    
    client_y_smart = copy.deepcopy(client_y)
    for pc in SMART_POISONED:
        lbl = client_y_smart[pc].clone()
        lbl[lbl == 1] = 0
        client_y_smart[pc] = lbl

    # 5A. Smart Sybil vs Apenas Sentinel-Flow (Sem PUF)
    print("\n" + "="*60 + "\nCENÁRIO 5A: Smart Sybil (Bypass no Sentinel-Flow)\n" + "="*60)
    model_smsf = create_model()
    
    for r in range(COMMUNICATION_ROUNDS):
        lw = []
        for i in range(NUM_CLIENTS):
            lm = create_model()
            lm.load_state_dict(model_smsf.state_dict())
            train_local(lm, client_X[i], client_y_smart[i])
            
            if i in SMART_POISONED:
                apply_weight_scaling(lm, model_smsf, sf_atk)
                trust = 1.0 # Fraude: O nó virtual ignora a física e força 100%
            else:
                trust = compute_trust_score(lm, canary_X, canary_y)

            if trust >= TRUST_THRESHOLD:
                lw.append(lm.state_dict())
                
        if lw: fedavg(model_smsf, lw)
        _, asr = evaluate(model_smsf)
        history_smsf_asr.append(asr)
        if (r + 1) % 10 == 0:
            print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
            evaluate(model_smsf, verbose=True)
            
    acc_smsf, asr_smsf = evaluate(model_smsf)

    # 5B. Smart Sybil vs Sinergia (Sentinel-Flow + PUF)
    print("\n" + "="*60 + "\nCENÁRIO 5B: Sinergia Completa (O PUF barra a fraude)\n" + "="*60)
    model_smpuf = create_model()
    
    for r in range(COMMUNICATION_ROUNDS):
        lw = []
        enrollment.rotate(r + 1)
        for i in range(NUM_CLIENTS):
            lm = create_model()
            lm.load_state_dict(model_smpuf.state_dict())
            train_local(lm, client_X[i], client_y_smart[i])
            
            if i in SMART_POISONED:
                apply_weight_scaling(lm, model_smpuf, sf_atk)
                trust = 1.0
                puf_ok = (i == physical_attacker_id) # A fraude esbarra no silício!
            else:
                trust = compute_trust_score(lm, canary_X, canary_y)
                puf_ok, _ = puf_verify(client_hardware[i], enrollment, enrolled_db)
                
            if puf_ok and trust >= TRUST_THRESHOLD:
                lw.append(lm.state_dict())
                
        if lw: fedavg(model_smpuf, lw)
        _, asr = evaluate(model_smpuf)
        history_smpuf_asr.append(asr)
        if (r + 1) % 10 == 0:
            print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
            evaluate(model_smpuf, verbose=True)
            
    acc_smpuf, asr_smpuf = evaluate(model_smpuf)

# =============================================================================
# EXPORTAÇÃO E TABELA FINAL
# =============================================================================
print("\n" + "="*60 + "\nCOMPARATIVO FINAL DE TODOS OS CENÁRIOS\n" + "="*60)
print("{:<35} {:>10} {:>10}".format("Cenário", "Acurácia", "ASR"))
print("-"*57)
print("{:<35} {:>9.2f}% {:>9.2f}%".format("1. Baseline Analógico", acc_base, asr_base))
print("{:<35} {:>9.2f}% {:>9.2f}%".format("2. Ataque Agressivo", acc_atk, asr_atk))
print("{:<35} {:>9.2f}% {:>9.2f}%".format("3. Sentinel-Flow", acc_sf, asr_sf))

if PUF_AVAILABLE:
    print("{:<35} {:>9.2f}% {:>9.2f}%".format("4. SF + PUF", acc_puf, asr_puf))
    print("{:<35} {:>9.2f}% {:>9.2f}%".format("5A. Fraude de Software (Sem PUF)", acc_smsf, asr_smsf))
    print("{:<35} {:>9.2f}% {:>9.2f}%".format("5B. Sinergia (Salvo pelo PUF)", acc_smpuf, asr_smpuf))

results = {
    "generated": datetime.datetime.now().isoformat(),
    "scenarios": {
        "1. Baseline": {"accuracy": acc_base, "asr": asr_base, "history": [{"round": i+1, "asr": v} for i, v in enumerate(history_base_asr)]},
        "2. Ataque 4.0x": {"accuracy": acc_atk, "asr": asr_atk, "history": [{"round": i+1, "asr": v} for i, v in enumerate(history_atk_asr)]},
        "3. Sentinel-Flow": {"accuracy": acc_sf, "asr": asr_sf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(history_sf_asr)]},
    }
}

if PUF_AVAILABLE:
    results["scenarios"].update({
        "4. SF + PUF": {"accuracy": acc_puf, "asr": asr_puf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(history_puf_asr)]},
        "5A. Fraude de Software (Sem PUF)": {"accuracy": acc_smsf, "asr": asr_smsf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(history_smsf_asr)]},
        "5B. Sinergia (Salvo pelo PUF)": {"accuracy": acc_smpuf, "asr": asr_smpuf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(history_smpuf_asr)]},
    })

with open("sentinel_puf_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nArquivo exportado com sucesso: sentinel_puf_results.json")
