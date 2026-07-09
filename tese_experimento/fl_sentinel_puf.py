"""
fl_sentinel_puf.py
==================
Arquitetura Assimétrica Edge-to-Cloud:
- Treinamento Federado (Digital): Evita a saturação de condutância do PCM.
- Inferência e NIDS (Analógico): Projeção dos pesos na matriz crossbar.
- Segurança (PUF): Atestação enraizada na entropia de fabricação do hardware.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── IBM AIHWKIT Imports ───────────────────────────────────────────────────────
try:
    from aihwkit.nn.conversion import convert_to_analog
    from aihwkit.simulator.presets import PCMPreset
    ANALOG_AVAILABLE = True
    print("[AIHWKIT] Módulos de inferência analógica carregados com sucesso.")
except ImportError as e:
    ANALOG_AVAILABLE = False
    print(f"[AIHWKIT] ERRO FATAL: {e}")
    sys.exit(1)

# ── PUF Imports ───────────────────────────────────────────────────────────────
PUF_ROOT = os.path.expanduser("~/aihwkit-puf")
sys.path.insert(0, PUF_ROOT)

try:
    from puf.identity import DeviceIdentity
    from puf.enrollment import PUFEnrollment
    PUF_AVAILABLE = True
    print("[PUF] Módulo de segurança em hardware carregado.")
except ImportError as e:
    PUF_AVAILABLE = False
    print(f"[PUF] AVISO: {e}")

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS
# =============================================================================
torch.manual_seed(42)
np.random.seed(42)

print("\n" + "="*60 + "\nCARREGANDO DATASET...\n" + "="*60)

data = pd.read_csv('dataset.txt', header=None)

# Assumindo que a penúltima coluna (-2) é a label devido a formatação do seu .txt
y = (data.iloc[:, -2] != 'normal').astype(int).values
X = pd.get_dummies(data.iloc[:, :-2]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

# --- A CORREÇÃO MATEMÁTICA CRÍTICA ---
# Normalização (Z-score scaling) impede que features grandesaturem os gradientes
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# -------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

INPUT_DIM = X_train.shape[1]

# Canary Set para Sentinel-Flow
attack_idx = (y_test == 1).nonzero(as_tuple=True)[0]
canary_X   = X_test[attack_idx[:100]]
canary_y   = y_test[attack_idx[:100]]

# =============================================================================
# 2. CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================
NUM_CLIENTS          = 20
COMMUNICATION_ROUNDS = 50 
LOCAL_EPOCHS         = 3  
POISONED_CLIENTS     = [15, 16, 17, 18, 19] # 25% de nós comprometidos
TRUST_THRESHOLD      = 0.60
PUF_THRESHOLD        = 0.90
PUF_CHALLENGES       = 256

client_data_size = len(X_train) // NUM_CLIENTS
client_X = [X_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]
client_y = [y_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]

# Clean-label poisoning: Inverte os labels forçando o modelo a achar que ataques são normais
client_y_poisoned = copy.deepcopy(client_y)
for pc in POISONED_CLIENTS:
    client_y_poisoned[pc] = torch.zeros_like(client_y_poisoned[pc])

# =============================================================================
# 3. LÓGICA ASSIMÉTRICA: TREINO DIGITAL -> INFERÊNCIA ANALÓGICA
# =============================================================================

def create_model():
    """ Modelo Digital Base """
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid()
    )

def train_local(model, X, y):
    """ Treinamento Digital no MCU do nó IoT """
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(LOCAL_EPOCHS):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return model.state_dict()

def fedavg(global_model, weights_list):
    """ Agregação Digital Pura na Nuvem (SDN) """
    avg_w = copy.deepcopy(weights_list[0])
    for key in avg_w.keys():
        avg_w[key] = torch.stack([w[key] for w in weights_list]).mean(dim=0)
    global_model.load_state_dict(avg_w)

def simulate_analog_inference(digital_model, X_eval):
    """ 
    Projeta os pesos digitais na matriz PCM para inferência. 
    Simula o ruído de programação e de leitura física.
    """
    analog_model = convert_to_analog(copy.deepcopy(digital_model), rpu_config=PCMPreset())
    analog_model.eval()
    with torch.no_grad():
        return analog_model(X_eval)

def evaluate_analog(digital_model, verbose=False):
    """ O Controlador avalia a acurácia global baseada na inferência analógica """
    preds = (simulate_analog_inference(digital_model, X_test) > 0.5).float()
    acc   = (preds == y_test).float().mean().item() * 100
    true_atk = (y_test == 1)
    asr   = ((preds[true_atk] == 0).sum().item() / true_atk.sum().item()) * 100
    
    if verbose:
        print(f"    Acurácia Analógica: {acc:.2f}%  |  ASR: {asr:.2f}%")
    return acc, asr

def compute_trust_score(digital_model):
    """ Sentinel-Flow: Audita o modelo testando a matriz analógica """
    preds = (simulate_analog_inference(digital_model, canary_X) > 0.5).float()
    return (preds == canary_y).float().mean().item()

def apply_weight_scaling(local_w, global_w, scale_factor):
    """ O Atacante intercepta os tensores e injeta o escalonamento agressivo """
    scaled_w = {}
    for k in local_w.keys():
        diff = local_w[k] - global_w[k]
        scaled_w[k] = global_w[k] + scale_factor * diff
    return scaled_w

# =============================================================================
# 4. PUF: SEGURANÇA NO NÍVEL DO SILÍCIO
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
    if did not in enrolled_db: return False
    sim = 1.0 - float(np.mean(enrollment.extract_signature(hw) != enrolled_db[did]))
    return sim >= threshold

# =============================================================================
# EXECUÇÃO DOS CENÁRIOS
# =============================================================================

print("\n" + "="*60 + "\nCENÁRIO 1: Baseline Honesto\n" + "="*60)
model_base = create_model()
hist_base = []
for r in range(COMMUNICATION_ROUNDS):
    lw = [train_local(copy.deepcopy(model_base), client_X[i], client_y[i]) for i in range(NUM_CLIENTS)]
    fedavg(model_base, lw)
    _, asr = evaluate_analog(model_base, verbose=(r+1)%10==0)
    hist_base.append(asr)
acc_base, asr_base = evaluate_analog(model_base)


print("\n" + "="*60 + "\nCENÁRIO 2: Ataque de Escalonamento (4.0x)\n" + "="*60)
model_atk = create_model()
hist_atk = []
sf_atk = NUM_CLIENTS / len(POISONED_CLIENTS)

for r in range(COMMUNICATION_ROUNDS):
    lw = []
    global_w = model_atk.state_dict()
    for i in range(NUM_CLIENTS):
        w = train_local(copy.deepcopy(model_atk), client_X[i], client_y_poisoned[i])
        if i in POISONED_CLIENTS:
            w = apply_weight_scaling(w, global_w, sf_atk)
        lw.append(w)
    fedavg(model_atk, lw)
    _, asr = evaluate_analog(model_atk, verbose=(r+1)%10==0)
    hist_atk.append(asr)
acc_atk, asr_atk = evaluate_analog(model_atk)


print("\n" + "="*60 + "\nCENÁRIO 3: Sentinel-Flow (Software)\n" + "="*60)
model_sf = create_model()
hist_sf = []

for r in range(COMMUNICATION_ROUNDS):
    lw = []
    global_w = model_sf.state_dict()
    for i in range(NUM_CLIENTS):
        temp_model = copy.deepcopy(model_sf)
        w = train_local(temp_model, client_X[i], client_y_poisoned[i])
        if i in POISONED_CLIENTS:
            w = apply_weight_scaling(w, global_w, sf_atk)
        
        temp_model.load_state_dict(w)
        if compute_trust_score(temp_model) >= TRUST_THRESHOLD:
            lw.append(w)
            
    if lw: fedavg(model_sf, lw)
    _, asr = evaluate_analog(model_sf, verbose=(r+1)%10==0)
    hist_sf.append(asr)
acc_sf, asr_sf = evaluate_analog(model_sf)


hist_puf = hist_smsf = hist_smpuf = []
acc_puf = asr_puf = acc_smsf = asr_smsf = acc_smpuf = asr_smpuf = 0.0

if PUF_AVAILABLE:
    print("\nInicializando infraestrutura de hardware (PUF Analógico)...")
    hardware = [ClientHardware(i) for i in range(NUM_CLIENTS)]
    enrollment = PUFEnrollment(n_challenges=PUF_CHALLENGES)
    enrolled_db = puf_enroll(hardware, enrollment)

    print("\n" + "="*60 + "\nCENÁRIO 4: Sentinel-Flow + PUF\n" + "="*60)
    model_puf = create_model()
    
    for r in range(COMMUNICATION_ROUNDS):
        lw = []
        global_w = model_puf.state_dict()
        enrollment.rotate(r + 1)
        
        for i in range(NUM_CLIENTS):
            temp_model = copy.deepcopy(model_puf)
            w = train_local(temp_model, client_X[i], client_y_poisoned[i])
            if i in POISONED_CLIENTS:
                w = apply_weight_scaling(w, global_w, sf_atk)
                
            temp_model.load_state_dict(w)
            if puf_verify(hardware[i], enrollment, enrolled_db) and compute_trust_score(temp_model) >= TRUST_THRESHOLD:
                lw.append(w)
                
        if lw: fedavg(model_puf, lw)
        _, asr = evaluate_analog(model_puf, verbose=(r+1)%10==0)
        hist_puf.append(asr)
    acc_puf, asr_puf = evaluate_analog(model_puf)

    # -------------------------------------------------------------------------
    # SMART SYBIL: 10 Atacantes virtuais rodando de 1 único chip roubado (id=15)
    # -------------------------------------------------------------------------
    SMART_POISONED = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    phys_hacker_id = 15
    y_smart = copy.deepcopy(client_y)
    for pc in SMART_POISONED: y_smart[pc] = torch.zeros_like(y_smart[pc])

    print("\n" + "="*60 + "\nCENÁRIO 5A: Smart Sybil (Fraude no Software)\n" + "="*60)
    model_smsf = create_model()
    for r in range(COMMUNICATION_ROUNDS):
        lw = []
        global_w = model_smsf.state_dict()
        for i in range(NUM_CLIENTS):
            temp = copy.deepcopy(model_smsf)
            w = train_local(temp, client_X[i], y_smart[i])
            if i in SMART_POISONED:
                w = apply_weight_scaling(w, global_w, sf_atk)
                trust = 1.0 # O Hacker burla a validação na nuvem
            else:
                temp.load_state_dict(w)
                trust = compute_trust_score(temp)

            if trust >= TRUST_THRESHOLD: lw.append(w)
                
        if lw: fedavg(model_smsf, lw)
        _, asr = evaluate_analog(model_smsf, verbose=(r+1)%10==0)
        hist_smsf.append(asr)
    acc_smsf, asr_smsf = evaluate_analog(model_smsf)

    print("\n" + "="*60 + "\nCENÁRIO 5B: Sinergia Suprema (Salvo pelo PUF)\n" + "="*60)
    model_smpuf = create_model()
    for r in range(COMMUNICATION_ROUNDS):
        lw = []
        global_w = model_smpuf.state_dict()
        enrollment.rotate(r + 1)
        for i in range(NUM_CLIENTS):
            temp = copy.deepcopy(model_smpuf)
            w = train_local(temp, client_X[i], y_smart[i])
            if i in SMART_POISONED:
                w = apply_weight_scaling(w, global_w, sf_atk)
                trust = 1.0
                puf_ok = (i == phys_hacker_id) # Hardware não pode ser emulado em escala
            else:
                temp.load_state_dict(w)
                trust = compute_trust_score(temp)
                puf_ok = puf_verify(hardware[i], enrollment, enrolled_db)
                
            if puf_ok and trust >= TRUST_THRESHOLD: lw.append(w)
                
        if lw: fedavg(model_smpuf, lw)
        _, asr = evaluate_analog(model_smpuf, verbose=(r+1)%10==0)
        hist_smpuf.append(asr)
    acc_smpuf, asr_smpuf = evaluate_analog(model_smpuf)

# =============================================================================
# EXPORTAÇÃO
# =============================================================================
print("\n" + "="*60 + "\nCOMPARATIVO FINAL\n" + "="*60)
print("{:<35} {:>10} {:>10}".format("Cenário", "Acurácia", "ASR"))
print("{:<35} {:>9.2f}% {:>9.2f}%".format("1. Baseline", acc_base, asr_base))
print("{:<35} {:>9.2f}% {:>9.2f}%".format("2. Ataque 4.0x", acc_atk, asr_atk))
print("{:<35} {:>9.2f}% {:>9.2f}%".format("3. Sentinel-Flow", acc_sf, asr_sf))
if PUF_AVAILABLE:
    print("{:<35} {:>9.2f}% {:>9.2f}%".format("4. SF + PUF", acc_puf, asr_puf))
    print("{:<35} {:>9.2f}% {:>9.2f}%".format("5A. Fraude de Software", acc_smsf, asr_smsf))
    print("{:<35} {:>9.2f}% {:>9.2f}%".format("5B. Sinergia Suprema", acc_smpuf, asr_smpuf))

results = {
    "generated": datetime.datetime.now().isoformat(),
    "scenarios": {
        "1. Baseline": {"accuracy": acc_base, "asr": asr_base, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_base)]},
        "2. Ataque 4.0x": {"accuracy": acc_atk, "asr": asr_atk, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_atk)]},
        "3. Sentinel-Flow": {"accuracy": acc_sf, "asr": asr_sf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_sf)]},
        "4. SF + PUF": {"accuracy": acc_puf, "asr": asr_puf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_puf)]},
        "5A. Fraude de Software (Sem PUF)": {"accuracy": acc_smsf, "asr": asr_smsf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_smsf)]},
        "5B. Sinergia (Salvo pelo PUF)": {"accuracy": acc_smpuf, "asr": asr_smpuf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_smpuf)]},
    }
}
with open("sentinel_puf_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nArquivo exportado: sentinel_puf_results.json")
