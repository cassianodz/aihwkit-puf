"""
fl_sentinel_puf.py
==================
Arquitetura Assimétrica Edge-to-Cloud:
- Treinamento Federado (Digital): Evita a saturação de condutância do PCM.
- Inferência e NIDS (Analógico): Projeção dos pesos na matriz crossbar.
- Segurança (PUF): Atestação enraizada na entropia de fabricação do hardware.
- Visualizações: Geração de gráficos prontos para artigos científicos e dump JSON.
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
from sklearn.metrics import confusion_matrix

# Imports para geração de gráficos científicos
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração de estilo para artigos científicos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

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
y = (data.iloc[:, -2] != 'normal').astype(int).values
X = pd.get_dummies(data.iloc[:, :-2]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

INPUT_DIM = X_train.shape[1]

attack_idx = (y_test == 1).nonzero(as_tuple=True)[0]
canary_X   = X_test[attack_idx[:100]]
canary_y   = y_test[attack_idx[:100]]

# =============================================================================
# 2. CONFIGURAÇÕES DO EXPERIMENTO
# =============================================================================
NUM_CLIENTS          = 20
COMMUNICATION_ROUNDS = 50 
LOCAL_EPOCHS         = 3  
POISONED_CLIENTS     = [15, 16, 17, 18, 19]
TRUST_THRESHOLD      = 0.60
PUF_THRESHOLD        = 0.90
PUF_CHALLENGES       = 256

client_data_size = len(X_train) // NUM_CLIENTS
client_X = [X_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]
client_y = [y_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]

client_y_poisoned = copy.deepcopy(client_y)
for pc in POISONED_CLIENTS:
    client_y_poisoned[pc] = torch.zeros_like(client_y_poisoned[pc])

# =============================================================================
# 3. LÓGICA ASSIMÉTRICA: TREINO DIGITAL -> INFERÊNCIA ANALÓGICA
# =============================================================================

def create_model():
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Sigmoid()
    )

def train_local(model, X, y):
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(LOCAL_EPOCHS):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return model.state_dict()

def fedavg(global_model, weights_list):
    avg_w = copy.deepcopy(weights_list[0])
    for key in avg_w.keys():
        avg_w[key] = torch.stack([w[key] for w in weights_list]).mean(dim=0)
    global_model.load_state_dict(avg_w)

def simulate_analog_inference(digital_model, X_eval, return_raw=False):
    analog_model = convert_to_analog(copy.deepcopy(digital_model), rpu_config=PCMPreset())
    analog_model.eval()
    with torch.no_grad():
        out = analog_model(X_eval)
        if return_raw: return out
        return out

def evaluate_analog(digital_model, verbose=False):
    preds = (simulate_analog_inference(digital_model, X_test) > 0.5).float()
    acc   = (preds == y_test).float().mean().item() * 100
    true_atk = (y_test == 1)
    asr   = ((preds[true_atk] == 0).sum().item() / true_atk.sum().item()) * 100
    if verbose:
        print(f"    Acurácia Analógica: {acc:.2f}%  |  ASR: {asr:.2f}%")
    return acc, asr

def compute_trust_score(digital_model):
    preds = (simulate_analog_inference(digital_model, canary_X) > 0.5).float()
    return (preds == canary_y).float().mean().item()

def apply_weight_scaling(local_w, global_w, scale_factor):
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
hist_asr_base, hist_acc_base = [], []
for r in range(COMMUNICATION_ROUNDS):
    lw = [train_local(copy.deepcopy(model_base), client_X[i], client_y[i]) for i in range(NUM_CLIENTS)]
    fedavg(model_base, lw)
    acc, asr = evaluate_analog(model_base, verbose=(r+1)%10==0)
    hist_asr_base.append(asr); hist_acc_base.append(acc)
acc_base, asr_base = evaluate_analog(model_base)

print("\n" + "="*60 + "\nCENÁRIO 2: Ataque de Escalonamento (4.0x)\n" + "="*60)
model_atk = create_model()
hist_asr_atk, hist_acc_atk = [], []
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
    acc, asr = evaluate_analog(model_atk, verbose=(r+1)%10==0)
    hist_asr_atk.append(asr); hist_acc_atk.append(acc)
acc_atk, asr_atk = evaluate_analog(model_atk)


print("\n" + "="*60 + "\nCENÁRIO 3: Sentinel-Flow (Software)\n" + "="*60)
model_sf = create_model()
hist_asr_sf, hist_acc_sf = [], []

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
    acc, asr = evaluate_analog(model_sf, verbose=(r+1)%10==0)
    hist_asr_sf.append(asr); hist_acc_sf.append(acc)
acc_sf, asr_sf = evaluate_analog(model_sf)

# CORREÇÃO DA INICIALIZAÇÃO APLICADA AQUI
hist_asr_puf, hist_asr_smsf, hist_asr_smpuf = [], [], []
hist_acc_puf, hist_acc_smsf, hist_acc_smpuf = [], [], []
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
        acc, asr = evaluate_analog(model_puf, verbose=(r+1)%10==0)
        hist_asr_puf.append(asr); hist_acc_puf.append(acc)
    acc_puf, asr_puf = evaluate_analog(model_puf)

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
                trust = 1.0 
            else:
                temp.load_state_dict(w)
                trust = compute_trust_score(temp)

            if trust >= TRUST_THRESHOLD: lw.append(w)
                
        if lw: fedavg(model_smsf, lw)
        acc, asr = evaluate_analog(model_smsf, verbose=(r+1)%10==0)
        hist_asr_smsf.append(asr); hist_acc_smsf.append(acc)
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
                puf_ok = (i == phys_hacker_id) 
            else:
                temp.load_state_dict(w)
                trust = compute_trust_score(temp)
                puf_ok = puf_verify(hardware[i], enrollment, enrolled_db)
                
            if puf_ok and trust >= TRUST_THRESHOLD: lw.append(w)
                
        if lw: fedavg(model_smpuf, lw)
        acc, asr = evaluate_analog(model_smpuf, verbose=(r+1)%10==0)
        hist_asr_smpuf.append(asr); hist_acc_smpuf.append(acc)
    acc_smpuf, asr_smpuf = evaluate_analog(model_smpuf)

# =============================================================================
# PLOTAGEM DE RESULTADOS CIENTÍFICOS
# =============================================================================
print("\n" + "="*60 + "\nGERANDO GRÁFICOS PARA O ARTIGO...\n" + "="*60)
os.makedirs("plots", exist_ok=True)
epochs_range = range(1, COMMUNICATION_ROUNDS + 1)

# 1. Evolução do Attack Success Rate (ASR)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, hist_asr_base, label='Baseline Honesto', color='green', linewidth=2.5)
plt.plot(epochs_range, hist_asr_atk, label='Ataque (4.0x) S/ Defesa', color='red', linestyle='--', linewidth=2.5)
plt.plot(epochs_range, hist_asr_sf, label='Sentinel-Flow', color='orange', linewidth=2)
if PUF_AVAILABLE:
    plt.plot(epochs_range, hist_asr_smsf, label='Smart Sybil (Falha SF)', color='darkred', linestyle='-.', linewidth=2.5)
    plt.plot(epochs_range, hist_asr_smpuf, label='Sinergia Lógico-Material', color='blue', linewidth=3)

plt.title('Evolução da Taxa de Sucesso do Ataque (ASR)')
plt.xlabel('Rodadas de Comunicação')
plt.ylabel('Attack Success Rate (%)')
plt.legend(loc='center right', bbox_to_anchor=(1.45, 0.5))
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/asr_evolution.png', dpi=300)
plt.close()

# 2. Evolução da Acurácia Global
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, hist_acc_base, label='Baseline Honesto', color='green', linewidth=2.5)
plt.plot(epochs_range, hist_acc_atk, label='Ataque (4.0x) S/ Defesa', color='red', linestyle='--', linewidth=2.5)
if PUF_AVAILABLE:
    plt.plot(epochs_range, hist_acc_smsf, label='Smart Sybil (Falha SF)', color='darkred', linestyle='-.', linewidth=2.5)
    plt.plot(epochs_range, hist_acc_smpuf, label='Sinergia Lógico-Material', color='blue', linewidth=3)

plt.title('Impacto do Ataque na Acurácia Global do NIDS')
plt.xlabel('Rodadas de Comunicação')
plt.ylabel('Acurácia Global (%)')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/accuracy_evolution.png', dpi=300)
plt.close()

# 3. Matriz de Confusão Analógica (Melhor Modelo: Sinergia)
final_model = model_smpuf if PUF_AVAILABLE else model_sf
with torch.no_grad():
    y_pred_probs = simulate_analog_inference(final_model, X_test)
    y_pred_classes = (y_pred_probs > 0.5).int().numpy()
    y_true_np = y_test.int().numpy()

cm = confusion_matrix(y_true_np, y_pred_classes)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['Benigno', 'Ataque'], yticklabels=['Benigno', 'Ataque'],
            annot_kws={"size": 16, "weight": "bold"})
plt.title('Matriz de Confusão NIDS (Inferência In-Memory)')
plt.ylabel('Rótulo Verdadeiro')
plt.xlabel('Rótulo Predito')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=300)
plt.close()

# 4. Distribuição do Ruído Físico (Assinatura Analógica vs Digital)
final_model.eval()
with torch.no_grad():
    digital_out = final_model(X_test).numpy()
    analog_out  = simulate_analog_inference(final_model, X_test, return_raw=True).numpy()
    
noise_delta = (analog_out - digital_out).flatten()

plt.figure(figsize=(9, 6))
sns.histplot(noise_delta, bins=50, kde=True, color='purple', stat="density")
plt.title('Distribuição Estocástica do Ruído Termodinâmico (Matriz PCM)')
plt.xlabel('Desvio de Inferência (Analógico - Digital)')
plt.ylabel('Densidade')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('plots/analog_noise_distribution.png', dpi=300)
plt.close()

# =============================================================================
# EXPORTAÇÃO DO JSON (NÃO FOI REMOVIDO)
# =============================================================================
print("\n" + "="*60 + "\nCOMPARATIVO FINAL E EXPORTAÇÃO JSON\n" + "="*60)
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
        "1. Baseline": {"accuracy": acc_base, "asr": asr_base, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_asr_base)]},
        "2. Ataque 4.0x": {"accuracy": acc_atk, "asr": asr_atk, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_asr_atk)]},
        "3. Sentinel-Flow": {"accuracy": acc_sf, "asr": asr_sf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_asr_sf)]},
        "4. SF + PUF": {"accuracy": acc_puf, "asr": asr_puf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_asr_puf)]},
        "5A. Fraude de Software (Sem PUF)": {"accuracy": acc_smsf, "asr": asr_smsf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_asr_smsf)]},
        "5B. Sinergia (Salvo pelo PUF)": {"accuracy": acc_smpuf, "asr": asr_smpuf, "history": [{"round": i+1, "asr": v} for i, v in enumerate(hist_asr_smpuf)]},
    }
}
with open("sentinel_puf_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nArquivo exportado: sentinel_puf_results.json")
print("[SUCESSO] Todos os processos foram finalizados.")
