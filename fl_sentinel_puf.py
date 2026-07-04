"""
fl_sentinel_puf.py
==================
Experimento completo: Sentinel-Flow com atestação por Analog PUF.

Cenários:
    1. FedAvg Honesto       — baseline sem ataque e sem defesa
    2. Weight Scaling Attack — clean-label poisoning + escalonamento
    3. Sentinel-Flow         — atestação comportamental via canary flows
    4. Sentinel-Flow + PUF  — comportamental + autenticação de hardware analógico

Bônus: diagnóstico de Sybil attack (1 chip físico → N identidades falsas)

Uso:
    conda activate aihwkit_puf
    python fl_sentinel_puf.py

Requisitos:
    pip install torch pandas scikit-learn matplotlib
    (aihwkit-puf deve estar em ~/aihwkit-puf ou ajuste PUF_ROOT abaixo)
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
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ── PUF imports ───────────────────────────────────────────────────────────────
PUF_ROOT = os.path.expanduser("~/aihwkit-puf")
sys.path.insert(0, PUF_ROOT)

try:
    from puf.identity import DeviceIdentity
    from puf.enrollment import PUFEnrollment
    PUF_AVAILABLE = True
    print("[PUF] Módulo carregado com sucesso.")
except ImportError as e:
    PUF_AVAILABLE = False
    print(f"[PUF] AVISO: módulo não encontrado ({e})")
    print("[PUF] Cenários 4 e Bônus serão desabilitados.")
    print("[PUF] Certifique-se de que ~/aihwkit-puf está instalado.")

# =============================================================================
# 1. PREPARAÇÃO DOS DADOS
# =============================================================================
torch.manual_seed(42)
np.random.seed(42)

print("\n" + "="*60)
print("CARREGANDO DATASET...")
print("="*60)

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

print("Dataset carregado.")
print(f"  Treino   : {X_train.shape}")
print(f"  Teste    : {X_test.shape}")
print(f"  Features : {INPUT_DIM}")
print(f"  Ataques no teste: {(y_test == 1).sum().item()}")
print(f"  Canary set: {len(canary_X)} amostras")

# =============================================================================
# 2. CONFIGURAÇÕES DO EXPERIMENTO (parâmetros do artigo)
# =============================================================================
NUM_CLIENTS          = 20
COMMUNICATION_ROUNDS = 200
LOCAL_EPOCHS         = 3
POISONED_CLIENTS     = [15, 16, 17, 18, 19]  # 5 atacantes
TRUST_THRESHOLD      = 0.60
PUF_THRESHOLD        = 0.90
PUF_CHALLENGES       = 256

# Prepara rótulos envenenados (clean-label: troca ataques por 0)
client_data_size = len(X_train) // NUM_CLIENTS
client_X = [X_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]
client_y = [y_train[i*client_data_size:(i+1)*client_data_size] for i in range(NUM_CLIENTS)]

client_y_poisoned = copy.deepcopy(client_y)
for pc in POISONED_CLIENTS:
    lbl = client_y_poisoned[pc].clone()
    lbl[lbl == 1] = 0
    client_y_poisoned[pc] = lbl

# =============================================================================
# 3. UTILITÁRIOS
# =============================================================================

def create_model():
    """Arquitetura NIDS do artigo: Linear(N,64) -> ReLU -> Linear(64,1) -> Sigmoid."""
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )


def evaluate(model, verbose=False):
    """Retorna (accuracy%, ASR%)."""
    model.eval()
    with torch.no_grad():
        preds = (model(X_test) > 0.5).float()
        acc   = (preds == y_test).float().mean().item() * 100
        true_atk = (y_test == 1)
        asr   = ((preds[true_atk] == 0).sum().item() / true_atk.sum().item()) * 100
    if verbose:
        print(f"    Acuracia: {acc:.2f}%  ASR: {asr:.2f}%")
    return acc, asr


def fedavg(global_model, local_weights_list):
    """FedAvg padrão — média simples dos pesos locais."""
    gw = copy.deepcopy(global_model.state_dict())
    for key in gw.keys():
        gw[key] = torch.stack([w[key] for w in local_weights_list], dim=0).mean(dim=0)
    global_model.load_state_dict(gw)


def apply_weight_scaling(local_model, global_model, scale_factor):
    """Weight scaling attack: amplifica gradientes para dominar FedAvg."""
    sd_local  = local_model.state_dict()
    sd_global = global_model.state_dict()
    for key in sd_local.keys():
        diff = sd_local[key] - sd_global[key]
        sd_local[key].copy_(sd_global[key] + scale_factor * diff)


def train_local(local_model, X, y, epochs=LOCAL_EPOCHS, lr=0.005):
    """Treinamento local padrão."""
    opt  = optim.Adam(local_model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss_fn(local_model(X), y).backward()
        opt.step()


def compute_trust_score(local_model, canary_X, canary_y):
    """Trust Score comportamental — Eq. 2 do Sentinel-Flow."""
    local_model.eval()
    with torch.no_grad():
        preds = (local_model(canary_X) > 0.5).float()
        return (preds == canary_y).float().mean().item()


# =============================================================================
# 4. PUF: HARDWARE DE AUTENTICAÇÃO ANALÓGICO
# =============================================================================

class ClientHardware:
    """
    Representa o chip analógico PCM de um cliente FL.

    Contém apenas os atributos necessários para PUFEnrollment.extract_signature():
        .device_identity  -> DeviceIdentity (parâmetros físicos únicos)
        .noise_model      -> PCMLikeNoiseModel configurado com a identidade

    O chip é INDEPENDENTE do modelo NIDS. O PUF atesta o HARDWARE
    que hospeda o modelo, não os pesos do modelo em si.

    Analogia: é o wobble do PS1 — uma propriedade física irreproduzível
    do suporte físico, não do conteúdo armazenado nele.
    """
    def __init__(self, device_id: int):
        self.device_identity = DeviceIdentity(device_id, "pcm")
        self.noise_model     = self.device_identity.to_noise_model()

    def __repr__(self):
        d = self.device_identity
        return f"ClientHardware(id={d.device_id}, g_prog_offset={float(d.g_prog_offset):.4f} uS)"


def puf_enroll(hardware_list, enrollment):
    """
    Fase de enrollment: extrai e armazena assinaturas PUF de todos os clientes.
    Executada UMA vez, antes do treinamento federado começar.

    Returns:
        dict {device_id: enrolled_signature (np.ndarray bool)}
    """
    db = {}
    for hw in hardware_list:
        did = hw.device_identity.device_id
        db[did] = enrollment.extract_signature(hw)
    return db


def puf_verify(hw, enrollment, enrolled_db, threshold=PUF_THRESHOLD):
    """
    Verifica a identidade do hardware de um cliente via PUF.

    Returns:
        (passed: bool, similarity: float)
    """
    did     = hw.device_identity.device_id
    if did not in enrolled_db:
        return False, 0.0
    current  = enrollment.extract_signature(hw)
    enrolled = enrolled_db[did]
    hamming  = float(np.mean(current != enrolled))
    sim      = 1.0 - hamming
    return sim >= threshold, round(sim, 4)


# =============================================================================
# 5. CENÁRIO 1 — FEDAVG HONESTO (BASELINE)
# =============================================================================
print("\n" + "="*60)
print("CENÁRIO 1: FedAvg Honesto (Baseline)")
print("="*60)

model_base  = create_model()
history_base_asr = []

for r in range(COMMUNICATION_ROUNDS):
    local_weights = []
    for i in range(NUM_CLIENTS):
        lm = create_model()
        lm.load_state_dict(copy.deepcopy(model_base.state_dict()))
        train_local(lm, client_X[i], client_y[i])
        local_weights.append(copy.deepcopy(lm.state_dict()))
    fedavg(model_base, local_weights)
    _, asr = evaluate(model_base)
    history_base_asr.append(asr)
    if (r + 1) % 50 == 0:
        print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
        evaluate(model_base, verbose=True)

acc_base, asr_base = evaluate(model_base)
print(f"RESULTADO BASELINE -> Acuracia: {acc_base:.2f}%  ASR: {asr_base:.2f}%")

# =============================================================================
# 6. CENÁRIO 2 — WEIGHT SCALING ATTACK
# =============================================================================
print("\n" + "="*60)
print("CENÁRIO 2: Weight Scaling Attack (sem defesa)")
print("="*60)

model_atk       = create_model()
history_atk_asr = []
scale_factor    = NUM_CLIENTS / len(POISONED_CLIENTS)

for r in range(COMMUNICATION_ROUNDS):
    local_weights = []
    for i in range(NUM_CLIENTS):
        lm = create_model()
        lm.load_state_dict(copy.deepcopy(model_atk.state_dict()))
        train_local(lm, client_X[i], client_y_poisoned[i])
        if i in POISONED_CLIENTS:
            apply_weight_scaling(lm, model_atk, scale_factor)
        local_weights.append(copy.deepcopy(lm.state_dict()))
    fedavg(model_atk, local_weights)
    _, asr = evaluate(model_atk)
    history_atk_asr.append(asr)
    if (r + 1) % 50 == 0:
        print(f"  Rodada {r+1}/{COMMUNICATION_ROUNDS}", end=" | ")
        evaluate(model_atk, verbose=True)

acc_atk, asr_atk = evaluate(model_atk)
print(f"RESULTADO ATAQUE -> Acuracia: {acc_atk:.2f}%  ASR: {asr_atk:.2f}%")

# =============================================================================
# 7. CENÁRIO 3 — SENTINEL-FLOW (COMPORTAMENTAL)
# =============================================================================
print("\n" + "="*60)
print("CENÁRIO 3: Sentinel-Flow — Atestação Comportamental")
print("="*60)

model_sf       = create_model()
history_sf_asr = []

for r in range(COMMUNICATION_ROUNDS):
    accepted_weights = []
    n_blocked = 0

    for i in range(NUM_CLIENTS):
        lm = create_model()
        lm.load_state_dict(copy.deepcopy(model_sf.state_dict()))
        train_local(lm, client_X[i], client_y_poisoned[i])
        if i in POISONED_CLIENTS:
            apply_weight_scaling(lm, model_sf, scale_factor)

        # ── Atestação comportamental (canary flows) ────────────
        trust = compute_trust_score(lm, canary_X, canary_y)
        if trust >= TRUST_THRESHOLD:
            accepted_weights.append(copy.deepcopy(lm.state_dict()))
        else:
            n_blocked += 1

    if accepted_weights:
        fedavg(model_sf, accepted_weights)

    _, asr = evaluate(model_sf)
    history_sf_asr.append(asr)
    if (r + 1) % 50 == 0:
        print(f"  Rodada {r+1} | Aceitos: {len(accepted_weights)} | Bloqueados: {n_blocked}", end=" | ")
        evaluate(model_sf, verbose=True)

acc_sf, asr_sf = evaluate(model_sf)
print(f"RESULTADO SENTINEL-FLOW -> Acuracia: {acc_sf:.2f}%  ASR: {asr_sf:.2f}%")

# =============================================================================
# 8. CENÁRIO 4 — SENTINEL-FLOW + PUF (se módulo disponível)
# =============================================================================
history_puf_asr = []
acc_puf = asr_puf = None

if PUF_AVAILABLE:
    print("\n" + "="*60)
    print("CENÁRIO 4: Sentinel-Flow + Analog PUF")
    print("="*60)

    # ── Cria hardware analógico para cada cliente ──────────────────────────────
    # Cada cliente tem um chip PCM único com device_id = índice do cliente
    client_hardware = [ClientHardware(device_id=i) for i in range(NUM_CLIENTS)]

    print(f"Hardware analógico criado para {NUM_CLIENTS} clientes:")
    for hw in client_hardware[:3]:
        print(f"  {hw}")
    print("  ...")

    # ── Enrollment PUF: executado UMA vez, antes do FL ────────────────────────
    print("\nFase de enrollment PUF...")
    enrollment = PUFEnrollment(n_challenges=PUF_CHALLENGES)
    enrolled_db = puf_enroll(client_hardware, enrollment)
    print(f"Enrollment concluído: {len(enrolled_db)} chips cadastrados.")

    # ── Treinamento FL com dupla atestação ────────────────────────────────────
    model_puf      = create_model()
    history_puf_asr = []

    for r in range(COMMUNICATION_ROUNDS):
        accepted_weights = []
        n_blocked_puf  = 0
        n_blocked_beh  = 0

        # Rotaciona os challenges PUF a cada rodada
        # Isso impede two-faced attacks: o adversário não sabe
        # quais condutâncias serão testadas nesta rodada
        enrollment.rotate(r + 1)

        for i in range(NUM_CLIENTS):
            hw = client_hardware[i]
            lm = create_model()
            lm.load_state_dict(copy.deepcopy(model_puf.state_dict()))
            train_local(lm, client_X[i], client_y_poisoned[i])
            if i in POISONED_CLIENTS:
                apply_weight_scaling(lm, model_puf, scale_factor)

            # ── Fase 1: Atestação PUF (hardware) ──────────────
            # O controlador SDN envia challenge ao chip analógico.
            # Verifica se a assinatura bate com o enrollment.
            puf_ok, puf_sim = puf_verify(hw, enrollment, enrolled_db)

            if not puf_ok:
                # Chip não reconhecido: pode ser Sybil, substituição
                # de hardware ou falha de dispositivo
                n_blocked_puf += 1
                continue

            # ── Fase 2: Atestação comportamental (canary flows) ──
            # Apenas clientes com hardware verificado passam para
            # a etapa comportamental
            trust = compute_trust_score(lm, canary_X, canary_y)

            if trust >= TRUST_THRESHOLD:
                accepted_weights.append(copy.deepcopy(lm.state_dict()))
            else:
                n_blocked_beh += 1

        if accepted_weights:
            fedavg(model_puf, accepted_weights)

        _, asr = evaluate(model_puf)
        history_puf_asr.append(asr)

        if (r + 1) % 50 == 0:
            print(f"  Rodada {r+1} | Aceitos: {len(accepted_weights)} | Bloq. PUF: {n_blocked_puf} | Bloq. Beh: {n_blocked_beh}", end=" | ")
            evaluate(model_puf, verbose=True)

    acc_puf, asr_puf = evaluate(model_puf)
    print(f"RESULTADO SF+PUF -> Acuracia: {acc_puf:.2f}%  ASR: {asr_puf:.2f}%")

    # =========================================================================
    # 9. DIAGNÓSTICO: SYBIL ATTACK
    # Demonstra que PUF detecta 1 atacante fingindo ser múltiplos clientes
    # =========================================================================
    print("\n" + "="*60)
    print("DIAGNÓSTICO: Sybil Attack com Analog PUF")
    print("="*60)
    print("Cenário: 1 adversário físico tenta controlar os 5 slots de atacantes")
    print("Mesmo chip (device_id=99) apresentado para 5 identidades diferentes\n")

    sybil_hw  = ClientHardware(device_id=99)  # 1 chip real do adversário
    enrollment.rotate(0)  # usa o mesmo challenge do enrollment original

    # Re-enrola com os challenges originais (round_seed=0)
    enrollment_orig = PUFEnrollment(n_challenges=PUF_CHALLENGES, round_seed=0)
    enrolled_orig   = puf_enroll(client_hardware, enrollment_orig)

    sybil_results = []
    for fake_id in POISONED_CLIENTS:
        # Sybil apresenta o MESMO chip, mas tenta ser reconhecido como
        # um dos 5 clientes comprometidos
        puf_ok, sim = puf_verify(sybil_hw, enrollment_orig, enrolled_orig)
        enrolled_sig = enrolled_orig[fake_id]
        current_sig  = enrollment_orig.extract_signature(sybil_hw)
        actual_sim   = round(1.0 - float(np.mean(current_sig != enrolled_sig)), 3)
        status = "PASSOU" if actual_sim >= PUF_THRESHOLD else "BLOQUEADO"
        sybil_results.append((fake_id, actual_sim, status))
        print(f"  Chip 99 tentando ser Device {fake_id}: sim={actual_sim:.3f}  [{status}]")

    n_detected = sum(1 for _, _, s in sybil_results if s == "BLOQUEADO")
    print(f"\nSybil detectado: {n_detected}/{len(POISONED_CLIENTS)} tentativas bloqueadas pelo PUF.")
    if n_detected == len(POISONED_CLIENTS):
        print("Resultado: TODOS bloqueados — ataque Sybil completamente neutralizado.")

# =============================================================================
# 10. TABELA COMPARATIVA
# =============================================================================
print("\n" + "="*60)
print("COMPARATIVO FINAL DE TODOS OS CENÁRIOS")
print("="*60)
print("{:<30} {:>12} {:>10}".format("Cenário", "Acurácia", "ASR"))
print("-"*54)
print("{:<30} {:>11.2f}% {:>9.2f}%".format("1. FedAvg Honesto", acc_base, asr_base))
print("{:<30} {:>11.2f}% {:>9.2f}%".format("2. Weight Scaling Attack", acc_atk, asr_atk))
print("{:<30} {:>11.2f}% {:>9.2f}%".format("3. Sentinel-Flow", acc_sf, asr_sf))
if acc_puf is not None:
    print("{:<30} {:>11.2f}% {:>9.2f}%".format("4. Sentinel-Flow + PUF", acc_puf, asr_puf))
print("="*60)

# =============================================================================
# 11. GRÁFICO
# =============================================================================
print("\nGerando gráfico...")

n_scenarios = 3 + (1 if history_puf_asr else 0)
rounds = list(range(1, COMMUNICATION_ROUNDS + 1))

fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 5), sharey=True)
if n_scenarios == 1:
    axes = [axes]

configs = [
    (history_base_asr, "1. FedAvg Honesto",        "tab:blue"),
    (history_atk_asr,  "2. Weight Scaling Attack", "tab:red"),
    (history_sf_asr,   "3. Sentinel-Flow",         "tab:green"),
]
if history_puf_asr:
    configs.append((history_puf_asr, "4. Sentinel-Flow + PUF", "tab:purple"))

for ax, (history, title, color) in zip(axes, configs):
    ax.plot(rounds, history, color=color, linewidth=1.5, alpha=0.9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Rodadas")
    ax.set_ylabel("ASR (%)")
    ax.set_ylim(-5, 105)
    ax.axhline(y=60, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='τ=60%')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.suptitle(
    f"Evolução da Taxa de Sucesso do Ataque (ASR)\n"
    f"{NUM_CLIENTS} Clientes | {len(POISONED_CLIENTS)} Atacantes | {COMMUNICATION_ROUNDS} Rodadas",
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig("sentinel_puf_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("Gráfico salvo: sentinel_puf_results.png")

# =============================================================================
# 12. EXPORTAR RESULTADOS PARA O VISUALIZADOR HTML (Integrado)
# =============================================================================
results = {
    "generated": datetime.datetime.now().isoformat(),
    "source": "fl_sentinel_puf.py",
    "config": {
        "num_clients":       NUM_CLIENTS,
        "rounds":            COMMUNICATION_ROUNDS,
        "poisoned_clients":  POISONED_CLIENTS,
        "trust_threshold":   TRUST_THRESHOLD,
        "puf_threshold":     PUF_THRESHOLD if PUF_AVAILABLE else None,
        "puf_available":     PUF_AVAILABLE,
    },
    "scenarios": {
        "baseline": {
            "accuracy": round(acc_base, 2),
            "asr":      round(asr_base, 2),
            "history":  [{"round": i + 1, "asr": round(v, 2)}
                         for i, v in enumerate(history_base_asr)],
        },
        "attack": {
            "accuracy": round(acc_atk, 2),
            "asr":      round(asr_atk, 2),
            "history":  [{"round": i + 1, "asr": round(v, 2)}
                         for i, v in enumerate(history_atk_asr)],
        },
        "sentinel": {
            "accuracy": round(acc_sf, 2),
            "asr":      round(asr_sf, 2),
            "history":  [{"round": i + 1, "asr": round(v, 2)}
                         for i, v in enumerate(history_sf_asr)],
        },
    },
}

# Adiciona cenário PUF se disponível
if PUF_AVAILABLE and acc_puf is not None:
    results["scenarios"]["puf"] = {
        "accuracy": round(acc_puf, 2),
        "asr":      round(asr_puf, 2),
        "history":  [{"round": i + 1, "asr": round(v, 2)}
                     for i, v in enumerate(history_puf_asr)],
    }

output_file = "sentinel_puf_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("EXPORTAÇÃO CONCLUÍDA")
print("=" * 60)
print(f"Arquivo: {output_file}")
print(f"Cenários: {list(results['scenarios'].keys())}")
print()
print("Para visualizar os dados reais no HTML:")
print("  1. Abra sentinel_puf_presentation.html no navegador")
print("  2. Clique na aba 'Importar Dados'")
print(f"  3. Cole o conteúdo de {output_file}")
print("=" * 60)
