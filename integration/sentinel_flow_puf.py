"""
integration/sentinel_flow_puf.py

SentinelFlowPUF: substitui o TEE de software do Sentinel-Flow original
por uma raiz de confiança física baseada em Analog PUF.

Protocolo estendido (Sentinel-Flow + PUF):
    1. SDN Controller envia challenge PUF ao cliente
    2. Cliente responde com assinatura do chip analogico
    3. Agregador verifica identidade via Hamming distance
    4. SDN Controller injeta canary flows (comportamental)
    5. Trust Score combinado: PUF_ok AND behavior_ok

Fecha 2 vetores nao cobertos pelo Sentinel-Flow original:
    - Sybil attack: 1 adversario fingindo ser N clientes
    - Two-faced attack: adversario que passa no canary mas
      modifica pesos entre rodadas

Refs: Zago et al. 2025 (Sentinel-Flow), Gao et al. 2020 (PUF),
      Maiti & Schaumont 2011 (metricas PUF)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClientRecord:
    device_id: int
    enrolled_signature: Optional[np.ndarray] = None
    trust_score: float = 0.0
    puf_similarity: float = 0.0
    puf_verified: bool = False
    behavioral_verified: bool = False
    rounds_participated: int = 0
    rounds_rejected_puf: int = 0
    rounds_rejected_behavior: int = 0

    @property
    def fully_trusted(self):
        return self.puf_verified and self.behavioral_verified


@dataclass
class RoundResult:
    round_id: int
    n_clients: int
    approved: list = field(default_factory=list)
    rejected_puf: list = field(default_factory=list)
    rejected_behavior: list = field(default_factory=list)
    rollback: bool = False

    @property
    def n_approved(self):
        return len(self.approved)

    @property
    def n_rejected(self):
        return len(self.rejected_puf) + len(self.rejected_behavior)


class SentinelFlowPUF:
    """
    Orquestrador do protocolo Sentinel-Flow + PUF.

    Cada rodada FL:
        1. rotate(round_id) nos challenges PUF (anti two-faced attack)
        2. Para cada cliente:
            a. Verifica assinatura PUF (autenticacao de hardware)
            b. Calcula Trust Score (canary flows comportamentais)
            c. Aceita se PUF_ok AND trust >= tau_safe
        3. FedAvg apenas sobre clientes aprovados
        4. Rollback se nenhum aprovado

    Usage:
        orch = SentinelFlowPUF(tau_safe=0.60, puf_threshold=0.90)
        orch.enroll_clients(hardware_list, enrollment)
        result = orch.attest_round(round_id, hardware_list, models,
                                   canary_X, canary_y, weights, enrollment)
        approved = [weights[i] for i in result.approved]
    """

    def __init__(
        self,
        tau_safe: float = 0.60,
        puf_threshold: float = 0.90,
        require_puf: bool = True,
        verbose: bool = True,
    ):
        self.tau_safe = tau_safe
        self.puf_threshold = puf_threshold
        self.require_puf = require_puf
        self.verbose = verbose
        self.clients = {}
        self.history = []

    # ── Enrollment ────────────────────────────────────────────────────────────

    def enroll_clients(self, hardware_list, enrollment):
        """
        Fase de enrollment: cadastra assinaturas PUF de todos os clientes.
        Executada UMA vez antes do inicio do treinamento federado.

        Args:
            hardware_list : lista de ClientHardware, um por cliente
            enrollment    : instancia de PUFEnrollment

        Returns:
            dict {device_id: enrolled_signature}
        """
        enrolled_db = {}
        for hw in hardware_list:
            did = hw.device_identity.device_id
            sig = enrollment.extract_signature(hw)
            self.clients[did] = ClientRecord(
                device_id=did,
                enrolled_signature=sig
            )
            enrolled_db[did] = sig

        if self.verbose:
            print("[ENROLLMENT] " + str(len(hardware_list)) +
                  " clientes cadastrados.")
        return enrolled_db

    # ── PUF verification ──────────────────────────────────────────────────────

    def _verify_puf(self, hw, enrollment):
        """
        Verifica a identidade do hardware via desafio-resposta PUF.

        Returns:
            (passed: bool, similarity: float)
        """
        did = hw.device_identity.device_id
        if did not in self.clients:
            return False, 0.0
        current = enrollment.extract_signature(hw)
        enrolled = self.clients[did].enrolled_signature
        hamming = float(np.mean(current != enrolled))
        sim = 1.0 - hamming
        return sim >= self.puf_threshold, round(sim, 4)

    # ── Behavioral attestation ────────────────────────────────────────────────

    def _compute_trust_score(self, model, canary_X, canary_y):
        """
        Trust Score comportamental (Eq. 2 do Sentinel-Flow).

        Trust(i) = |{k : M_i(x_atk_k) = Ataque}| / K

        Args:
            model    : modelo PyTorch do cliente (nn.Module)
            canary_X : tensor (K, features) com trafego malicioso
            canary_y : tensor (K, 1) com labels = 1 (ataque)

        Returns:
            float em [0.0, 1.0]
        """
        import torch
        model.eval()
        with torch.no_grad():
            preds = (model(canary_X) > 0.5).float()
            return (preds == canary_y).float().mean().item()

    # ── Main attestation round ────────────────────────────────────────────────

    def attest_round(
        self,
        round_id,
        hardware_list,
        models,
        canary_X,
        canary_y,
        weights,
        enrollment,
    ):
        """
        Executa um ciclo completo de atestacao para uma rodada FL.

        Args:
            round_id      : indice da rodada (int)
            hardware_list : lista de ClientHardware (um por cliente)
            models        : lista de nn.Module (modelos treinados localmente)
            canary_X      : tensor com trafego de ataque para canary test
            canary_y      : tensor com labels dos canary flows
            weights       : lista de state_dict (pesos locais)
            enrollment    : PUFEnrollment (ja rotacionado para esta rodada)

        Returns:
            RoundResult com listas de aprovados/rejeitados e flag de rollback
        """
        result = RoundResult(round_id=round_id, n_clients=len(models))

        for i, (hw, model) in enumerate(zip(hardware_list, models)):
            did = hw.device_identity.device_id

            # ── Fase 1: Verificacao PUF ───────────────────────────────────
            if self.require_puf:
                puf_ok, puf_sim = self._verify_puf(hw, enrollment)
            else:
                puf_ok, puf_sim = True, 1.0

            # ── Fase 2: Trust Score comportamental ───────────────────────
            trust = self._compute_trust_score(model, canary_X, canary_y)

            # ── Atualiza registro ─────────────────────────────────────────
            if did in self.clients:
                c = self.clients[did]
                c.trust_score = trust
                c.puf_similarity = puf_sim
                c.puf_verified = puf_ok
                c.behavioral_verified = trust >= self.tau_safe

            # ── Decisao ───────────────────────────────────────────────────
            if not puf_ok:
                result.rejected_puf.append(i)
                if did in self.clients:
                    self.clients[did].rounds_rejected_puf += 1
                if self.verbose:
                    print("  [R" + str(round_id) + "] Device " + str(did) +
                          " REJEITADO (PUF)  sim=" + str(puf_sim) +
                          " < " + str(self.puf_threshold))

            elif trust < self.tau_safe:
                result.rejected_behavior.append(i)
                if did in self.clients:
                    self.clients[did].rounds_rejected_behavior += 1
                if self.verbose:
                    print("  [R" + str(round_id) + "] Device " + str(did) +
                          " REJEITADO (Behavior)  trust=" +
                          str(round(trust, 3)) + " < " + str(self.tau_safe))

            else:
                result.approved.append(i)
                if did in self.clients:
                    self.clients[did].rounds_participated += 1
                if self.verbose:
                    print("  [R" + str(round_id) + "] Device " + str(did) +
                          " APROVADO  puf=" + str(puf_sim) +
                          "  trust=" + str(round(trust, 3)))

        # ── Rollback ──────────────────────────────────────────────────────────
        if len(result.approved) == 0:
            result.rollback = True
            if self.verbose:
                print("  [R" + str(round_id) + "] ROLLBACK — nenhum cliente aprovado.")

        self.history.append(result)
        return result

    # ── Aggregation ───────────────────────────────────────────────────────────

    def fedavg_approved(self, result, weights, dataset_sizes=None):
        """
        FedAvg ponderado sobre clientes aprovados (Eq. 4 do Sentinel-Flow).

        Args:
            result        : RoundResult da rodada corrente
            weights       : lista de state_dict (todos os clientes)
            dataset_sizes : lista de tamanhos de dataset local.
                            Se None, usa media simples (peso uniforme).

        Returns:
            state_dict agregado, ou None se rollback
        """
        import torch

        if result.rollback:
            return None

        idx = result.approved

        if dataset_sizes is not None:
            sizes = [dataset_sizes[i] for i in idx]
            total = sum(sizes)
            w_factors = [s / total for s in sizes]
        else:
            w_factors = [1.0 / len(idx)] * len(idx)

        import copy
        agg = copy.deepcopy(weights[idx[0]])
        for key in agg:
            agg[key] = agg[key].float() * w_factors[0]

        for j, i in enumerate(idx[1:], start=1):
            for key in agg:
                agg[key] += weights[i][key].float() * w_factors[j]

        return agg

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary(self):
        """Retorna dict com estatisticas agregadas de todas as rodadas."""
        if not self.history:
            return {}
        total_rounds = len(self.history)
        return {
            "total_rounds": total_rounds,
            "total_approved": sum(r.n_approved for r in self.history),
            "total_rejected": sum(r.n_rejected for r in self.history),
            "rollbacks": sum(1 for r in self.history if r.rollback),
            "avg_approved_per_round": round(
                sum(r.n_approved for r in self.history) / total_rounds, 2
            ),
            "clients": {
                did: {
                    "participated": c.rounds_participated,
                    "rejected_puf": c.rounds_rejected_puf,
                    "rejected_behavior": c.rounds_rejected_behavior,
                    "last_trust_score": round(c.trust_score, 4),
                    "last_puf_similarity": round(c.puf_similarity, 4),
                }
                for did, c in self.clients.items()
            },
        }

    def print_summary(self):
        s = self.summary()
        if not s:
            print("Sem historico de rodadas.")
            return
        print("=" * 60)
        print("SENTINEL-FLOW + PUF — RESUMO")
        print("=" * 60)
        print("Rodadas totais   : " + str(s["total_rounds"]))
        print("Aprovacoes totais: " + str(s["total_approved"]))
        print("Rejeicoes totais : " + str(s["total_rejected"]))
        print("Rollbacks        : " + str(s["rollbacks"]))
        print("Aprovados/rodada : " + str(s["avg_approved_per_round"]))
        print("=" * 60)
        print("Detalhes por cliente:")
        for did, c in s["clients"].items():
            print("  Device " + str(did) + ": " +
                  "part=" + str(c["participated"]) +
                  " rej_puf=" + str(c["rejected_puf"]) +
                  " rej_beh=" + str(c["rejected_behavior"]) +
                  " trust=" + str(c["last_trust_score"]) +
                  " puf_sim=" + str(c["last_puf_similarity"]))
        print("=" * 60)
