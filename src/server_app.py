"""
server_app.py
=============

Serveur Flower — stratégie séquentielle XGBoost + soft voting pondéré.

Chaque round Flower = 1 client entraîne NUM_LOCAL_ROUNDS arbres.
Un round sémantique = NUM_CLIENTS rounds Flower consécutifs.
L'ordre des clients est mélangé à chaque round sémantique
(seed = 42 + round_sémantique) — identique à la simulation originale.

Après chaque round sémantique complet :
  - Soft voting pondéré sur les modèles locaux de tous les clients
  - Évaluation sur le val set (log_loss, accuracy, f1_macro)
  - Sauvegarde du meilleur modèle global (log_loss minimal)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb


from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Context, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)

from task import (
    NUM_CLIENTS, _val_dmatrix, _n_classes,
    serialize_model, deserialize_model,
    soft_voting, evaluate_global, merge_xgb_trees,
)

MODELS_PATH  = Path(__file__).resolve().parent.parent / "models"  / "federated"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

N_SEMANTIC_ROUNDS = 20  # 20 × 5 clients × 1 arbre = 100 arbres


class SequentialXGBoostStrategy(Strategy):
    """
    Stratégie Flower reproduisant l'entraînement séquentiel + soft voting.

    Chaque round Flower sélectionne 1 seul client (la partition ``target_partition``
    déterminée par l'ordre mélangé du round sémantique courant).
    Le modèle global est mis à jour immédiatement après chaque client,
    exactement comme dans run_federated() de l'implémentation originale.

    Après chaque round sémantique (tous les n_clients rounds Flower), un
    soft voting pondéré est réalisé sur les modèles intermédiaires accumulés
    — identique à la boucle originale.
    """

    def __init__(self, n_clients: int = NUM_CLIENTS, n_semantic_rounds: int = N_SEMANTIC_ROUNDS):
        self.n_clients         = n_clients
        self.n_semantic_rounds = n_semantic_rounds
        self.global_bst: Optional[xgb.Booster] = None
        self.best_log_loss     = float("inf")
        self.best_bst: Optional[xgb.Booster] = None

        # Précalcul : flower_round -> (sem_round, étape, target_partition)
        # Même graine que run_federated : seed = 42 + round_sémantique
        self.round_info: Dict[int, Tuple[int, int, int]] = {}
        for sem in range(1, n_semantic_rounds + 1):
            rng   = np.random.default_rng(42 + sem)
            order = rng.permutation(n_clients).tolist()
            for step, partition in enumerate(order):
                fl_round = (sem - 1) * n_clients + step + 1
                self.round_info[fl_round] = (sem, step, partition)

        # sem_round -> {partition: (booster_intermédiaire, n_train)}
        # Conserve les modèles de chaque client pour le soft voting de fin de round.
        self._round_models: Dict[int, Dict[int, Tuple[xgb.Booster, int]]] = {}

        print(f"\n{'='*60}")
        print(f"SIMULATION FÉDÉRÉE FLOWER — Séquentiel + Soft Voting")
        print(f"  Clients        : {n_clients}")
        print(f"  Rounds sém.    : {n_semantic_rounds}")
        print(f"  Rounds Flower  : {n_clients * n_semantic_rounds}")
        print(f"  Stratégie      : séquentiel (ordre mélangé chaque round)")
        print(f"  Agrégation     : Soft Voting pondéré par n_samples")
        print(f"{'='*60}\n")

    # ── Interface Strategy de Flower ──────────────────────────────────────────────────

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None  # Les clients entraînent depuis zéro au round 1

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,  # noqa: ARG002
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sem_round, step, target_partition = self.round_info[server_round]

        if self.global_bst is not None:
            model_array = np.frombuffer(serialize_model(self.global_bst), dtype=np.uint8).copy()
            cur_params  = ndarrays_to_parameters([model_array])
        else:
            cur_params  = ndarrays_to_parameters([np.array([], dtype=np.uint8)])

        config = {
            "target_partition": float(target_partition),
            "current_round":    float(sem_round),
        }

        print(
            f"\n── Round Sémantique {sem_round}/{self.n_semantic_rounds} "
            f"(étape {step + 1}/{self.n_clients}) "
            f"── Partition {target_partition} ──"
        )

        # Sélectionne le seul client disponible (num_supernodes=1) et lui envoie
        # le modèle global + la config (target_partition, current_round).
        proxy = client_manager.sample(num_clients=1, min_num_clients=1)[0]
        return [(proxy, FitIns(cur_params, config))]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],  # noqa: ARG002
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        sem_round, step, target_partition = self.round_info[server_round]
        _, fit_res = results[0]

        # Désérialiser et conserver le modèle du client pour le soft voting
        arrays      = parameters_to_ndarrays(fit_res.parameters)
        client_bst  = deserialize_model(bytes(arrays[0].tobytes())) if arrays and len(arrays[0]) > 0 else None

        if client_bst is not None:
            # Mise à jour immédiate du modèle global (séquentiel)
            self.global_bst = client_bst
            self._round_models.setdefault(sem_round, {})[target_partition] = (
                client_bst, fit_res.num_examples
            )

        # Fin d'un round sémantique : soft voting + évaluation
        if step == self.n_clients - 1:
            self._end_of_semantic_round(sem_round)

        if self.global_bst is not None:
            model_array = np.frombuffer(serialize_model(self.global_bst), dtype=np.uint8).copy()
            return ndarrays_to_parameters([model_array]), {}
        return None, {}

    def configure_evaluate(
        self,
        server_round: int,   # noqa: ARG002: pas d'évaluation client à configurer
        parameters: Parameters,  # noqa: ARG002
        client_manager: ClientManager,  # noqa: ARG002
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []  # Évaluation globale faite côté serveur

    def aggregate_evaluate(
        self,
        server_round: int,   # noqa: ARG002: pas d'évaluation client à agréger
        results: List[Tuple[ClientProxy, EvaluateRes]],  # noqa: ARG002
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],  # noqa: ARG002
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self,
        server_round: int,   # noqa: ARG002: pas d'évaluation client à évaluer
        parameters: Parameters,  # noqa: ARG002
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    # ── Logique interne ─────────────────────────────────────────────────────

    def _end_of_semantic_round(self, sem_round: int) -> None:
        """Soft voting sur les modèles intermédiaires + évaluation."""

        models_dict = self._round_models.get(sem_round, {})
        if not models_dict:
            return

        probas  = [
            bst.predict(_val_dmatrix).reshape(-1, _n_classes)
            for bst, _ in models_dict.values()
        ]
        weights = [n for _, n in models_dict.values()]
        agg_proba = soft_voting(probas, weights)
        n_trees = self.global_bst.num_boosted_rounds() if self.global_bst else 0

        result  = evaluate_global(agg_proba, sem_round, n_trees, "federated_metrics.csv")

        if result["log_loss"] < self.best_log_loss:
            self.best_log_loss = result["log_loss"]
            self.best_bst      = self.global_bst
            if self.best_bst:
                self.best_bst.save_model(str(MODELS_PATH / "best_global_model.cubj"))
            print(f"  Best LogLoss : {self.best_log_loss:.4f}")

        del self._round_models[sem_round]


class FedAvgXGBoostStrategy(Strategy):
    """
    Stratégie FedAvg parallèle pour XGBoost avec fusion d'arbres (Tree Merging).

    Différence clé vs SequentialXGBoostStrategy :
      - Séquentiel : chaque client build sur le modèle mis à jour par le précédent
        → Client 2 corrige les erreurs de Client 1 dans le même round sémantique.
      - Parallèle  : tous les clients reçoivent le MÊME modèle de départ du round,
        entraînent indépendamment, puis leurs nouveaux arbres sont fusionnés.

    Le parallélisme est simulé avec num_supernodes=1 :
    configure_fit() envoie _round_start_bst (snapshot) à tous les clients du round,
    ignorant les mises à jour intermédiaires de global_bst.
    """

    def __init__(self, n_clients: int = NUM_CLIENTS, n_semantic_rounds: int = N_SEMANTIC_ROUNDS):
        self.n_clients         = n_clients
        self.n_semantic_rounds = n_semantic_rounds
        self.global_bst: Optional[xgb.Booster] = None
        self.best_log_loss     = float("inf")
        self.best_bst: Optional[xgb.Booster]   = None

        # server_round -> (sem_round, étape, target_partition)
        self.round_info: Dict[int, Tuple[int, int, int]] = {}
        for sem in range(1, n_semantic_rounds + 1):
            rng   = np.random.default_rng(42 + sem)
            order = rng.permutation(n_clients).tolist()
            for step, partition in enumerate(order):
                fl_round = (sem - 1) * n_clients + step + 1
                self.round_info[fl_round] = (sem, step, partition)

        # Snapshot du modèle au début de chaque round sémantique
        self._round_start_bst: Optional[xgb.Booster] = None
        # sem_round -> [(client_bst, n_samples), ...]
        self._pending: Dict[int, List[Tuple[xgb.Booster, int]]] = {}

        print(f"\n{'='*60}")
        print(f"SIMULATION FÉDÉRÉE FLOWER — FedAvg Parallèle + Tree Merging")
        print(f"  Clients        : {n_clients}")
        print(f"  Rounds sém.    : {n_semantic_rounds}")
        print(f"  Rounds Flower  : {n_clients * n_semantic_rounds}")
        print(f"  Stratégie      : parallèle (même modèle de départ par round sém.)")
        print(f"  Agrégation     : Tree Merging + Soft Voting (évaluation)")
        print(f"{'='*60}\n")

    # ── Interface Strategy de Flower ──────────────────────────────────────────────────

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return None

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,  # noqa: ARG002
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        
        # Récupérer les infos du round courant pour déterminer la partition cible et le modèle de départ
        sem_round, step, target_partition = self.round_info[server_round]

        # Snapshot du modèle global au début du round sémantique
        if step == 0:
            self._round_start_bst = self.global_bst

        # Tous les clients reçoivent le modèle de DÉBUT de round (parallélisme)
        if self._round_start_bst is not None:
            arr        = np.frombuffer(serialize_model(self._round_start_bst), dtype=np.uint8).copy()
            cur_params = ndarrays_to_parameters([arr])
        else:
            cur_params = ndarrays_to_parameters([np.array([], dtype=np.uint8)])

        config = {
            "target_partition": float(target_partition),
            "current_round":    float(sem_round),
        }
        print(
            f"\n[FedAvg] Round Sém. {sem_round}/{self.n_semantic_rounds} "
            f"(étape {step + 1}/{self.n_clients}) ── Partition {target_partition} ──"
        )
        
        
        # Sélectionne le seul client disponible (num_supernodes=1) et lui envoie
        # le snapshot du début de round + la config (target_partition, current_round).
        proxy = client_manager.sample(num_clients=1, min_num_clients=1)[0]
        return [(proxy, FitIns(cur_params, config))]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],  # noqa: ARG002
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        sem_round, step, _ = self.round_info[server_round]
        _, fit_res = results[0]

        arrays     = parameters_to_ndarrays(fit_res.parameters)
        client_bst = deserialize_model(bytes(arrays[0].tobytes())) if arrays and len(arrays[0]) > 0 else None

        if client_bst is not None:
            self._pending.setdefault(sem_round, []).append((client_bst, fit_res.num_examples))

        # Fin du round sémantique : fusion des arbres + évaluation
        if step == self.n_clients - 1:
            self._end_of_semantic_round(sem_round)

        if self.global_bst is not None:
            arr = np.frombuffer(serialize_model(self.global_bst), dtype=np.uint8).copy()
            return ndarrays_to_parameters([arr]), {}
        return None, {}

    def configure_evaluate(
        self,
        server_round: int,   # noqa: ARG002
        parameters: Parameters,  # noqa: ARG002
        client_manager: ClientManager,  # noqa: ARG002
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,   # noqa: ARG002
        results: List[Tuple[ClientProxy, EvaluateRes]],  # noqa: ARG002
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],  # noqa: ARG002
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self,
        server_round: int,   # noqa: ARG002
        parameters: Parameters,  # noqa: ARG002
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    # ── Logique interne ─────────────────────────────────────────────────────

    def _end_of_semantic_round(self, sem_round: int) -> None:
        """Fusion des arbres parallèles + évaluation par soft voting."""
        pending = self._pending.get(sem_round, [])
        if not pending:
            return

        # Tree merging : fusion des nouveaux arbres de tous les clients
        self.global_bst = merge_xgb_trees(self._round_start_bst, pending)

        # Évaluation : soft voting sur les modèles individuels des clients
        probas  = [bst.predict(_val_dmatrix).reshape(-1, _n_classes) for bst, _ in pending]
        weights = [n for _, n in pending]
        agg_proba = soft_voting(probas, weights)

        n_trees = self.global_bst.num_boosted_rounds() if self.global_bst else 0
        result  = evaluate_global(agg_proba, sem_round, n_trees, "fedavg_metrics.csv")

        if result["log_loss"] < self.best_log_loss:
            self.best_log_loss = result["log_loss"]
            self.best_bst      = self.global_bst
            if self.best_bst:
                self.best_bst.save_model(str(MODELS_PATH / "best_global_model_fedavg.cubj"))
            print(f"  [FedAvg] Best LogLoss : {self.best_log_loss:.4f}")

        del self._pending[sem_round]


# ── Point d'entrée Flower ────────────────────────────────────────────────────

# ── Point d'entrée pour la stratégie séquentielle ─────────────────────────
def sequential_server_fn(_context: Context) -> ServerAppComponents:
    return ServerAppComponents(
        strategy = SequentialXGBoostStrategy(
            n_clients         = NUM_CLIENTS,
            n_semantic_rounds = N_SEMANTIC_ROUNDS,
        ),
        config = ServerConfig(num_rounds=NUM_CLIENTS * N_SEMANTIC_ROUNDS),
    )

sequential_app = ServerApp(server_fn=sequential_server_fn)


# ── Point d'entrée pour la stratégie FedAvg ─────────────────────────
def fedavg_server_fn(_context: Context) -> ServerAppComponents:
    return ServerAppComponents(
        strategy = FedAvgXGBoostStrategy(
            n_clients         = NUM_CLIENTS,
            n_semantic_rounds = N_SEMANTIC_ROUNDS,
        ),
        config = ServerConfig(num_rounds=NUM_CLIENTS * N_SEMANTIC_ROUNDS),
    )

fedavg_app = ServerApp(server_fn=fedavg_server_fn)
