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
from sklearn.metrics import log_loss, accuracy_score, f1_score

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Context, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays,
)

from task import (
    load_data, NUM_CLIENTS,
    serialize_model, deserialize_model,
)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics"
MODELS_PATH  = Path(__file__).resolve().parent.parent / "models"  / "federated"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

N_SEMANTIC_ROUNDS = 20  # 20 × 5 clients × 1 arbre = 100 arbres

# Val set chargé une seule fois pour l'évaluation serveur
_, _val_df, _, _feature_cols, _le = load_data()
_X_val       = _val_df[_feature_cols].values
_y_val       = _le.transform(_val_df["SampleType"].values)
_val_dmatrix = xgb.DMatrix(_X_val, label=_y_val)
_n_classes   = len(_le.classes_)

history: List[Dict] = []


def soft_voting(probas: List[np.ndarray], weights: List[int]) -> np.ndarray:
    """Agrège les probabilités par soft voting pondéré : p = Σ (n_i/N) * p_i."""
    total = sum(weights)
    agg   = np.zeros_like(probas[0])
    for proba, w in zip(probas, weights):
        agg += (w / total) * proba
    return agg


def evaluate_global(agg_proba: np.ndarray, sem_round: int, n_trees: int) -> Dict:
    """Évalue les probabilités agrégées sur le val set et sauvegarde."""
    y_pred = agg_proba.argmax(axis=1)
    ll  = float(log_loss(_y_val, agg_proba, labels=list(range(_n_classes))))
    acc = float(accuracy_score(_y_val, y_pred))
    f1  = float(f1_score(_y_val, y_pred, average="macro", zero_division=0))

    result = {
        "round": sem_round, "log_loss": ll,
        "accuracy": acc, "f1_macro": f1, "n_trees": n_trees,
    }
    history.append(result)
    pd.DataFrame(history).to_csv(RESULTS_PATH / "federated_metrics.csv", index=False)
    print(f"  -> Global | LogLoss={ll:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Arbres={n_trees}")
    return result


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

    # ── Interface Strategy ──────────────────────────────────────────────────

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
        server_round: int,   # noqa: ARG002
        parameters: Parameters,  # noqa: ARG002
        client_manager: ClientManager,  # noqa: ARG002
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return []  # Évaluation globale faite côté serveur

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
        result  = evaluate_global(agg_proba, sem_round, n_trees)

        if result["log_loss"] < self.best_log_loss:
            self.best_log_loss = result["log_loss"]
            self.best_bst      = self.global_bst
            if self.best_bst:
                self.best_bst.save_model(str(MODELS_PATH / "best_global_model.cubj"))
            print(f"  Best LogLoss : {self.best_log_loss:.4f}")

        del self._round_models[sem_round]


# ── Point d'entrée Flower ────────────────────────────────────────────────────

def server_fn(_context: Context) -> ServerAppComponents:
    return ServerAppComponents(
        strategy = SequentialXGBoostStrategy(
            n_clients         = NUM_CLIENTS,
            n_semantic_rounds = N_SEMANTIC_ROUNDS,
        ),
        config = ServerConfig(num_rounds=NUM_CLIENTS * N_SEMANTIC_ROUNDS),
    )


app = ServerApp(server_fn=server_fn)
