"""
client_app.py
=============

Client Flower XGBoost — entraînement séquentiel + soft voting pondéré.

Stratégie : chaque round Flower = 1 client entraîne NUM_LOCAL_ROUNDS arbres
sur ses données locales en continuant depuis le modèle global courant.
Le serveur orchestre l'ordre séquentiel (mélangé par round sémantique).

Le client est sans état persistent pour la partition : il reçoit
``target_partition`` dans FitIns.config et charge les données correspondantes
à la demande (avec cache local).
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Tuple

import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, f1_score

from flwr.client import Client, ClientApp
from flwr.common import (
    Code, Context, FitIns, FitRes, EvaluateIns, EvaluateRes,
    Status, ndarrays_to_parameters, parameters_to_ndarrays,
)

from task import (
    get_client_partition, load_data,
    XGBOOST_PARAMS, NUM_LOCAL_ROUNDS, NUM_CLIENTS,
    serialize_model, deserialize_model,
)


class XGBoostFlowerClient(Client):
    """
    Client Flower XGBoost sans état persistent sur la partition.

    Reçoit ``target_partition`` (int) dans FitIns.config à chaque round.
    Charge et met en cache les données de la partition demandée.
    Entraîne NUM_LOCAL_ROUNDS arbres supplémentaires sur le modèle global reçu.

    Paramètres d'entraînement identiques à l'implémentation originale :
      Round 1 : xgb.train() from scratch (global_bst absent).
      Round > 1 : xgb.train(xgb_model=global_bst) — boosting incrémental.
    """

    def __init__(self) -> None:
        self._train_df   = None
        self._val_df     = None
        self._feature_cols = None
        self._le         = None
        self._n_classes  = XGBOOST_PARAMS["num_class"]
        # cache : partition_id -> (train_dmat, val_dmat, y_val, n_train)
        self._cache: Dict[int, Tuple] = {}

    # ── Chargement des données ──────────────────────────────────────────────

    def _load_global(self) -> None:
        if self._train_df is None:
            self._train_df, self._val_df, _, self._feature_cols, self._le = load_data()

    def _get_partition(self, pid: int) -> Tuple:
        if pid not in self._cache:
            self._load_global()
            train_dmat, _ = get_client_partition(
                pid, self._train_df, self._val_df,
                self._feature_cols, self._le, NUM_CLIENTS,
            )
            X_val   = self._val_df[self._feature_cols].values
            y_val   = self._le.transform(self._val_df["SampleType"].values)
            val_dmat = xgb.DMatrix(X_val, label=y_val)
            self._cache[pid] = (train_dmat, val_dmat, y_val, train_dmat.num_row())
        return self._cache[pid]

    # ── Interface Flower ────────────────────────────────────────────────────

    def fit(self, ins: FitIns) -> FitRes:
        pid           = int(ins.config.get("target_partition", 0))
        current_round = int(ins.config.get("current_round",    1))

        train_dmat, val_dmat, y_val, n_train = self._get_partition(pid)

        # Désérialiser le modèle global (absent au round 1)
        arrays     = parameters_to_ndarrays(ins.parameters)
        global_bst = deserialize_model(bytes(arrays[0].tobytes())) if arrays and len(arrays[0]) > 0 else None

        # Entraînement local — même logique que XGBoostClient.fit()
        if global_bst is None:
            bst = xgb.train(
                XGBOOST_PARAMS, train_dmat,
                num_boost_round=NUM_LOCAL_ROUNDS,
                verbose_eval=False,
            )
        else:
            bst = xgb.train(
                XGBOOST_PARAMS, train_dmat,
                num_boost_round=NUM_LOCAL_ROUNDS,
                xgb_model=global_bst,
                verbose_eval=False,
            )

        # Métriques locales pour affichage
        val_proba = bst.predict(val_dmat).reshape(-1, self._n_classes)
        metrics   = _compute_metrics(val_proba, y_val, self._n_classes)

        print(
            f"  Client {pid} | R{current_round:>2} | {n_train:>4} samples "
            f"| LogLoss={metrics['log_loss']:.4f} "
            f"| Acc={metrics['accuracy']:.4f} "
            f"| Arbres={bst.num_boosted_rounds()}"
        )

        model_array = np.frombuffer(serialize_model(bst), dtype=np.uint8).copy()

        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([model_array]),
            num_examples=n_train,
            metrics={
                "log_loss":        metrics["log_loss"],
                "accuracy":        metrics["accuracy"],
                "f1_macro":        metrics["f1_macro"],
                "partition_id":    float(pid),
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:  # noqa: ARG002
        # L'évaluation globale est faite côté serveur (soft voting sur val set).
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=0.0,
            num_examples=0,
        )


# ── Helpers ─────────────────────────────────────────────────────────────────

def _compute_metrics(proba: np.ndarray, y_val: np.ndarray, n_classes: int) -> Dict:
    pred = proba.argmax(axis=1)
    return {
        "log_loss": float(log_loss(y_val, proba, labels=list(range(n_classes)))),
        "accuracy": float(accuracy_score(y_val, pred)),
        "f1_macro": float(f1_score(y_val, pred, average="macro", zero_division=0)),
    }


# ── Point d'entrée Flower ────────────────────────────────────────────────────

def client_fn(context: Context) -> Client:  # noqa: ARG001: context non utilisé
    return XGBoostFlowerClient()


app = ClientApp(client_fn=client_fn)
