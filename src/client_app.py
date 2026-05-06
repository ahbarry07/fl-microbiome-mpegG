"""
client_app.py
=============

Logique client XGBoost pour l'apprentissage fédéré multiclasse.

Stratégie : entraînement séquentiel + Soft Voting pondéré par n_samples.
  À chaque round, les clients enrichissent le modèle global l'un après l'autre
  (ordre mélangé). Après tous les clients, global_bst a vu TOUTES les données.
  Les prédictions finales sont agrégées par soft voting pondéré :

  p_global(x) = Σ_i (n_i / N) * p_i(x):
    où n_i est la taille du dataset local du client i, N = Σ_i n_i, et p_i(x) les probabilités prédites par le modèle local du client i.

  En pratique : chaque client reçoit global_bst, entraîne 1 arbre sur ses
  données locales, met à jour global_bst immédiatement, puis passe au suivant.
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, f1_score

from task import (
    get_client_partition,
    XGBOOST_PARAMS,
    NUM_LOCAL_ROUNDS,
    NUM_CLIENTS,
)


class XGBoostClient:
    """
    Client fédéré XGBoost — entraînement local multi:softprob.

    Chaque client reçoit une partition de sujets (round-robin SubjectID)
    avec les 4 types d'échantillons -> les 4 classes présentes.

    À chaque round :
    - fit()      : entraîne un XGBoost local, retourne ses probabilités
                   prédites sur le val set + le modèle sérialisé
    - evaluate() : évalue le modèle local sur le val set

    Parameters
    ----------
    client_id        : int
    train_dmatrix    : partition locale (train + val combinés)
    val_dmatrix      : val set commun pour évaluation
    y_val            : labels val encodés
    params           : hyperparamètres XGBoost
    num_local_rounds : arbres entraînés par round FL
    """

    def __init__(self,
                 client_id: int,
                 train_dmatrix: xgb.DMatrix,
                 val_dmatrix: xgb.DMatrix,
                 y_val: np.ndarray,
                 params: Dict,
                 num_local_rounds: int = NUM_LOCAL_ROUNDS):

        self.client_id        = client_id
        self.train_dmatrix    = train_dmatrix
        self.val_dmatrix      = val_dmatrix
        self.y_val            = y_val
        self.params           = params
        self.num_local_rounds = num_local_rounds
        self.n_classes        = params["num_class"]
        self.n_train          = train_dmatrix.num_row()
        self.bst: Optional[xgb.Booster] = None
        self.last_val_loss: float = float("inf")

    def fit(self, current_round: int, global_bst: Optional[xgb.Booster] = None) -> Tuple[np.ndarray, int, Dict]:
        """
        Entraîne le modèle local XGBoost.

        Round 1 : xgb.train() from scratch.
        Round > 1 : xgb.train() avec xgb_model=global_bst pour
                    continuer le boosting depuis le meilleur modèle global.

        Returns
        -------
        val_proba   : np.ndarray (n_val, n_classes) — probabilités sur val
        n_samples   : int — taille du dataset local (pour pondération)
        metrics     : dict — log_loss, accuracy, f1_macro locaux
        """
        if global_bst is None:
            self.bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round = self.num_local_rounds,
                verbose_eval    = False,
            )
        else:
            self.bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round = self.num_local_rounds,
                xgb_model       = global_bst,
                verbose_eval    = False,
            )

        # Probabilités sur le val set pour le soft voting
        val_proba = self.bst.predict(self.val_dmatrix).reshape(-1, self.n_classes)
        metrics   = self._compute_metrics(val_proba)
        self.last_val_loss = metrics["log_loss"]

        print(f"  Client {self.client_id} "
              f"| R{current_round:>2} "
              f"| {self.n_train:>4} samples "
              f"| LogLoss={metrics['log_loss']:.4f} "
              f"| Acc={metrics['accuracy']:.4f} "
              f"| Arbres={self.bst.num_boosted_rounds()}")

        return val_proba, self.n_train, metrics


    def _compute_metrics(self, proba: np.ndarray) -> Dict:
        pred = proba.argmax(axis=1)
        return {
            "log_loss": float(log_loss(self.y_val, proba, labels=list(range(self.n_classes)))),
            "accuracy": float(accuracy_score(self.y_val, pred)),
            "f1_macro": float(f1_score(self.y_val, pred, average="macro", zero_division=0)),
        }


def create_clients(train_df: pd.DataFrame,
                   val_df: pd.DataFrame,
                   feature_cols: List[str],
                   le,
                   params: Dict = None,
                   num_local_rounds: int = NUM_LOCAL_ROUNDS,
                   n_clients: int = NUM_CLIENTS) -> List[XGBoostClient]:
    """
    Crée n_clients XGBoostClient avec partitionnement round-robin par SubjectID.

    Chaque client utilise train + val pour l'entraînement local.
    Le val set original est conservé pour l'évaluation du modèle global.
    """
    params = params or XGBOOST_PARAMS

    X_val    = val_df[feature_cols].values
    y_val    = le.transform(val_df["SampleType"].values)
    val_dmat = xgb.DMatrix(X_val, label=y_val)

    full_df  = pd.concat([train_df, val_df], ignore_index=True)
    subjects = sorted(full_df["SubjectID"].unique())

    clients = []
    print(f"Création des clients ({n_clients} — train+val, round-robin SubjectID) :")
    for client_id in range(n_clients):
        train_dmat, y_loc = get_client_partition(
            client_id, train_df, val_df, feature_cols, le, n_clients
        )
        n_classes_local = len(np.unique(y_loc))
        client_subjects = [s for j, s in enumerate(subjects) if j % n_clients == client_id]

        clients.append(XGBoostClient(
            client_id        = client_id,
            train_dmatrix    = train_dmat,
            val_dmatrix      = val_dmat,
            y_val            = y_val,
            params           = params,
            num_local_rounds = num_local_rounds,
        ))
        print(f"  Client {client_id} : {len(client_subjects):>2} sujets "
              f"| {train_dmat.num_row():>4} échantillons "
              f"| {n_classes_local} classes")

    return clients