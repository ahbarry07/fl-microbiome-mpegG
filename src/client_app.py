"""
client_app.py
=============

Logique client XGBoost pour l'apprentissage fédéré multiclasse.

Agrégation : Soft Voting pondéré par n_samples.
  Le serveur ne manipule pas les arbres XGBoost — il agrège directement
  les probabilités prédites sur le val set. Cela évite les problèmes
  de compatibilité de FedXgbBagging avec multi:softprob / num_class > 2.

  p_global(x) = Σ_i (n_i / N) * p_i(x)

  Le modèle global est le modèle du client avec le plus de données
  (le mieux représentatif), amélioré par les probabilités des autres.
  En pratique : chaque client entraîne son XGBoost local, envoie ses
  probabilités sur le val set, le serveur calcule le vote pondéré.
"""

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import log_loss, accuracy_score, f1_score

from task import (
    load_data,
    get_client_partition,
    evaluate_global_model,
    XGBOOST_PARAMS,
    NUM_LOCAL_ROUNDS,
    NUM_CLIENTS,
)


class XGBoostClient:
    """
    Client fédéré XGBoost — entraînement local multi:softprob.

    Chaque client reçoit une partition de sujets (round-robin SubjectID)
    avec les 4 types d'échantillons → les 4 classes présentes.

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

    def evaluate(self) -> Dict:
        """Évalue le modèle local courant sur le val set."""
        if self.bst is None:
            return {"log_loss": 999.0, "accuracy": 0.0, "f1_macro": 0.0}
        val_proba = self.bst.predict(self.val_dmatrix).reshape(-1, self.n_classes)
        return self._compute_metrics(val_proba)

    def _compute_metrics(self, proba: np.ndarray) -> Dict:
        pred = proba.argmax(axis=1)
        return {
            "log_loss": float(log_loss(self.y_val, proba,
                                       labels=list(range(self.n_classes)))),
            "accuracy": float(accuracy_score(self.y_val, pred)),
            "f1_macro": float(f1_score(self.y_val, pred,
                                       average="macro", zero_division=0)),
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
    for i in range(n_clients):
        train_dmat, y_loc = get_client_partition(
            i, train_df, val_df, feature_cols, le, n_clients
        )
        n_classes_local = len(np.unique(y_loc))
        client_subjects = [s for j, s in enumerate(subjects) if j % n_clients == i]

        clients.append(XGBoostClient(
            client_id        = i,
            train_dmatrix    = train_dmat,
            val_dmatrix      = val_dmat,
            y_val            = y_val,
            params           = params,
            num_local_rounds = num_local_rounds,
        ))
        print(f"  Client {i} : {len(client_subjects):>2} sujets "
              f"| {train_dmat.num_row():>4} échantillons "
              f"| {n_classes_local} classes")

    return clients