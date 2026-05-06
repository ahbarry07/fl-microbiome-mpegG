"""
task.py
=======

Utilitaires partagés entre client_app.py et server_app.py.

Contenu :
- load_data()          : charge train, val, test et construit le LabelEncoder
- get_client_partition(): partitionne train_df en N partitions équilibrées
                          par SubjectID — chaque partition contient les 4 classes
- get_test_dmatrix()   : DMatrix du test set
- evaluate_global_model(): log_loss, accuracy, f1_macro sur un val DMatrix

STRATÉGIE DE PARTITIONNEMENT
------------------------------
On partitionne par SubjectID.
Chaque partition reçoit un sous-ensemble de sujets avec leurs 4 types
d'échantillons (Mouth, Nasal, Skin, Stool). Cela garantit :
  - Chaque client connaît les 4 classes -> peut entraîner un XGBoost valide
  - Distribution proche de l'IID -> convergence vers le modèle centralisé

NUM_CLIENTS = 5 est un bon compromis pour :
  - suffisamment de clients pour simuler un contexte fédéré réaliste
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"

NUM_CLIENTS = 5   # nombre de clients fédérés

XGBOOST_PARAMS: Dict = {
    "objective":          "multi:softprob", 
    "num_class":          4,
    "eval_metric":        "mlogloss",
    "eta":                0.05, # alias learning_rate
    "max_depth":          6, # profondeur maximale des arbres
    "subsample":          0.8, # échantillonnage des données pour chaque arbre
    "colsample_bytree":   0.8, # échantillonnage des caractéristiques pour chaque arbre
    "min_child_weight":   1, # poids minimum d'une feuille pour éviter le surapprentissage
    "tree_method":        "hist", 
    "nthread":            8,
    "seed":               42,
}

NUM_LOCAL_ROUNDS = 1  # séquentiel : 5 clients × 1 arbre × 100 rounds = 500 arbres


def load_data(processed_path: Path = PROCESSED_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], LabelEncoder]:
    """
    Charge train, val et test depuis le dossier processed.

    Returns
    -------
    train_df, val_df, test_df, feature_cols, le
    """
    train_df     = pd.read_csv(processed_path / "train_engineered.csv")
    val_df       = pd.read_csv(processed_path / "val_engineered.csv")
    test_df      = pd.read_csv(processed_path / "test_engineered.csv")
    feature_cols = pd.read_csv(processed_path / "feature_cols.csv")["feature"].tolist()

    le = LabelEncoder()
    le.fit(train_df["SampleType"].values)

    return train_df, val_df, test_df, feature_cols, le


def get_client_partition(client_id: int,
                         train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         feature_cols: List[str],
                         le: LabelEncoder,
                         n_clients: int = NUM_CLIENTS
                         ) -> Tuple[xgb.DMatrix, np.ndarray]:
    """
    Retourne la partition DMatrix du client `client_id`.

    Les clients utilisent train + val pour l'entraînement local.

    Partitionnement par SubjectID en round-robin :
    Client i reçoit les sujets d'indices [i, i+n, i+2n, ...] depuis
    train + val combinés. Chaque sujet ayant des échantillons des 4 sites,
    chaque client dispose des 4 classes.

    Parameters
    ----------
    client_id   : int (0 à n_clients-1)
    train_df    : DataFrame train
    val_df      : DataFrame val
    feature_cols: liste des colonnes features
    le          : LabelEncoder global
    n_clients   : nombre total de clients

    Returns
    -------
    train_dmatrix : xgb.DMatrix
    y_local       : np.ndarray labels encodés
    """
    full_df  = pd.concat([train_df, val_df], ignore_index=True)
    subjects = sorted(full_df["SubjectID"].unique())

    # Round-robin : client i reçoit les sujets d'indices [i, i+n, i+2n, ...]
    client_subjects = [s for j, s in enumerate(subjects) if j % n_clients == client_id]

    local = full_df[full_df["SubjectID"].isin(client_subjects)] # partition locale du client
    X_loc = local[feature_cols].values
    y_loc = le.transform(local["SampleType"].values)

    return xgb.DMatrix(X_loc, label=y_loc), y_loc


def get_test_dmatrix(test_df: pd.DataFrame, feature_cols: List[str]) -> xgb.DMatrix:
    """Retourne le DMatrix du test set (sans labels)."""

    return xgb.DMatrix(test_df[feature_cols].values)


def evaluate_global_model(bst: xgb.Booster,
                           val_dmatrix: xgb.DMatrix,
                           y_val: np.ndarray,
                           n_classes: int = 4) -> Dict[str, float]:
    """
    Évalue le modèle global sur le val set.

    Returns
    -------
    dict : log_loss, accuracy, f1_macro
    """
    proba  = bst.predict(val_dmatrix).reshape(-1, n_classes)
    y_pred = proba.argmax(axis=1) # prédiction de la classe la plus probable

    return {
        "log_loss": float(log_loss(y_val, proba, labels=list(range(n_classes)))),
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
    }