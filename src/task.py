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

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"

NUM_CLIENTS = 5   # nombre de clients fédérés

XGBOOST_PARAMS: Dict = {
    "objective":        "multi:softprob",
    "num_class":        4,
    "eval_metric":      "mlogloss",
    "eta":              0.05,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "tree_method":      "hist",
    "nthread":          8,
    "seed":             42,
}

NUM_LOCAL_ROUNDS = 1  # séquentiel : 5 clients × 1 arbre × N_SEMANTIC_ROUNDS


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


def serialize_model(bst: xgb.Booster) -> bytes:
    """Sérialise un Booster XGBoost en bytes (format JSON — UTF-8 valide).

    UBJ est exclu : XGBoost 3.x détecte mal le format en mémoire (UBJ et JSON
    commencent tous deux par `{`) et lève un UnicodeDecodeError sur les bytes
    binaires intégrés dans le message d'erreur C++.
    JSON est ~3× plus grand mais garanti décodable et auto-détectable.
    """
    return bst.save_raw("json")


def deserialize_model(data: bytes) -> xgb.Booster:
    """Désérialise un Booster XGBoost depuis des bytes JSON."""
    bst = xgb.Booster()
    bst.load_model(bytearray(data))
    return bst


