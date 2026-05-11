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
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, f1_score


PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

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

history:        List[Dict] = []
history_fedavg: List[Dict] = []


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



def soft_voting(probas: List[np.ndarray], weights: List[int]) -> np.ndarray:
    """Agrège les probabilités par soft voting pondéré : p = Σ (n_i/N) * p_i."""
    total = sum(weights)
    agg   = np.zeros_like(probas[0])
    for proba, w in zip(probas, weights):
        agg += (w / total) * proba
    return agg


def evaluate_global(agg_proba: np.ndarray, sem_round: int, n_trees: int, filename: str = "federated_metrics.csv") -> Dict:
    """Évalue les probabilités agrégées sur le val set et sauvegarde."""
    y_pred = agg_proba.argmax(axis=1)
    ll  = float(log_loss(_y_val, agg_proba, labels=list(range(_n_classes))))
    acc = float(accuracy_score(_y_val, y_pred))
    f1  = float(f1_score(_y_val, y_pred, average="macro", zero_division=0))

    result = {
        "round": sem_round, "n_trees": n_trees,
        "log_loss": ll, "accuracy": acc, "f1_macro": f1, 
    }
    history.append(result)
    pd.DataFrame(history).to_csv(RESULTS_PATH / filename, index=False)
    print(f"  -> Global | LogLoss={ll:.4f} | Acc={acc:.4f} | F1={f1:.4f} | Arbres={n_trees}")
    return result



def merge_xgb_trees(start_bst: Optional[xgb.Booster], client_models: List[Tuple[xgb.Booster, int]],) -> Optional[xgb.Booster]:
    """
    Fusionne les nouveaux arbres de chaque client dans le modèle global.

    Tous les clients ont entraîné depuis le même start_bst (parallélisme simulé).
    Pour chaque client, on extrait les arbres ajoutés au-delà du modèle de départ
    et on les concatène dans le modèle fusionné.

    Le tree_info (indice de classe par arbre) est conservé tel quel,
    ce qui garantit des prédictions cohérentes pour multi:softprob.
    """
    if not client_models:
        return start_bst

    if start_bst is None:
        # Round 1 : clients ont entraîné depuis zéro
        n_start_trees = 0
        base_raw      = client_models[0][0].save_raw("json") # Juste pour obtenir la structure de base du modèle (paramètres, tree_info) sans les arbres
        merged_data   = json.loads(base_raw)
        m = merged_data["learner"]["gradient_booster"]["model"] 
        # Lire le step AVANT de vider les arbres
        orig_indptr = m.get("iteration_indptr", [])
        m["trees"]     = []
        m["tree_info"] = []
        m["gbtree_model_param"]["num_trees"] = "0"
    else:
        start_raw   = start_bst.save_raw("json")
        merged_data = json.loads(start_raw)
        m = merged_data["learner"]["gradient_booster"]["model"]
        n_start_trees = int(m["gbtree_model_param"]["num_trees"])
        orig_indptr = m.get("iteration_indptr", []) 

    # Pas par itération = num_class * num_parallel_tree (4 pour softprob 4 classes)
    step = (orig_indptr[1] - orig_indptr[0]) if len(orig_indptr) >= 2 else _n_classes

    all_new_trees: list = []
    all_new_info:  list = []
    for client_bst, _ in client_models:
        c = json.loads(client_bst.save_raw("json"))["learner"]["gradient_booster"]["model"]  
        all_new_trees.extend(c["trees"][n_start_trees:])
        all_new_info.extend(c["tree_info"][n_start_trees:])

    if not all_new_trees: 
        return start_bst

    m["trees"].extend(all_new_trees)
    m["tree_info"].extend(all_new_info)

    # Re-indexer les arbres fusionnés : XGBoost vérifie que les IDs sont consécutifs à partir de 0
    for i, tree in enumerate(m["trees"]):
        tree["id"] = i
    total_trees = len(m["trees"])
    m["gbtree_model_param"]["num_trees"] = str(total_trees)

    # Recalculer iteration_indptr pour refléter les nouveaux arbres fusionnés
    m["iteration_indptr"] = list(range(0, total_trees + 1, step))

    merged_bst = xgb.Booster()
    merged_bst.load_model(bytearray(json.dumps(merged_data).encode()))
    return merged_bst




# Val set chargé une seule fois pour l'évaluation serveur
_, _val_df, _, _feature_cols, _le = load_data()
_X_val       = _val_df[_feature_cols].values
_y_val       = _le.transform(_val_df["SampleType"].values)
_val_dmatrix = xgb.DMatrix(_X_val, label=_y_val)
_n_classes   = len(_le.classes_)