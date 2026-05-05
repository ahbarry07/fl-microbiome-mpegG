"""
server_app.py
=============

Agrégation fédérée par soft voting pondéré et évaluation centralisée.

Contenu :
- soft_voting()    : agrège les probabilités des clients en un vecteur
                     global pondéré par n_samples de chaque client
- select_best_bst(): retourne le Booster du client le plus représentatif
                     (le plus de données) comme modèle global pour le
                     round suivant
- evaluate_global(): calcule log_loss, accuracy, f1_macro sur le val set
- run_federated()  : boucle principale de simulation fédérée
- MODELS_PATH      : dossier de sauvegarde du meilleur modèle global
- RESULTS_PATH     : dossier de sauvegarde de l'historique CSV
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, accuracy_score, f1_score

from task import load_data, PROCESSED_PATH

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "metrics"
MODELS_PATH  = Path(__file__).resolve().parent.parent / "models"  / "federated"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Val set chargé une seule fois pour l'évaluation serveur
_train_df, _val_df, _test_df, _feature_cols, _le = load_data()
_X_val       = _val_df[_feature_cols].values
_y_val       = _le.transform(_val_df["SampleType"].values)
_val_dmatrix = xgb.DMatrix(_X_val, label=_y_val)
_n_classes   = len(_le.classes_)

history: List[Dict] = []


def soft_voting(probas: List[np.ndarray],
                weights: List[int]) -> np.ndarray:
    """
    Agrège les probabilités des clients par soft voting pondéré.

    p_global(x) = Σ_i (n_i / N) * p_i(x)

    Chaque client contribue proportionnellement à la taille de son
    dataset local. Cela donne plus de poids aux clients qui ont plus
    de données → convergence vers le modèle centralisé.

    Parameters
    ----------
    probas  : list of np.ndarray (n_val, n_classes) — probas de chaque client
    weights : list of int — n_samples de chaque client

    Returns
    -------
    np.ndarray (n_val, n_classes) — probabilités agrégées
    """
    total = sum(weights)
    agg   = np.zeros_like(probas[0])
    for proba, w in zip(probas, weights):
        agg += (w / total) * proba
    return agg


def select_best_bst(clients) -> Optional[xgb.Booster]:
    """
    Retourne le Booster du client ayant le meilleur val log_loss.

    Choisir le modèle qui généralise le mieux sur le val set (et non
    simplement le plus gros client) donne un meilleur point de départ
    pour le round suivant → convergence plus rapide vers le centralisé.
    """
    fitted = [(c.last_val_loss, c.bst) for c in clients if c.bst is not None]
    if not fitted:
        return None
    _, best_bst = min(fitted, key=lambda x: x[0])
    return best_bst


def evaluate_global(agg_proba: np.ndarray,
                    server_round: int,
                    n_trees: int) -> Dict:
    """
    Évalue les probabilités agrégées sur le val set et sauvegarde.

    Parameters
    ----------
    agg_proba    : np.ndarray (n_val, n_classes)
    server_round : int
    n_trees      : int — nombre d'arbres du modèle global

    Returns
    -------
    dict : round, log_loss, accuracy, f1_macro, n_trees
    """
    y_pred = agg_proba.argmax(axis=1)
    ll     = float(log_loss(_y_val, agg_proba,
                            labels=list(range(_n_classes))))
    acc    = float(accuracy_score(_y_val, y_pred))
    f1     = float(f1_score(_y_val, y_pred, average="macro", zero_division=0))

    result = {"round": server_round, "log_loss": ll,
              "accuracy": acc, "f1_macro": f1, "n_trees": n_trees}
    history.append(result)

    pd.DataFrame(history).to_csv(
        RESULTS_PATH / "federated_metrics.csv", index=False
    )
    print(f"  → Global | LogLoss={ll:.4f} | Acc={acc:.4f} "
          f"| F1={f1:.4f} | Arbres={n_trees}")

    return result


def run_federated(clients,
                  n_rounds: int = 20) -> Tuple[xgb.Booster, List[Dict]]:
    """
    Boucle de simulation fédérée avec soft voting pondéré.

    À chaque round :
    1. Chaque client entraîne son XGBoost local (depuis le meilleur
       modèle global du round précédent)
    2. Soft voting pondéré sur les probabilités val → p_global
    3. Évaluation de p_global sur le val set
    4. Le meilleur modèle global est sauvegardé

    Parameters
    ----------
    clients  : list of XGBoostClient
    n_rounds : int

    Returns
    -------
    best_bst : meilleur Booster global
    history  : métriques par round
    """
    best_log_loss = float("inf")
    best_bst      = None
    global_bst    = None   # None au round 1

    print(f"\n{'='*60}")
    print(f"SIMULATION FÉDÉRÉE — Entraînement Séquentiel + Soft Voting")
    print(f"  Clients    : {len(clients)}")
    print(f"  Rounds     : {n_rounds}")
    print(f"  Stratégie  : séquentiel (ordre mélangé chaque round)")
    print(f"  Agrégation : Soft Voting pondéré par n_samples")
    print(f"{'='*60}\n")

    for round_idx in range(1, n_rounds + 1):
        print(f"── Round {round_idx}/{n_rounds} ──────────────────────────")

        # Ordre aléatoire reproductible — évite qu'un client domine toujours
        rng   = np.random.default_rng(42 + round_idx)
        order = rng.permutation(len(clients)).tolist()

        # Entraînement séquentiel : chaque client enrichit global_bst immédiatement.
        # Après tous les clients, global_bst a vu TOUTES les données du round.
        for i in order:
            clients[i].fit(round_idx, global_bst)
            global_bst = clients[i].bst

        # Soft voting sur les modèles locaux finaux du round
        probas  = [c.bst.predict(_val_dmatrix).reshape(-1, _n_classes)
                   for c in clients if c.bst is not None]
        weights = [c.n_train for c in clients if c.bst is not None]
        agg_proba = soft_voting(probas, weights)

        n_trees = global_bst.num_boosted_rounds() if global_bst else 0
        result  = evaluate_global(agg_proba, round_idx, n_trees)

        if result["log_loss"] < best_log_loss:
            best_log_loss = result["log_loss"]
            best_bst      = global_bst
            if best_bst:
                best_bst.save_model(str(MODELS_PATH / "best_global_model.ubj"))
            print(f"  🏆 Nouveau meilleur LogLoss : {best_log_loss:.4f}")

    print(f"\n{'='*60}")
    print(f"🏆 Meilleur LogLoss fédéré : {best_log_loss:.4f}")
    print(f"{'='*60}")

    return best_bst, history