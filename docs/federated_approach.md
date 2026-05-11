# Approche d'Apprentissage Fédéré — Classification du Microbiome

## 1. Le problème de départ

L'objectif est de prédire d'où vient un échantillon biologique humain parmi 4 sites :
**Bouche**, **Nez**, **Peau** ou **Intestin (Stool)**.

Pour cela, on dispose de données métagénomiques — c'est-à-dire des informations extraites de
séquences d'ADN microbien — ainsi que des caractéristiques statistiques des fichiers de
séquençage (longueur des lectures, composition en bases, présence de bactéries spécifiques, etc.).

---

## 2. Qu'est-ce que l'apprentissage fédéré ?

Dans un scénario classique, toutes les données sont centralisées sur un seul serveur pour
entraîner un modèle. C'est le **modèle centralisé**.

L'**apprentissage fédéré** simule une situation où les données sont **réparties entre plusieurs
institutions** (hôpitaux, laboratoires, etc.) qui ne peuvent ou ne veulent pas partager leurs
données brutes pour des raisons de confidentialité ou de réglementation.

Dans ce projet, on simule **5 clients** (institutions fictives), chacun possédant une partie
des données. Le but est de faire collaborer ces 5 clients pour construire un bon modèle commun,
sans qu'aucun client n'ait accès aux données des autres.

---

## 3. Comment les données sont-elles réparties ?

Les données sont découpées par **identifiant de sujet (SubjectID)**.

Il y a **66 sujets au total** (54 train + 12 validation), tous utilisés pour
l'entraînement local des clients. Chaque sujet a fourni des échantillons des 4 sites corporels.
On distribue les sujets entre les 5 clients selon un principe de **round-robin** :

- Client 0 → sujets d'indices 0, 5, 10, 15, ... (14 sujets)
- Client 1 → sujets d'indices 1, 6, 11, 16, ... (13 sujets)
- etc.

Ce découpage garantit que **chaque client a des échantillons des 4 classes** (Bouche, Nez,
Peau, Intestin), ce qui est indispensable pour que chaque client puisse entraîner un modèle
capable de distinguer les 4 sites.

Chaque client utilise train + val pour l'entraînement local.
L'évaluation honnête se fait sur le jeu de test Zindi, complètement indépendant.

| Client | Sujets | Échantillons |
|--------|--------|--------------|
| 0      | 14     | 561          |
| 1      | 13     | 551          |
| 2      | 13     | 353          |
| 3      | 13     | 559          |
| 4      | 13     | 877          |

---

## 4. Le modèle utilisé : XGBoost

Chaque client entraîne un **XGBoost** avec l'objectif `multi:softprob` — un algorithme
de gradient boosting qui produit des **probabilités** pour chaque classe plutôt qu'une
simple prédiction binaire.

XGBoost fonctionne en construisant des **arbres de décision successifs**, chacun corrigeant
les erreurs du précédent. C'est l'un des algorithmes les plus performants pour les données
tabulaires (structurées en lignes et colonnes).

**Hyperparamètres utilisés :**

| Paramètre          | Valeur | Rôle |
|--------------------|--------|------|
| `eta`              | 0.05   | Taux d'apprentissage — petits pas pour mieux généraliser |
| `max_depth`        | 6      | Profondeur maximale des arbres |
| `subsample`        | 0.8    | 80 % des données utilisées par arbre (réduit le surapprentissage) |
| `colsample_bytree` | 0.8    | 80 % des features utilisées par arbre |
| `min_child_weight` | 1      | Taille minimale d'un groupe feuille |
| `num_class`        | 4      | Nombre de classes à prédire |

---

## 5. Les deux stratégies fédérées comparées

Deux stratégies d'agrégation sont implémentées et comparables via `server_app.py`.
La différence fondamentale porte sur **à partir de quel résiduel chaque arbre est entraîné**.

### Stratégie 1 — Entraînement séquentiel (Sequential XGBoost)

À chaque **round sémantique**, les clients s'enchaînent **l'un après l'autre**.
Chaque client reçoit le modèle **mis à jour par le client précédent** :

```
Round sémantique k (ordre mélangé : ex. [C3, C0, C4, C1, C2]) :

  Départ : M (modèle global courant)
  C3 reçoit M    → entraîne 1 arbre sur résiduel(M,    données C3) → M'
  C0 reçoit M'   → entraîne 1 arbre sur résiduel(M',   données C0) → M''
  C4 reçoit M''  → entraîne 1 arbre sur résiduel(M'',  données C4) → M'''
  C1 reçoit M''' → entraîne 1 arbre sur résiduel(M''', données C1) → M''''
  C2 reçoit M''''→ entraîne 1 arbre sur résiduel(M'''',données C2) → M_new
```

Chaque arbre corrige le **résiduel cumulatif** — il intègre les corrections de tous les
clients précédents dans le même round. C'est du gradient boosting distribué pur.

L'ordre est **mélangé aléatoirement à chaque round** (seed = 42 + numéro_round,
reproductible) pour éviter qu'un client ne domine systématiquement.

**Résultat** : après 20 rounds × 5 clients × 1 arbre = **100 arbres**, le modèle global
a appris séquentiellement de l'ensemble des 2 901 échantillons — équivalent à un XGBoost
centralisé, sans jamais partager les données brutes.

### Stratégie 2 — FedAvg parallèle avec Tree Merging

Tous les clients reçoivent le **même modèle de départ** au début du round sémantique
et entraînent indépendamment. Leurs nouveaux arbres sont ensuite **fusionnés** :

```
Round sémantique k (même ordre, mais tous partent de M) :

  Snapshot : M (modèle global courant)
  C3 reçoit M → entraîne 1 arbre sur résiduel(M, données C3) → arbre_C3
  C0 reçoit M → entraîne 1 arbre sur résiduel(M, données C0) → arbre_C0
  C4 reçoit M → entraîne 1 arbre sur résiduel(M, données C4) → arbre_C4
  C1 reçoit M → entraîne 1 arbre sur résiduel(M, données C1) → arbre_C1
  C2 reçoit M → entraîne 1 arbre sur résiduel(M, données C2) → arbre_C2

  Fusion : M_new = M + arbre_C3 + arbre_C0 + arbre_C4 + arbre_C1 + arbre_C2
```

Tous les arbres du round corrigent **le même résiduel de départ M** — ils sont parallèles,
pas séquentiels. La fusion consiste à **concaténer** (pas moyenner) les nouveaux arbres
dans le modèle global, en conservant le `tree_info` (indice de classe par arbre).

> **Pourquoi Tree Merging et pas FedAvg standard ?**
> Le FedAvg de Flower fait une moyenne pondérée de paramètres numpy, ce qui n'a de sens
> que pour les réseaux de neurones. XGBoost n'a pas de paramètres numériques à moyenner —
> ses "paramètres" sont des arbres de décision. La fusion d'arbres est l'adaptation naturelle
> de FedAvg pour les modèles à base d'arbres.

### Comparaison des deux approches

| Critère | Séquentiel | FedAvg parallèle |
|---------|-----------|-----------------|
| Résiduel de chaque arbre | cumulatif (intègre les clients précédents) | identique pour tous (même M de départ) |
| Qualité du gradient boosting | optimale | redondante par round |
| Influence de l'ordre des clients | oui | non |
| Équivalent centralisé | ≈ XGBoost normal | ≈ XGBoost avec arbres ajoutés en batch |
| Agrégation | mise à jour immédiate | concaténation en fin de round |

---

## 6. Implémentation avec Flower

La simulation est orchestrée par le framework **Flower (flwr)**, qui gère le cycle de vie
des rounds, la communication serveur-clients et la sérialisation des modèles.

### Architecture générale

```
Flower Server (SequentialXGBoostStrategy  ou  ParallelXGBoostStrategy)
        │
        │  configure_fit()  →  FitIns(modèle_global, {target_partition, current_round})
        ▼
Flower Client (XGBoostFlowerClient)    ← num_supernodes=1, stateless
        │
        │  fit()  →  FitRes(modèle_entraîné, métriques_locales)
        ▼
Flower Server
        │
        │  aggregate_fit()  →  met à jour global_bst
        │                      → à la fin d'un round sémantique : _end_of_semantic_round()
        ▼
```

### Correspondance rounds Flower ↔ rounds sémantiques

| Concept             | Valeur | Description |
|---------------------|--------|-------------|
| `N_SEMANTIC_ROUNDS` | 20     | Nombre de rounds sémantiques (boucles complètes sur tous les clients) |
| `NUM_CLIENTS`       | 5      | Nombre de clients / partitions |
| `NUM_LOCAL_ROUNDS`  | 1      | Arbres ajoutés par client par round Flower |
| Rounds Flower totaux | 100   | `N_SEMANTIC_ROUNDS × NUM_CLIENTS` |

Chaque **round Flower** correspond à **1 étape** d'un round sémantique :
un seul client entraîne 1 arbre supplémentaire.

```
Round Flower 1  →  sem_round=1, étape 1/5, partition=?
Round Flower 2  →  sem_round=1, étape 2/5, partition=?
...
Round Flower 5  →  sem_round=1, étape 5/5  → fin du round sémantique → évaluation
Round Flower 6  →  sem_round=2, étape 1/5
...
Round Flower 100 → sem_round=20, étape 5/5 → fin du round sémantique → évaluation
```

La table de correspondance `round_info` est précalculée à l'initialisation (identique
dans les deux stratégies) :

```python
# Précalcul : flower_round -> (sem_round, étape, target_partition)
for sem in range(1, n_semantic_rounds + 1):
    rng   = np.random.default_rng(42 + sem)
    order = rng.permutation(n_clients).tolist()
    for step, partition in enumerate(order):
        fl_round = (sem - 1) * n_clients + step + 1
        self.round_info[fl_round] = (sem, step, partition)
```

### Client stateless (`num_supernodes=1`)

La simulation utilise **un seul supernode Flower** (`num_supernodes=1`).
Le client `XGBoostFlowerClient` est **sans état persistent sur la partition** :
il reçoit `target_partition` dans `FitIns.config` à chaque round et charge les données
correspondantes à la demande (avec cache local `_cache`).

Cela évite d'instancier 5 processus séparés tout en permettant d'orchestrer 5 partitions.
Le parallélisme de la stratégie FedAvg est donc **simulé** : `configure_fit` envoie
le snapshot `_round_start_bst` à chaque client du round (au lieu du modèle progressif),
ce qui reproduit le comportement d'un entraînement parallèle réel.

### Méthodes de `SequentialXGBoostStrategy`

| Méthode | Rôle |
|---------|------|
| `initialize_parameters` | Retourne `None` — les clients partent de zéro au round 1 |
| `configure_fit` | Sélectionne 1 client, envoie `global_bst` courant + `target_partition` |
| `aggregate_fit` | Désérialise le modèle, met à jour `global_bst` immédiatement, accumule `_round_models` ; déclenche `_end_of_semantic_round` en fin de round |
| `configure_evaluate` | Retourne `[]` — évaluation faite côté serveur |
| `aggregate_evaluate` | Retourne `None` — non utilisé |
| `evaluate` | Retourne `None` — non utilisé |

### Méthodes de `ParallelXGBoostStrategy`

| Méthode | Rôle |
|---------|------|
| `initialize_parameters` | Retourne `None` |
| `configure_fit` | Prend un snapshot `_round_start_bst` à l'étape 0, envoie CE snapshot à tous les clients du round (parallélisme simulé) |
| `aggregate_fit` | Accumule les modèles clients dans `_pending` ; déclenche `_end_of_semantic_round` en fin de round |
| `configure_evaluate` | Retourne `[]` |
| `aggregate_evaluate` | Retourne `None` |
| `evaluate` | Retourne `None` |

### Évaluation serveur (`_end_of_semantic_round`)

À la fin de chaque round sémantique (quand les 5 clients ont entraîné), le serveur :

**Séquentiel :**
1. Récupère les modèles intermédiaires dans `_round_models[sem_round]`
2. Fait un **soft voting pondéré** sur leurs prédictions (évaluation uniquement)
3. Calcule `log_loss`, `accuracy`, `f1_macro` sur le val set global
4. Sauvegarde les métriques dans `results/metrics/federated_metrics.csv`
5. Sauvegarde le meilleur modèle dans `models/federated/best_global_model.cubj`

**FedAvg parallèle :**
1. Appelle `merge_xgb_trees(_round_start_bst, _pending[sem_round])` → `global_bst` fusionné
2. Fait un **soft voting pondéré** sur les modèles individuels des clients (évaluation uniquement)
3. Calcule `log_loss`, `accuracy`, `f1_macro` sur le val set global
4. Sauvegarde les métriques dans `results/metrics/fedavg_metrics.csv`
5. Sauvegarde le meilleur modèle dans `models/federated/best_global_model_fedavg.cubj`

```python
# Déclenchement dans aggregate_fit() à chaque fin de round sémantique
if step == self.n_clients - 1:
    self._end_of_semantic_round(sem_round)
```

### Tree Merging — détail technique

La fusion des arbres XGBoost manipule directement le format JSON interne du modèle :

```python
def merge_xgb_trees(start_bst, client_models):
    # Extraire les arbres ajoutés par chaque client au-delà du snapshot de départ
    for client_bst, _ in client_models:
        c = json.loads(client_bst.save_raw("json"))["learner"]["gradient_booster"]["model"]
        all_new_trees.extend(c["trees"][n_start_trees:])
        all_new_info.extend(c["tree_info"][n_start_trees:])

    # Concaténer dans le modèle fusionné
    m["trees"].extend(all_new_trees)
    m["tree_info"].extend(all_new_info)
    # Renommer les ids (séquentiels) et reconstruire iteration_indptr
    for i, tree in enumerate(m["trees"]): tree["id"] = i
    m["iteration_indptr"] = list(range(0, len(m["trees"]) + 1, step))
```

Deux points critiques :
- Les ids de chaque arbre doivent être **renumérotés séquentiellement** après fusion
  (chaque client retourne des arbres avec les mêmes ids → doublons → segfault XGBoost).
- `iteration_indptr` doit être **reconstruit** : XGBoost vérifie que
  `indptr[-1] == num_trees` ; la concaténation augmente `num_trees` sans toucher `indptr`.

### Boosting incrémental (côté client)

Le client reçoit le modèle global courant et l'utilise comme point de départ :

```python
# Round 1 : entraînement from scratch
bst = xgb.train(XGBOOST_PARAMS, train_dmat, num_boost_round=1)

# Rounds suivants : ajout d'un arbre au modèle global
bst = xgb.train(XGBOOST_PARAMS, train_dmat, num_boost_round=1, xgb_model=global_bst)
```

Le modèle est sérialisé en bytes (`serialize_model` / `deserialize_model`) pour être
transmis via les paramètres Flower (`ndarrays_to_parameters`).

---

## 7. L'évaluation par Soft Voting

Le **soft voting pondéré** est utilisé **uniquement pour l'évaluation** à la fin de chaque
round sémantique — il ne sert pas à construire le modèle global.

À la fin du round, on dispose des modèles intermédiaires de chaque client. Plutôt que
d'évaluer un seul de ces modèles (ce qui biaiserait vers le dernier client dans l'ordre),
on moyenne leurs prédictions en probabilité :

```
p_global(x) = Σ_i  (n_i / N) × p_i(x)

  où :
    p_i(x)   = probabilités prédites par le modèle du client i
    n_i      = nombre d'échantillons du client i
    N        = nombre total d'échantillons (somme de tous les clients)
```

Un client avec plus de données a plus de poids dans la métrique d'évaluation.
Cette agrégation n'implique aucun échange de données brutes — seulement des vecteurs
de probabilités calculés sur le val set global (chargé une seule fois côté serveur).

---

## 8. Calibration en température

Après l'entraînement fédéré, une étape de **calibration en température** est appliquée
pour améliorer la qualité des probabilités prédites.

Le log loss (métrique de Zindi) pénalise les modèles mal calibrés — c'est-à-dire ceux qui
sont trop confiants sur de mauvaises prédictions ou pas assez confiants sur les bonnes.

La calibration en température consiste à trouver un scalaire **T** qui ajuste la "confiance"
des probabilités :

```
p_calibrée(x) = softmax( log(p(x)) / T )

  - T < 1 → probabilités plus tranchées (le modèle est plus affirmatif)
  - T > 1 → probabilités plus douces (le modèle est plus prudent)
```

T est optimisé automatiquement sur le jeu de validation pour minimiser le log loss,
puis appliqué aux prédictions du jeu de test.

---

## 9. Résultats obtenus

| Modèle                                              | Log Loss (val) | Accuracy (val) | F1-macro (val) |
|-----------------------------------------------------|----------------|----------------|----------------|
| Centralisé (XGBoost)                                | 0.0368         | 0.9863         | 0.9867         |
| Fédéré v1 (20 rounds, parallèle, eta=0.1)           | 0.2107         | 0.9840         | 0.9849         |
| Fédéré v2 (100 rounds, parallèle, eta=0.05)         | 0.0382         | 0.9886         | 0.9889         |
| **Fédéré v3 — Séquentiel (100 rounds, eta=0.05)**   | **0.0076**     | **0.9977**     | **0.9978**     |
| Fédéré v4 — FedAvg Tree Merging (100 rounds)        | en cours...    | en cours...    | en cours...    |

**Scores Zindi (jeu de test public/privé) :**

| Modèle       | Public       | Privé        |
|--------------|-------------|-------------|
| Centralisé   | 0.003741    | 0.024795    |
| Fédéré v2    | 0.043446    | 0.049010    |
| Fédéré v3    | en cours... | en cours... |
| Fédéré v4    | en cours... | en cours... |

---

## 10. Résumé visuel du pipeline

```
Données brutes (MPEG-G .mgb)
        │
        ▼
Extraction de features (Kraken2, k-mers, stats qualité)
        │
        ▼
Partitionnement round-robin par SubjectID → 5 partitions
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│         SIMULATION FLOWER (100 rounds Flower × 2 stratégies)     │
│                                                                  │
│  ── Stratégie A : Séquentielle ──────────────────────────────    │
│  Round Flower r :                                                │
│    configure_fit()  → envoie global_bst courant + partition      │
│    client.fit()     → xgb.train(global_bst, 1 arbre)            │
│    aggregate_fit()  → global_bst = modèle client (immédiat)     │
│  Fin round sém.    → soft voting (éval) + save best_model.cubj  │
│                                                                  │
│  ── Stratégie B : FedAvg Parallèle ──────────────────────────    │
│  Round Flower r :                                                │
│    configure_fit()  → envoie _round_start_bst (snapshot) + part.│
│    client.fit()     → xgb.train(snapshot, 1 arbre)              │
│    aggregate_fit()  → accumule modèles dans _pending            │
│  Fin round sém.    → merge_xgb_trees() + soft voting (éval)     │
│                      + save best_model_fedavg.cubj              │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
Calibration en température (T optimisé sur val)
        │
        ▼
Prédictions test → soumission Zindi (CSV)
```

---

## 11. Fichiers du projet

```
src/
  task.py          — chargement des données, partitionnement round-robin,
                     hyperparamètres XGBoost, sérialisation/désérialisation modèle,
                     soft_voting(), evaluate_global(), merge_xgb_trees()
  client_app.py    — XGBoostFlowerClient : fit() (entraînement local incrémental),
                     evaluate() (stub, évaluation faite côté serveur)
  server_app.py    — SequentialXGBoostStrategy (séquentiel + soft voting)
                     ParallelXGBoostStrategy (FedAvg parallèle + tree merging)
                     app / parallel_app : points d'entrée ServerApp

notebook/
  05_federated_model.ipynb — simulation Flower (run_simulation × 2 stratégies),
                             visualisations comparatives, calibration, soumission Zindi

models/federated/
  best_global_model.cubj        — meilleur modèle séquentiel (log_loss minimal sur val)
  best_global_model_fedavg.cubj — meilleur modèle FedAvg (log_loss minimal sur val)

results/metrics/
  federated_metrics.csv  — log_loss, accuracy, F1-macro par round sémantique (séquentiel)
  fedavg_metrics.csv     — log_loss, accuracy, F1-macro par round sémantique (FedAvg)
```
