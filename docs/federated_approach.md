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

## 5. La stratégie fédérée : entraînement séquentiel

### Le problème avec l'approche parallèle naïve

Une approche simple serait de faire entraîner tous les clients **en parallèle** depuis le
même modèle global, puis de sélectionner le meilleur modèle local comme nouveau modèle global.

Le problème : le "modèle global" ne verrait alors que les données d'**un seul client** par round.
Les 4 autres clients contribueraient uniquement à l'évaluation, pas à la structure du modèle.
Avec 500-877 échantillons par client au lieu des 2901 totaux, la qualité du modèle serait
nettement inférieure au centralisé.

### La solution : entraînement séquentiel avec ordre mélangé

À chaque **round sémantique**, les clients s'enchaînent **l'un après l'autre** pour enrichir
le modèle global :

```
Round sémantique k (ordre mélangé : ex. [Client 3, Client 0, Client 4, Client 1, Client 2]) :

  → Client 3 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 0 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 4 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 1 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 2 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Fin du round : le modèle global a vu TOUTES les données ce round
```

L'ordre est **mélangé aléatoirement à chaque round** (seed = 42 + numéro_round, reproductible)
pour éviter qu'un client ne domine systématiquement.

**Résultat** : après 20 rounds sémantiques × 5 clients × 1 arbre = **100 arbres**, le modèle
global a appris de l'ensemble des 2901 échantillons de manière distribuée — comme le centralisé,
mais sans que les données ne soient jamais partagées.

---

## 6. Implémentation avec Flower

La simulation est orchestrée par le framework **Flower (flwr)**, qui gère le cycle de vie
des rounds, la communication serveur-clients et la sérialisation des modèles.

### Architecture générale

```
Flower Server (SequentialXGBoostStrategy)
        │
        │  configure_fit()  →  FitIns(modèle_global, {target_partition, current_round})
        ▼
Flower Client (XGBoostFlowerClient)    ← num_supernodes=1, stateless
        │
        │  fit()  →  FitRes(modèle_entraîné, métriques_locales)
        ▼
Flower Server
        │
        │  aggregate_fit()  →  met à jour global_bst, accumule _round_models
        │                      → à la fin d'un round sémantique : _end_of_semantic_round()
        ▼
```

### Correspondance rounds Flower ↔ rounds sémantiques

| Concept           | Valeur | Description |
|-------------------|--------|-------------|
| `N_SEMANTIC_ROUNDS` | 20   | Nombre de rounds sémantiques (boucles complètes sur tous les clients) |
| `NUM_CLIENTS`       | 5    | Nombre de clients / partitions |
| `NUM_LOCAL_ROUNDS`  | 1    | Arbres ajoutés par client par round Flower |
| Rounds Flower totaux | 100 | `N_SEMANTIC_ROUNDS × NUM_CLIENTS` |

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

La table de correspondance `round_info` est précalculée à l'initialisation :

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

### Méthodes de la stratégie

| Méthode | Rôle |
|---------|------|
| `initialize_parameters` | Retourne `None` — les clients partent de zéro au round 1 |
| `configure_fit` | Sélectionne 1 client, sérialise le modèle global, envoie `target_partition` |
| `aggregate_fit` | Désérialise le modèle client, met à jour `global_bst`, accumule `_round_models` ; déclenche `_end_of_semantic_round` à chaque fin de round sémantique |
| `configure_evaluate` | Retourne `[]` — l'évaluation est faite côté serveur |
| `aggregate_evaluate` | Retourne `None` — non utilisé |
| `evaluate` | Retourne `None` — non utilisé (évaluation gérée dans `aggregate_fit`) |

### Évaluation serveur (`_end_of_semantic_round`)

À la fin de chaque round sémantique (quand les 5 clients ont entraîné), le serveur :

1. Récupère les modèles intermédiaires accumulés dans `_round_models[sem_round]`
2. Fait un **soft voting pondéré** sur ces modèles (voir section 7)
3. Calcule `log_loss`, `accuracy`, `f1_macro` sur le val set global (chargé une seule fois)
4. Sauvegarde les métriques dans `results/metrics/federated_metrics.csv`
5. Sauvegarde le meilleur modèle (log_loss minimal) dans `models/federated/best_global_model.cubj`

```python
# Évaluation déclenchée dans aggregate_fit() à chaque fin de round sémantique
if step == self.n_clients - 1:
    self._end_of_semantic_round(sem_round)
```

### Boosting incrémental

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

## 7. L'agrégation des prédictions : Soft Voting

Une fois tous les clients entraînés dans un round sémantique, les **prédictions finales**
sont produites par **soft voting pondéré** : chaque client prédit des probabilités pour les
4 classes, et ces probabilités sont moyennées en pondérant par la taille du dataset local.

```
p_global(x) = Σ_i  (n_i / N) × p_i(x)

  où :
    p_i(x)   = probabilités prédites par le modèle intermédiaire du client i
    n_i      = nombre d'échantillons du client i
    N        = nombre total d'échantillons (somme de tous les clients)
```

Un client avec plus de données a plus de poids dans la prédiction finale. Cette agrégation
n'implique aucun échange de données brutes — seulement des vecteurs de probabilités.

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
| **Fédéré v3 (100 rounds, séquentiel, eta=0.05)**    | **0.0076**     | **0.9977**     | **0.9978**     |

**Scores Zindi (jeu de test public/privé) :**

| Modèle       | Public       | Privé        |
|--------------|-------------|-------------|
| Centralisé   | 0.003741    | 0.024795    |
| Fédéré v2    | 0.043446    | 0.049010    |
| Fédéré v3    | en cours... | en cours... |

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
┌──────────────────────────────────────────────────────────┐
│       SIMULATION FLOWER (100 rounds Flower)              │
│                                                          │
│  Round Flower r  (= étape d'un round sémantique) :       │
│    configure_fit()  → envoie global_bst + target_partition│
│    client.fit()     → xgb.train(global_bst, 1 arbre)     │
│    aggregate_fit()  → met à jour global_bst              │
│                                                          │
│  Toutes les 5 étapes (fin de round sémantique) :         │
│    Soft voting pondéré sur _round_models                 │
│    Évaluation val set → log_loss / accuracy / F1         │
│    Sauvegarde du meilleur modèle                         │
└──────────────────────────────────────────────────────────┘
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
                     hyperparamètres XGBoost, sérialisation/désérialisation modèle
  client_app.py    — XGBoostFlowerClient : fit() (entraînement local incrémental),
                     evaluate() (stub, évaluation faite côté serveur)
  server_app.py    — SequentialXGBoostStrategy : configure_fit, aggregate_fit,
                     _end_of_semantic_round (soft voting + éval), evaluate_global

notebook/
  05_federated_model.ipynb — simulation Flower (run_simulation), visualisations,
                             calibration en température, soumission Zindi

models/federated/
  best_global_model.cubj   — meilleur modèle global (log_loss minimal sur val)

results/metrics/
  federated_metrics.csv    — log_loss, accuracy, F1-macro par round sémantique
```
