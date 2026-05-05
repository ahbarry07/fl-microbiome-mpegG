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

| Paramètre        | Valeur | Rôle |
|------------------|--------|------|
| `eta`            | 0.05   | Taux d'apprentissage — petits pas pour mieux généraliser |
| `max_depth`      | 6      | Profondeur maximale des arbres |
| `subsample`      | 0.8    | 80 % des données utilisées par arbre (réduit le surapprentissage) |
| `colsample_bytree` | 0.8  | 80 % des features utilisées par arbre |
| `min_child_weight` | 1    | Taille minimale d'un groupe feuille |
| `num_class`      | 4      | Nombre de classes à prédire |

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

À chaque round, les clients s'enchaînent **l'un après l'autre** pour enrichir le modèle global :

```
Round k (ordre mélangé : ex. [Client 3, Client 0, Client 4, Client 1, Client 2]) :

  → Client 3 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 0 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 4 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 1 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Client 2 entraîne 1 arbre sur ses données → modèle global mis à jour
  → Fin du round : le modèle global a vu TOUTES les données ce round
```

L'ordre est **mélangé aléatoirement à chaque round** (mais de façon reproductible via une
graine fixe) pour éviter qu'un client ne domine systématiquement.

**Résultat** : après 100 rounds × 5 clients × 1 arbre = **500 arbres**, le modèle global
a appris de l'ensemble des 2901 échantillons de manière distribuée — comme le centralisé,
mais sans que les données ne soient jamais partagées.

---

## 6. L'agrégation des prédictions : Soft Voting

Une fois tous les clients entraînés, les **prédictions finales** sont produites par
**soft voting pondéré** : chaque client prédit des probabilités pour les 4 classes, et
ces probabilités sont moyennées en pondérant par la taille du dataset local de chaque client.

```
p_global(x) = Σ_i  (n_i / N) × p_i(x)

  où :
    p_i(x)   = probabilités prédites par le client i pour l'échantillon x
    n_i      = nombre d'échantillons du client i
    N        = nombre total d'échantillons (somme de tous les clients)
```

Un client avec plus de données a plus de poids dans la prédiction finale. Cette agrégation
n'implique aucun échange de données brutes — seulement des vecteurs de probabilités.

---

## 7. Calibration en température

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

## 8. Résultats obtenus

| Modèle                      | Log Loss (val) | Accuracy (val) | F1-macro (val) |
|-----------------------------|----------------|----------------|----------------|
| Centralisé (XGBoost)        | 0.0368         | 0.9863         | 0.9867         |
| Fédéré — v1 (20 rounds, parallèle, eta=0.1) | 0.2107 | 0.9840 | 0.9849 |
| Fédéré — v2 (100 rounds, parallèle, eta=0.05) | 0.0382 | 0.9886 | 0.9889 |
| **Fédéré — v3 (100 rounds, séquentiel, eta=0.05)** | **0.0076** | **0.9977** | **0.9978** |

**Scores Zindi (jeu de test public/privé) :**

| Modèle       | Public       | Privé        |
|--------------|-------------|-------------|
| Centralisé   | 0.003741    | 0.024795    |
| Fédéré v2    | 0.043446    | 0.049010    |
| Fédéré v3    | en cours... | en cours... |

---

## 9. Résumé visuel du pipeline

```
Données brutes (MPEG-G .mgb)
        │
        ▼
Extraction de features (Kraken2, k-mers, stats qualité)
        │
        ▼
Partitionnement round-robin par SubjectID → 5 clients
        │
        ▼
┌─────────────────────────────────────────────┐
│         BOUCLE FÉDÉRÉE (100 rounds)         │
│                                             │
│  Round k :                                  │
│    ordre mélangé [c_i, c_j, c_k, c_l, c_m] │
│    Pour chaque client (séquentiel) :        │
│      xgb.train(global_bst) → global_bst    │
│    Soft voting → p_global                   │
│    Évaluation log_loss / accuracy / F1      │
└─────────────────────────────────────────────┘
        │
        ▼
Calibration en température (T optimisé sur val)
        │
        ▼
Prédictions test → soumission Zindi (CSV)
```

---

## 10. Fichiers du projet

```
src/
  task.py          — chargement des données, partitionnement, hyperparamètres XGBoost
  client_app.py    — classe XGBoostClient (entraînement local, évaluation)
  server_app.py    — soft voting, boucle fédérée, évaluation globale

notebook/
  05_federated_model.ipynb — simulation complète, visualisations, soumission

models/federated/
  best_global_model.ubj    — meilleur modèle global sauvegardé

results/metrics/
  federated_metrics.csv    — log_loss, accuracy, F1 par round
```
