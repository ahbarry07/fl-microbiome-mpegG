# Description des features — Dataset final (94 features)

Le dataset final est produit par `src/feature_engineering.py` et sauvegardé dans
`data/processed/train_engineered.csv` / `val_engineered.csv` / `test_engineered.csv`.

Chaque ligne correspond à un échantillon FASTQ (un prélèvement microbiome).  
Les features sont regroupées en 6 familles.

---

## Récapitulatif

| Famille | Features | N |
|---------|----------|---|
| Fractions nucléotidiques brutes | pct_A, pct_T, pct_C, pct_G, pct_GC | 5 |
| Qualité de séquençage | avg_quality, num_reads, avg_read_length | 3 |
| Ratios biologiques dérivés | gc_skew, at_skew, purine_pyrimidine_ratio, nucleotide_entropy | 4 |
| K-mers (k=3) | kmer_AAA … kmer_TTT | 64 |
| Dinucléotides relatifs rho | di_AA … di_TT | 16 |
| Qualité Phred différenciée | pct_bases_q20, pct_bases_q30 | 2 |
| **TOTAL** | | **94** |

---

## 1. Fractions nucléotidiques brutes (5 features)

Calculées directement par `data_processing.py` sur l'ensemble des reads de l'échantillon.

| Feature | Définition | Plage |
|---------|-----------|-------|
| `pct_A` | Fraction de bases Adénine : `count(A) / total_bases` | [0, 1] |
| `pct_T` | Fraction de bases Thymine : `count(T) / total_bases` | [0, 1] |
| `pct_C` | Fraction de bases Cytosine : `count(C) / total_bases` | [0, 1] |
| `pct_G` | Fraction de bases Guanine : `count(G) / total_bases` | [0, 1] |
| `pct_GC` | Fraction GC : `pct_G + pct_C` | [0, 1] |

> **Note :** `pct_A + pct_T + pct_C + pct_G = 1` par construction (les N et autres bases ambiguës sont exclus).  
> `pct_GC` est conservé malgré la redondance car c'est un indicateur biologique standard largement cité dans la littérature microbiome.

---

## 2. Qualité de séquençage (3 features)

Métriques globales du fichier FASTQ, calculées par `data_processing.py`.

| Feature | Définition | Plage |
|---------|-----------|-------|
| `num_reads` | Nombre total de reads dans le fichier FASTQ | ≥ 1 |
| `avg_read_length` | Longueur moyenne des reads (en bp) | > 0 |
| `avg_quality` | Score Phred moyen sur toutes les bases de tous les reads | [0, 40+] |

> **Score Phred :** Q = −10 × log₁₀(P_erreur). Q20 = 1 % d'erreur, Q30 = 0,1 % d'erreur.

---

## 3. Ratios biologiques dérivés (4 features)

Calculés par `feature_engineering.py` (`build_features`) à partir des fractions brutes.

### `gc_skew`

```
gc_skew = (G − C) / (G + C)
```

Mesure l'asymétrie de composition entre G et C sur le brin séquencé.  
Liée à la direction de réplication bactérienne et à la pression de sélection sur le brin codant.  
Plage : [−1, +1]. Valeur ~0 = symétrie parfaite.  
*(Lobry, 1996, Nucleic Acids Research)*

### `at_skew`

```
at_skew = (A − T) / (A + T)
```

Asymétrie complémentaire du GC-skew pour les bases A et T.  
Ensemble, `gc_skew` et `at_skew` caractérisent l'asymétrie de brin de chaque microbiome.  
Plage : [−1, +1].  
*(Lobry & Sueoka, 2002, Genome Biology)*

### `purine_pyrimidine_ratio`

```
R/Y = (A + G) / (C + T)
```

Rapport purines (A, G — deux cycles) sur pyrimidines (C, T — un cycle).  
Proxy de la pression de sélection sur la composition en bases.  
Valeur ~1 = équilibre de Chargaff. Des écarts indiquent un biais de composition caractéristique.  
*(Forsdyke & Mortimer, 2000, Gene)*

### `nucleotide_entropy`

```
H = −Σ pᵢ × log₂(pᵢ)    avec pᵢ ∈ {pct_A, pct_T, pct_C, pct_G}
```

Entropie de Shannon sur la composition nucléotidique (base 2, en bits).  
Maximum théorique : 2 bits (distribution uniforme 25/25/25/25).  
Une entropie faible indique un fort biais de composition (ex. génomes à très fort GC).  
Plage : [0, 2].

---

## 4. K-mers (k=3) — 64 features

Calculés par `extract_fastq_features` → accumulation par blocs via `np.bincount`.

### Définition

Un **k-mer** est une sous-séquence de longueur k. Pour k=3 et l'alphabet {A, C, G, T},
il existe 4³ = **64 trinucléotides** possibles (AAA, AAC, …, TTT).

```
kmer_XYZ = count(XYZ) / Σ count(tous les trinucléotides valides)
```

Seules les positions où les 3 bases sont dans {A, C, G, T} sont comptées
(les bases ambiguës N sont exclues ; les k-mers inter-reads sont empêchés par un séparateur N).

### Liste des 64 features

Les features sont nommées `kmer_` + le trinucléotide en ordre lexicographique :

```
kmer_AAA  kmer_AAC  kmer_AAG  kmer_AAT
kmer_ACA  kmer_ACC  kmer_ACG  kmer_ACT
kmer_AGA  kmer_AGC  kmer_AGG  kmer_AGT
kmer_ATA  kmer_ATC  kmer_ATG  kmer_ATT
kmer_CAA  kmer_CAC  kmer_CAG  kmer_CAT
kmer_CCA  kmer_CCC  kmer_CCG  kmer_CCT
kmer_CGA  kmer_CGC  kmer_CGG  kmer_CGT
kmer_CTA  kmer_CTC  kmer_CTG  kmer_CTT
kmer_GAA  kmer_GAC  kmer_GAG  kmer_GAT
kmer_GCA  kmer_GCC  kmer_GCG  kmer_GCT
kmer_GGA  kmer_GGC  kmer_GGG  kmer_GGT
kmer_GTA  kmer_GTC  kmer_GTG  kmer_GTT
kmer_TAA  kmer_TAC  kmer_TAG  kmer_TAT
kmer_TCA  kmer_TCC  kmer_TCG  kmer_TCT
kmer_TGA  kmer_TGC  kmer_TGG  kmer_TGT
kmer_TTA  kmer_TTC  kmer_TTG  kmer_TTT
```

### Justification biologique

Les fréquences de trinucléotides capturent le **contexte local de la séquence** sans nécessiter
d'assignation taxonomique. Pour des reads 16S rRNA, elles sont la représentation la plus
informative pour discriminer les sites corporels (bouche, peau, nasopharynx, intestin).  
*(Woloszynek et al., 2019, PLoS Comput Biol ; MicroPheno — Reiman et al., 2018, Bioinformatics)*

---

## 5. Dinucléotides relatifs rho (16 features)

Calculés par `extract_fastq_features` → accumulation mono/di-nucleotide par blocs.

### Définition

```
rho(XY) = f(XY) / (f(X) × f(Y))
```

où `f(XY)` est la fréquence observée du dinucléotide XY,
et `f(X)`, `f(Y)` sont les fréquences des bases individuelles.

| Valeur de rho | Interprétation |
|--------------|----------------|
| rho = 1 | XY distribué de façon aléatoire (indépendance) |
| rho < 1 | XY sous-représenté (ex. CpG dans les génomes bactériens) |
| rho > 1 | XY sur-représenté (pression de sélection positive) |

### Liste des 16 features

```
di_AA  di_AC  di_AG  di_AT
di_CA  di_CC  di_CG  di_CT
di_GA  di_GC  di_GG  di_GT
di_TA  di_TC  di_TG  di_TT
```

### Justification biologique

Les déviations par rapport à l'indépendance (rho ≠ 1) sont une **signature génomique conservée**
propre à chaque espèce bactérienne. Ces patterns sont stables au sein d'un même site corporel
et discriminants entre sites.  
*(Karlin & Burge, 1995, Trends Genet ; Deschavanne et al., 1999, Mol Biol Evol)*

> **Exemple caractéristique :** `di_CG` (dinucléotide CpG) est systématiquement sous-représenté
> (rho < 1) dans les génomes bactériens en raison de la méthylation et de la mutation C→T.

---

## 6. Qualité Phred différenciée (2 features)

Calculées par `extract_fastq_features` sur les lignes de qualité des fichiers FASTQ.

| Feature | Définition | Plage |
|---------|-----------|-------|
| `pct_bases_q20` | Fraction de bases avec score Phred ≥ 20 (P_erreur ≤ 1 %) | [0, 1] |
| `pct_bases_q30` | Fraction de bases avec score Phred ≥ 30 (P_erreur ≤ 0,1 %) | [0, 1] |

```
pct_bases_q20 = count(bases avec Q ≥ 20) / total_bases
pct_bases_q30 = count(bases avec Q ≥ 30) / total_bases
```

> **Décodage Phred+33 :** chaque caractère ASCII dans la ligne de qualité FASTQ est converti
> en score Phred par `Q = ASCII_value − 33`.

### Justification biologique

La qualité de séquençage peut varier selon le **site corporel** :
- Les prélèvements à forte densité microbienne (intestin) produisent généralement une qualité homogène.
- Les prélèvements nasaux ou cutanés peuvent présenter plus d'inhibiteurs PCR, abaissant le Q moyen.

Ces deux features capturent la **distribution** de la qualité (pas seulement sa moyenne `avg_quality`),
ce qui permet de détecter des échantillons partiellement dégradés.  
*(Standard Illumina de qualité séquençage)*

---

## Source et pipeline de calcul

```
data/raw/TrainFiles/*.fastq
        │
        ▼
data_processing.py          → pct_A/T/C/G/GC, avg_quality, num_reads, avg_read_length
        │
        ▼
feature_engineering.py
  build_features()          → gc_skew, at_skew, purine_pyrimidine_ratio, nucleotide_entropy
  extract_fastq_features()  → kmer_*, di_*, pct_bases_q20, pct_bases_q30
        │
        ▼
data/processed/train_engineered.csv   (N_train × 94)
data/processed/val_engineered.csv     (N_val   × 94)
data/processed/test_engineered.csv    (N_test  × 94)
```
