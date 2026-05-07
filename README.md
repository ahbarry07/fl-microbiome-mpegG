
# **Federated Learning for Microbiome Body-Site Classification from MPEG-G Data**

> MPEG-G Microbiome Classification Challenge — Zindi Africa  
> Sequential XGBoost Federated Learning with Soft Voting Aggregation via Flower

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [Notebook 01 — Data Exploration](#notebook-01--data-exploration)
  - [Notebook 02 — MPEG-G Decompression & Raw Feature Extraction](#notebook-02--mpeg-g-decompression--raw-feature-extraction)
  - [Notebook 03 — Feature Engineering & Dataset Split](#notebook-03--feature-engineering--dataset-split)
  - [Notebook 04 — Centralized Baseline Models](#notebook-04--centralized-baseline-models)
  - [Notebook 05 — Federated Learning Simulation](#notebook-05--federated-learning-simulation)
- [Federated Architecture](#federated-architecture)
- [Feature Catalog](#feature-catalog)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Overview

This project tackles the [MPEG-G Microbiome Classification Challenge](https://zindi.africa/competitions/mpeg-g-microbiome-classification-challenge) on Zindi Africa. The objective is to classify **metagenomics sequencing samples** into **4 body-site categories** — Stool, Skin, Nasal, and Mouth — using **federated machine learning**, with no raw patient data ever shared between clients.

### Key Contributions

- **End-to-end pipeline** from compressed MPEG-G files (`.mgb`) to competition-ready submission
- **208 biologically informed features** spanning nucleotide composition, k-mers, dinucleotide relative frequencies, sequence complexity, and Kraken2 taxonomic classification
- **Sequential federated XGBoost** strategy via [Flower](https://flower.ai/): clients improve a shared model one-by-one per round, rather than averaging independent models — resulting in better convergence than classical parallel aggregation
- **Soft voting aggregation** preserving probability calibration across clients
- **Subject-level train/validation split** to prevent data leakage (no samples from the same subject appear in both sets)
- Federated model **matches or outperforms the centralized baseline** after 20 semantic rounds

---

## Results

| Model | Log Loss | Accuracy | F1-macro |
|---|---|---|---|
| Centralized XGBoost (baseline) | 0.0368 | 98.63% | 0.9867 |
| **Federated v3 — Round 20 (final)** | **0.0379** | **99.32%** | **0.9932** |
| Federated v3 — Calibrated (T=0.6979) | **0.0261** | — | — |

**Zindi Public Leaderboard scores (test set):**

| Submission | Public Score | Private Score |
|---|---|---|
| Centralized XGBoost | 0.003741144 | 0.024794668 |
| Federated  | 0.016999233 | 0.03279972 |

> Validation metrics computed on a subject-held-out set (12 subjects, 438 samples). Lower log loss is better.

---

## Dataset

| Source | Link |
|---|---|
| Zindi Africa (official) | https://zindi.africa/competitions/mpeg-g-microbiome-classification-challenge/data |
| Kaggle mirror | https://www.kaggle.com/datasets/ahbarry01edudkr1124/mpeg-dataset |

**Dataset statistics:**

| Split | Samples | Subjects |
|---|---|---|
| Train | 2,901 | 66 |
| Test | 1,068 | — |

**Classes (body sites):** Stool (28%), Skin (27%), Nasal (24%), Mouth (20%) — relatively balanced (imbalance ratio 1.37×).

Each sample is a compressed MPEG-G `.mgb` file containing a 16S/metagenomics sequencing read set. Subject metadata (BMI, age, FPG glucose, gender, ethnicity, diabetes classification) is provided in `Train_Subjects.csv`.

**Place the raw data under:**
```
data/raw/
├── Train.csv
├── Test.csv
├── Train_Subjects.csv
├── TrainFiles/          # .mgb files for training
└── TestFiles/           # .mgb files for test
```

---

## Project Structure

```
fl-microbiome-mpegG/
├── notebook/
│   ├── 01_exploration_data.ipynb            # EDA: distributions, metadata
│   ├── 02_mgb_extraction_features.ipynb     # MPEG-G → FASTQ + raw features
│   ├── 03_preprocessing_engineering.ipynb  # 208-feature engineering + split
│   ├── 04_centralized_model.ipynb          # Centralized baselines (RF, XGBoost)
│   └── 05_federated_model.ipynb            # Flower FL simulation + submission
├── src/
│   ├── client_app.py          # Flower client (XGBoostFlowerClient)
│   ├── server_app.py          # Flower server (SequentialXGBoostStrategy)
│   ├── task.py                # Shared utilities, constants, data loading
│   ├── data_processing.py     # MPEG-G decompression, raw feature extraction
│   └── feature_engineering.py # K-mers, dinucleotides, taxonomy, complexity
├── docs/
│   ├── federated_approach.md  # Detailed FL architecture documentation
│   └── feature_description.md # Full feature catalog with biology justification
├── data/
│   ├── raw/                   # Original CSVs + .mgb archives (not tracked in git)
│   └── processed/             # Generated feature CSVs (not tracked in git)
├── models/
│   ├── centralized/           # xgboost_centralized.joblib, label_encoder.joblib
│   └── federated/             # best_global_model.cubj
├── results/
│   ├── metrics/               # centralized_metrics.csv, federated_metrics.csv
│   └── figures/               # EDA plots, model comparison charts
├── main.py                    # Entry point for Flower FL simulation
├── pyproject.toml             # Project dependencies (uv)
└── requirements.txt           # Alternative pip requirements
```

---

## Pipeline

The project follows a **5-notebook sequential pipeline**. Each notebook produces artifacts consumed by the next.

```
Raw MPEG-G files (.mgb)
        │
        ▼
[01] Exploration ─────────────── dataset statistics, class balance, metadata
        │
        ▼
[02] MPEG-G Decompression ─────── Docker (Genie) .mgb → .fastq
        │                         9 raw nucleotide/quality features
        ▼
[03] Feature Engineering ──────── 208-feature dataset
        │   K-mers (k=3)     64   │
        │   Dinucleotides     16   │
        │   Biological ratios  4   │
        │   Quality bins        2   │
        │   Complexity (LZ76)   7   │
        │   Kraken2 taxonomy  103   │
        │
        │   Subject-level split (no leakage)
        │   → 2,463 train / 438 val / 1,068 test
        ▼
[04] Centralized Baselines ─────── RF, XGBoost → log_loss 0.0368
        │
        ▼
[05] Federated Simulation ──────── 5 clients, 20 semantic rounds, 100 trees
        │   Sequential training + soft voting
        │   → log_loss 0.0379, accuracy 99.32%
        │
        ▼
    Temperature Calibration (T=0.6979) → log_loss 0.0261
        │
        ▼
    Zindi Submission
```

---

### Notebook 01 — Data Exploration

**File:** [`notebook/01_exploration_data.ipynb`](notebook/01_exploration_data.ipynb)

Exploratory data analysis of the raw metadata and dataset structure.

**Key outputs:**
- Train set: 2,901 samples from 66 unique subjects; test set: 1,068 samples
- Target distribution: Stool 28% / Skin 27% / Nasal 24% / Mouth 20% (balanced)
- 65 out of 66 subjects contributed samples from multiple body sites
- Subject metadata variables: FPG (fasting plasma glucose), BMI, age, gender, ethnicity, diabetes classification
- Confirms that a **subject-level split** is mandatory to avoid data leakage

---

### Notebook 02 — MPEG-G Decompression & Raw Feature Extraction

**File:** [`notebook/02_mgb_extraction_features.ipynb`](notebook/02_mgb_extraction_features.ipynb)

Converts compressed MPEG-G binary files into readable FASTQ sequences and extracts 9 low-level nucleotide features.

**Decompression pipeline:**
1. Unzip `TrainFiles.zip` and `TestFiles.zip` (2,901 + 1,068 `.mgb` files)
2. Run [Genie](https://github.com/MueFab/genie) via Docker (`muefab/genie:latest`) to decode `.mgb → .fastq`
3. Parse FASTQ with BioPython for feature extraction

**Features extracted per file (9):**

| Feature | Description |
|---|---|
| `num_reads` | Sequencing depth (read count) |
| `avg_read_length` | Mean read length in base pairs |
| `avg_quality` | Mean Phred quality score |
| `pct_A`, `pct_T`, `pct_C`, `pct_G` | Nucleotide base fractions |
| `pct_GC` | GC content |

**Key finding:** GC content varies significantly by body site (Nasal avg 55.95% vs Mouth 52.13%), already providing discriminant signal before any advanced feature engineering.

**Outputs:**
- `data/processed/train_fastq_features.csv` (2,901 × 12)
- `data/processed/test_fastq_features.csv` (1,068 × 9)

---

### Notebook 03 — Feature Engineering & Dataset Split

**File:** [`notebook/03_preprocessing_engineering.ipynb`](notebook/03_preprocessing_engineering.ipynb)

Builds the final 208-feature ML-ready dataset using parallel processing and Docker-based taxonomic classification. This is the most computationally intensive step.

#### Feature families

**Biological ratios (4 features):**
- `gc_skew` = (G−C)/(G+C) — strand asymmetry (Lobry 1996)
- `at_skew` = (A−T)/(A+T) — replication-related asymmetry
- `purine_pyrimidine_ratio` — R/Y balance
- `nucleotide_entropy` — Shannon entropy over A/T/C/G fractions

**K-mers k=3 (64 features):**
- All 64 trinucleotide frequencies (AAA…TTT), normalized per file
- Captures local sequence composition patterns (Woloszynek et al. 2019)

**Dinucleotide relative frequencies / rho (16 features):**
- Observed/expected ratios for all 16 dinucleotides (Karlin & Burge 1995)
- CpG under-representation visible across human microbiome sites

**Quality bins (2 features):**
- `pct_bases_q20`, `pct_bases_q30` — fraction of bases above quality thresholds

**Sequence complexity (7 features):**
- `lz_complexity` — Lempel-Ziv LZ76 complexity (diversity proxy, sampled on 2,000 reads)
- `pct_ambiguous` — fraction of non-ATCG bases
- Read length distribution: std, min, max, Q25, Q75

**Kraken2 taxonomic classification (103 features):**
- Run via Docker (`staphb/kraken2`) against Silva 16S database
- ~2,680 genera detected across all samples
- Filtered to **prevalence ≥ 5%** → 101 retained genera
- `kraken_unclassified` (unclassified fraction)
- `kraken_n_genera` (number of genera detected per sample)
- Top discriminant genera: *Cutibacterium* (skin), *Staphylococcus* (skin), *Corynebacterium* (nasal)

#### Train/Validation split

Split is performed **by SubjectID** (stratified, 80/20), not by sample, to ensure no subject appears in both train and validation:

| Set | Samples | Subjects |
|---|---|---|
| Train | 2,463 | 54 |
| Validation | 438 | 12 |
| Test | 1,068 | — |

**Outputs:**
- `data/processed/train_engineered.csv` (2,463 × 208)
- `data/processed/val_engineered.csv` (438 × 208)
- `data/processed/test_engineered.csv` (1,068 × 205)
- `data/processed/feature_cols.csv` (204 feature names)

---

### Notebook 04 — Centralized Baseline Models

**File:** [`notebook/04_centralized_model.ipynb`](notebook/04_centralized_model.ipynb)

Trains centralized models on the full training set to establish a performance ceiling for comparison with the federated approach.

**Models trained:**

| Model | Configuration |
|---|---|
| Random Forest | 500 trees, balanced class weights, `n_jobs=-1` |
| **XGBoost** | 500 rounds, `eta=0.05`, `max_depth=6`, `objective=multi:softprob` |

**Best model — XGBoost (validation set):**

| Metric | Score |
|---|---|
| Log Loss | 0.0368 |
| Accuracy | 98.63% |
| F1-macro | 0.9867 |
| Per-class precision/recall | ≥ 0.97 for all 4 sites |

**Feature importance analysis:**
- Kraken2 genera dominate (especially *Cutibacterium*, *Staphylococcus*)
- Secondary: `at_skew`, dinucleotide rho patterns
- Tertiary: k-mer frequencies

**Outputs:**
- `models/centralized/xgboost_centralized.joblib`
- `models/centralized/label_encoder.joblib`
- `results/metrics/centralized_metrics.csv`
- `data/submission/submission_xgboost.csv`

---

### Notebook 05 — Federated Learning Simulation

**File:** [`notebook/05_federated_model.ipynb`](notebook/05_federated_model.ipynb)

Implements and evaluates a **federated XGBoost classifier** using the [Flower](https://flower.ai/) framework (`flwr[simulation]`).

#### Client partitioning

Training samples are distributed across **5 clients** via round-robin assignment by SubjectID. This ensures:
- Each client sees all 4 body-site classes
- Partitions reflect realistic heterogeneous data distributions

| Client | Samples |
|---|---|
| 0 | 561 |
| 1 | 551 |
| 2 | 353 |
| 3 | 559 |
| 4 | 877 |

#### Federated strategy — Sequential XGBoost

Unlike classical **parallel** FL (FedAvg), where each client trains independently and their models are averaged, this project uses **sequential training**:

1. Within each semantic round, clients are randomly ordered (seed = 42 + round)
2. Each client receives the current shared model and trains **1 additional tree** on top of it
3. After all 5 clients have trained, the server aggregates via **soft voting**

This ensures every sample contributes to model improvement every round, rather than only one fifth of the data at a time.

**Configuration:**

| Parameter | Value |
|---|---|
| Number of clients | 5 |
| Semantic rounds | 20 |
| Trees per client per round | 1 |
| Total trees (final model) | 100 |
| Total Flower rounds | 100 (20 × 5) |
| XGBoost `eta` | 0.05 |
| XGBoost `max_depth` | 6 |
| Objective | `multi:softprob` (4 classes) |

#### Soft voting aggregation

At the end of each semantic round, the server collects per-client probability predictions on the validation set and aggregates them using **sample-weighted soft voting**:

```
p_global = Σ (n_i / N) × p_i
```

where `n_i` = number of training samples for client `i`, `N` = total training samples, `p_i` = client probability predictions. This weighting gives more influence to clients with more representative data.

#### Training convergence

| Round | Log Loss | Accuracy |
|---|---|---|
| 1 | 1.18 | — |
| 2 | 0.86 | — |
| 5 | 0.40 | — |
| 10 | ~0.07 | — |
| 20 (final) | **0.0379** | **99.32%** |

Model stabilizes around round 10; rounds 10–20 provide fine-grained refinement.

#### Temperature calibration

Post-training probability calibration via temperature scaling:

- Grid search over T ∈ [0.1, 3.0] on validation set
- **Optimal temperature: T = 0.6979**
- Calibrated log loss: **0.0261** (improvement of ~0.012 vs uncalibrated)

#### Comparison: Federated vs. Centralized

| Approach | Log Loss | Accuracy | F1-macro |
|---|---|---|---|
| Centralized XGBoost | 0.0368 | 98.63% | 0.9867 |
| Federated (round 20) | 0.0379 | 99.32% | 0.9932 |

The federated model achieves **higher accuracy and F1** than the centralized baseline while keeping raw data local to each client.

**Outputs:**
- `models/federated/best_global_model.cubj`
- `results/metrics/federated_metrics.csv`
- `data/submission/submission_federated.csv`

---

## Federated Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Flower Server                          │
│                                                          │
│  SequentialXGBoostStrategy                               │
│  ┌────────────────────────────────┐                      │
│  │  round_info: pre-calculated    │                      │
│  │  mapping flower_round →        │                      │
│  │  (sem_round, step, partition)  │                      │
│  │                                │                      │
│  │  configure_fit() →             │                      │
│  │    send global_bst + target    │                      │
│  │                                │                      │
│  │  aggregate_fit() →             │                      │
│  │    update global_bst           │                      │
│  │    accumulate _round_models    │                      │
│  │                                │                      │
│  │  _end_of_semantic_round() →    │                      │
│  │    soft_voting()               │                      │
│  │    evaluate_global()           │                      │
│  │    save best model             │                      │
│  └────────────────────────────────┘                      │
└──────────────────────────────────────────────────────────┘
        │   FitIns (global_bst, target_partition)
        ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ...
│  Client 0    │  │  Client 1    │  │  Client 2    │
│              │  │              │  │              │
│ 561 samples  │  │ 551 samples  │  │ 353 samples  │
│              │  │              │  │              │
│ fit():       │  │ fit():       │  │ fit():       │
│  load part.  │  │  load part.  │  │  load part.  │
│  train 1 tree│  │  train 1 tree│  │  train 1 tree│
│  return bst  │  │  return bst  │  │  return bst  │
└──────────────┘  └──────────────┘  └──────────────┘
        │   FitRes (serialized booster, n_samples)
        ▼
┌──────────────────────────────────────────────────────────┐
│  Semantic round end → soft_voting() → val evaluation     │
└──────────────────────────────────────────────────────────┘
```

---

## Feature Catalog

| Family | Count | Key features |
|---|---|---|
| Raw nucleotide fractions | 5 | `pct_A`, `pct_T`, `pct_C`, `pct_G`, `pct_GC` |
| Sequencing quality | 3 | `avg_quality`, `num_reads`, `avg_read_length` |
| Biological ratios | 4 | `gc_skew`, `at_skew`, `purine_pyrimidine_ratio`, `nucleotide_entropy` |
| K-mers (k=3) | 64 | `kmer_AAA` … `kmer_TTT` |
| Dinucleotides rho | 16 | `di_AA` … `di_TT` |
| Quality bins | 2 | `pct_bases_q20`, `pct_bases_q30` |
| Sequence complexity | 7 | `lz_complexity`, `pct_ambiguous`, read-length stats |
| Kraken2 taxonomy | 103 | 101 genera + `kraken_unclassified` + `kraken_n_genera` |
| **Total** | **204** | |

See [`docs/feature_description.md`](docs/feature_description.md) for the full catalog with biological justification and literature references.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/ahbarry07/fl-microbiome-mpegG.git
cd fl-microbiome-mpegG

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**System requirements:**

- Python 3.11+
- [Docker](https://docs.docker.com/get-docker/) (required for MPEG-G decompression and Kraken2 classification)
- ~1.8 GB disk space for the Kraken2 Silva 16S database
- ~24 GB disk space for raw `.mgb` files

**Docker images used (pulled automatically):**

```bash
docker pull muefab/genie:latest    # MPEG-G decompression
docker pull staphb/kraken2:latest  # 16S taxonomic classification
```

---

## Usage

### Run the full notebook pipeline

Execute notebooks **in order**:

```bash
# 01 - Explore the dataset
jupyter notebook notebook/01_exploration_data.ipynb

# 02 - Decompress MPEG-G and extract raw features (requires Docker)
jupyter notebook notebook/02_mgb_extraction_features.ipynb

# 03 - Feature engineering (parallel processing + Kraken2 via Docker)
jupyter notebook notebook/03_preprocessing_engineering.ipynb

# 04 - Train centralized baselines
jupyter notebook notebook/04_centralized_model.ipynb

# 05 - Run federated learning simulation
jupyter notebook notebook/05_federated_model.ipynb
```

### Run the federated simulation standalone

```bash
# From project root, with uv
uv run python main.py
```

This launches the Flower simulation with 5 clients, 20 semantic rounds, and logs per-round metrics to `results/metrics/federated_metrics.csv`.

---

## Dependencies

**Core ML:**

| Package | Purpose |
|---|---|
| `xgboost` | Gradient boosting classifier (centralized + federated) |
| `scikit-learn` | Preprocessing, metrics, Random Forest baseline |
| `flwr[simulation]` | Flower federated learning framework |

**Bioinformatics:**

| Package | Purpose |
|---|---|
| `biopython` | FASTQ parsing |
| `scikit-bio` | Diversity metrics |
| Docker / `muefab/genie` | MPEG-G → FASTQ decompression |
| Docker / `staphb/kraken2` | 16S taxonomic classification |

**Data & utilities:**

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scipy` | Statistical utilities |
| `joblib` | Model serialization, parallel processing |
| `tqdm` | Progress bars |
| `matplotlib`, `seaborn`, `plotly` | Visualization |

Full dependency list: [`pyproject.toml`](pyproject.toml)

---

## Documentation

- [`docs/federated_approach.md`](docs/federated_approach.md) — detailed FL architecture, strategy design decisions, hyperparameter choices, and results progression (v1 → v3)
- [`docs/feature_description.md`](docs/feature_description.md) — complete feature catalog with biological justification and literature references for each feature family

---

## Repository

[https://github.com/ahbarry07/fl-microbiome-mpegG](https://github.com/ahbarry07/fl-microbiome-mpegG)
