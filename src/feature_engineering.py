"""
feature_engineering.py
=======================

Construction des 94 features ML à partir des fichiers FASTQ et des colonnes
pct_*/qualité issues de data_processing.py.

CATALOGUE DES FEATURES (94 au total pour k=3)
----------------------------------------------

Famille              | Features                              | N
---------------------|---------------------------------------|----
Fractions brutes     | pct_A, pct_T, pct_C, pct_G, pct_GC   |  5
Qualité brute        | avg_quality, num_reads, avg_read_length|  3
Ratios biologiques   | gc_skew, at_skew, R/Y, entropie       |  4
K-mers (k=3)         | kmer_AAA … kmer_TTT                   | 64
Dinucléotides rho    | di_AA … di_TT                         | 16
Qualité différenciée | pct_bases_q20, pct_bases_q30          |  2

PIPELINE DE TRAITEMENT RAPIDE
------------------------------
Le chemin critique utilise `extract_fastq_features` (conçue pour
ProcessPoolExecutor) :
  1. `_parse_fastq_bytes`     — lecture FASTQ native en bytes (pas BioPython)
  2. `_encode_sequences`      — table de correspondance ASCII → {0,1,2,3,4}
                                appliquée en une passe numpy sur la
                                concaténation de tous les reads
  3. `_kmer_from_encoded`     — k décalages numpy + bincount (pas de tableau 2D)
  4. `_di_from_encoded`       — bincount sur paires consécutives valides
  5. qualité                  — np.frombuffer sur la concaténation des quals,
                                soustraction Phred+33 vectorisée

Les fonctions Sections 2-4 (Python pur, Counter) sont conservées comme
référence de lisibilité ; `extract_fastq_features` est à utiliser en production.

RÉFÉRENCES
----------
K-mers  : Woloszynek et al. (2019) PLoS Comput Biol ; Reiman et al. (2018) Bioinformatics
Di-nucl.: Karlin & Burge (1995) Trends Genet ; Deschavanne et al. (1999) Mol Biol Evol
Skew    : Lobry (1996) Nucleic Acids Research ; Forsdyke & Mortimer (2000) Gene
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import product
from typing import Tuple
from scipy.stats import entropy as scipy_entropy
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1 : FEATURES DÉRIVÉES
# ============================================================

def compute_gc_skew(df: pd.DataFrame,
                    pct_g_col: str = 'pct_G',
                    pct_c_col: str = 'pct_C') -> pd.Series:
    """
    GC-skew = (G - C) / (G + C).

    Mesure l'asymétrie de composition entre G et C.
    Liée à la direction de réplication bactérienne et à la
    pression de sélection sur le brin codant.
    (Lobry, 1996, Nucleic Acids Research)

    Parameters
    ----------
    df : pd.DataFrame contenant pct_G et pct_C
    pct_g_col, pct_c_col : str

    Returns
    -------
    pd.Series (valeurs entre -1 et +1)
    """
    gc_sum = (df[pct_g_col] + df[pct_c_col]).replace(0, np.nan)
    return ((df[pct_g_col] - df[pct_c_col]) / gc_sum).rename('gc_skew')


def compute_at_skew(df: pd.DataFrame,
                    pct_a_col: str = 'pct_A',
                    pct_t_col: str = 'pct_T') -> pd.Series:
    """
    AT-skew = (A - T) / (A + T).

    Asymétrie complémentaire du GC-skew.
    Ensemble, gc_skew et at_skew forment la signature
    d'asymétrie de brin du microbiome de l'échantillon.
    (Lobry & Sueoka, 2002, Genome Biology)

    Parameters
    ----------
    df : pd.DataFrame contenant pct_A et pct_T
    pct_a_col, pct_t_col : str

    Returns
    -------
    pd.Series (valeurs entre -1 et +1)
    """
    at_sum = (df[pct_a_col] + df[pct_t_col]).replace(0, np.nan)
    return ((df[pct_a_col] - df[pct_t_col]) / at_sum).rename('at_skew')


def compute_purine_pyrimidine_ratio(df: pd.DataFrame,
                                     pct_a_col: str = 'pct_A',
                                     pct_g_col: str = 'pct_G',
                                     pct_c_col: str = 'pct_C',
                                     pct_t_col: str = 'pct_T') -> pd.Series:
    """
    Ratio purine/pyrimidine = (A + G) / (C + T).

    Purines (A, G) et pyrimidines (C, T) ont des structures chimiques
    différentes. Ce ratio, dit R/Y, est un proxy de la pression de
    sélection sur la composition en bases. Une valeur ~1 correspond
    à l'équilibre de Chargaff.
    (Forsdyke & Mortimer, 2000, Gene)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series
    """
    purines    = df[pct_a_col] + df[pct_g_col]
    pyrimidines = (df[pct_c_col] + df[pct_t_col]).replace(0, np.nan)
    return (purines / pyrimidines).rename('purine_pyrimidine_ratio')


def compute_nucleotide_entropy(df: pd.DataFrame,
                                pct_a_col: str = 'pct_A',
                                pct_t_col: str = 'pct_T',
                                pct_c_col: str = 'pct_C',
                                pct_g_col: str = 'pct_G') -> pd.Series:
    """
    Entropie de Shannon sur la composition nucléotidique (base 2).

    H = -Σ p_i * log2(p_i), avec p_i = fraction de chaque nucléotide.

    Valeur maximale : 2 bits (distribution uniforme 25/25/25/25).
    Une entropie faible indique un fort biais de composition.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series (bits, entre 0 et 2)
    """
    fracs = df[[pct_a_col, pct_t_col, pct_c_col, pct_g_col]].clip(lower=1e-10)
    entropy_vals = fracs.apply(
        lambda row: scipy_entropy(row.values, base=2), axis=1
    )
    return entropy_vals.rename('nucleotide_entropy')


# ============================================================
# SECTION 2 : K-MERS (depuis séquences brutes)
# ============================================================
 
def _all_kmers(k):
    """Génère la liste de tous les k-mers possibles sur {A,C,G,T} (ordre lexicographique)."""
    return [''.join(p) for p in product('ACGT', repeat=k)]
 

# ============================================================
# SECTION 4b : VERSIONS NUMPY (RAPIDES)
# ============================================================

# Table de correspondance ASCII → indice base (A=0, C=1, G=2, T=3, autre=4)
# uint8 : 4× moins de mémoire que int32 sur le tableau encoded
_BASE_TABLE = np.full(256, 4, dtype=np.uint8)
for _char, _val in zip('AaCcGgTt', [0, 0, 1, 1, 2, 2, 3, 3]):
    _BASE_TABLE[ord(_char)] = _val


def _encode_sequences(sequences):
    """
    Concatène les séquences avec un séparateur 'N' (index 4, invalide),
    puis mappe chaque octet ASCII vers {A=0, C=1, G=2, T=3, autre=4}
    via une lookup table numpy en une seule opération vectorisée.

    Le séparateur 'N' entre deux reads garantit qu'aucun k-mer ne
    chevauche une frontière de read.

    Accepte des séquences str ou bytes.
    """
    if sequences and isinstance(sequences[0], (bytes, bytearray)):
        combined = np.frombuffer(b'N'.join(sequences), dtype=np.uint8)
    else:
        combined = np.frombuffer('N'.join(sequences).encode(), dtype=np.uint8)
    return _BASE_TABLE[combined]


def compute_kmer_frequencies_fast(sequences, k=3):
    """
    Fréquences relatives des 4^k k-mers — version numpy vectorisée.

    Algorithme :
      1. `_encode_sequences` : concaténation + lookup table → tableau uint8 de longueur n
      2. k tableaux décalés (vues O(1) sans copie) : encoded[0:n-k+1], ..., encoded[k-1:n]
      3. Masque booléen : positions où les k bases sont toutes dans {A,C,G,T} (< 4)
      4. Indice base-4 : sum(shift[i] * 4^(k-1-i)) pour les positions valides
      5. np.bincount → comptage en C pur

    Parameters
    ----------
    sequences : list of str ou list of bytes
    k : int  — longueur des k-mers (3 → 64 features)

    Returns
    -------
    dict : {f'kmer_{km}': fréquence_relative} pour les 4^k k-mers
    """
    encoded   = _encode_sequences(sequences)
    all_kmers = _all_kmers(k)
    m         = len(encoded) - k + 1

    if m <= 0:
        return {f'kmer_{km}': 0.0 for km in all_kmers}

    shifts = [encoded[i:i + m] for i in range(k)]
    valid  = shifts[0] < 4
    for s in shifts[1:]:
        valid &= s < 4

    # Dtype minimal pour les indices : max théorique = 4*(4^k-1)/3
    # k=3 → 84 (uint8), k=4 → 340 (uint16), k≥5 → int32
    max_idx   = sum(4 * 4 ** (k - 1 - i) for i in range(k))
    idx_dtype = np.uint8 if max_idx <= 255 else np.uint16 if max_idx <= 65535 else np.int32
    powers    = (4 ** np.arange(k - 1, -1, -1)).astype(idx_dtype)
    indices   = np.zeros(m, dtype=idx_dtype)
    for s, p in zip(shifts, powers):
        indices += s.astype(idx_dtype) * p

    vi = indices[valid]
    if not len(vi):
        return {f'kmer_{km}': 0.0 for km in all_kmers}

    counts = np.bincount(vi, minlength=4 ** k)
    total  = counts.sum()
    return {f'kmer_{km}': int(counts[i]) / total for i, km in enumerate(all_kmers)}


def compute_relative_dinucleotide_frequencies_fast(sequences):
    """
    Fréquences relatives des 16 dinucléotides — version numpy vectorisée.

    rho(XY) = f(XY) / (f(X) · f(Y))

    Algorithme :
      1. `_encode_sequences` → tableau encodé de longueur n
      2. np.bincount sur les bases valides (< 4) → fréquences mononucléotidiques
      3. Masque des paires valides : encoded[i] < 4 ET encoded[i+1] < 4
         (les paires chevauchant un séparateur 'N' sont exclues automatiquement)
      4. Indice de dinucléotide : encoded[i]*4 + encoded[i+1]
      5. np.bincount → f(XY), division vectorisée pour obtenir rho(XY)

    Parameters
    ----------
    sequences : list of str ou list of bytes

    Returns
    -------
    dict : {f'di_{XY}': rho(XY)} pour les 16 dinucléotides
    """
    encoded = _encode_sequences(sequences)
    all_di  = _all_kmers(2)
    valid   = encoded < 4

    if valid.sum() < 2:
        return {f'di_{d}': 1.0 for d in all_di}

    mono_counts = np.bincount(encoded[valid], minlength=4).astype(float)
    valid_pairs = valid[:-1] & valid[1:]
    di_indices  = encoded[:-1][valid_pairs] * 4 + encoded[1:][valid_pairs]
    di_counts   = np.bincount(di_indices, minlength=16).astype(float)

    total_mono, total_di = mono_counts.sum(), di_counts.sum()
    if total_mono == 0 or total_di == 0:
        return {f'di_{d}': 1.0 for d in all_di}

    f_mono = mono_counts / total_mono
    f_di   = di_counts   / total_di

    result = {}
    for i, di in enumerate(all_di):
        x, y  = 'ACGT'.index(di[0]), 'ACGT'.index(di[1])
        denom = f_mono[x] * f_mono[y]
        result[f'di_{di}'] = float(f_di[i] / denom) if denom > 0 else 1.0
    return result



def extract_fastq_features(fastq_path, k=3, chunk_size=200_000):
    """
    Worker autonome : lit un FASTQ par blocs et retourne les 82 features de séquence.

    Traitement par blocs de `chunk_size` reads pour éviter les pics RAM
    sur les gros fichiers (jusqu'à 35 M reads × 400 bp = 14 GB par fichier).
    Chaque bloc libère sa mémoire avant le suivant ; le pic par bloc est ~160 MB
    quelle que soit la taille du fichier.

    Les comptages bincount (k-mers, dinucléotides) sont additifs : on accumule
    les comptes bruts sur tous les blocs et on normalise une seule fois à la fin.

    Pipeline par bloc :
      1. Lecture de `chunk_size` reads (binaire natif, sans BioPython)
      2. `_encode_sequences`  — concaténation + lookup table → uint8
         → del seqs_chunk     — libère ~80 MB
      3. bincount k-mers      — accumulation dans kmer_counts (int64, 64 valeurs)
      4. bincount di-nucl.    — accumulation dans mono_accum / di_accum
         → del encoded        — libère ~80 MB
      5. qualité              — b''.join + frombuffer - 33, sommes cumulées
         → del quals_chunk, q_bytes

    Normalisation finale depuis les comptes totaux → fréquences et rho(XY).

    Parameters
    ----------
    fastq_path : str ou Path
    k : int  — longueur des k-mers (défaut 3 → 64 features)
    chunk_size : int  — nombre de reads par bloc (défaut 200 000, ~160 MB/bloc)

    Returns
    -------
    dict : 82 features  (64 kmer + 16 di + 2 qual) ou None
    """
    all_kmers    = _all_kmers(k)
    all_di       = _all_kmers(2)
    kmer_counts  = np.zeros(4 ** k, dtype=np.int64)
    mono_accum   = np.zeros(4, dtype=np.int64)
    di_accum     = np.zeros(16, dtype=np.int64)
    q20_sum      = 0
    q30_sum      = 0
    total_bases  = 0
    n_reads      = 0

    # Dtype minimal pour les indices k-mers (calculé une seule fois)
    max_idx   = sum(4 * 4 ** (k - 1 - i) for i in range(k))
    idx_dtype = np.uint8 if max_idx <= 255 else np.uint16 if max_idx <= 65535 else np.int32
    powers    = (4 ** np.arange(k - 1, -1, -1)).astype(idx_dtype)

    try:
        with open(fastq_path, 'rb') as fh:
            while True:
                seqs_chunk  = []
                quals_chunk = []
                for _ in range(chunk_size):
                    header = fh.readline()
                    if not header:
                        break
                    seqs_chunk.append(fh.readline().rstrip(b'\n'))
                    fh.readline()                          # ligne +
                    quals_chunk.append(fh.readline().rstrip(b'\n'))

                if not seqs_chunk:
                    break
                n_reads += len(seqs_chunk)

                # --- k-mers + dinucléotides ---
                encoded = _encode_sequences(seqs_chunk)
                del seqs_chunk

                m = len(encoded) - k + 1
                if m > 0:
                    shifts  = [encoded[i:i + m] for i in range(k)]
                    valid   = shifts[0] < 4
                    for s in shifts[1:]:
                        valid &= s < 4
                    indices = np.zeros(m, dtype=idx_dtype)
                    for s, p in zip(shifts, powers):
                        indices += s.astype(idx_dtype) * p
                    vi = indices[valid]
                    if len(vi):
                        kmer_counts += np.bincount(vi, minlength=4 ** k)

                valid_bases = encoded < 4
                if valid_bases.sum() >= 2:
                    mono_accum += np.bincount(encoded[valid_bases],
                                              minlength=4).astype(np.int64)
                    valid_pairs = valid_bases[:-1] & valid_bases[1:]
                    di_idx = (encoded[:-1][valid_pairs].astype(np.uint8) * 4
                              + encoded[1:][valid_pairs].astype(np.uint8))
                    di_accum += np.bincount(di_idx, minlength=16).astype(np.int64)

                del encoded

                # --- qualité ---
                q_bytes = b''.join(quals_chunk)
                del quals_chunk
                q_arr = np.frombuffer(q_bytes, dtype=np.uint8).astype(np.int16) - 33
                del q_bytes
                q20_sum     += int((q_arr >= 20).sum())
                q30_sum     += int((q_arr >= 30).sum())
                total_bases += len(q_arr)

    except Exception:
        return None

    if n_reads == 0:
        return None

    # --- Normalisation k-mers ---
    kmer_total = kmer_counts.sum()
    kmer_f = ({f'kmer_{km}': int(kmer_counts[i]) / kmer_total
               for i, km in enumerate(all_kmers)}
              if kmer_total > 0 else
              {f'kmer_{km}': 0.0 for km in all_kmers})

    # --- Normalisation dinucléotides : rho(XY) = f(XY) / (f(X)·f(Y)) ---
    total_mono = mono_accum.sum()
    total_di   = di_accum.sum()
    if total_mono > 0 and total_di > 0:
        f_mono = mono_accum.astype(float) / total_mono
        f_di   = di_accum.astype(float)  / total_di
        di_f = {}
        for i, di in enumerate(all_di):
            x, y  = 'ACGT'.index(di[0]), 'ACGT'.index(di[1])
            denom = f_mono[x] * f_mono[y]
            di_f[f'di_{di}'] = float(f_di[i] / denom) if denom > 0 else 1.0
    else:
        di_f = {f'di_{d}': 1.0 for d in all_di}

    qual_f = {
        'pct_bases_q20': q20_sum / total_bases if total_bases > 0 else 0.0,
        'pct_bases_q30': q30_sum / total_bases if total_bases > 0 else 0.0,
    }
    return {**kmer_f, **di_f, **qual_f}


# ============================================================
# SECTION 5 : PIPELINE PRINCIPAL
# ============================================================

def build_features(df: pd.DataFrame,
                   pct_a_col: str = 'pct_A',
                   pct_t_col: str = 'pct_T',
                   pct_c_col: str = 'pct_C',
                   pct_g_col: str = 'pct_G',
                   pct_gc_col: str = 'pct_GC',
                   avg_quality_col: str = 'avg_quality',
                   num_reads_col: str = 'num_reads',
                   avg_length_col: str = 'avg_read_length') -> pd.DataFrame:
    """
    Construit le DataFrame de features ML complet.

    Assemble en un seul DataFrame :
    - Fractions nucléotidiques brutes     : pct_A, pct_T, pct_C, pct_G, pct_GC
    - Features de qualité séquentielle    : avg_quality, num_reads, avg_read_length
    - Ratios biologiques (dérivés)        : gc_skew, at_skew, purine_pyrimidine_ratio
    - Complexité séquentielle (dérivée)   : nucleotide_entropy

    Note : pct_GC = pct_G + pct_C est conservé comme feature directe
    car il est biologiquement interprétable et largement utilisé dans
    la littérature microbiome. pct_A n'est pas supprimé car, contrairement
    à pct_GC qui est une somme de deux autres colonnes présentes,
    les quatre fractions individuelles portent chacune une information
    distincte pour les ratios et l'entropie.

    Parameters
    ----------
    df : pd.DataFrame
        Sortie de data_processing.extract_all_fastq_features() mergée avec Train.csv

    Returns
    -------
    pd.DataFrame : features ML, index aligné sur df
    """
    features = pd.DataFrame(index=df.index)

    # -- Fractions brutes (passées telles quelles) --
    for col in [pct_a_col, pct_t_col, pct_c_col, pct_g_col, pct_gc_col]:
        if col in df.columns:
            features[col] = df[col].values

    # -- Qualité séquentielle --
    for col, alias in [(avg_quality_col, 'avg_quality'),
                       (num_reads_col,   'num_reads'),
                       (avg_length_col,  'avg_read_length')]:
        if col in df.columns:
            features[alias] = df[col].values

    # -- Ratios biologiques dérivés --
    features['gc_skew']                = compute_gc_skew(df, pct_g_col, pct_c_col).values
    features['at_skew']                = compute_at_skew(df, pct_a_col, pct_t_col).values
    features['purine_pyrimidine_ratio'] = compute_purine_pyrimidine_ratio(df, pct_a_col, pct_g_col, pct_c_col, pct_t_col).values

    # -- Complexité --
    features['nucleotide_entropy'] = compute_nucleotide_entropy(df, pct_a_col, pct_t_col, pct_c_col, pct_g_col).values

    print(f"✅ {features.shape[1]} features construites :")
    print(f"   Brutes    : pct_A, pct_T, pct_C, pct_G, pct_GC, avg_quality, num_reads, avg_read_length")
    print(f"   Dérivées  : gc_skew, at_skew, purine_pyrimidine_ratio, nucleotide_entropy")

    return features


# ============================================================
# SECTION 6 : SPLIT SANS DATA LEAKAGE
# ============================================================

def split_by_subject(df: pd.DataFrame,
                     subject_col: str = 'SubjectID',
                     target_col: str = 'SampleType',
                     val_size: float = 0.2,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sépare train et validation au niveau des sujets (SubjectID).

    Un même sujet ayant fourni des échantillons de plusieurs sites,
    un split aléatoire par échantillon placerait des données du même
    patient dans les deux partitions. Le modèle mémoriserait alors
    des signatures individuelles plutôt que des patterns généralisables.

    Stratégie : les sujets sont répartis entre train et val (jamais
    les deux). La stratification s'appuie sur le SampleType dominant
    de chaque sujet pour équilibrer la distribution des classes.

    Parameters
    ----------
    df : pd.DataFrame
    subject_col : str
    target_col : str
    val_size : float (default : 0.2)
    random_state : int

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """
    rng = np.random.default_rng(random_state)

    subject_profile = (
        df.groupby(subject_col)[target_col]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={target_col: 'dominant_type'})
    )

    val_subjects = []
    for _, group in subject_profile.groupby('dominant_type'):
        subjects = group[subject_col].values.copy()
        rng.shuffle(subjects)
        n_val = max(1, int(len(subjects) * val_size))
        val_subjects.extend(subjects[:n_val].tolist())

    val_subjects   = set(val_subjects)
    train_subjects = set(subject_profile[subject_col]) - val_subjects

    train_df = df[df[subject_col].isin(train_subjects)].copy()
    val_df   = df[df[subject_col].isin(val_subjects)].copy()

    print("=" * 55)
    print("SPLIT TRAIN / VALIDATION (par SubjectID)")
    print("=" * 55)
    print(f"  Sujets train : {len(train_subjects):>3}  |  Échantillons : {len(train_df):>4}")
    print(f"  Sujets val   : {len(val_subjects):>3}  |  Échantillons : {len(val_df):>4}")
    print()
    for label, data in [("TRAIN", train_df), ("VAL", val_df)]:
        print(f"  Distribution SampleType — {label} :")
        for t, c in data[target_col].value_counts().items():
            print(f"    {t:<10} : {c:>4} ({100*c/len(data):.1f}%)")
    print("=" * 55)

    overlap = set(train_df[subject_col]) & set(val_df[subject_col])
    assert len(overlap) == 0, f"DATA LEAKAGE — sujets en commun : {overlap}"
    print("✅ Vérification anti-leakage : OK")

    return train_df, val_df

