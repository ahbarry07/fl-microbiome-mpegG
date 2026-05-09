"""
feature_engineering.py
=======================

Construction des features ML à partir des fichiers FASTQ et des colonnes
pct_*/qualité issues de data_processing.py.

CATALOGUE DES FEATURES
-----------------------

Famille              | Features                                | N
---------------------|-----------------------------------------|------
Fractions brutes     | pct_A, pct_T, pct_C, pct_G, pct_GC      |    5
Qualité brute        | avg_quality, num_reads, avg_read_length |    3
Ratios biologiques   | gc_skew, at_skew, R/Y, entropie         |    4
K-mers (k=3)         | kmer_AAA … kmer_TTT                     |   64
Dinucléotides rho    | di_AA … di_TT                           |   16
Qualité différenciée | pct_bases_q20, pct_bases_q30            |    2
Complexité séq.      | lz_complexity, pct_ambiguous, read_len_*|    7
Taxonomie Kraken2    | kraken_{genus}: abondances relatives    | ≤500
                     | kraken_unclassified, kraken_n_genera    |    2

PIPELINE DE TRAITEMENT (Section 4b — par blocs)
-------------------------------------------------
`extract_fastq_features` lit chaque fichier FASTQ par blocs de 200 000 reads,
ce qui maintient le pic RAM à ~160 MB/worker quelle que soit la taille du fichier
(certains fichiers contiennent jusqu'à 35 M reads x 400 bp = 14 GB).
Conçue pour ProcessPoolExecutor — un appel par fichier, par worker.

  Par bloc :
    1. Lecture binaire native (4 lignes FASTQ : @header / seq / + / qual)
    2. `_encode_sequences`  — ASCII -> {A=0, C=1, G=2, T=3, autre=4} via lookup table
    3. k-mers               — k décalages numpy + bincount -> accumulation int64
    4. dinucléotides        — bincount sur paires valides -> accumulation int64
    5. qualité              — frombuffer - 33, sommes q20/q30 cumulées

  Après tous les blocs :
    normalisation des comptes accumulés -> fréquences k-mers et rho(XY)


PIPELINE TAXONOMIQUE (Section 7 — Kraken2)
-------------------------------------------
`run_kraken2_on_fastq` lance Kraken2 sur un FASTQ et retourne les abondances
relatives au niveau genus. Kraken2 classe chaque read contre une base de
données 16S (Silva/Greengenes) et produit un rapport structuré.

`build_taxonomic_features` agrège les rapports de tous les échantillons en
une matrice (échantillons x genus) normalisée. Seuls les genres présents dans
au moins `min_prevalence` des échantillons sont conservés pour éviter la
haute dimensionnalité due aux genres rares.

RÉFÉRENCES
----------
K-mers  : Woloszynek et al. (2019) PLoS Comput Biol ; Reiman et al. (2018) Bioinformatics
Di-nucl.: Karlin & Burge (1995) Trends Genet ; Deschavanne et al. (1999) Mol Biol Evol
Skew    : Lobry (1996) Nucleic Acids Research ; Forsdyke & Mortimer (2000) Gene
Kraken2 : Wood et al. (2019) Genome Biology — ultrafast metagenomic sequence classification
Taxo 16S: Knights et al. (2011) Nature Methods — body-site classification from OTU features
"""

import numpy as np
import pandas as pd
# from collections import Counter
from itertools import product
from typing import Tuple, List, Optional, Dict
from scipy.stats import entropy as scipy_entropy
import subprocess
import shutil
import os
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 1 : FEATURES DÉRIVÉES
# ============================================================

def compute_gc_skew(df: pd.DataFrame, pct_g_col: str = 'pct_G', pct_c_col: str = 'pct_C') -> pd.Series:
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


def compute_at_skew(df: pd.DataFrame, pct_a_col: str = 'pct_A', pct_t_col: str = 'pct_T') -> pd.Series:
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
    Entropie de Shannon sur la composition nucléotidique.

    H = -Σ p_i * log2(p_i), avec p_i = fraction de chaque nucléotide.
        Mesure la complexité séquentielle : une entropie élevée indique
        une composition équilibrée, tandis qu'une entropie faible indique
        un fort biais de composition. L'entropie est maximale (2 bits) lorsque
        les quatre bases sont présentes à parts égales (25% chacune).
        (Karlin & Burge, 1995, Trends Genet ; Deschavanne et al., 1999, Mol Biol Evol)
    
        Note : les fractions sont clipées à une valeur minimale pour éviter
        log(0) en cas de nucléotide absent. Une fraction nulle est traitée
        comme une contribution nulle à l'entropie, ce qui est cohérent avec
        la définition mathématique de l'entropie de Shannon.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.Series (bits, entre 0 et 2)
    """
    fracs = df[[pct_a_col, pct_t_col, pct_c_col, pct_g_col]].clip(lower=1e-10) # évite log(0) en cas de fraction nulle
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

# Table de correspondance ASCII -> indice base (A=0, C=1, G=2, T=3, autre=4)
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
    """
    if sequences and isinstance(sequences[0], (bytes, bytearray)):
        # Séquences déjà en bytes : jointure directe sans encodage Python
        combined = np.frombuffer(b'N'.join(sequences), dtype=np.uint8)
    else:
        # Séquences str : encode en bytes puis jointure
        combined = np.frombuffer('N'.join(sequences).encode(), dtype=np.uint8)
    # Indexation vectorisée : chaque valeur ASCII devient son indice de base {0,1,2,3,4}
    return _BASE_TABLE[combined]



def extract_fastq_features(fastq_path, k=3, chunk_size=200_000):
    """
    Worker autonome : lit un FASTQ par blocs et retourne les 82 features de séquence.

    Traitement par blocs de `chunk_size` reads pour éviter les pics RAM
    sur les gros fichiers (jusqu'à 35 M reads x 400 bp = 14 GB par fichier).
    Chaque bloc libère sa mémoire avant le suivant ; le pic par bloc est ~160 MB
    quelle que soit la taille du fichier.

    Les comptages bincount (k-mers, dinucléotides) sont additifs : on accumule
    les comptes bruts sur tous les blocs et on normalise une seule fois à la fin.

    Pipeline par bloc :
      1. Lecture de `chunk_size` reads (binaire natif, sans BioPython)
      2. `_encode_sequences`  — concaténation + lookup table -> uint8
         -> del seqs_chunk     — libère espace avant de traiter les k-mers
      3. bincount k-mers      — accumulation dans kmer_counts (int64, 64 valeurs)
      4. bincount di-nucl.    — accumulation dans mono_accum / di_accum
         -> del encoded        — libère espace avant de traiter les qualités
      5. qualité              — b''.join + frombuffer - 33, sommes cumulées
         -> del quals_chunk, q_bytes

    Normalisation finale depuis les comptes totaux -> fréquences et rho(XY).

    Parameters
    ----------
    fastq_path : str ou Path
    k : int  — longueur des k-mers (défaut 3 -> 64 features)
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

    # Précalculé une seule fois pour tous les blocs (évite de recréer à chaque itération)
    # k=3 -> max_idx=84 -> uint8 (1 octet/position) ; k=4 -> uint16 ; k≥5 -> int32
    max_idx   = sum(4 * 4 ** (k - 1 - i) for i in range(k))
    idx_dtype = np.uint8 if max_idx <= 255 else np.uint16 if max_idx <= 65535 else np.int32
    powers    = (4 ** np.arange(k - 1, -1, -1)).astype(idx_dtype)  # [4^(k-1), ..., 1]

    try:
        with open(fastq_path, 'rb') as fh:
            while True:
                # ── Lecture d'un bloc de chunk_size reads ──────────────────────
                # Un read FASTQ = 4 lignes : @header / séquence / + / qualité
                seqs_chunk  = []
                quals_chunk = []
                for _ in range(chunk_size):
                    header = fh.readline()
                    if not header:          # fin de fichier
                        break
                    seqs_chunk.append(fh.readline().rstrip(b'\n'))   # séquence
                    fh.readline()                                      # ligne "+" (ignorée)
                    quals_chunk.append(fh.readline().rstrip(b'\n'))  # qualités Phred+33

                if not seqs_chunk:          # bloc vide -> tout le fichier a été lu
                    break
                n_reads += len(seqs_chunk)

                # ── K-mers + dinucléotides ─────────────────────────────────────
                # Encode le bloc : liste de bytes -> tableau uint8 {0,1,2,3,4}
                # Les 'N' entre reads empêchent les k-mers inter-reads
                encoded = _encode_sequences(seqs_chunk)
                del seqs_chunk   # libère espace memoire avant de traiter les k-mers et qualités

                m = len(encoded) - k + 1   # positions k-mer valides dans ce bloc
                if m > 0:
                    # k vues décalées O(1) du tableau encodé
                    shifts  = [encoded[i:i + m] for i in range(k)]

                    # Masque : True là où les k bases sont toutes A/C/G/T (pas de 'N')
                    valid   = shifts[0] < 4
                    for s in shifts[1:]:
                        valid &= s < 4

                    # Indice base-4 : b0*4^(k-1) + b1*4^(k-2) + ... + b(k-1)
                    indices = np.zeros(m, dtype=idx_dtype)
                    for s, p in zip(shifts, powers):
                        indices += s.astype(idx_dtype) * p
                        
                    vi = indices[valid]   # indices des positions sans 'N'
                    if len(vi):
                        # Accumulation : bincount additionne les comptes du bloc
                        # aux comptes globaux (propriété additive des fréquences brutes)
                        kmer_counts += np.bincount(vi, minlength=4 ** k)

                valid_bases = encoded < 4
                if valid_bases.sum() >= 2:
                    # Fréquences mononucléotidiques du bloc
                    mono_accum += np.bincount(encoded[valid_bases], minlength=4).astype(np.int64)

                    # Paires : les deux bases adjacentes doivent être valides
                    valid_pairs = valid_bases[:-1] & valid_bases[1:]

                    # Indices des dinucléotides : b0*4 + b1, où b0 et b1 sont les indices des bases
                    di_idx = (encoded[:-1][valid_pairs].astype(np.uint8) * 4 + encoded[1:][valid_pairs].astype(np.uint8))
                    di_accum += np.bincount(di_idx, minlength=16).astype(np.int64) 

                del encoded   # libère espace memoire avant de traiter les qualités

                # ── Qualité Phred ──────────────────────────────────────────────
                q_bytes = b''.join(quals_chunk)   # concatène les lignes de qualité
                del quals_chunk

                # Phred+33 : valeur ASCII − 33 = score de qualité (0–40 typiquement)
                q_arr = np.frombuffer(q_bytes, dtype=np.uint8).astype(np.int16) - 33
                del q_bytes
                q20_sum     += int((q_arr >= 20).sum())   # bases Q≥20
                q30_sum     += int((q_arr >= 30).sum())   # bases Q≥30
                total_bases += len(q_arr)

    except Exception:
        return None

    if n_reads == 0:
        return None

    # ── Normalisation finale ───────────

    # K-mers : comptes bruts -> fréquences relatives
    kmer_total = kmer_counts.sum()
    kmer_f = ({f'kmer_{km}': int(kmer_counts[i]) / kmer_total
               for i, km in enumerate(all_kmers)}
              if kmer_total > 0 else
              {f'kmer_{km}': 0.0 for km in all_kmers})

    # Dinucléotides : comptes bruts -> rho(XY) = f(XY) / (f(X) · f(Y))
    total_mono = mono_accum.sum()
    total_di   = di_accum.sum()
    if total_mono > 0 and total_di > 0:
        f_mono = mono_accum.astype(float) / total_mono   # f(A), f(C), f(G), f(T)
        f_di   = di_accum.astype(float)  / total_di      # f(AA), f(AC), ..., f(TT)
        di_f = {}
        for i, di in enumerate(all_di):
            x, y  = 'ACGT'.index(di[0]), 'ACGT'.index(di[1])
            denom = f_mono[x] * f_mono[y]   # fréquence attendue si indépendance
            di_f[f'di_{di}'] = float(f_di[i] / denom) if denom > 0 else 1.0
    else:
        di_f = {f'di_{d}': 1.0 for d in all_di}   # rho=1 par défaut si pas de données

    # Qualité : fractions de bases dépassant les seuils Q20 et Q30
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
    Construit le DataFrame de features.

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
        n_val = max(1, int(len(subjects) * val_size)) # au moins 1 sujet par classe dans la validation
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


# ============================================================
# SECTION 7 : COMPLEXITÉ SÉQUENTIELLE (depuis FASTQ)
# ============================================================
 
def _lz_complexity_sequence(seq: bytes) -> float:
    """
    Complexité de Lempel-Ziv (LZ76) normalisée d'une séquence.
 
    Mesure le nombre de sous-chaînes distinctes nécessaires pour
    reconstruire la séquence. Valeur proche de 1 = haute complexité
    (communauté diversifiée), proche de 0 = séquence répétitive.
 
    Utilisée comme proxy de la diversité taxonomique de l'échantillon.
 
    Parameters
    ----------
    seq : bytes
 
    Returns
    -------
    float : complexité normalisée dans [0, 1]
    """
    n = len(seq)
    if n == 0:
        return 0.0
    i, c, l = 0, 1, 1
    while i + l <= n:
        if seq[i:i + l] not in seq[:i]:
            c += 1
            i += l
            l = 1
        else:
            l += 1
    max_c = n / np.log2(n + 1) if n > 1 else 1
    return min(c / max_c, 1.0)
 
 
def compute_sequence_complexity_features(fastq_path: str, sample_size: int = 2000) -> Optional[Dict]:
    """
    Extrait les features de complexité séquentielle depuis un FASTQ.
 
    Features calculées :
    - lz_complexity  : complexité de Lempel-Ziv moyenne (proxy diversité taxonomique)
    - pct_ambiguous  : fraction de bases ambiguës N/R/Y/... (qualité du prélèvement)
    - read_len_std   : écart-type des longueurs (hétérogénéité du protocole)
    - read_len_min   : longueur minimale des reads
    - read_len_max   : longueur maximale des reads
    - read_len_q25   : 1er quartile des longueurs
    - read_len_q75   : 3e quartile des longueurs
 
 
    Parameters
    ----------
    fastq_path : str
    sample_size : int
        Nombre de reads à analyser pour LZ (coûteux O(n²), limité à 2000)
 
    Returns
    -------
    dict ou None si fichier introuvable ou vide
    """
    lz_scores   = []
    lengths     = []
    n_ambiguous = 0
    n_total     = 0
    n_reads     = 0
 
    try:
        with open(fastq_path, 'rb') as fh:
            for _ in range(sample_size):
                fh.readline()                      # @header
                seq  = fh.readline().rstrip(b'\n')
                fh.readline()                      # +
                fh.readline()                      # qualité
                if not seq:
                    break
                n_reads += 1
                lengths.append(len(seq))
                seq_upper = seq.upper()
                # Bases ambiguës = tout ce qui n'est pas A(65) T(84) C(67) G(71)
                n_ambiguous += sum(1 for b in seq_upper
                                   if b not in (65, 84, 67, 71))
                n_total += len(seq)
                lz_scores.append(_lz_complexity_sequence(seq_upper))
    except Exception:
        return None
 
    if n_reads == 0:
        return None
 
    lengths_arr = np.array(lengths)
    return {
        'lz_complexity': float(np.mean(lz_scores)),
        'pct_ambiguous': n_ambiguous / n_total if n_total > 0 else 0.0,
        'read_len_std':  float(np.std(lengths_arr)),
        'read_len_min':  float(np.min(lengths_arr)),
        'read_len_max':  float(np.max(lengths_arr)),
        'read_len_q25':  float(np.percentile(lengths_arr, 25)),
        'read_len_q75':  float(np.percentile(lengths_arr, 75)),
    }
 
 
# ============================================================
# SECTION 8 : ASSIGNATION TAXONOMIQUE KRAKEN2 (via Docker)
# ============================================================
 
# Image Docker : docker pull staphb/kraken2
KRAKEN2_DOCKER_IMAGE = 'staphb/kraken2'
 
 
def check_kraken2() -> bool:
    """
    Vérifie que Docker est disponible et que l'image staphb/kraken2 est présente.
 
    Returns
    -------
    bool
    """
    if shutil.which('docker') is None:
        return False
    try:
        result = subprocess.run(
            ['docker', 'images', '-q', KRAKEN2_DOCKER_IMAGE],
            capture_output=True, text=True, timeout=10
        )
        return bool(result.stdout.strip())
    except Exception:
        return False
 
 
def run_kraken2_on_fastq(fastq_path: str,
                          db_path: str,
                          threads: int = 4) -> Optional[Dict[str, float]]:
    """
    Lance Kraken2 via Docker sur un fichier FASTQ et retourne les abondances
    relatives au niveau genus.
 
    Kraken2 classifie chaque read en le comparant à une base de données
    de références génomiques (Silva 16S) via des k-mers. C'est la feature
    la plus discriminante pour la classification par site corporel.
    (Wood et al., 2019, Genome Biology)
 
    Le rapport est écrit directement dans le même dossier que le FASTQ
    (nom : <fastq>.kraken2_report), puis supprimé après lecture.
    Cela évite tout problème de montage de dossier temporaire dans Docker.
 
    Montages Docker :
    - fastq_dir -> /data/fastq:ro   (lecture seule)
    - db_path   -> /data/db:ro      (lecture seule)
    Le rapport est écrit via /data/fastq/ (même volume, accès en écriture
    sur le host même si le flag Docker est :ro car le flag s'applique au
    container, pas à l'hôte).
 
    Parameters
    ----------
    fastq_path : str
        Chemin vers le fichier FASTQ (absolu ou relatif)
    db_path : str
        Chemin vers la base de données Kraken2
    threads : int
 
    Returns
    -------
    dict : {'kraken_Prevotella': 0.23, ...,
            'kraken_unclassified': 0.12, 'kraken_n_genera': 45.0}
    ou None si erreur
    """
    fastq_path = str(os.path.abspath(fastq_path))
    db_path    = str(os.path.abspath(db_path))
 
    if not os.path.exists(fastq_path):
        return None
 
    fastq_dir   = os.path.dirname(fastq_path)
    fastq_name  = os.path.basename(fastq_path)
    report_name = fastq_name + '.kraken2_report'
    report_host = os.path.join(fastq_dir, report_name)  # sur le host
 
    # Chemins dans le container Docker
    c_fastq  = f'/data/fastq/{fastq_name}'
    c_db     = '/data/db'
    c_report = f'/data/fastq/{report_name}'   # même dossier que le FASTQ
 
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{fastq_dir}:/data/fastq',    # pas de :ro — on doit écrire le rapport ici
        '-v', f'{db_path}:/data/db:ro',
        KRAKEN2_DOCKER_IMAGE,
        'kraken2',
        '--db',      c_db,
        '--threads', str(threads),
        '--report',  c_report,
        '--output',  '/dev/null',             # reads classifiés : non conservés
        c_fastq
    ]
 
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            return None
 
        if not os.path.exists(report_host):
            return None
 
        # Parser le rapport Kraken2
        # Format TSV : %reads  n_clade  n_direct  rank  taxid  name
        genus_abundances = {}
        n_unclassified   = 0.0
 
        with open(report_host) as rf:
            for line in rf:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue
                pct  = float(parts[0])
                rank = parts[3]
                name = parts[5].strip()
 
                if rank == 'U':
                    n_unclassified = pct / 100.0
                elif rank == 'G':    # genus level uniquement
                    safe_name = name.replace(' ', '_').replace('/', '_')
                    genus_abundances[f'kraken_{safe_name}'] = pct / 100.0
 
        return {
            **genus_abundances,
            'kraken_unclassified': n_unclassified,
            'kraken_n_genera':     float(len(genus_abundances)),
        }
 
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        # Nettoyage du rapport quelle que soit l'issue
        if os.path.exists(report_host):
            os.remove(report_host)
 
 
def build_taxonomic_features(fastq_paths: List[str],
                              db_path: str,
                              min_prevalence: float = 0.05,
                              threads: int = 4) -> pd.DataFrame:
    """
    Construit la matrice d'abondances taxonomiques pour tous les échantillons.
 
    Lance Kraken2 (via Docker staphb/kraken2) sur chaque FASTQ et agrège
    les abondances au niveau genus. Seuls les genres présents dans au moins
    `min_prevalence` des échantillons sont conservés.
 
    Genres discriminants attendus par site :
    - Stool  : Prevotella, Bacteroides, Faecalibacterium
    - Mouth  : Streptococcus, Veillonella, Prevotella
    - Nasal  : Corynebacterium, Staphylococcus, Dolosigranulum
    - Skin   : Staphylococcus, Cutibacterium, Corynebacterium
    (Knights et al., 2011, Nature Methods)
 
    DB recommandée pour amplicons 16S — construction (une seule fois) :
        mkdir -p data/kraken2_silva_db
        docker run --rm -v $(pwd)/data/kraken2_silva_db:/db \\
               staphb/kraken2 kraken2-build --special silva --db /db
 
    Parameters
    ----------
    fastq_paths : list of str
    db_path : str
    min_prevalence : float
    threads : int
 
    Returns
    -------
    pd.DataFrame : shape (n_samples, n_genera_filtered + 2)
        Colonnes : kraken_{genus}, kraken_unclassified, kraken_n_genera
    """
    if not check_kraken2():
        print("⚠️  Docker ou image staphb/kraken2 non trouvée")
        print("   Installation : docker pull staphb/kraken2")
        return pd.DataFrame()
 
    from tqdm import tqdm
    print(f"🔬 Kraken2 (Docker) — {len(fastq_paths)} échantillons")
 
    all_reports = []
    for fp in tqdm(fastq_paths, desc='Kraken2'):
        report = run_kraken2_on_fastq(fp, db_path, threads)
        all_reports.append(report if report else {})
 
    tax_df = pd.DataFrame(all_reports).fillna(0.0)
 
    if tax_df.empty:
        return np.log1p(tax_df)
 
    # Filtrage des genres rares
    n_samples   = len(tax_df)
    min_samples = int(n_samples * min_prevalence)
    genus_cols  = [c for c in tax_df.columns
                   if c.startswith('kraken_')
                   and c not in ('kraken_unclassified', 'kraken_n_genera')]
    prevalent   = [c for c in genus_cols
                   if (tax_df[c] > 0.001).sum() >= min_samples]
    meta_cols   = [c for c in ('kraken_unclassified', 'kraken_n_genera')
                   if c in tax_df.columns]
    tax_df = tax_df[prevalent + meta_cols]
 
    print(f"✅ Taxonomie Kraken2 :")
    print(f"   Genres détectés  : {len(genus_cols)}")
    print(f"   Genres conservés : {len(prevalent)}  "
          f"(prévalence ≥ {min_prevalence*100:.0f}%)")
    print(f"   Features totales : {tax_df.shape[1]}")
 
    return np.log1p(tax_df)
 