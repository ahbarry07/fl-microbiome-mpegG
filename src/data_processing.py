"""
data_processing.py
==================

Chargement des données, décompression MPEG-G → FASTQ,
extraction des mesures brutes par échantillon, merge et utilitaires I/O.
"""

import os
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from Bio import SeqIO
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================
# SECTION 1 : CHARGEMENT DES DONNÉES
# ============================================

def load_data(data_path: str = '../data/raw', file_name: str = 'Train.csv') -> pd.DataFrame:
    """
    Charge un fichier CSV depuis le dossier spécifié.

    Parameters
    ----------
    data_path : str
        Chemin vers le dossier contenant les données
    file_name : str
        Nom du fichier (ex : 'Train.csv', 'Test.csv', 'Train_Subjects.csv')

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> train_df    = load_data('../data/raw', 'Train.csv')
    >>> test_df     = load_data('../data/raw', 'Test.csv')
    >>> subjects_df = load_data('../data/raw', 'Train_Subjects.csv')
    """
    file_path = Path(data_path) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"{file_name} non trouvé dans {data_path}")
    df = pd.read_csv(file_path)
    print(f"✅ {file_name} chargé : {df.shape}")
    return df


# ============================================
# SECTION 2 : DÉCOMPRESSION MPEG-G
# ============================================

def decode_mgb_to_fastq(mgb_file_path: Path,
                         output_fastq_path: Path,
                         timeout: int = 600) -> bool:
    """
    Décompresse un fichier .mgb en .fastq via Docker Genie.

    Parameters
    ----------
    mgb_file_path : Path
        Fichier source .mgb
    output_fastq_path : Path
        Fichier de sortie .fastq
    timeout : int
        Timeout en secondes (default : 600)

    Returns
    -------
    bool : True si succès, False sinon
    """
    # Dossier parent
    host_dir = str(mgb_file_path.parent.absolute())
    container_dir = "/data"
    
    # Noms de fichiers
    mgb_filename = mgb_file_path.name
    fastq_filename = output_fastq_path.name

    command = [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:{container_dir}",
        "muefab/genie:latest", "run",
        "-f",
        "-i", f"{container_dir}/{mgb_filename}",
        "-o", f"{container_dir}/{fastq_filename}"
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True
        print(f"❌ Erreur pour {mgb_filename}: {result.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout pour {mgb_filename}")
        return False
    except Exception as e:
        print(f"❌ Exception pour {mgb_filename}: {str(e)}")
        return False


def decode_all_mgb_files(mgb_files: List[Path],
                          skip_existing: bool = True,
                          desc: str = "Decompression") -> int:
    """
    Décompresse une liste de fichiers .mgb en .fastq.

    Parameters
    ----------
    mgb_files : list of Path
    skip_existing : bool
        Si True, ignore les fichiers déjà décompressés
    desc : str
        Label de la barre de progression

    Returns
    -------
    int : Nombre de fichiers décompressés avec succès
    """
    success_count = 0

    for mgb_file in tqdm(mgb_files, desc=desc):
        fastq_file = mgb_file.parent / (mgb_file.stem + '.fastq')
        
        # Skip si déjà décompressé
        if skip_existing and fastq_file.exists():
            success_count += 1
            continue
        
        if decode_mgb_to_fastq(mgb_file, fastq_file):
            success_count += 1
    
    print(f"\n✅ {success_count}/{len(mgb_files)} fichiers décompressés avec succès")
    return success_count


# ============================================
# SECTION 3 : EXTRACTION FEATURES FASTQ
# ============================================

def _nucleotide_fractions(sequence: str) -> Dict[str, float]:
    """
    Calcule les fractions et la fréquence GC d'une séquence ADN concaténée.

    Parameters
    ----------
    sequence : str
        Séquence ADN (tous les reads d'un échantillon concaténés)

    Returns
    -------
    dict avec les clés : pct_A, pct_T, pct_C, pct_G, pct_GC
    """
    seq = sequence.upper()
    total = len(seq)
    if total == 0:
        return {'pct_A': 0.0, 'pct_T': 0.0, 'pct_C': 0.0, 'pct_G': 0.0, 'pct_GC': 0.0}

    count_A = seq.count('A')
    count_T = seq.count('T')
    count_C = seq.count('C')
    count_G = seq.count('G')

    return {
        'pct_A':  count_A / total,
        'pct_T':  count_T / total,
        'pct_C':  count_C / total,
        'pct_G':  count_G / total,
        'pct_GC': (count_G + count_C) / total,
    }


def extract_features_from_fastq(fastq_path: Path) -> Optional[Dict]:
    """
    Extrait les mesures brutes d'un fichier FASTQ.

    Mesures extraites :
    - num_reads       : nombre de séquences
    - avg_read_length : longueur moyenne des reads (pb)
    - avg_quality     : qualité Phred moyenne
    - pct_A, pct_T, pct_C, pct_G : fractions nucléotidiques
    - pct_GC          : fraction GC = pct_G + pct_C

    Parameters
    ----------
    fastq_path : Path

    Returns
    -------
    dict ou None si erreur / fichier vide
    """
    if not fastq_path.exists():
        return None

    reads, lengths, qualities = [], [], []

    try:
        # Lire toutes les séquences du FASTQ
        for record in SeqIO.parse(fastq_path, "fastq"):
            reads.append(str(record.seq))
            lengths.append(len(record.seq))
            qualities.append(np.mean(record.letter_annotations['phred_quality']))

        if not reads:
            return None

        # Composition nucléotidique globale (concaténation de tous les reads)
        fractions = _nucleotide_fractions(''.join(reads))

        return {
            'num_reads':       len(reads),
            'avg_read_length': np.mean(lengths),
            'avg_quality':     np.mean(qualities),
            **fractions,
        }

    except Exception as e:
        print(f"❌ Erreur extraction {fastq_path.name}: {e}")
        return None


def extract_all_fastq_features(mgb_files: List[Path],
                                desc: str = "Extraction Features") -> pd.DataFrame:
    """
    Extrait les mesures brutes de tous les fichiers FASTQ correspondant aux .mgb.

    Parameters
    ----------
    mgb_files : list of Path
        Liste des fichiers .mgb (les .fastq correspondants doivent exister)
    desc : str
        Label de la barre de progression

    Returns
    -------
    pd.DataFrame, colonnes :
        filename, num_reads, avg_read_length, avg_quality,
        pct_A, pct_T, pct_C, pct_G, pct_GC
    """
    all_features = []

    for mgb_file in tqdm(mgb_files, desc=desc):
        fastq_file = mgb_file.parent / (mgb_file.stem + '.fastq')

        # Extraire features du FASTQ correspondant
        features = extract_features_from_fastq(fastq_file)

        if features is not None:
            features['filename'] = mgb_file.name
            all_features.append(features)

    df = pd.DataFrame(all_features)

    # Réordonner colonnes
    ordered_cols = ['filename', 'num_reads', 'avg_read_length', 'avg_quality',
                    'pct_A', 'pct_T', 'pct_C', 'pct_G', 'pct_GC']
    df = df[ordered_cols]

    print(f"✅ Features extraites : {df.shape}")
    return df


# ============================================
# SECTION 4 : NETTOYAGE ET MERGE
# ============================================

def merge_with_metadata(fastq_df: pd.DataFrame,
                         train_df: pd.DataFrame,
                         subjects_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Fusionne les features FASTQ avec les métadonnées Train.csv
    et optionnellement Train_Subjects.csv.

    Parameters
    ----------
    fastq_df : pd.DataFrame
        Sortie de extract_all_fastq_features()
    train_df : pd.DataFrame
        Train.csv (colonnes : filename, SampleType, SubjectID, SampleID)
    subjects_df : pd.DataFrame, optionnel
        Train_Subjects.csv (métadonnées patients)

    Returns
    -------
    pd.DataFrame fusionné
    """
    merged = train_df.merge(fastq_df, on='filename', how='inner')
    print(f"✅ Merge Train × FASTQ : {merged.shape}")

    if subjects_df is not None:
        merged = merged.merge(subjects_df, on='SubjectID', how='left')
        print(f"✅ Merge × Subjects   : {merged.shape}")

    return merged


def encode_categorical_variables(df: pd.DataFrame,
                                  categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Encode les variables catégorielles (Gender, Ethnicity, etc.) par label encoding.

    Parameters
    ----------
    df : pd.DataFrame
    categorical_cols : list, optionnel
        Colonnes à encoder (None = auto-détection, hors colonnes ID)

    Returns
    -------
    pd.DataFrame
    """
    df_encoded = df.copy()

    if categorical_cols is None:
        # Exclure les colonnes ID et filename de l'encodage
        exclude = {'filename', 'SubjectID', 'SampleID', 'SampleType'}
        categorical_cols = [
            col for col in df_encoded.select_dtypes(include=['object']).columns
            if col not in exclude
        ]

    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            print(f"   ✓ {col} encodé")

    print(f"✅ {len(categorical_cols)} colonnes encodées")
    return df_encoded


# ============================================
# SECTION 6 : UTILITAIRES I/O
# ============================================

def get_mgb_files(folder_path: str) -> List[Path]:
    """
    Retourne la liste de tous les fichiers .mgb d'un dossier.

    Parameters
    ----------
    folder_path : str

    Returns
    -------
    list of Path
    """
    folder = Path(folder_path)
    mgb_files = list(folder.glob('*.mgb'))
    print(f"📁 {folder} — {len(mgb_files)} fichiers .mgb")
    return mgb_files


def save_processed_data(df: pd.DataFrame,
                         filename: str,
                         output_dir: str = '../data/processed') -> None:
    """
    Sauvegarde un DataFrame en CSV dans le dossier processed.

    Parameters
    ----------
    df : pd.DataFrame
    filename : str
    output_dir : str
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    print(f"✅ Sauvegardé : {filepath} {df.shape}")


