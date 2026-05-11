"""
carbon_tracking.py
==================

Mesure les émissions de CO2 des entraînements centralisé et fédérés
avec CodeCarbon (EmissionsTracker).

Usage depuis le notebook :
    %run ../src/carbon_tracking.py

Sorties :
    results/metrics/carbon_emissions.csv   — tableau comparatif
    emissions_centralized.csv              — détail CodeCarbon centralisé
    emissions_sequential.csv              — détail CodeCarbon séquentiel
    emissions_fedavg.csv                  — détail CodeCarbon FedAvg
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import xgboost as xgb
from codecarbon import EmissionsTracker

from task import load_data, XGBOOST_PARAMS, NUM_CLIENTS

RESULTS_PATH = ROOT / "results" / "metrics"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# ── Chargement des données ──────────────────────────────────────────────────

print("Chargement des données...")
train_df, val_df, test_df, feature_cols, le = load_data()

# Centralisé : train + val ensemble (comme dans notebook 04)
full_train = pd.concat([train_df, val_df], ignore_index=True)
X_train = full_train[feature_cols].values
y_train = le.transform(full_train["SampleType"].values)
X_val   = val_df[feature_cols].values
y_val   = le.transform(val_df["SampleType"].values)

dtrain  = xgb.DMatrix(X_train, label=y_train)
dval    = xgb.DMatrix(X_val,   label=y_val)

CENTRALIZED_ROUNDS = 500   # identique au notebook 04
N_SEMANTIC_ROUNDS  = 20    # identique à la simulation Flower


# ── Helper ──────────────────────────────────────────────────────────────────

def make_tracker(name: str) -> EmissionsTracker:
    return EmissionsTracker(
        project_name=name,
        output_dir=str(RESULTS_PATH),
        output_file=f"emissions_{name}.csv",
        log_level="error",
        save_to_file=True,
        measure_power_secs=1,
    )


# ── 1. Modèle centralisé ────────────────────────────────────────────────────

print("\n" + "="*55)
print("Tracking : Modèle Centralisé (XGBoost 500 rounds)")
print("="*55)

tracker_c = make_tracker("centralized")
tracker_c.start()

bst_central = xgb.train(
    XGBOOST_PARAMS,
    dtrain,
    num_boost_round=CENTRALIZED_ROUNDS,
    verbose_eval=False,
)

emissions_c  = tracker_c.stop()   # kg CO2eq
duration_c   = tracker_c.final_emissions_data.duration          # secondes
energy_c     = tracker_c.final_emissions_data.energy_consumed   # kWh

print(f"  CO2       : {emissions_c*1000:.4f} gCO2eq")
print(f"  Énergie   : {energy_c*1000:.4f} Wh")
print(f"  Durée     : {duration_c:.1f} s")


# ── 2. Modèle fédéré séquentiel ─────────────────────────────────────────────

print("\n" + "="*55)
print("Tracking : Fédéré Séquentiel (Flower, 100 rounds)")
print("="*55)

import task as _task
from flwr.simulation import run_simulation
from server_app import sequential_app, fedavg_app
from client_app import app       as client_app

_task.history.clear()       # évite l'accumulation entre runs

tracker_s = make_tracker("sequential")
tracker_s.start()

run_simulation(
    server_app=sequential_app,
    client_app=client_app,
    num_supernodes=1,
    verbose_logging=False,
)

emissions_s  = tracker_s.stop()
duration_s   = tracker_s.final_emissions_data.duration
energy_s     = tracker_s.final_emissions_data.energy_consumed

print(f"  CO2       : {emissions_s*1000:.4f} gCO2eq")
print(f"  Énergie   : {energy_s*1000:.4f} Wh")
print(f"  Durée     : {duration_s:.1f} s")


# ── 3. Modèle FedAvg Tree Merging ───────────────────────────────────────────

print("\n" + "="*55)
print("Tracking : FedAvg Tree Merging (Flower, 100 rounds)")
print("="*55)


_task.history.clear()       # repart de zéro pour fedavg_metrics.csv

tracker_f = make_tracker("fedavg")
tracker_f.start()

run_simulation(
    server_app=fedavg_app,
    client_app=client_app,
    num_supernodes=1,
    verbose_logging=False,
)

emissions_f  = tracker_f.stop()
duration_f   = tracker_f.final_emissions_data.duration
energy_f     = tracker_f.final_emissions_data.energy_consumed

print(f"  CO2       : {emissions_f*1000:.4f} gCO2eq")
print(f"  Énergie   : {energy_f*1000:.4f} Wh")
print(f"  Durée     : {duration_f:.1f} s")


# ── Tableau comparatif ──────────────────────────────────────────────────────

df_carbon = pd.DataFrame([
    {
        "model":          "Centralisé (XGBoost 500 rounds)",
        "co2_kg":         round(emissions_c, 6),
        "co2_g":          round(emissions_c * 1000, 4),
        "energy_wh":      round(energy_c * 1000, 4),
        "duration_s":     round(duration_c, 1),
    },
    {
        "model":          "Fédéré Séquentiel (100 rounds Flower)",
        "co2_kg":         round(emissions_s, 6),
        "co2_g":          round(emissions_s * 1000, 4),
        "energy_wh":      round(energy_s * 1000, 4),
        "duration_s":     round(duration_s, 1),
    },
    {
        "model":          "FedAvg Tree Merging (100 rounds Flower)",
        "co2_kg":         round(emissions_f, 6),
        "co2_g":          round(emissions_f * 1000, 4),
        "energy_wh":      round(energy_f * 1000, 4),
        "duration_s":     round(duration_f, 1),
    },
])

out_path = RESULTS_PATH / "carbon_emissions.csv"
df_carbon.to_csv(out_path, index=False)

print("\n" + "="*55)
print("RÉSUMÉ DES ÉMISSIONS")
print("="*55)
print(df_carbon.to_string(index=False))
print(f"\nSauvegardé : {out_path}")
