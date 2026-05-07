"""
main.py — Point d'entrée de la simulation fédérée Flower.

Lance run_simulation() avec :
  - server_app : SequentialXGBoostStrategy (séquentiel + soft voting)
  - client_app : XGBoostFlowerClient (stateless, charge la partition depuis config)
  - num_supernodes=1 : un seul nœud virtuel, orchestre toutes les partitions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flwr.simulation import run_simulation

from src.server_app import app as server_app
from src.client_app import app as client_app
def main() -> None:
    run_simulation(
        server_app   = server_app,
        client_app   = client_app,
        num_supernodes = 1,  # client stateless — gère toutes les partitions via config
        backend_config = {"client_resources": {"num_cpus": 8, "num_gpus": 0}},
    )


if __name__ == "__main__":
    main()
