"""
Paramètres d'entraînement et chemins de données.

Les chemins par défaut pointent vers des dossiers relatifs au projet pour la portabilité
(mémoire / déploiement Afrique). Surcharge via variables d'environnement :

- PNEUMONIA_DATA_ROOT : racine des données (contient GOGETA, SPLITS, etc.)
- PNEUMONIA_FAST_RUN : "1" active petits batchs, 1 epoch, sous-échantillonnage optionnel
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Racine du package (dossier parent de config/)."""
    return Path(__file__).resolve().parent.parent


@dataclass
class TrainingConfig:
    """Hyperparamètres et chemins pour l'entraînement CNN (binaire NORMAL vs PNEUMONIA)."""

    # Chemins (défaut : sous-dossiers data/ à la racine du projet)
    data_root: Path = field(default_factory=lambda: _project_root() / "data")
    gogeta_dir: Path | None = None
    splits_dir: Path | None = None
    models_dir: Path | None = None

    batch_size: int = 32
    epochs: int = 20
    img_size: tuple[int, int] = (224, 224)
    learning_rate: float = 1e-3
    random_seed: int = 42

    # Mode debug / exécution rapide (sans exiger un gros dataset)
    fast_run: bool = False
    fast_run_max_train_batches: int | None = 2
    fast_run_max_val_batches: int | None = 1
    fast_run_sample_per_class: int = 4  # pour création dataset mock

    def __post_init__(self) -> None:
        root = Path(self.data_root)
        if self.gogeta_dir is None:
            self.gogeta_dir = root / "GOGETA"
        if self.splits_dir is None:
            self.splits_dir = root / "SPLITS"
        if self.models_dir is None:
            self.models_dir = _project_root() / "models" / "checkpoints"


def load_training_config() -> TrainingConfig:
    """
    Charge la configuration depuis l'environnement.

    PNEUMONIA_DATA_ROOT : chemin absolu ou relatif vers la racine données.
    PNEUMONIA_FAST_RUN : "1", "true", "yes" → active fast_run.
    """
    cfg = TrainingConfig()
    env_root = os.environ.get("PNEUMONIA_DATA_ROOT")
    if env_root:
        cfg.data_root = Path(env_root).expanduser().resolve()
        cfg.gogeta_dir = cfg.data_root / "GOGETA"
        cfg.splits_dir = cfg.data_root / "SPLITS"

    fast = os.environ.get("PNEUMONIA_FAST_RUN", "").lower() in ("1", "true", "yes")
    if fast:
        cfg.fast_run = True
        cfg.epochs = 1
        cfg.batch_size = min(cfg.batch_size, 4)

    return cfg
