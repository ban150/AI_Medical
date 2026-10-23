"""Configuration centralisée (chemins, hyperparamètres, mode debug)."""

from AI_Medical.config.settings import TrainingConfig, load_training_config

__all__ = ["TrainingConfig", "load_training_config"]
