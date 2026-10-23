"""
Gestion des accélérateurs : TensorFlow (entraînement actuel) et PyTorch (compatibilité future).

TensorFlow utilise automatiquement le GPU si CUDA/cuDNN sont installés ; on limite la croissance
mémoire pour éviter d'allouer toute la VRAM d'un coup.
"""

from __future__ import annotations

import os


def configure_tensorflow_gpu() -> None:
    """
    Configure TensorFlow pour GPU si disponible (sinon CPU transparent).

    Désactive le GPU si TF_FORCE_GPU_ALLOW_GROWTH ou CUDA_VISIBLE_DEVICES=-1 selon besoin utilisateur.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[device] Avertissement configuration GPU TensorFlow : {e}")


def get_pytorch_device():
    """
    Retourne torch.device('cuda') si PyTorch + CUDA disponibles, sinon cpu.

    Utile si vous migrez le modèle vers PyTorch ; optionnel si torch n'est pas installé.
    """
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        return None


def print_device_summary() -> None:
    """Affiche un résumé lisible pour le mémoire (TF + PyTorch si présent)."""
    import tensorflow as tf

    print("[device] TensorFlow version :", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[device] TensorFlow GPU : {len(gpus)} périphérique(s) détecté(s)")
        for g in gpus:
            print(f"         - {g}")
    else:
        print("[device] TensorFlow : exécution CPU (aucun GPU détecté ou drivers manquants)")

    dev = get_pytorch_device()
    if dev is not None:
        print(f"[device] PyTorch (optionnel) : {dev}")
    else:
        print("[device] PyTorch non installé — ignoré (le pipeline actuel est TensorFlow/Keras).")
