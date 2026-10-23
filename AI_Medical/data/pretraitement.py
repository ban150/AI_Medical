"""
Prétraitement unique, appliqué à TOUTES les sources (PAUL, RSNA, jeu nigérian, PediCXR).

Choix documentés (voir rapport_gogeta_clean.md) :
  - Lecture en niveaux de gris. DICOM : table de modalité (pente / ordonnée) puis
    inversion si MONOCHROME1. Aucun fenêtrage de TDM : ce sont des radiographies.
  - Normalisation par percentiles : écrêtage à [p0,5 ; p99,5] puis mise à l'échelle
    linéaire vers [0, 255]. Plus robuste que le min-max aux marqueurs et textes incrustés.
  - Redimensionnement direct en TAILLE x TAILLE (512) par interpolation INTER_AREA.
    Pas de bandes noires, qui coderaient la source à travers le rapport largeur/hauteur.
  - PAS de CLAHE, pour aucune source : il amplifie le bruit et les textures propres
    à chaque appareil (risque de raccourci « source = classe ») et il est irréversible.
    Il reste testable plus tard comme variante d'entraînement, à appliquer à la volée.

pHash : 64 bits (DCT 32x32, bloc 8x8 basse fréquence, seuil à la médiane), calculé
sur l'image PRÉTRAITÉE, pour comparer toutes les sources dans le même espace.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

VERSION = "v1-percentile0.5-99.5_area512_sansCLAHE"
TAILLE = 512
P_BAS, P_HAUT = 0.5, 99.5

_N = 32
_DCT = np.array([[np.cos(np.pi * (2 * x + 1) * u / (2 * _N)) for x in range(_N)] for u in range(_N)])


def sha256_fichier(chemin: str | Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def lire_niveaux_de_gris(chemin: str | Path) -> np.ndarray:
    """Retourne l'image brute en float32 (H, W). Lève ValueError si illisible."""
    chemin = str(chemin)
    if chemin.lower().endswith(".dcm"):
        import pydicom
        from pydicom.pixels import apply_modality_lut

        ds = pydicom.dcmread(chemin)
        img = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if img.ndim == 3:  # DICOM couleur : improbable pour une radio, mais géré
            img = img.mean(axis=-1)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = img.max() - img  # MONOCHROME1 : 0 = blanc, il faut inverser
        return img
    brut = np.fromfile(chemin, dtype=np.uint8)  # supporte les chemins non ASCII sous Windows
    img = cv2.imdecode(brut, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"illisible : {chemin}")
    if img.ndim == 3:
        img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def normaliser_redimensionner(img: np.ndarray) -> np.ndarray:
    """Écrêtage par percentiles, mise à l'échelle 0-255, redimensionnement TAILLE x TAILLE (uint8)."""
    bas, haut = np.percentile(img, [P_BAS, P_HAUT])
    if haut - bas < 1e-3:
        raise ValueError("image uniforme")
    img = np.clip((img - bas) / (haut - bas), 0.0, 1.0) * 255.0
    img = cv2.resize(img, (TAILLE, TAILLE), interpolation=cv2.INTER_AREA)
    return np.rint(img).astype(np.uint8)


def pretraiter(chemin: str | Path) -> np.ndarray:
    return normaliser_redimensionner(lire_niveaux_de_gris(chemin))


def phash(img_uint8: np.ndarray) -> int:
    petite = cv2.resize(img_uint8, (_N, _N), interpolation=cv2.INTER_AREA).astype(np.float64)
    d = _DCT @ petite @ _DCT.T
    bas = d[:8, :8].flatten()
    bits = bas > np.median(bas[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def vignette(img_uint8: np.ndarray, cote: int = 64) -> np.ndarray:
    """Vignette centrée-réduite (float32, cote*cote), pour confirmer un quasi-doublon."""
    v = cv2.resize(img_uint8, (cote, cote), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Corrélation de Pearson entre deux vignettes : ≈ 1 pour la même radio, ≤ 0,95 pour deux radios distinctes."""
    return float(np.mean(a * b))


def distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def ecrire_png(img_uint8: np.ndarray, destination: Path) -> None:
    """Écriture atomique : fichier temporaire puis remplacement."""
    ok, tampon = cv2.imencode(".png", img_uint8)
    if not ok:
        raise ValueError(f"encodage PNG impossible : {destination}")
    tmp = destination.with_suffix(".png.tmp")
    tmp.write_bytes(tampon.tobytes())
    tmp.replace(destination)
