"""Fonctions partagées par audit_donnees.py et audit_complements.py.

Lecture seule : aucune fonction ici n'écrit, ne déplace, ne renomme
et ne supprime quoi que ce soit dans l'arborescence auditée.
"""

import csv
import hashlib
import os

import cv2
import numpy as np

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

# --- Hachage exact -----------------------------------------------------

def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                h.update(chunk)
    except (OSError, IOError):
        return None
    return h.hexdigest()


# --- Parcours de dossiers -----------------------------------------------

def list_images(folder):
    """Liste (non récursive) les fichiers images d'un dossier. Vide si le dossier n'existe pas."""
    if not os.path.isdir(folder):
        return []
    out = []
    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(IMG_EXT):
                out.append(entry.name)
    return out


def walk_find(root, dirname_predicate):
    """Retourne la liste des chemins de dossiers sous root dont le nom satisfait dirname_predicate."""
    found = []
    for dirpath, dirnames, _filenames in os.walk(root):
        for d in dirnames:
            if dirname_predicate(d):
                found.append(os.path.join(dirpath, d))
    return found


def find_macosx_dirs(root):
    return walk_find(root, lambda d: d == '__MACOSX')


def count_recursive(root, extensions=IMG_EXT):
    """Compte récursivement les fichiers par extension sous root. Retourne dict ext->count et total."""
    counts = {}
    total = 0
    if not os.path.isdir(root):
        return counts, total
    for _dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in extensions:
                counts[ext] = counts.get(ext, 0) + 1
                total += 1
    return counts, total


# --- pHash (DCT, 64 bits) -----------------------------------------------

def phash_from_gray(gray, hash_size=8, highfreq_factor=4):
    """pHash classique par DCT. gray: image 2D (numpy), niveaux de gris, n'importe quelle taille/dtype numérique."""
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    bits = (dct_low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def phash_from_path(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return phash_from_gray(img)


_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def hamming_batch(query_hash, hash_array_uint64):
    """Distance de Hamming entre un hash (int python, 64 bits) et un tableau numpy uint64 de hashes."""
    xored = np.bitwise_xor(hash_array_uint64, np.uint64(query_hash))
    bytes_view = xored.view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_TABLE[bytes_view].sum(axis=1)


def band_keys(hash_int, n_bands=4, band_bits=16):
    """Découpe un hash 64 bits en n_bands bandes pour un LSH (recherche de quasi-doublons par bandes)."""
    mask = (1 << band_bits) - 1
    return tuple((hash_int >> (band_bits * i)) & mask for i in range(n_bands))


# --- CSV helpers ---------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def append_csv(path, header, rows):
    exists = os.path.isfile(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerows(rows)


def detect_source_from_filename(filename):
    """Devine la source d'une image GOGETA à partir du préfixe de nom de fichier (convention test_1.py)."""
    name = filename.lower()
    if name.startswith('kermany_'):
        return 'kermany(PAUL)'
    if name.startswith('rsna_'):
        return 'rsna'
    if name.startswith('tb_'):
        return 'tb_rahman'
    return 'inconnu(probable Fusion.py sans préfixe)'
