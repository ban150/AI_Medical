"""Compléments d'audit en LECTURE SEULE.

(a) Zip Zenodo (jeu nigérian, https://doi.org/10.5281/zenodo.14185822) :
    liste du contenu, README, comptage par classe/dossier — SANS extraction sur disque
    (lecture via zipfile, en mémoire uniquement).
(b) Comparaison pHash entre les images NORMAL de TB_RAHMAN et les images RSNA
    (RSNA est en .dcm : décodage en mémoire uniquement, aucune conversion écrite sur disque).
(c) Détection de dossiers imbriqués (ex: PAUL/chest_xray/chest_xray) et de dossiers
    __MACOSX qui fausseraient les comptages.

N'écrit jamais dans --root. N'écrit que dans --out.

Usage :
    python audit_complements.py --root C:\\Z_Soutenance\\DATA --out audit_out \\
        [--zenodo-zip "C:\\chemin\\vers\\fichier.zip"] [--rsna-sample 5000] [--phash-threshold 5]

Si --zenodo-zip n'est pas fourni, le script cherche un .zip plausible sous --root
et sous le dossier Téléchargements de l'utilisateur.
"""

import argparse
import json
import os
import zipfile
from collections import defaultdict

import numpy as np
import pydicom
from tqdm import tqdm

import audit_lib as lib

IMG_EXT_ZIP = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def log(msg):
    print(msg, flush=True)


# --- (a) Zip Zenodo --------------------------------------------------------

def find_zenodo_zip(root):
    candidates = []
    search_dirs = [root, os.path.join(os.path.expanduser('~'), 'Downloads'),
                    os.path.join(os.path.expanduser('~'), 'Téléchargements')]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for dirpath, _dirnames, filenames in os.walk(d):
            for f in filenames:
                if f.lower().endswith('.zip') and ('zenodo' in f.lower() or 'nigeria' in f.lower()
                                                     or '14185822' in f):
                    candidates.append(os.path.join(dirpath, f))
    return candidates


def audit_zenodo_zip(zip_path, out_dir):
    result = {'zip_path': zip_path}
    if not zip_path or not os.path.isfile(zip_path):
        result['present'] = False
        return result
    result['present'] = True

    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        result['n_entries_total'] = len(names)

        readme_names = [n for n in names if 'readme' in os.path.basename(n).lower()]
        result['readme_files'] = readme_names
        readme_texts = {}
        for rn in readme_names:
            try:
                raw = zf.read(rn)
                readme_texts[rn] = raw.decode('utf-8', errors='replace')
            except Exception as e:
                readme_texts[rn] = f"<erreur lecture: {e}>"
        if readme_texts:
            with open(os.path.join(out_dir, 'zenodo_readme.txt'), 'w', encoding='utf-8') as f:
                for rn, text in readme_texts.items():
                    f.write(f"=== {rn} ===\n{text}\n\n")

        image_entries = [n for n in names if n.lower().endswith(IMG_EXT_ZIP) and not n.endswith('/')]
        result['n_images'] = len(image_entries)

        per_folder_class = defaultdict(int)
        rows = []
        for n in image_entries:
            parent = os.path.dirname(n)
            lower = n.lower()
            if 'normal' in lower:
                classe = 'NORMAL'
            elif 'pneumonia' in lower:
                classe = 'PNEUMONIA'
            elif 'tuberculos' in lower or 'tb' in os.path.basename(lower):
                classe = 'TUBERCULOSIS?'
            else:
                classe = 'INCONNUE'
            per_folder_class[(parent, classe)] += 1
            info = zf.getinfo(n)
            rows.append([n, parent, classe, info.file_size])

        result['par_dossier_classe'] = {f"{k[0]} | {k[1]}": v for k, v in per_folder_class.items()}

        lib.write_csv(os.path.join(out_dir, 'zenodo_contenu.csv'),
                      ['chemin_dans_zip', 'dossier_parent', 'classe_detectee', 'taille_octets'], rows)

    with open(os.path.join(out_dir, 'zenodo_resume.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# --- (b) pHash TB_RAHMAN (NORMAL) vs RSNA -----------------------------------

def load_tb_normal_hashes(root, tb_dirname_candidates=('TB_RAHMAN', 'tb_rahman')):
    tb_dir = None
    for cand in tb_dirname_candidates:
        p = os.path.join(root, cand, 'TB_Chest_Radiography_Database', 'Normal')
        if os.path.isdir(p):
            tb_dir = p
            break
    if tb_dir is None:
        for d in os.listdir(root):
            p = os.path.join(root, d, 'TB_Chest_Radiography_Database', 'Normal')
            if os.path.isdir(p):
                tb_dir = p
                break
    if tb_dir is None:
        return [], [], None

    files = lib.list_images(tb_dir)
    hashes, paths = [], []
    for fname in tqdm(files, desc='  pHash TB_RAHMAN/Normal'):
        path = os.path.join(tb_dir, fname)
        h = lib.phash_from_path(path)
        if h is not None:
            hashes.append(h)
            paths.append(path)
    return hashes, paths, tb_dir


def load_rsna_hashes(root, sample=None):
    rsna_dir = None
    images_dir = None
    for cand in ('RSNA', 'rsna'):
        p1 = os.path.join(root, cand, 'stage_2_train_images')
        p2 = os.path.join(root, 'stage_2_train_images')
        if os.path.isdir(p1):
            images_dir = p1
            rsna_dir = os.path.join(root, cand)
            break
        if os.path.isdir(p2):
            images_dir = p2
            rsna_dir = root
            break
    if images_dir is None:
        return [], [], None

    dcm_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.dcm')]
    if sample is not None and sample < len(dcm_files):
        import random
        random.seed(42)
        dcm_files = random.sample(dcm_files, sample)

    hashes, paths = [], []
    for fname in tqdm(dcm_files, desc='  pHash RSNA (.dcm, en mémoire)'):
        path = os.path.join(images_dir, fname)
        try:
            ds = pydicom.dcmread(path)
            img = ds.pixel_array.astype(np.float32)
            if img.max() - img.min() < 1:
                continue
            gray = (img - img.min()) / (img.max() - img.min()) * 255.0
            h = lib.phash_from_gray(gray)
            hashes.append(h)
            paths.append(path)
        except Exception:
            continue
    return hashes, paths, rsna_dir


def compare_tb_rsna(root, out_dir, rsna_sample, threshold):
    log("Chargement pHash TB_RAHMAN/Normal...")
    tb_hashes, tb_paths, tb_dir = load_tb_normal_hashes(root)
    log(f"  -> {len(tb_hashes)} images hachées (dossier: {tb_dir})")

    log("Chargement pHash RSNA (.dcm, en mémoire uniquement, rien n'est écrit)...")
    rsna_hashes, rsna_paths, rsna_dir = load_rsna_hashes(root, sample=rsna_sample)
    log(f"  -> {len(rsna_hashes)} images hachées (dossier: {rsna_dir})"
        + (f" [échantillon de {rsna_sample}]" if rsna_sample else " [ensemble complet]"))

    result = {'n_tb_normal_hashes': len(tb_hashes), 'n_rsna_hashes': len(rsna_hashes),
              'tb_dir': tb_dir, 'rsna_dir': rsna_dir, 'rsna_sample': rsna_sample}

    if not tb_hashes or not rsna_hashes:
        result['n_quasi_doublons'] = 0
        return result

    tb_arr = np.array(tb_hashes, dtype=np.uint64)
    rows = []
    for ridx, rh in enumerate(tqdm(rsna_hashes, desc='  Comparaison Hamming TB vs RSNA')):
        dists = lib.hamming_batch(rh, tb_arr)
        matches = np.where(dists <= threshold)[0]
        for midx in matches:
            rows.append([rh, tb_hashes[midx], int(dists[midx]),
                         'NORMAL(TB_RAHMAN)', tb_paths[midx], 'tb_rahman',
                         'RSNA(non étiqueté ici)', rsna_paths[ridx], 'rsna'])

    lib.append_csv(os.path.join(out_dir, 'quasi_doublons_phash.csv'),
                   ['hash1', 'hash2', 'hamming', 'classe1', 'chemin1', 'source1', 'classe2', 'chemin2', 'source2'],
                   rows)
    result['n_quasi_doublons'] = len(rows)
    return result


# --- (c) Anomalies de structure ---------------------------------------------

def structure_anomalies(root, out_dir):
    rows = []
    macosx_dirs = lib.find_macosx_dirs(root)
    for d in macosx_dirs:
        rows.append(['__MACOSX', d])

    for d in os.listdir(root):
        paul_like = os.path.join(root, d, 'chest_xray', 'chest_xray')
        if os.path.isdir(paul_like):
            rows.append(['PAUL_nested_chest_xray_chest_xray', paul_like])

    lib.write_csv(os.path.join(out_dir, 'anomalies_structure.csv'), ['type_anomalie', 'chemin'], rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help=r'Racine des données, ex: C:\Z_Soutenance\DATA')
    parser.add_argument('--out', default='audit_out', help='Dossier de sortie (créé si besoin)')
    parser.add_argument('--zenodo-zip', default=None, help='Chemin explicite vers le zip Zenodo si non trouvé automatiquement')
    parser.add_argument('--rsna-sample', type=int, default=None,
                         help='Limite le nombre de DICOM RSNA utilisés pour le pHash (défaut: tous, plus lent)')
    parser.add_argument('--phash-threshold', type=int, default=5, help='Seuil de distance de Hamming (défaut 5)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if not os.path.isdir(args.root):
        log(f"ERREUR : le dossier racine n'existe pas : {args.root}")
        return

    log("=" * 70)
    log("(a) ZIP ZENODO (jeu nigérian)")
    log("=" * 70)
    zip_path = args.zenodo_zip
    if not zip_path:
        candidates = find_zenodo_zip(args.root)
        if candidates:
            zip_path = candidates[0]
            log(f"Zip trouvé automatiquement : {zip_path}")
            if len(candidates) > 1:
                log(f"  (autres candidats ignorés : {candidates[1:]})")
        else:
            log("Aucun zip Zenodo trouvé automatiquement sous --root ni dans Téléchargements.")
            log("Relance avec --zenodo-zip \"chemin_exact.zip\" si tu connais son emplacement.")
    zenodo_result = audit_zenodo_zip(zip_path, args.out)
    log(json.dumps(zenodo_result, indent=2, ensure_ascii=False, default=str)[:3000])

    log("\n" + "=" * 70)
    log("(b) pHash TB_RAHMAN/Normal vs RSNA")
    log("=" * 70)
    tb_rsna_result = compare_tb_rsna(args.root, args.out, args.rsna_sample, args.phash_threshold)
    log(json.dumps(tb_rsna_result, indent=2, ensure_ascii=False, default=str))

    log("\n" + "=" * 70)
    log("(c) Anomalies de structure (__MACOSX, dossiers imbriqués)")
    log("=" * 70)
    anomalies = structure_anomalies(args.root, args.out)
    for a in anomalies:
        log(f"  {a[0]} : {a[1]}")
    if not anomalies:
        log("  Aucune anomalie détectée.")

    log("\n" + "=" * 70)
    log(f"TERMINÉ. Sorties ajoutées/écrites dans : {os.path.abspath(args.out)}")
    log("Fichiers : zenodo_contenu.csv, zenodo_readme.txt, zenodo_resume.json,")
    log("           quasi_doublons_phash.csv (lignes TB vs RSNA ajoutées),")
    log("           anomalies_structure.csv")
    log("=" * 70)


if __name__ == '__main__':
    main()
