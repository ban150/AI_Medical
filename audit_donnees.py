"""Audit en LECTURE SEULE du dossier de données du mémoire (PAUL, RSNA, TB_RAHMAN, GOGETA, SPLITS).

N'écrit jamais dans --root. N'écrit que dans --out (créé si besoin).
Ne supprime, ne déplace, ne renomme et ne modifie aucun fichier source.

Usage :
    python audit_donnees.py --root C:\\Z_Soutenance\\DATA --out audit_out

Peut prendre plusieurs minutes (RSNA ~26 000 fichiers DICOM, GOGETA potentiellement
~35 000+ images). Peut être lancé en arrière-plan.
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd
import pydicom
from tqdm import tqdm

import audit_lib as lib

EXPECTED_TOPDIRS = {
    'PAUL': ['paul'],
    'RSNA': ['rsna'],
    'TB_RAHMAN': ['tb_rahman', 'tb-rahman', 'tbrahman'],
    'GOGETA': ['gogeta'],
    'SPLITS': ['splits'],
}

EXPECTED_COUNTS = {
    'PAUL_total_min': 5856,
    'PAUL_total_max': 5863,
    'RSNA_total': 26684,
    'RSNA_pneumonia': 6012,
    'TB_RAHMAN_total': 4200,
    'TB_RAHMAN_tuberculosis': 700,
    'TB_RAHMAN_normal': 3500,
}


def log(msg):
    print(msg, flush=True)


# --- Étape 0 : accès et présence ----------------------------------------

def resolve_topdirs(root):
    """Associe chaque clé attendue au nom réel trouvé sous root (insensible à la casse)."""
    if not os.path.isdir(root):
        return None, []
    actual = {d.lower(): d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))}
    resolved = {}
    for key, aliases in EXPECTED_TOPDIRS.items():
        found_name = None
        for alias in aliases:
            if alias in actual:
                found_name = actual[alias]
                break
        resolved[key] = found_name
    known_lower = {a for aliases in EXPECTED_TOPDIRS.values() for a in aliases}
    unexpected = [actual[k] for k in actual if k not in known_lower]
    return resolved, unexpected


def presence_report(root, resolved, out_dir):
    rows = []
    for key, found_name in resolved.items():
        if found_name is None:
            rows.append([key, '', False, '', 0])
            continue
        path = os.path.join(root, found_name)
        _counts, total = lib.count_recursive(path)
        rows.append([key, found_name, True, path, total])
    lib.write_csv(os.path.join(out_dir, 'presence.csv'),
                  ['dossier_attendu', 'nom_trouve', 'existe', 'chemin', 'nb_fichiers_images_recursif'], rows)
    return rows


# --- PAUL ----------------------------------------------------------------

def audit_paul(root_path):
    """root_path = chemin vers le dossier PAUL (ou équivalent)."""
    result = {'present': os.path.isdir(root_path)}
    if not result['present']:
        return result
    nested = os.path.join(root_path, 'chest_xray', 'chest_xray')
    result['nested_chest_xray_chest_xray'] = os.path.isdir(nested)
    macosx = lib.find_macosx_dirs(root_path)
    result['macosx_dirs'] = macosx
    counts = {}
    base = os.path.join(root_path, 'chest_xray')
    for split in ('train', 'val', 'test'):
        for cls in ('NORMAL', 'PNEUMONIA'):
            folder = os.path.join(base, split, cls)
            counts[f'{split}/{cls}'] = len(lib.list_images(folder))
    result['counts'] = counts
    result['total'] = sum(counts.values())
    return result


# --- RSNA ------------------------------------------------------------------

def audit_rsna(root_path):
    result = {'present': os.path.isdir(root_path)}
    if not result['present']:
        return result
    macosx = lib.find_macosx_dirs(root_path)
    result['macosx_dirs'] = macosx

    candidates_images = [
        os.path.join(root_path, 'stage_2_train_images'),
        os.path.join(root_path, 'RSNA', 'stage_2_train_images'),
    ]
    candidates_csv = [
        os.path.join(root_path, 'stage_2_train_labels.csv'),
        os.path.join(root_path, 'RSNA', 'stage_2_train_labels.csv'),
    ]
    images_dir = next((p for p in candidates_images if os.path.isdir(p)), None)
    labels_csv = next((p for p in candidates_csv if os.path.isfile(p)), None)

    result['images_dir'] = images_dir
    result['labels_csv'] = labels_csv

    n_dcm = 0
    if images_dir:
        _counts, n_dcm = lib.count_recursive(images_dir, extensions=('.dcm',))
    result['n_dcm_files'] = n_dcm

    if labels_csv:
        df = pd.read_csv(labels_csv)
        df_grouped = df.groupby('patientId')['Target'].max().reset_index()
        result['n_patients_csv'] = len(df_grouped)
        result['n_normal_csv'] = int((df_grouped['Target'] == 0).sum())
        result['n_pneumonia_csv'] = int((df_grouped['Target'] == 1).sum())
        if images_dir:
            missing = 0
            sample_check = df_grouped['patientId'].tolist()
            for pid in sample_check:
                if not os.path.isfile(os.path.join(images_dir, pid + '.dcm')):
                    missing += 1
            result['patients_sans_dcm'] = missing
    return result


# --- TB_RAHMAN --------------------------------------------------------------

def audit_tb_rahman(root_path):
    result = {'present': os.path.isdir(root_path)}
    if not result['present']:
        return result
    macosx = lib.find_macosx_dirs(root_path)
    result['macosx_dirs'] = macosx
    base = os.path.join(root_path, 'TB_Chest_Radiography_Database')
    result['base_found'] = os.path.isdir(base)
    counts = {}
    for cls, folder_name in (('NORMAL', 'Normal'), ('TUBERCULOSIS', 'Tuberculosis')):
        folder = os.path.join(base, folder_name)
        counts[cls] = len(lib.list_images(folder))
    result['counts'] = counts
    result['total'] = sum(counts.values())
    # Rahman distribue souvent des métadonnées xlsx ; simple présence, pas de lecture du contenu
    result['metadata_files'] = [f for f in os.listdir(base) if f.lower().endswith(('.xlsx', '.csv'))] if result['base_found'] else []
    return result


# --- GOGETA ------------------------------------------------------------------

def audit_gogeta(root_path, out_dir, do_phash, phash_threshold):
    result = {'present': os.path.isdir(root_path)}
    if not result['present']:
        return result

    files_by_class = {}
    for cls in ('NORMAL', 'PNEUMONIA'):
        folder = os.path.join(root_path, cls)
        files_by_class[cls] = lib.list_images(folder)
    result['counts'] = {cls: len(fs) for cls, fs in files_by_class.items()}
    result['total'] = sum(result['counts'].values())

    # Hachage SHA-256 de toutes les images GOGETA, avec classe + source détectée
    hash_index = defaultdict(list)  # sha256 -> list of (classe, chemin)
    origine_rows = []
    all_entries = []  # (classe, chemin, nom_fichier)
    for cls, fs in files_by_class.items():
        for fname in fs:
            all_entries.append((cls, os.path.join(root_path, cls, fname), fname))

    log(f"  GOGETA : hachage SHA-256 de {len(all_entries)} images...")
    for cls, path, fname in tqdm(all_entries, desc='  SHA-256 GOGETA'):
        h = lib.sha256_of_file(path)
        if h is None:
            continue
        hash_index[h].append((cls, path))
        size = os.path.getsize(path)
        source = lib.detect_source_from_filename(fname)
        origine_rows.append([fname, cls, source, h, size])

    lib.write_csv(os.path.join(out_dir, 'origine_gogeta.csv'),
                  ['fichier', 'classe', 'source_detectee', 'sha256', 'taille_octets'], origine_rows)

    # Doublons exacts (toutes occurrences, y compris même classe)
    doublons_rows = []
    contradictoires_rows = []
    for h, occ in hash_index.items():
        if len(occ) > 1:
            classes = ';'.join(sorted({c for c, _ in occ}))
            paths = ';'.join(p for _, p in occ)
            doublons_rows.append([h, len(occ), classes, paths])
            classes_set = {c for c, _ in occ}
            if 'NORMAL' in classes_set and 'PNEUMONIA' in classes_set:
                normals = ';'.join(p for c, p in occ if c == 'NORMAL')
                pneumos = ';'.join(p for c, p in occ if c == 'PNEUMONIA')
                contradictoires_rows.append([h, normals, pneumos])

    lib.write_csv(os.path.join(out_dir, 'doublons.csv'),
                  ['sha256', 'n_occurrences', 'classes', 'chemins'], doublons_rows)
    lib.write_csv(os.path.join(out_dir, 'etiquettes_contradictoires.csv'),
                  ['sha256', 'chemins_NORMAL', 'chemins_PNEUMONIA'], contradictoires_rows)
    result['n_doublons_exacts_groupes'] = len(doublons_rows)
    result['n_images_contradictoires'] = len(contradictoires_rows)

    # Quasi-doublons par pHash (optionnel, LSH par bandes pour rester praticable)
    if do_phash:
        log(f"  GOGETA : calcul pHash sur {len(all_entries)} images (LSH, seuil Hamming <= {phash_threshold})...")
        hashes = []
        meta = []
        for cls, path, fname in tqdm(all_entries, desc='  pHash GOGETA'):
            h = lib.phash_from_path(path)
            if h is None:
                continue
            hashes.append(h)
            meta.append((cls, path, lib.detect_source_from_filename(fname)))

        buckets = defaultdict(list)
        for idx, h in enumerate(hashes):
            for band_idx, band_val in enumerate(lib.band_keys(h)):
                buckets[(band_idx, band_val)].append(idx)

        import numpy as np
        candidate_pairs = set()
        for key, idxs in buckets.items():
            if len(idxs) < 2 or len(idxs) > 2000:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    candidate_pairs.add((a, b) if a < b else (b, a))

        quasi_rows = []
        for a, b in candidate_pairs:
            dist = bin(hashes[a] ^ hashes[b]).count('1')
            if dist <= phash_threshold:
                cls_a, path_a, src_a = meta[a]
                cls_b, path_b, src_b = meta[b]
                quasi_rows.append([hashes[a], hashes[b], dist, cls_a, path_a, src_a, cls_b, path_b, src_b])

        lib.write_csv(os.path.join(out_dir, 'quasi_doublons_phash.csv'),
                      ['hash1', 'hash2', 'hamming', 'classe1', 'chemin1', 'source1', 'classe2', 'chemin2', 'source2'],
                      quasi_rows)
        result['n_quasi_doublons_phash'] = len(quasi_rows)
    else:
        # Fichier vide avec en-tête pour cohérence si --phash non demandé
        if not os.path.isfile(os.path.join(out_dir, 'quasi_doublons_phash.csv')):
            lib.write_csv(os.path.join(out_dir, 'quasi_doublons_phash.csv'),
                          ['hash1', 'hash2', 'hamming', 'classe1', 'chemin1', 'source1', 'classe2', 'chemin2', 'source2'], [])
        result['n_quasi_doublons_phash'] = None

    return result


# --- SPLITS (fuite train/val/test) ------------------------------------------

def audit_splits(root_path, gogeta_path, out_dir):
    result = {'present': os.path.isdir(root_path)}
    if not result['present']:
        return result

    counts = {}
    hash_locations = defaultdict(list)  # sha256 -> list of (split, classe, chemin)
    for split in ('train', 'val', 'test'):
        for cls in ('NORMAL', 'PNEUMONIA'):
            folder = os.path.join(root_path, split, cls)
            files = lib.list_images(folder)
            counts[f'{split}/{cls}'] = len(files)
            for fname in files:
                path = os.path.join(folder, fname)
                h = lib.sha256_of_file(path)
                if h:
                    hash_locations[h].append((split, cls, path))
    result['counts'] = counts
    result['total'] = sum(counts.values())

    fuite_rows = []
    for h, occ in hash_locations.items():
        splits_present = {s for s, _, _ in occ}
        if len(splits_present) > 1:
            classes = ';'.join(sorted({c for _, c, _ in occ}))
            splits_str = ';'.join(sorted(splits_present))
            paths = ';'.join(p for _, _, p in occ)
            fuite_rows.append([h, splits_str, classes, paths])

    lib.write_csv(os.path.join(out_dir, 'fuite_splits.csv'),
                  ['sha256', 'splits', 'classes', 'chemins'], fuite_rows)
    result['n_images_en_fuite'] = len(fuite_rows)

    # Vérification rapide : les fichiers de SPLITS existent-ils aussi dans GOGETA (cohérence) ?
    if os.path.isdir(gogeta_path):
        gogeta_hashes = set()
        for cls in ('NORMAL', 'PNEUMONIA'):
            folder = os.path.join(gogeta_path, cls)
            for fname in lib.list_images(folder):
                h = lib.sha256_of_file(os.path.join(folder, fname))
                if h:
                    gogeta_hashes.add(h)
        not_in_gogeta = sum(1 for h in hash_locations if h not in gogeta_hashes)
        result['n_hashes_splits_absents_de_gogeta'] = not_in_gogeta

    return result


# --- Tableau attendu / trouvé -------------------------------------------

def build_expected_vs_found(results, out_dir):
    rows = []

    def add(dataset, mesure, attendu, trouve):
        ecart = None
        ecart_pct = None
        if trouve is not None and attendu is not None:
            ecart = trouve - attendu
            ecart_pct = (ecart / attendu * 100) if attendu else None
        rows.append([dataset, mesure, attendu, trouve, ecart,
                    f"{ecart_pct:.1f}%" if ecart_pct is not None else ''])

    paul = results.get('PAUL', {})
    add('PAUL', f"total (attendu {EXPECTED_COUNTS['PAUL_total_min']}-{EXPECTED_COUNTS['PAUL_total_max']})",
        EXPECTED_COUNTS['PAUL_total_min'], paul.get('total'))

    rsna = results.get('RSNA', {})
    add('RSNA', 'total (.dcm trouvés)', EXPECTED_COUNTS['RSNA_total'], rsna.get('n_dcm_files'))
    add('RSNA', 'pneumonie (CSV, Target=1)', EXPECTED_COUNTS['RSNA_pneumonia'], rsna.get('n_pneumonia_csv'))

    tb = results.get('TB_RAHMAN', {})
    add('TB_RAHMAN', 'total', EXPECTED_COUNTS['TB_RAHMAN_total'], tb.get('total'))
    tb_counts = tb.get('counts', {})
    add('TB_RAHMAN', 'tuberculose', EXPECTED_COUNTS['TB_RAHMAN_tuberculosis'], tb_counts.get('TUBERCULOSIS'))
    add('TB_RAHMAN', 'normal', EXPECTED_COUNTS['TB_RAHMAN_normal'], tb_counts.get('NORMAL'))

    gogeta = results.get('GOGETA', {})
    attendu_gogeta_si_propre = None
    if paul.get('total') is not None and rsna.get('n_dcm_files') is not None and tb.get('total') is not None:
        attendu_gogeta_si_propre = paul['total'] + rsna['n_dcm_files'] + tb['total']
    add('GOGETA', 'total (somme sources si fusion propre, sans dédoublonnage)',
        attendu_gogeta_si_propre, gogeta.get('total'))

    lib.write_csv(os.path.join(out_dir, 'attendu_vs_trouve.csv'),
                  ['dataset', 'mesure', 'attendu', 'trouve', 'ecart', 'ecart_pct'], rows)


# --- Source x classe ------------------------------------------------------

def build_source_class_table(out_dir):
    origine_path = os.path.join(out_dir, 'origine_gogeta.csv')
    if not os.path.isfile(origine_path):
        return
    df = pd.read_csv(origine_path)
    table = df.groupby(['source_detectee', 'classe']).size().reset_index(name='n_images')
    table.to_csv(os.path.join(out_dir, 'source_x_classe.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help=r'Racine des données, ex: C:\Z_Soutenance\DATA')
    parser.add_argument('--out', default='audit_out', help='Dossier de sortie (créé si besoin)')
    parser.add_argument('--phash', action='store_true',
                         help='Active la recherche de quasi-doublons par pHash sur GOGETA (plus lent)')
    parser.add_argument('--phash-threshold', type=int, default=5, help='Seuil de distance de Hamming (défaut 5)')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    log("=" * 70)
    log("ÉTAPE 0 : vérification d'accès")
    log("=" * 70)
    if not os.path.isdir(args.root):
        log(f"ERREUR : le dossier racine n'existe pas : {args.root}")
        log("ARRÊT. Vérifie le chemin passé à --root.")
        sys.exit(1)

    resolved, unexpected = resolve_topdirs(args.root)
    log(f"Contenu de {args.root} :")
    for entry in sorted(os.listdir(args.root)):
        log(f"  - {entry}")

    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        log(f"ATTENTION : dossiers attendus non trouvés (sous ces noms) : {missing}")
    if unexpected:
        log(f"ATTENTION : dossiers présents non reconnus (noms à vérifier/adapter) : {unexpected}")

    presence_rows = presence_report(args.root, resolved, args.out)
    for row in presence_rows:
        key, found_name, exists, path, total = row
        log(f"  {key:12s} -> trouvé='{found_name}' existe={exists} images_recursif={total}")

    if all(v is None for v in resolved.values()):
        log("ERREUR : aucun des dossiers attendus (PAUL, RSNA, TB_RAHMAN, GOGETA, SPLITS) n'a été trouvé.")
        log("ARRÊT. Les noms de dossiers diffèrent probablement : adapte EXPECTED_TOPDIRS dans audit_donnees.py.")
        sys.exit(1)

    results = {}

    log("\n" + "=" * 70)
    log("AUDIT PAUL")
    log("=" * 70)
    if resolved['PAUL']:
        results['PAUL'] = audit_paul(os.path.join(args.root, resolved['PAUL']))
        log(json.dumps(results['PAUL'], indent=2, ensure_ascii=False))
    else:
        results['PAUL'] = {'present': False}

    log("\n" + "=" * 70)
    log("AUDIT RSNA")
    log("=" * 70)
    if resolved['RSNA']:
        results['RSNA'] = audit_rsna(os.path.join(args.root, resolved['RSNA']))
        log(json.dumps(results['RSNA'], indent=2, ensure_ascii=False))
    else:
        results['RSNA'] = {'present': False}

    log("\n" + "=" * 70)
    log("AUDIT TB_RAHMAN")
    log("=" * 70)
    if resolved['TB_RAHMAN']:
        results['TB_RAHMAN'] = audit_tb_rahman(os.path.join(args.root, resolved['TB_RAHMAN']))
        log(json.dumps(results['TB_RAHMAN'], indent=2, ensure_ascii=False))
    else:
        results['TB_RAHMAN'] = {'present': False}

    log("\n" + "=" * 70)
    log("AUDIT GOGETA (SHA-256, doublons, étiquettes contradictoires" +
        (", pHash" if args.phash else "") + ")")
    log("=" * 70)
    if resolved['GOGETA']:
        results['GOGETA'] = audit_gogeta(os.path.join(args.root, resolved['GOGETA']), args.out,
                                          args.phash, args.phash_threshold)
        log(json.dumps({k: v for k, v in results['GOGETA'].items() if k != 'counts'},
                       indent=2, ensure_ascii=False))
    else:
        results['GOGETA'] = {'present': False}

    log("\n" + "=" * 70)
    log("AUDIT SPLITS (fuite train/val/test)")
    log("=" * 70)
    if resolved['SPLITS']:
        gogeta_path = os.path.join(args.root, resolved['GOGETA']) if resolved['GOGETA'] else ''
        results['SPLITS'] = audit_splits(os.path.join(args.root, resolved['SPLITS']), gogeta_path, args.out)
        log(json.dumps(results['SPLITS'], indent=2, ensure_ascii=False))
    else:
        results['SPLITS'] = {'present': False}

    build_expected_vs_found(results, args.out)
    build_source_class_table(args.out)

    with open(os.path.join(args.out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 70)
    log(f"TERMINÉ. Sorties écrites dans : {os.path.abspath(args.out)}")
    log("Fichiers : presence.csv, attendu_vs_trouve.csv, origine_gogeta.csv, source_x_classe.csv,")
    log("           doublons.csv, etiquettes_contradictoires.csv, fuite_splits.csv,")
    log("           quasi_doublons_phash.csv" + (" (vide, --phash non demandé)" if not args.phash else ""))
    log("           summary.json")
    log("=" * 70)


if __name__ == '__main__':
    main()
