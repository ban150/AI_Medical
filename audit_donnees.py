"""
AUDIT EN LECTURE SEULE des données (PAUL, RSNA, TB_RAHMAN, GOGETA, SPLITS).

Ce script ne supprime, ne déplace et ne modifie AUCUN fichier de données.
Il ne fait que lire les images et écrire ses résultats dans --out (dossier à part).

Usage :
    python audit_donnees.py --root C:\\Z_Soutenance\\DATA --out audit_out
Dépendances : numpy, Pillow (obligatoires). Le pHash est calculé en interne (DCT numpy).
"""
import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

IMG_EXT = ('.png', '.jpg', '.jpeg')
DCM_EXT = ('.dcm',)
CLASSES = ('NORMAL', 'PNEUMONIA')
SPLITS = ('train', 'val', 'test')

# Motifs de noms pour retrouver l'origine d'un fichier sans préfixe (Fusion.py)
RE_UUID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.png$', re.I)
RE_TB = re.compile(r'^(Normal|Tuberculosis)-\d+\.(png|jpg|jpeg)$', re.I)
RE_KERMANY = re.compile(r'^(IM-|NORMAL2-IM-|person\d+_)', re.I)


def list_files(folder, exts):
    out = []
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


_N = 32
_DCT = np.array([[np.cos(np.pi * (2 * x + 1) * u / (2 * _N)) for x in range(_N)] for u in range(_N)])


def phash_file(path):
    """pHash 64 bits : gris 32x32 -> DCT 2D -> bloc 8x8 -> seuil à la médiane."""
    try:
        with Image.open(path) as im:
            a = np.asarray(im.convert('L').resize((_N, _N), Image.LANCZOS), dtype=np.float64)
        d = _DCT @ a @ _DCT.T
        low = d[:8, :8].flatten()
        bits = low > np.median(low[1:])
        return int(''.join('1' if b else '0' for b in bits), 2)
    except Exception:
        return None


def parallel(fn, paths, workers):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, paths, chunksize=64))


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='dossier DATA contenant PAUL, RSNA, TB_RAHMAN, GOGETA, SPLITS')
    ap.add_argument('--out', default='audit_out')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--no-phash', action='store_true')
    ap.add_argument('--phash-dist', type=int, default=4, help='distance de Hamming max pour un quasi-doublon')
    args = ap.parse_args()
    root, out = args.root, args.out
    os.makedirs(out, exist_ok=True)

    # ---------- 1. Inventaire ----------
    groups = {  # étiquette -> dossier
        'PAUL': os.path.join(root, 'PAUL'),
        'RSNA': os.path.join(root, 'RSNA'),
        'TB_RAHMAN': os.path.join(root, 'TB_RAHMAN'),
        'GOGETA/NORMAL': os.path.join(root, 'GOGETA', 'NORMAL'),
        'GOGETA/PNEUMONIA': os.path.join(root, 'GOGETA', 'PNEUMONIA'),
    }
    for s in SPLITS:
        for c in CLASSES:
            groups[f'SPLITS/{s}/{c}'] = os.path.join(root, 'SPLITS', s, c)
    # RSNA peut aussi être posé directement dans DATA (cf. Fusion.py)
    extra_rsna = os.path.join(root, 'stage_2_train_images')

    files = {}   # label -> [chemins d'images]
    dcms = {}
    for label, folder in groups.items():
        files[label] = list_files(folder, IMG_EXT) if os.path.isdir(folder) else []
        dcms[label] = list_files(folder, DCM_EXT) if os.path.isdir(folder) else []
    if os.path.isdir(extra_rsna):
        dcms['RSNA'] = dcms['RSNA'] + list_files(extra_rsna, DCM_EXT)

    # comptage par sous-dossier de premier niveau pour les sources
    rows = []
    for label in ('PAUL', 'RSNA', 'TB_RAHMAN'):
        by_sub = Counter()
        for p in files[label]:
            rel = os.path.relpath(p, groups[label])
            by_sub[os.path.dirname(rel) or '.'] += 1
        for sub, n in sorted(by_sub.items()):
            rows.append((label, sub, n))
        if dcms[label]:
            rows.append((label, '(fichiers .dcm)', len(dcms[label])))
    for label in groups:
        if label.startswith(('GOGETA', 'SPLITS')):
            rows.append((label, '.', len(files[label])))
    write_csv(os.path.join(out, 'comptages.csv'), ['ensemble', 'sous_dossier', 'nb_images'], rows)

    n_src = sum(len(files[k]) + len(dcms[k]) for k in ('PAUL', 'RSNA', 'TB_RAHMAN'))
    n_gog = len(files['GOGETA/NORMAL']) + len(files['GOGETA/PNEUMONIA'])
    n_spl = sum(len(files[k]) for k in files if k.startswith('SPLITS'))

    # ---------- 2. SHA-256 ----------
    all_paths = [p for k, v in files.items() for p in v]
    print(f'Hash SHA-256 de {len(all_paths)} images...')
    sha = dict(zip(all_paths, parallel(sha256_file, all_paths, args.workers)))
    write_csv(os.path.join(out, 'hashes.csv'), ['chemin', 'sha256'], sorted(sha.items()))

    # a. doublons exacts dans GOGETA
    gog_paths = files['GOGETA/NORMAL'] + files['GOGETA/PNEUMONIA']
    by_hash = defaultdict(list)
    for p in gog_paths:
        by_hash[sha[p]].append(p)
    dup_rows = []
    n_extra = 0
    for h, ps in by_hash.items():
        if len(ps) > 1:
            n_extra += len(ps) - 1
            for p in sorted(ps):
                dup_rows.append((p, h, len(ps), os.path.basename(os.path.dirname(p))))
    dup_rows.sort(key=lambda r: (-r[2], r[1], r[0]))
    write_csv(os.path.join(out, 'doublons.csv'), ['chemin', 'hash', 'nombre_de_copies', 'dossier'], dup_rows)
    copies_hist = Counter(len(ps) for ps in by_hash.values())

    # b. étiquettes contradictoires
    cls_of = defaultdict(set)
    for p in gog_paths:
        cls_of[sha[p]].add(os.path.basename(os.path.dirname(p)))
    contradict = [(p, sha[p]) for p in gog_paths if len(cls_of[sha[p]]) > 1]
    write_csv(os.path.join(out, 'etiquettes_contradictoires.csv'), ['chemin', 'hash'], sorted(contradict))

    # c. fuite entre splits
    split_of = defaultdict(set)
    for k, v in files.items():
        if k.startswith('SPLITS/'):
            for p in v:
                split_of[sha[p]].add(k.split('/')[1])
    leak_hashes = {h for h, s in split_of.items() if len(s) > 1}
    leak_rows = []
    for k, v in files.items():
        if k.startswith('SPLITS/'):
            for p in v:
                if sha[p] in leak_hashes:
                    leak_rows.append((p, sha[p], '+'.join(sorted(split_of[sha[p]]))))
    write_csv(os.path.join(out, 'fuite_splits.csv'), ['chemin', 'hash', 'ensembles_concernes'], sorted(leak_rows))
    pair_count = Counter('+'.join(sorted(s)) for h, s in split_of.items() if len(s) > 1)
    # même nom de fichier dans plusieurs splits (indice d'un SPLITS non purgé entre deux exécutions)
    name_split = defaultdict(set)
    for k, v in files.items():
        if k.startswith('SPLITS/'):
            for p in v:
                name_split[os.path.basename(p)].add(k.split('/')[1])
    same_name_multi = sum(1 for s in name_split.values() if len(s) > 1)

    # collisions de noms entre sources (sources avec le même basename) -> écrasement silencieux par shutil.copy
    name_src = defaultdict(list)
    for k in ('PAUL', 'RSNA', 'TB_RAHMAN'):
        for p in files[k]:
            name_src[os.path.basename(p)].append(p)
    coll = [(n, len(ps), ' | '.join(ps[:4])) for n, ps in name_src.items() if len(ps) > 1]
    write_csv(os.path.join(out, 'collisions_noms_sources.csv'), ['nom', 'occurrences', 'exemples'], sorted(coll))

    # ---------- 3. Origine des images de GOGETA ----------
    sha_src = {}
    for k in ('PAUL', 'RSNA', 'TB_RAHMAN'):
        for p in files[k]:
            sha_src.setdefault(sha[p], k)

    def origin(p):
        fn = os.path.basename(p)
        low = fn.lower()
        if low.startswith('kermany_'):
            return 'PAUL', 'préfixe'
        if low.startswith('tb_'):
            return 'TB_RAHMAN', 'préfixe'
        if low.startswith('rsna_'):
            return 'RSNA', 'préfixe'
        if sha[p] in sha_src:
            return sha_src[sha[p]], 'hash identique à la source'
        if RE_UUID.match(fn):
            return 'RSNA', 'motif de nom (UUID)'
        if RE_TB.match(fn):
            return 'TB_RAHMAN', 'motif de nom'
        if RE_KERMANY.match(fn):
            return 'PAUL', 'motif de nom'
        return 'INCONNUE', ''

    orig_rows, uniq = [], defaultdict(set)
    for p in gog_paths:
        s, how = origin(p)
        c = os.path.basename(os.path.dirname(p))
        orig_rows.append((p, c, s, how, sha[p]))
        uniq[(s, c)].add(sha[p])
    write_csv(os.path.join(out, 'origine_gogeta.csv'), ['chemin', 'classe', 'source', 'methode', 'hash'], sorted(orig_rows))
    sources = sorted({s for s, _ in uniq})

    # ---------- 4. pHash / quasi-doublons ----------
    near_groups = []
    if not args.no_phash:
        reps = {}  # un représentant par sha dans GOGETA
        for p in gog_paths:
            reps.setdefault(sha[p], p)
        rep_paths = list(reps.values())
        print(f'pHash de {len(rep_paths)} images uniques...')
        ph = dict(zip(rep_paths, parallel(phash_file, rep_paths, args.workers)))
        ph = {p: v for p, v in ph.items() if v is not None}
        # bucketing par 8 morceaux de 8 bits (principe des tiroirs : distance <= 7 => 1 morceau identique)
        buckets = defaultdict(list)
        for p, v in ph.items():
            for i in range(8):
                buckets[(i, (v >> (8 * i)) & 0xFF)].append(p)
        parent = {p: p for p in ph}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for b in buckets.values():
            if len(b) > 3000:
                continue  # image quasi uniforme : bucket non discriminant
            for i in range(len(b)):
                for j in range(i + 1, len(b)):
                    if bin(ph[b[i]] ^ ph[b[j]]).count('1') <= args.phash_dist:
                        parent[find(b[i])] = find(b[j])
        comp = defaultdict(list)
        for p in ph:
            comp[find(p)].append(p)
        near_groups = [sorted(v) for v in comp.values() if len(v) > 1]
        rows = []
        for gi, g in enumerate(sorted(near_groups, key=len, reverse=True), 1):
            for p in g:
                rows.append((gi, p, sha[p], format(ph[p], '016x'), os.path.basename(os.path.dirname(p))))
        write_csv(os.path.join(out, 'quasi_doublons_phash.csv'),
                  ['groupe', 'chemin', 'sha256', 'phash', 'classe'], rows)

    # ---------- Rapport chiffré ----------
    L = []
    L.append('# Résultats chiffrés de l\'audit (généré par audit_donnees.py)\n')
    L.append(f'Racine auditée : `{root}`\n')
    L.append('## Comptages\n')
    L.append('| Ensemble | Images |\n|---|---:|')
    for k in ('PAUL', 'RSNA', 'TB_RAHMAN'):
        extra = f' (+ {len(dcms[k])} .dcm)' if dcms[k] else ''
        L.append(f'| {k} | {len(files[k])}{extra} |')
    L.append(f'| GOGETA/NORMAL | {len(files["GOGETA/NORMAL"])} |')
    L.append(f'| GOGETA/PNEUMONIA | {len(files["GOGETA/PNEUMONIA"])} |')
    L.append(f'| **GOGETA total** | **{n_gog}** |')
    for s in SPLITS:
        for c in CLASSES:
            L.append(f'| SPLITS/{s}/{c} | {len(files[f"SPLITS/{s}/{c}"])} |')
    L.append(f'| **SPLITS total** | **{n_spl}** |')
    L.append(f'\nÉcart GOGETA − SPLITS : {n_gog - n_spl:+d}  (0 attendu si SPLITS vient exactement de GOGETA)')
    L.append(f'\nSources (images + .dcm) : {n_src}. Écart GOGETA − sources : {n_gog - n_src:+d} '
             '(des écarts négatifs sont normaux : filtre qualité, équilibrage RSNA de test_1.py ; '
             'positifs = doublons ou fichiers hors sources).')
    L.append('\n## Doublons exacts dans GOGETA (SHA-256)\n')
    L.append(f'- Images uniques : {len(by_hash)} / {n_gog}')
    L.append(f'- Fichiers excédentaires (copies en trop) : {n_extra}')
    L.append('- Répartition : ' + (', '.join(f'{n} copie(s) : {c} hash' for n, c in sorted(copies_hist.items())) or 'n/a'))
    L.append(f'\n## Étiquettes contradictoires (même hash dans NORMAL et PNEUMONIA)\n')
    L.append(f'- Hash concernés : {sum(1 for s in cls_of.values() if len(s) > 1)} ; fichiers : {len(contradict)}')
    L.append('\n## Fuite entre train / val / test (hash communs)\n')
    L.append(f'- Hash présents dans plusieurs splits : {len(leak_hashes)} ; fichiers concernés : {len(leak_rows)}')
    for k, v in sorted(pair_count.items()):
        L.append(f'  - {k} : {v} hash')
    L.append(f'- Noms de fichiers présents dans plusieurs splits : {same_name_multi}')
    if not args.no_phash:
        L.append(f'\n## Quasi-doublons pHash (distance ≤ {args.phash_dist}, hash SHA distincts)\n')
        L.append(f'- Groupes : {len(near_groups)} ; images dans ces groupes : {sum(len(g) for g in near_groups)}')
    L.append(f'\n## Collisions de noms entre dossiers sources\n- Noms présents dans plusieurs dossiers sources : {len(coll)}')
    L.append('\n## Images UNIQUES par source × classe (GOGETA)\n')
    L.append('| Source | NORMAL | PNEUMONIA | % PNEUMONIA |\n|---|---:|---:|---:|')
    for s in sources:
        n, p = len(uniq.get((s, 'NORMAL'), ())), len(uniq.get((s, 'PNEUMONIA'), ()))
        L.append(f'| {s} | {n} | {p} | {100 * p / (n + p) if n + p else 0:.1f} |')
        if n == 0 or p == 0:
            L.append(f'|  ⚠ {s} ne fournit qu\'une seule classe : risque de raccourci « source = classe » | | | |')
    unk = sum(1 for r in orig_rows if r[2] == 'INCONNUE')
    L.append(f'\nImages d\'origine inconnue : {unk}')
    with open(os.path.join(out, 'resultats_audit.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nFichiers écrits dans : {os.path.abspath(out)}')


if __name__ == '__main__':
    main()
