"""
FUSION : construit DATA/GOGETA_CLEAN à partir de PAUL et RSNA uniquement.

Ne lit QUE les sources brutes. N'écrit QUE dans GOGETA_CLEAN (et, avec --preparer-tests,
dans NIGERIA_TEST et PEDICXR_TEST). Ne touche jamais à GOGETA, SPLITS, PAUL, RSNA ni TB_RAHMAN.

Règles (dans cet ordre, déterministes) :
  1. Inventaire
     - PAUL : chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}. On ignore __MACOSX, ._*, .DS_Store
       et la copie imbriquée chest_xray/chest_xray.
     - RSNA : stage_2_train_images + stage_2_detailed_class_info.csv.
       « Normal » → NORMAL, « Lung Opacity » → PNEUMONIA,
       « No Lung Opacity / Not Normal » → exclu_not_normal (listé dans le manifeste).
  2. Images illisibles ou uniformes            → exclu_illisible
  3. Doublons exacts (SHA-256 du fichier source) → exclu_doublon_sha (on garde le plus petit
     chemin source) ; étiquettes contradictoires → exclu_conflit_etiquette (toutes les copies)
  Quasi-doublon (règles 4, 5, 7) = pHash ≤ --seuil-phash (candidat) ET corrélation des vignettes
  64×64 ≥ --seuil-correlation (confirmation). Le pHash seul ne suffit pas : des radios thoraciques
  distinctes ont souvent un pHash à distance 2 ou 4 (vérifié à l'œil, audit_out/planche_*).
  4. TB_RAHMAN : SHA identique ou quasi-doublon d'une image TB     → exclu_jumeau_tb
  5. Jeu nigérian : SHA identique ou quasi-doublon                 → exclu_doublon_nigeria
  6. Exclusions manuelles (config/exclusions_manuelles.csv)       → exclu_manuel
  7. Quasi-doublons internes (groupes par union-find)             → exclu_doublon_phash
     (on garde le plus petit hash) ; groupe à étiquettes mixtes → exclu_conflit_etiquette
  Toutes les paires candidates et leur décision : GOGETA_CLEAN/controle_quasi_doublons.csv

Sorties : GOGETA_CLEAN/{NORMAL,PNEUMONIA}/<source>_<classe>_<sha256[:16]>.png et manifest.csv.
Idempotent : une image déjà présente n'est pas réécrite ; le manifeste est trié et sans date ;
les valeurs « split » déjà attribuées sont conservées si l'ensemble des images retenues est inchangé.

Usage (depuis C:\\Users\\User\\PycharmProjects\\AI_Medical) :
    python -m AI_Medical.fusion_clean --dry-run
    python -m AI_Medical.fusion_clean
    python -m AI_Medical.fusion_clean --reset
    python -m AI_Medical.fusion_clean --preparer-tests   (extrait le zip nigérian dans NIGERIA_TEST)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import stat
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from AI_Medical.data import pretraitement as pt

RACINE_DEFAUT = Path(os.environ.get("PNEUMONIA_DATA_ROOT", r"C:\Project_D\Z_Soutenance\DATA"))
ZIP_NIGERIA_DEFAUT = Path(r"C:\Project_D\Z_Soutenance\NG Pneum Dataset.zip")
MD5_NIGERIA = "a04ce100eb5ef6df4cc0352efa25028c"  # publié sur Zenodo (DOI 10.5281/zenodo.14185822)
EXCLUSIONS_MANUELLES = Path(__file__).resolve().parent / "config" / "exclusions_manuelles.csv"
COLONNES = ["chemin", "source", "classe", "hash", "split", "patient",
            "classe_source", "chemin_source", "phash", "doublon_de"]
IMG_EXT = (".png", ".jpg", ".jpeg")
RSNA_CLASSES = {"Normal": "NORMAL", "Lung Opacity": "PNEUMONIA",
                "No Lung Opacity / Not Normal": "exclu_not_normal"}


# ---------------------------------------------------------------------------
# Identifiants patients
# ---------------------------------------------------------------------------

def patient_paul(nom: str) -> str:
    """
    Kermany : person<N>_... (pneumonie) → person<N> ; [NORMAL2-]IM-<NNNN>-... (normales) → IM-<NNNN>.
    Choix prudent : IM-0427 et NORMAL2-IM-0427 sont regroupés (au pire on regroupe deux patients
    distincts, ce qui ne crée jamais de fuite).
    """
    m = re.match(r"^person(\d+)_", nom, re.I)
    if m:
        return f"PAUL:person{int(m.group(1))}"
    m = re.match(r"^(?:NORMAL2-)?IM-(\d{4})-", nom, re.I)
    if m:
        return f"PAUL:IM-{m.group(1)}"
    raise ValueError(f"motif de nom PAUL inconnu : {nom}")


# ---------------------------------------------------------------------------
# Calcul parallèle (avec cache)
# ---------------------------------------------------------------------------

def _travail(args: tuple[str, bool]) -> tuple[str, str, str, str]:
    """(chemin, avec_image) → (chemin, sha256, phash_hex, erreur)."""
    chemin, avec_image = args
    try:
        sha = pt.sha256_fichier(chemin)
    except OSError as e:
        return chemin, "", "", f"lecture: {e}"
    if not avec_image:
        return chemin, sha, "", ""
    try:
        return chemin, sha, format(pt.phash(pt.pretraiter(chemin)), "016x"), ""
    except Exception as e:  # image illisible ou uniforme
        return chemin, sha, "", str(e)[:200]


def _ecrire(args: tuple[str, str]) -> None:
    source, destination = args
    pt.ecrire_png(pt.pretraiter(source), Path(destination))


def _vignette(chemin: str) -> bytes:
    return pt.vignette(pt.pretraiter(chemin)).tobytes()


def vignettes(chemins: list[str], cache_pkl: Path, ecrire_cache: bool, workers: int) -> dict:
    """Vignettes 64×64 normalisées (float32), avec cache disque pour les relances."""
    import pickle

    import numpy as np

    def cle(p: str) -> str:
        s = os.stat(p)
        return f"{p}|{s.st_size}|{s.st_mtime_ns}|{pt.VERSION}"

    cache = pickle.loads(cache_pkl.read_bytes()) if cache_pkl.is_file() else {}
    manquants = [p for p in chemins if cle(p) not in cache]
    if manquants:
        print(f"  vignettes à calculer : {len(manquants)} ({len(chemins) - len(manquants)} en cache)", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for p, b in zip(manquants, ex.map(_vignette, manquants, chunksize=32)):
                cache[cle(p)] = b
        if ecrire_cache:
            tmp = cache_pkl.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps(cache, protocol=4))
            tmp.replace(cache_pkl)
    return {p: np.frombuffer(cache[cle(p)], dtype=np.float32) for p in chemins}


def calculer(chemins: list[tuple[str, bool]], cache_csv: Path, ecrire_cache: bool, workers: int) -> dict:
    cache = {}
    if cache_csv.is_file():
        with open(cache_csv, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                cache[r["cle"]] = r

    def cle(p: str, img: bool) -> str:
        s = os.stat(p)
        return f"{p}|{s.st_size}|{s.st_mtime_ns}|{pt.VERSION if img else 'sha'}"

    res, a_faire = {}, []
    for p, img in chemins:
        k = cle(p, img)
        if k in cache:
            r = cache[k]
            res[p] = (r["sha256"], r["phash"], r["erreur"])
        else:
            a_faire.append((p, img))
    if a_faire:
        print(f"  calcul de {len(a_faire)} fichiers ({len(chemins) - len(a_faire)} en cache)...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (p, sha, ph, err) in enumerate(ex.map(_travail, a_faire, chunksize=32), 1):
                res[p] = (sha, ph, err)
                if i % 5000 == 0:
                    print(f"    {i}/{len(a_faire)}", flush=True)
    if ecrire_cache and a_faire:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        for p, img in a_faire:
            sha, ph, err = res[p]
            cache[cle(p, img)] = {"cle": cle(p, img), "sha256": sha, "phash": ph, "erreur": err}
        tmp = cache_csv.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cle", "sha256", "phash", "erreur"])
            w.writeheader()
            for k in sorted(cache):
                w.writerow(cache[k])
        tmp.replace(cache_csv)
    return res


# ---------------------------------------------------------------------------
# Recherche de voisins pHash (exacte pour une distance ≤ 7)
# ---------------------------------------------------------------------------

class IndexPhash:
    def __init__(self, hashes: list[int]):
        self.h = hashes
        self.tiroirs = defaultdict(list)
        for j, v in enumerate(hashes):
            for i in range(8):
                self.tiroirs[(i, (v >> (8 * i)) & 0xFF)].append(j)

    def voisins(self, v: int, seuil: int) -> list[tuple[int, int]]:
        assert seuil <= 7, "le principe des tiroirs n'est exact que pour une distance ≤ 7"
        vus, out = set(), []
        for i in range(8):
            for j in self.tiroirs.get((i, (v >> (8 * i)) & 0xFF), ()):
                if j not in vus:
                    vus.add(j)
                    d = pt.distance(v, self.h[j])
                    if d <= seuil:
                        out.append((d, j))
        return sorted(out)

    def plus_proche(self, v: int) -> int | None:
        r = self.voisins(v, 7)
        return r[0][0] if r else None


# ---------------------------------------------------------------------------
# Jeux de tests externes
# ---------------------------------------------------------------------------

def preparer_tests(racine: Path, zip_ng: Path) -> None:
    import hashlib

    ng = racine / "NIGERIA_TEST"
    (racine / "PEDICXR_TEST").mkdir(exist_ok=True)
    if ng.exists():
        print(f"NIGERIA_TEST existe déjà : {ng} (rien à faire)")
        return
    md5 = hashlib.md5(zip_ng.read_bytes()).hexdigest()
    if md5 != MD5_NIGERIA:
        raise SystemExit(f"MD5 du zip inattendu : {md5} (Zenodo : {MD5_NIGERIA})")
    rows = []
    with zipfile.ZipFile(zip_ng) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            if info.is_dir() or not info.filename.lower().endswith(IMG_EXT):
                continue
            parts = info.filename.split("/")
            classe = parts[-2].upper()
            if classe not in ("NORMAL", "PNEUMONIA"):
                raise SystemExit(f"classe inattendue dans le zip : {info.filename}")
            dst = ng / classe / parts[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = z.read(info)
            dst.write_bytes(data)
            rows.append((f"{classe}/{parts[-1]}", classe, hashlib.sha256(data).hexdigest()))
    with open(ng / "manifest_nigeria.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chemin", "classe", "hash"])
        w.writerows(rows)
    for p in ng.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)  # lecture seule
    print(f"NIGERIA_TEST : {len(rows)} images extraites en lecture seule ({Counter(r[1] for r in rows)})")


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--racine", type=Path, default=RACINE_DEFAUT)
    ap.add_argument("--dry-run", action="store_true", help="calcule et affiche, n'écrit rien")
    ap.add_argument("--reset", action="store_true", help="vide GOGETA_CLEAN (images et manifeste) avant reconstruction")
    ap.add_argument("--preparer-tests", action="store_true", help="extrait le zip nigérian dans NIGERIA_TEST et crée PEDICXR_TEST")
    ap.add_argument("--zip-nigeria", type=Path, default=ZIP_NIGERIA_DEFAUT)
    ap.add_argument("--seuil-phash", type=int, default=6,
                    help="distance pHash maximale pour PROPOSER une paire candidate (≤ 7)")
    ap.add_argument("--seuil-correlation", type=float, default=0.99,
                    help="corrélation minimale des vignettes 64×64 pour CONFIRMER un doublon. Calibrage : "
                         "les 3 500 jumelles TB↔RSNA sont ≥ 0,99 ; entre 0,95 et 0,99 ce sont des radios "
                         "distinctes d'un même patient probable (regroupées par split_clean, pas supprimées)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()

    R = a.racine
    if a.preparer_tests:
        preparer_tests(R, a.zip_nigeria)
        return
    sortie = R / "GOGETA_CLEAN"
    assert sortie.name == "GOGETA_CLEAN"
    cache_csv = sortie / "_cache" / "index_sources.csv"

    # ---- 1. Inventaire -------------------------------------------------------
    items = []  # dict(source, classe_source, classe, chemin_source(abs), patient)
    paul = R / "PAUL" / "chest_xray"
    for sp in ("train", "val", "test"):
        for cl in ("NORMAL", "PNEUMONIA"):
            for f in sorted(os.listdir(paul / sp / cl)):
                if f.startswith(".") or not f.lower().endswith(IMG_EXT):
                    continue
                items.append({"source": "PAUL", "classe_source": f"{sp}/{cl}", "classe": cl,
                              "abs": str(paul / sp / cl / f), "patient": patient_paul(f)})
    rs = R / "RSNA"
    det = pd.read_csv(rs / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    lab = pd.read_csv(rs / "stage_2_train_labels.csv").groupby("patientId")["Target"].max()
    for pid, c in sorted(zip(det["patientId"], det["class"])):
        classe = RSNA_CLASSES[c]
        cible = int(lab[pid])
        assert (classe == "PNEUMONIA") == (cible == 1), f"incohérence RSNA {pid}"
        items.append({"source": "RSNA", "classe_source": c, "classe": classe,
                      "abs": str(rs / "stage_2_train_images" / f"{pid}.dcm"), "patient": f"RSNA:{pid}"})
    tb = sorted(str(p) for p in (R / "TB_RAHMAN").rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith("."))
    ngd = R / "NIGERIA_TEST"
    if not ngd.is_dir():
        ngd = R / "NG Pneum Dataset"
    ng = sorted(str(p) for p in ngd.rglob("*") if p.suffix.lower() in IMG_EXT)
    print(f"Inventaire : PAUL {sum(i['source'] == 'PAUL' for i in items)}, RSNA {sum(i['source'] == 'RSNA' for i in items)}, "
          f"TB (référence d'exclusion) {len(tb)}, nigérian (référence d'exclusion) {len(ng)} [{ngd}]")

    # ---- calcul SHA / pHash --------------------------------------------------
    taches = [(i["abs"], i["classe"] != "exclu_not_normal") for i in items] + [(p, True) for p in tb + ng]
    res = calculer(taches, cache_csv, ecrire_cache=not a.dry_run, workers=a.workers)
    for i in items:
        i["hash"], ph, i["erreur"] = res[i["abs"]]
        i["phash"] = int(ph, 16) if ph else None
        i["exclusion"], i["doublon_de"] = "", ""
        if i["classe"] == "exclu_not_normal":
            i["exclusion"] = "exclu_not_normal"

    def actifs():
        return [i for i in items if not i["exclusion"]]

    rel = lambda p: os.path.relpath(p, R)

    # ---- 2. illisibles --------------------------------------------------------
    for i in actifs():
        if i["erreur"] or i["phash"] is None:
            i["exclusion"] = "exclu_illisible"
            i["doublon_de"] = i["erreur"]

    # ---- 3. doublons SHA --------------------------------------------------------
    par_sha = defaultdict(list)
    for i in actifs():
        par_sha[i["hash"]].append(i)
    for groupe in par_sha.values():
        if len(groupe) < 2:
            continue
        groupe.sort(key=lambda x: rel(x["abs"]))
        if len({g["classe"] for g in groupe}) > 1:
            for g in groupe:
                g["exclusion"], g["doublon_de"] = "exclu_conflit_etiquette", "SHA identique, classes différentes"
        else:
            for g in groupe[1:]:
                g["exclusion"], g["doublon_de"] = "exclu_doublon_sha", rel(groupe[0]["abs"])

    # ---- Quasi-doublons : le pHash (≤ seuil) PROPOSE des candidats, la corrélation des
    #      vignettes 64×64 (≥ seuil) CONFIRME. Le pHash seul confond des radios distinctes.
    ref_tb = [(p, int(res[p][1], 16)) for p in tb if res[p][1]]
    ref_ng = [(p, int(res[p][1], 16)) for p in ng if res[p][1]]
    idx_tb, idx_ng = IndexPhash([h for _, h in ref_tb]), IndexPhash([h for _, h in ref_ng])
    act0 = actifs()
    idx0 = IndexPhash([i["phash"] for i in act0])
    besoin = set()
    for k, i in enumerate(act0):
        v_tb = idx_tb.voisins(i["phash"], a.seuil_phash)
        v_ng = idx_ng.voisins(i["phash"], a.seuil_phash)
        v_in = [j for _, j in idx0.voisins(i["phash"], a.seuil_phash) if j != k]
        if v_tb or v_ng or v_in:
            besoin.add(i["abs"])
        besoin.update(ref_tb[j][0] for _, j in v_tb)
        besoin.update(ref_ng[j][0] for _, j in v_ng)
    vig = vignettes(sorted(besoin), sortie / "_cache" / "vignettes.pkl", not a.dry_run, a.workers)
    controle = []  # (type, image, candidate, distance, corrélation, décision)

    def correlation(p: str, q: str) -> float:
        return round(pt.correlation(vig[p], vig[q]), 4)

    # ---- 4 et 5. TB_RAHMAN et jeu nigérian ------------------------------------------
    def exclure_contre(ref: list[tuple[str, int]], refs_toutes: list[str], idx: IndexPhash, motif: str, typ: str):
        ref_sha = {res[p][0]: p for p in refs_toutes}
        dist_min = []
        for i in actifs():
            if i["hash"] in ref_sha:
                i["exclusion"], i["doublon_de"] = motif, rel(ref_sha[i["hash"]]) + " (SHA identique)"
                continue
            d = idx.plus_proche(i["phash"])
            dist_min.append(d if d is not None else 99)
            for dist, j in idx.voisins(i["phash"], a.seuil_phash):
                r = correlation(i["abs"], ref[j][0])
                ok = r >= a.seuil_correlation
                controle.append((typ, rel(i["abs"]), rel(ref[j][0]), dist, r, motif if ok else "conservée"))
                if ok and not i["exclusion"]:
                    i["exclusion"], i["doublon_de"] = motif, f"{rel(ref[j][0])} (pHash d={dist}, r={r:.4f})"
        return dist_min

    dist_tb = exclure_contre(ref_tb, tb, idx_tb, "exclu_jumeau_tb", "TB_RAHMAN")
    dist_ng = exclure_contre(ref_ng, ng, idx_ng, "exclu_doublon_nigeria", "NIGERIA")

    # ---- 6. exclusions manuelles ------------------------------------------------------
    manu = {}
    if EXCLUSIONS_MANUELLES.is_file():
        with open(EXCLUSIONS_MANUELLES, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                manu[r["hash_ou_chemin_source"].strip()] = r.get("motif", "").strip()
    for i in actifs():
        for k in (i["hash"], rel(i["abs"])):
            if k in manu:
                i["exclusion"], i["doublon_de"] = "exclu_manuel", manu[k]

    # ---- 7. quasi-doublons pHash internes -----------------------------------------------
    act = actifs()
    idx = IndexPhash([i["phash"] for i in act])
    parent = list(range(len(act)))

    def trouver(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    dist_interne = []
    for k, i in enumerate(act):
        v = [(d, j) for d, j in idx.voisins(i["phash"], 7) if j != k]
        dist_interne.append(v[0][0] if v else 99)
        for d, j in v:
            if d <= a.seuil_phash and j > k:
                r = correlation(i["abs"], act[j]["abs"])
                ok = r >= a.seuil_correlation
                controle.append(("interne", rel(i["abs"]), rel(act[j]["abs"]), d, r, "doublon" if ok else "conservée"))
                if ok:
                    parent[trouver(k)] = trouver(j)
    comp = defaultdict(list)
    for k in range(len(act)):
        comp[trouver(k)].append(act[k])
    n_groupes = 0
    for g in comp.values():
        if len(g) < 2:
            continue
        n_groupes += 1
        g.sort(key=lambda x: x["hash"])
        if len({x["classe"] for x in g}) > 1:
            for x in g:
                x["exclusion"], x["doublon_de"] = "exclu_conflit_etiquette", "pHash proche, classes différentes"
        else:
            for x in g[1:]:
                x["exclusion"], x["doublon_de"] = "exclu_doublon_phash", rel(g[0]["abs"])

    # ---- Bilan -----------------------------------------------------------------
    def histo(ds):
        c = Counter(ds)
        return ", ".join(f"{d}:{c.get(d, 0)}" for d in range(0, 8)) + f", >7:{sum(v for k, v in c.items() if k > 7)}"

    print("\nDistances pHash minimales (0..7) :")
    print(f"  interne PAUL+RSNA : {histo(dist_interne)}")
    print(f"  vers TB_RAHMAN    : {histo(dist_tb)}")
    print(f"  vers nigérian     : {histo(dist_ng)}")
    print(f"Groupes de doublons confirmés (pHash ≤ {a.seuil_phash} et r ≥ {a.seuil_correlation}) : {n_groupes}")
    dec = Counter((t, d) for t, _, _, _, _, d in controle)
    print("Paires candidates examinées : " + ", ".join(f"{t}/{d} {n}" for (t, d), n in sorted(dec.items())))
    limites = [c for c in controle if 0.95 <= c[4] < 0.99]
    print(f"Paires limites (0,95 ≤ r < 0,99), à vérifier à l'œil : {len(limites)}")

    lignes = []
    for i in items:
        garde = not i["exclusion"]
        nom = f"{i['source'].lower()}_{i['classe'].lower()}_{i['hash'][:16]}.png" if garde else ""
        lignes.append({
            "chemin": f"{i['classe']}/{nom}" if garde else "",
            "source": i["source"],
            "classe": i["classe"] if garde else i["exclusion"],
            "hash": i["hash"],
            "split": "",
            "patient": i["patient"],
            "classe_source": i["classe_source"],
            "chemin_source": rel(i["abs"]),
            "phash": format(i["phash"], "016x") if i["phash"] is not None else "",
            "doublon_de": i["doublon_de"],
        })
    lignes.sort(key=lambda r: (r["source"], r["chemin_source"]))
    bilan = Counter((r["source"], r["classe"]) for r in lignes)
    print("\nBilan source × classe :")
    for (s, c), n in sorted(bilan.items()):
        print(f"  {s:5s} {c:26s} {n:6d}")
    garde = [r for r in lignes if r["chemin"]]
    noms = Counter(r["chemin"] for r in garde)
    assert max(noms.values()) == 1, "collision de noms de fichiers"
    print(f"Images retenues : {len(garde)}")

    if a.dry_run:
        print("\n--dry-run : rien n'a été écrit.")
        return

    # ---- Écriture --------------------------------------------------------------
    if a.reset and sortie.exists():
        for sous in ("NORMAL", "PNEUMONIA"):
            if (sortie / sous).exists():
                shutil.rmtree(sortie / sous)
        for f in ("manifest.csv", "resume_fusion.md"):
            (sortie / f).unlink(missing_ok=True)
        print(f"--reset : {sortie} vidé (le cache _cache est conservé).")
    for sous in ("NORMAL", "PNEUMONIA"):
        (sortie / sous).mkdir(parents=True, exist_ok=True)

    abs_par_hash = {i["hash"]: i["abs"] for i in items}
    a_ecrire = [r for r in garde if not (sortie / r["chemin"]).is_file()]
    print(f"Écriture de {len(a_ecrire)} images ({len(garde) - len(a_ecrire)} déjà présentes)...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        travaux = [(abs_par_hash[r["hash"]], str(sortie / r["chemin"])) for r in a_ecrire]
        for k, _ in enumerate(ex.map(_ecrire, travaux, chunksize=32), 1):
            if k % 2000 == 0:
                print(f"  {k}/{len(a_ecrire)}", flush=True)

    attendus = {r["chemin"] for r in garde}
    presents = {f"{s}/{f}" for s in ("NORMAL", "PNEUMONIA") for f in os.listdir(sortie / s)}
    orphelins = sorted(presents - attendus)
    if orphelins:
        print(f"ATTENTION : {len(orphelins)} fichiers dans GOGETA_CLEAN ne sont pas dans le manifeste "
              f"(ex. {orphelins[:3]}). Relancer avec --reset pour repartir de zéro.")

    # conserver les splits si l'ensemble retenu est inchangé
    mpath = sortie / "manifest.csv"
    if mpath.is_file():
        ancien = pd.read_csv(mpath, dtype=str, keep_default_na=False)
        anc_garde = ancien[ancien["chemin"] != ""]
        if set(anc_garde["hash"]) == {r["hash"] for r in garde}:
            sp = dict(zip(anc_garde["hash"], anc_garde["split"]))
            for r in garde:
                r["split"] = sp.get(r["hash"], "")
        elif (anc_garde["split"] != "").any():
            print("ATTENTION : l'ensemble des images retenues a changé → colonne split vidée, relancer split_clean.")
    tmp = mpath.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLONNES, lineterminator="\n")
        w.writeheader()
        w.writerows(lignes)
    tmp.replace(mpath)
    with open(sortie / "controle_quasi_doublons.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["type", "image", "candidate", "distance_phash", "correlation", "decision"])
        w.writerows(sorted(controle))

    with open(sortie / "resume_fusion.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Résumé de fusion\n\nPrétraitement : `{pt.VERSION}`. Doublon = SHA-256 identique, ou pHash ≤ "
                f"{a.seuil_phash} ET corrélation des vignettes 64×64 ≥ {a.seuil_correlation}.\n\n"
                "| Source | Classe | Images |\n|---|---|---:|\n")
        for (s, c), n in sorted(bilan.items()):
            f.write(f"| {s} | {c} | {n} |\n")
        f.write(f"\nImages retenues : {len(garde)}\n")
    print(f"Manifeste écrit : {mpath}")


if __name__ == "__main__":
    sys.exit(main())
