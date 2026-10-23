"""
SPLIT : attribue train / val / test aux images retenues de GOGETA_CLEAN (colonne « split » de manifest.csv).

- Reproductible : aucune fonction aléatoire ; l'ordre des patients vient de sha256(graine|patient).
- Découpage PAR PATIENT : un patient (RSNA : PatientID du DICOM ; PAUL : person<N> ou IM-<NNNN>)
  n'apparaît que dans un seul split.
- « Même patient probable » : le PatientID de RSNA identifie une IMAGE, pas une personne (RSNA vient de
  NIH ChestX-ray14, qui contient des examens de suivi). Deux images d'une même source dont les vignettes
  sont très corrélées (r ≥ 0,92) ou dont le pHash est à ≤ 2 sont donc mises dans le même split
  (complément de sécurité ; le vrai patient RSNA vient de la correspondance NIH, voir fusion_clean).
- Stratifié par source ET par classe : chaque strate (source, classe) est découpée séparément,
  à 70 / 15 / 15 % en nombre d'images, de façon gloutonne (le split le plus en retard reçoit le patient suivant).
- Contrôles bloquants : zéro hash source commun, zéro patient commun, zéro fichier PNG identique
  entre train, val et test ; quasi-doublons pHash entre splits comptés et signalés.
- Évaluation croisée entre sources (utilisée par entrainement.py --experience) :
    paul_vers_rsna : entraînement + validation sur PAUL (train / val), test sur RSNA (test)
    rsna_vers_paul : entraînement + validation sur RSNA (train / val), test sur PAUL (test)

Usage :
    python -m AI_Medical.split_clean --dry-run
    python -m AI_Medical.split_clean
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from AI_Medical.data import pretraitement as pt

RACINE_DEFAUT = Path(os.environ.get("PNEUMONIA_DATA_ROOT", r"C:\Project_D\Z_Soutenance\DATA"))
SPLITS = ("train", "val", "test")
EXPERIENCES = {
    "principal": (("PAUL", "RSNA"), ("PAUL", "RSNA")),
    "paul_vers_rsna": (("PAUL",), ("RSNA",)),
    "rsna_vers_paul": (("RSNA",), ("PAUL",)),
}


def attribuer(df: pd.DataFrame, ratios: dict[str, float], graine: int) -> dict[str, str]:
    """Retourne {patient: split}. Les patients sont répartis strate par strate."""
    classes_patient = df.groupby("patient")["classe"].agg(lambda s: "+".join(sorted(set(s))))
    source_patient = df.groupby("patient")["source"].agg(lambda s: "+".join(sorted(set(s))))
    taille = df.groupby("patient").size()
    strates = defaultdict(list)
    for p in taille.index:
        strates[(source_patient[p], classes_patient[p])].append(p)
    affectation = {}
    for cle in sorted(strates):
        patients = sorted(strates[cle], key=lambda p: hashlib.sha256(f"{graine}|{p}".encode()).hexdigest())
        compte = dict.fromkeys(SPLITS, 0)
        for p in patients:
            total = sum(compte.values()) + int(taille[p])
            s = max(SPLITS, key=lambda s: (ratios[s] * total - compte[s], -SPLITS.index(s)))
            affectation[p] = s
            compte[s] += int(taille[p])
    return affectation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--racine", type=Path, default=RACINE_DEFAUT)
    ap.add_argument("--graine", type=int, default=42)
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seuil-meme-patient", type=float, default=0.92,
                    help="corrélation des vignettes au-delà de laquelle deux images d'une même source vont "
                         "dans le même split (0,90 crée des chaînes de plus de 100 images)")
    ap.add_argument("--distance-meme-patient", type=int, default=2,
                    help="distance pHash en deçà de laquelle deux images d'une même source vont dans le même split")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sans-verif-fichiers", action="store_true", help="ne pas re-hacher les PNG de sortie")
    a = ap.parse_args()

    sortie = a.racine / "GOGETA_CLEAN"
    mpath = sortie / "manifest.csv"
    man = pd.read_csv(mpath, dtype=str, keep_default_na=False)
    garde = man[man["chemin"] != ""].copy()
    ratios = dict(zip(SPLITS, a.ratios))
    assert abs(sum(ratios.values()) - 1) < 1e-9

    # Patients probablement identiques : images distinctes mais très corrélées (suivis d'un même
    # patient, fréquents dans RSNA dont le PatientID est un identifiant d'IMAGE). On les regroupe.
    ctrl = sortie / "controle_quasi_doublons.csv"
    n_fusion = 0
    if ctrl.is_file():
        c = pd.read_csv(ctrl)
        meme_source = c["image"].str.split("\\").str[0] == c["candidate"].str.split("\\").str[0]
        proche = (c["correlation"] >= a.seuil_meme_patient) | (c["distance_phash"] <= a.distance_meme_patient)
        c = c[(c["type"] == "interne") & meme_source & proche]
        pat_de = dict(zip(garde["chemin_source"], garde["patient"]))
        parent = {p: p for p in garde["patient"].unique()}

        def trouver(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for x, y in zip(c["image"], c["candidate"]):
            if x in pat_de and y in pat_de:
                rx, ry = trouver(pat_de[x]), trouver(pat_de[y])
                if rx != ry:
                    parent[max(rx, ry)] = min(rx, ry)
                    n_fusion += 1
        garde["patient_origine"] = garde["patient"]
        garde["patient"] = garde["patient"].map(trouver)
    print(f"Regroupements « même patient probable » (même source, r ≥ {a.seuil_meme_patient} "
          f"ou pHash ≤ {a.distance_meme_patient}) : {n_fusion}")

    multi = garde.groupby("patient")["classe"].nunique()
    print(f"Images retenues : {len(garde)} ; patients : {garde['patient'].nunique()} "
          f"(dont {int((multi > 1).sum())} avec deux classes, traités comme une strate à part)")
    aff = attribuer(garde, ratios, a.graine)
    garde["split"] = garde["patient"].map(aff)

    # ---- contrôles bloquants ----
    def communs(col: str) -> int:
        n = garde.groupby(col)["split"].nunique()
        return int((n > 1).sum())

    erreurs = []
    if communs("hash"):
        erreurs.append(f"{communs('hash')} hash source dans plusieurs splits")
    if communs("patient"):
        erreurs.append(f"{communs('patient')} patients dans plusieurs splits")
    if not a.sans_verif_fichiers:
        print("Vérification SHA-256 des PNG de sortie...", flush=True)
        garde["sha_png"] = [pt.sha256_fichier(sortie / c) for c in garde["chemin"]]
        if communs("sha_png"):
            erreurs.append(f"{communs('sha_png')} fichiers PNG identiques dans plusieurs splits")
    # quasi-doublons pHash entre splits (information)
    ph = [int(h, 16) for h in garde["phash"]]
    sp = list(garde["split"])
    tiroirs = defaultdict(list)
    for j, v in enumerate(ph):
        for i in range(8):
            tiroirs[(i, (v >> (8 * i)) & 0xFF)].append(j)
    croise = Counter()
    for k, v in enumerate(ph):
        vus = set()
        for i in range(8):
            for j in tiroirs[(i, (v >> (8 * i)) & 0xFF)]:
                if j > k and j not in vus and sp[j] != sp[k]:
                    vus.add(j)
                    d = pt.distance(v, ph[j])
                    if d <= 6:
                        croise[d] += 1
    print("Paires pHash proches entre splits différents (distance : nombre) :",
          ", ".join(f"{d}:{croise.get(d, 0)}" for d in range(7)))
    if erreurs:
        raise SystemExit("ÉCHEC des contrôles : " + " ; ".join(erreurs))
    print("Contrôles OK : zéro hash source, zéro patient" + ("" if a.sans_verif_fichiers else ", zéro PNG")
          + " commun entre train, val et test.")

    # ---- bilan ----
    tab = garde.pivot_table(index=["source", "classe"], columns="split", values="hash", aggfunc="count",
                            fill_value=0)[list(SPLITS)]
    tab["total"] = tab.sum(axis=1)
    print("\n" + tab.to_string())
    lignes_exp = []
    for nom, (src_tr, src_te) in EXPERIENCES.items():
        tr = garde[garde["source"].isin(src_tr) & (garde["split"] == "train")]
        va = garde[garde["source"].isin(src_tr) & (garde["split"] == "val")]
        te = garde[garde["source"].isin(src_te) & (garde["split"] == "test")]
        lignes_exp.append((nom, len(tr), len(va), len(te),
                           f"{(te['classe'] == 'NORMAL').sum()}/{(te['classe'] == 'PNEUMONIA').sum()}"))
    print("\nExpériences : nom, train, val, test, test NORMAL/PNEUMONIA")
    for l in lignes_exp:
        print("  ", l)

    if a.dry_run:
        print("\n--dry-run : manifeste inchangé.")
        return
    man.loc[garde.index, "split"] = garde["split"]  # la colonne patient du manifeste reste l'identifiant d'origine
    tmp = mpath.with_suffix(".tmp")
    man.to_csv(tmp, index=False, encoding="utf-8", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    tmp.replace(mpath)
    with open(sortie / "resume_split.md", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Résumé du split\n\nGraine {a.graine}, ratios {a.ratios}, découpage par patient, "
                f"strates source × classe.\n\n")
        f.write("| Source | Classe | " + " | ".join(SPLITS) + " | total |\n|---|---|" + "---:|" * 4 + "\n")
        for (s, c), r in tab.iterrows():
            f.write(f"| {s} | {c} | " + " | ".join(str(int(r[k])) for k in (*SPLITS, "total")) + " |\n")
        f.write("\n")
        f.write("| Expérience | train | val | test | test N/P |\n|---|---:|---:|---:|---|\n")
        for l in lignes_exp:
            f.write("| " + " | ".join(map(str, l)) + " |\n")
    print(f"\nManifeste mis à jour : {mpath}")


if __name__ == "__main__":
    main()
