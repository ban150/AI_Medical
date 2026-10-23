"""
TEST DE BIAIS DE SOURCE : un petit classifieur devine-t-il la SOURCE (PAUL ou RSNA) d'une image ?

Si oui (AUC proche de 1), un CNN peut apprendre « source = classe » au lieu de la pneumonie,
puisque la proportion de pneumonies diffère beaucoup entre sources (PAUL ≈ 73 %, RSNA ≈ 40 %).
Le test se fait aussi À CLASSE FIXÉE (NORMAL seules, puis PNEUMONIA seules) : une source
encore reconnaissable à classe fixée prouve que le signal ne vient pas de la maladie.

Caractéristiques volontairement simples : vignette 32×32 et histogramme d'intensité (32 classes),
régression logistique (scikit-learn). Entraîné sur le split train, mesuré sur le split test.
Pour comparaison, on affiche l'AUC qu'obtiendrait un modèle qui prédit la classe à partir de la
seule source (borne du raccourci).

Usage :
    python -m AI_Medical.biais_source --dry-run
    python -m AI_Medical.biais_source
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RACINE_DEFAUT = Path(os.environ.get("PNEUMONIA_DATA_ROOT", r"C:\Project_D\Z_Soutenance\DATA"))


def caracteristiques(fichier: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(fichier, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    vignette = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
    histo = np.histogram(img, bins=32, range=(0, 256))[0].astype(np.float32)
    return np.concatenate([vignette, histo / histo.sum()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--racine", type=Path, default=RACINE_DEFAUT)
    ap.add_argument("--max-par-groupe", type=int, default=3000, help="plafond d'images par (split, source, classe)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = pd.read_csv(a.racine / "GOGETA_CLEAN" / "manifest.csv", dtype=str, keep_default_na=False)
    d = man[(man["chemin"] != "") & man["split"].isin(["train", "test"])]
    d = (d.sort_values("hash").groupby(["split", "source", "classe"], group_keys=False)
         .head(a.max_par_groupe).reset_index(drop=True))
    print(d.groupby(["split", "source", "classe"]).size().to_string())
    if a.dry_run:
        print("--dry-run : aucun calcul.")
        return

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    fichiers = [str(a.racine / "GOGETA_CLEAN" / c) for c in d["chemin"]]
    with ThreadPoolExecutor(8) as ex:
        X = np.stack(list(ex.map(caracteristiques, fichiers)))
    y_src = (d["source"] == "RSNA").astype(int).to_numpy()
    y_cls = (d["classe"] == "PNEUMONIA").astype(int).to_numpy()
    tr, te = (d["split"] == "train").to_numpy(), (d["split"] == "test").to_numpy()

    print("\nPrédire la SOURCE (PAUL=0, RSNA=1) :")
    for nom, masque in (("toutes classes", np.ones(len(d), bool)),
                        ("NORMAL seules", (y_cls == 0)), ("PNEUMONIA seules", (y_cls == 1))):
        m_tr, m_te = tr & masque, te & masque
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(X[m_tr], y_src[m_tr])
        s = clf.predict_proba(X[m_te])[:, 1]
        print(f"  {nom:17s} : AUC = {roc_auc_score(y_src[m_te], s):.3f}  exactitude = "
              f"{accuracy_score(y_src[m_te], (s >= 0.5).astype(int)):.3f}  (n test = {int(m_te.sum())})")
    prev = pd.Series(y_cls[tr]).groupby(y_src[tr]).mean()
    s_src = np.where(y_src[te] == 1, prev[1], prev[0])
    print(f"\nRaccourci : prédire la classe par la seule source → AUC = {roc_auc_score(y_cls[te], s_src):.3f} "
          f"(PNEUMONIA en train : PAUL {prev[0]:.1%}, RSNA {prev[1]:.1%})")


if __name__ == "__main__":
    main()
