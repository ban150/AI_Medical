"""
FIGURES ET TABLEAU DU MÉMOIRE, à partir des fichiers de resultats/ (aucun entraînement).

Produit dans resultats/memoire/ :
  - tableau_recapitulatif.csv et tableau_recapitulatif.md : toutes les expériences, seuil 0,5 et seuil de validation ;
  - roc_test_interne.png, roc_nigeria.png : courbes ROC v1 contre v2 (ResNet50 et MobileNetV2) ;
  - roc_croise.png : courbes ROC des évaluations croisées v1 et v2.

Usage : python -m AI_Medical.figures_memoire
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

RES = Path(__file__).resolve().parent.parent / "resultats"
OUT = RES / "memoire"
EXPS = ["principal", "principal_v2", "paul_vers_rsna", "paul_vers_rsna_v2", "rsna_vers_paul", "rsna_vers_paul_v2"]
MODELES = ["resnet50", "mobilenetv2"]


def roc(ax, exp, modele, jeu, style):
    f = RES / exp / f"scores_{modele}_{jeu}.csv"
    if not f.is_file():
        return
    s = pd.read_csv(f)
    fpr, tpr, _ = roc_curve(s.y, s.score)
    ax.plot(fpr, tpr, style, label=f"{exp} {modele} (AUC {roc_auc_score(s.y, s.score):.3f})")


def figure(nom, titre, paires):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for exp, modele, jeu, style in paires:
        roc(ax, exp, modele, jeu, style)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="hasard")
    ax.set_xlabel("1 − spécificité")
    ax.set_ylabel("sensibilité")
    ax.set_title(titre)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / nom, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lignes = []
    for exp in EXPS:
        f = RES / exp / "resultats.csv"
        if f.is_file():
            r = pd.read_csv(f)
            if "seuil" not in r:
                r["seuil"] = "0.5"
            r["version"] = "v2" if exp.endswith("_v2") else "v1"
            r["experience"] = exp.removesuffix("_v2")
            lignes.append(r)
    t = pd.concat(lignes, ignore_index=True)
    cols = ["experience", "version", "modele", "jeu", "seuil", "roc_auc", "auc_ic95_bas", "auc_ic95_haut",
            "sensibilite", "specificite", "accuracy", "f1", "n", "epoque_retenue", "auc_val"]
    t = t[[c for c in cols if c in t]].sort_values(["experience", "jeu", "modele", "version", "seuil"])
    t.to_csv(OUT / "tableau_recapitulatif.csv", index=False)
    with open(OUT / "tableau_recapitulatif.md", "w", encoding="utf-8") as fh:
        fh.write("| Expérience | Version | Modèle | Jeu | Seuil | AUC [IC95 %] | Sensibilité | Spécificité | Exactitude |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in t.iterrows():
            fh.write(f"| {r.experience} | {r.version} | {r.modele} | {r.jeu} | {r.seuil} | "
                     f"{r.roc_auc:.3f} [{r.auc_ic95_bas:.3f} ; {r.auc_ic95_haut:.3f}] | "
                     f"{r.sensibilite:.3f} | {r.specificite:.3f} | {r.accuracy:.3f} |\n")

    styles = {("v1", "resnet50"): "C0--", ("v2", "resnet50"): "C0-", ("v1", "mobilenetv2"): "C1--",
              ("v2", "mobilenetv2"): "C1-"}
    for jeu, nom, titre in (("test", "roc_test_interne.png", "Test interne (PAUL + RSNA) : v1 contre v2"),
                            ("nigeria", "roc_nigeria.png", "Test externe nigérian (190) : v1 contre v2")):
        figure(nom, titre, [(e, m, jeu, styles[(v, m)]) for m in MODELES
                            for v, e in (("v1", "principal"), ("v2", "principal_v2"))])
    figure("roc_croise.png", "Évaluations croisées (test sur l'autre source)",
           [(e + suf, "resnet50", "test", st) for e, base in (("paul_vers_rsna", "C2"), ("rsna_vers_paul", "C3"))
            for suf, st in (("", base + "--"), ("_v2", base + "-"))])
    print(f"Écrit dans {OUT} : {sorted(p.name for p in OUT.iterdir())}")


if __name__ == "__main__":
    main()
