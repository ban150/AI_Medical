"""
GRAD-CAM : où le modèle regarde-t-il sur la radio ?

Pour un modèle entraîné (checkpoint), produit des planches « radio | carte de chaleur superposée »
sur des images de test choisies de façon reproductible :
  - test interne : vrais positifs, faux positifs, faux négatifs, vrais négatifs, pour PAUL et pour RSNA ;
  - jeu nigérian : mêmes catégories.
Une zone chaude sur les poumons est rassurante ; une zone chaude sur les lettres (L/R), les bords
ou les câbles signale un raccourci (biais de source).

Usage :
    python -m AI_Medical.gradcam --experience principal_v2 --modele resnet50
    python -m AI_Medical.gradcam --experience principal_v2 --modele mobilenetv2 --par-categorie 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from AI_Medical.data import pretraitement as pt

RACINE_DEFAUT = Path(os.environ.get("PNEUMONIA_DATA_ROOT", r"C:\Project_D\Z_Soutenance\DATA"))
ICI = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--racine", type=Path, default=RACINE_DEFAUT)
    ap.add_argument("--experience", default="principal_v2", help="nom du dossier de résultats (ex. principal, principal_v2)")
    ap.add_argument("--modele", choices=["resnet50", "mobilenetv2"], default="resnet50")
    ap.add_argument("--par-categorie", type=int, default=2, help="images par catégorie (VP, FP, FN, VN) et par source")
    ap.add_argument("--taille", type=int, default=224)
    ap.add_argument("--resultats", type=Path, default=ICI.parent / "resultats")
    ap.add_argument("--checkpoints", type=Path, default=ICI / "models" / "checkpoints_clean")
    a = ap.parse_args()

    import torch

    from AI_Medical.models.transfer_cnn import build_binary_transfer_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = build_binary_transfer_model(a.modele)
    etat = torch.load(a.checkpoints / f"{a.experience}_{a.modele}_best.pt", map_location=device)
    model.load_state_dict(etat["model_state_dict"])
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = True
    couche = model.layer4 if a.modele == "resnet50" else model.features[-1]
    activ, grad = {}, {}
    couche.register_forward_hook(lambda m, i, o: activ.__setitem__("a", o))
    couche.register_full_backward_hook(lambda m, gi, go: grad.__setitem__("g", go[0]))

    moy = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    ect = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def charger(fichier: str) -> np.ndarray:
        if "GOGETA_CLEAN" in fichier:
            img = cv2.imdecode(np.fromfile(fichier, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            img = pt.pretraiter(fichier)  # test externe : même prétraitement
        return cv2.resize(img, (a.taille, a.taille), interpolation=cv2.INTER_AREA)

    def gradcam(img: np.ndarray) -> tuple[np.ndarray, float]:
        x = (np.repeat(img[None].astype(np.float32) / 255.0, 3, axis=0) - moy) / ect
        x = torch.from_numpy(x)[None].to(device)
        model.zero_grad()
        logit = model(x).view(-1)[0]
        logit.backward()
        poids = grad["g"].mean(dim=(2, 3), keepdim=True)
        carte = torch.relu((poids * activ["a"]).sum(dim=1))[0].detach().cpu().numpy()
        carte = cv2.resize(carte, (a.taille, a.taille))
        carte = carte / carte.max() if carte.max() > 0 else carte
        return carte, float(torch.sigmoid(logit))

    dossier = a.resultats / a.experience
    seuil = 0.5
    res = dossier / "resultats.csv"
    if res.is_file():
        r = pd.read_csv(res)
        r = r[(r["modele"] == a.modele) & (r["jeu"] == "test")]
        if "seuil" in r and (r["seuil"] == "validation").any():
            seuil = float(r[r["seuil"] == "validation"]["valeur_seuil"].iloc[0])
    print(f"Seuil utilisé pour classer VP/FP/FN/VN : {seuil:.3f}")

    man = pd.read_csv(a.racine / "GOGETA_CLEAN" / "manifest.csv", dtype=str, keep_default_na=False)
    test = man[(man["chemin"] != "") & (man["split"] == "test")].reset_index(drop=True)
    test = test.assign(fichier=[str(a.racine / "GOGETA_CLEAN" / c) for c in test["chemin"]])
    ng = pd.read_csv(a.racine / "NIGERIA_TEST" / "manifest_nigeria.csv", dtype=str)
    ng = ng.assign(fichier=[str(a.racine / "NIGERIA_TEST" / c) for c in ng["chemin"]], source="NIGERIA")

    jeux = {}
    for nom, table, fichier_scores in (("test", test, f"scores_{a.modele}_test.csv"),
                                       ("nigeria", ng, f"scores_{a.modele}_nigeria.csv")):
        sc = pd.read_csv(dossier / fichier_scores)
        assert len(sc) == len(table), f"{fichier_scores} ne correspond pas au manifeste"
        t = table.assign(y=sc["y"].to_numpy(), score=sc["score"].to_numpy())
        t["cat"] = np.select([(t.y == 1) & (t.score >= seuil), (t.y == 0) & (t.score >= seuil),
                              (t.y == 1) & (t.score < seuil)], ["VP", "FP", "FN"], "VN")
        jeux[nom] = t

    sortie = a.resultats / "gradcam"
    sortie.mkdir(parents=True, exist_ok=True)
    for nom, t in jeux.items():
        lignes = []
        for src in sorted(t["source"].unique()):
            for cat in ("VP", "FN", "FP", "VN"):
                sel = t[(t.source == src) & (t.cat == cat)].sort_values("fichier")
                if sel.empty:
                    continue
                idx = np.linspace(0, len(sel) - 1, min(a.par_categorie, len(sel))).astype(int)
                for _, row in sel.iloc[idx].iterrows():
                    img = charger(row["fichier"])
                    carte, p = gradcam(img)
                    chaud = cv2.applyColorMap((carte * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    sup = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, chaud, 0.4, 0)
                    vignette = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), sup])
                    bandeau = np.full((26, vignette.shape[1], 3), 255, np.uint8)
                    vrai = "PNEUMONIA" if row["y"] == 1 else "NORMAL"
                    cv2.putText(bandeau, f"{src} {cat} vrai={vrai} p={p:.2f}", (4, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    lignes.append(np.vstack([bandeau, vignette]))
        par_ligne = 2
        while len(lignes) % par_ligne:
            lignes.append(np.full_like(lignes[0], 255))
        planche = np.vstack([np.hstack(lignes[i:i + par_ligne]) for i in range(0, len(lignes), par_ligne)])
        out = sortie / f"gradcam_{a.experience}_{a.modele}_{nom}.png"
        cv2.imencode(".png", planche)[1].tofile(str(out))
        print(f"écrit : {out} ({len(lignes)} cases)")


if __name__ == "__main__":
    main()
