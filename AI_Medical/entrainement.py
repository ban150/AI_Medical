"""
ENTRAÎNEMENT : ResNet50 et MobileNetV2 sur GOGETA_CLEAN, à partir de manifest.csv.

Remplace l'ancien entrainement.py, qui codait les effectifs en dur (22463 / 20808) :
les effectifs et la pondération de la classe positive sont LUS dans le manifeste.

Expériences (--experience) :
  principal       : train / val / test sur PAUL + RSNA
  paul_vers_rsna  : train / val sur PAUL, test sur RSNA (évaluation croisée entre sources)
  rsna_vers_paul  : train / val sur RSNA, test sur PAUL
Le jeu nigérian (NIGERIA_TEST) est un test externe : jamais utilisé pour l'entraînement
ni pour la validation, évalué une seule fois à la fin avec --tester-nigeria.

Le choix du meilleur modèle se fait sur l'AUC de validation (le rappel seul récompense
un modèle qui prédit « pneumonie » partout).

Usage :
    python -m AI_Medical.entrainement --dry-run                   (effectifs et pondération, sans entraîner)
    python -m AI_Medical.entrainement --experience principal
    python -m AI_Medical.entrainement --experience paul_vers_rsna --tester-nigeria
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from AI_Medical.data import pretraitement as pt
from AI_Medical.split_clean import EXPERIENCES

RACINE_DEFAUT = Path(os.environ.get("PNEUMONIA_DATA_ROOT", r"C:\Project_D\Z_Soutenance\DATA"))
LABEL = {"NORMAL": 0, "PNEUMONIA": 1}


def charger_tables(racine: Path, experience: str) -> dict[str, pd.DataFrame]:
    man = pd.read_csv(racine / "GOGETA_CLEAN" / "manifest.csv", dtype=str, keep_default_na=False)
    garde = man[man["chemin"] != ""]
    if (garde["split"] == "").any():
        raise SystemExit("La colonne split est vide : lancer d'abord python -m AI_Medical.split_clean")
    src_tr, src_te = EXPERIENCES[experience]
    t = {
        "train": garde[garde["source"].isin(src_tr) & (garde["split"] == "train")],
        "val": garde[garde["source"].isin(src_tr) & (garde["split"] == "val")],
        "test": garde[garde["source"].isin(src_te) & (garde["split"] == "test")],
    }
    for k in t:
        t[k] = t[k].assign(fichier=[str(racine / "GOGETA_CLEAN" / c) for c in t[k]["chemin"]],
                           label=t[k]["classe"].map(LABEL))
    ng = pd.read_csv(racine / "NIGERIA_TEST" / "manifest_nigeria.csv", dtype=str)
    t["nigeria"] = ng.assign(fichier=[str(racine / "NIGERIA_TEST" / c) for c in ng["chemin"]],
                             label=ng["classe"].map(LABEL), source="NIGERIA")
    return t


def resume(t: dict[str, pd.DataFrame]) -> float:
    for k, d in t.items():
        par = d.groupby(["source", "classe"]).size().to_dict()
        print(f"  {k:8s} {len(d):6d} images  {par}")
    neg = int((t["train"]["label"] == 0).sum())
    pos = int((t["train"]["label"] == 1).sum())
    pos_weight = neg / max(pos, 1)
    print(f"  pondération classe PNEUMONIA (pos_weight = NORMAL/PNEUMONIA en train) = {neg}/{pos} = {pos_weight:.4f}")
    return pos_weight


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--racine", type=Path, default=RACINE_DEFAUT)
    ap.add_argument("--experience", choices=sorted(EXPERIENCES), default="principal")
    ap.add_argument("--backbones", nargs="+", default=["resnet50", "mobilenetv2"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--taille", type=int, default=224)
    ap.add_argument("--graine", type=int, default=42)
    ap.add_argument("--tester-nigeria", action="store_true", help="évaluation finale sur NIGERIA_TEST")
    ap.add_argument("--sortie", type=Path, default=Path(__file__).resolve().parent / "models" / "checkpoints_clean")
    ap.add_argument("--resultats", type=Path, default=Path(__file__).resolve().parent.parent / "resultats",
                    help="dossier des scores, courbes et matrices de confusion (un sous-dossier par expérience)")
    ap.add_argument("--version", default="v1",
                    help="nom de la variante ; v1 = réglages d'origine. Les résultats vont dans "
                         "resultats/<experience>_<version> (sauf v1), pour ne rien écraser")
    ap.add_argument("--degeler", type=int, default=0,
                    help="nombre de derniers blocs du réseau à ré-entraîner (0 = réseau gelé, comme en v1)")
    ap.add_argument("--lr-backbone", type=float, default=1e-5,
                    help="taux d'apprentissage des blocs dégelés (plus faible que celui de la tête)")
    ap.add_argument("--augmentation", choices=["legere", "forte"], default="legere",
                    help="forte = contraste, luminosité, gamma, flou, bruit, recadrage (imite la diversité des appareils)")
    ap.add_argument("--essai", action="store_true", help="essai rapide : 64 images par jeu, 1 époque")
    ap.add_argument("--autoriser-cpu", action="store_true", help="accepter un entraînement sans carte graphique (très lent)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    t = charger_tables(a.racine, a.experience)
    if a.essai:  # vérification rapide de bout en bout : 64 images par jeu, 1 époque, résultats à part
        t = {k: d.sample(n=min(64, len(d)), random_state=0) for k, d in t.items()}
        a.epochs, a.resultats, a.sortie = 1, a.resultats / "_essai", a.sortie / "_essai"
    print(f"Expérience « {a.experience} » (effectifs lus dans manifest.csv) :")
    pos_weight = resume(t)
    if a.dry_run:
        print("--dry-run : aucun entraînement.")
        return

    import random

    import cv2
    import torch
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm

    from AI_Medical.evaluation.medical_metrics import classification_metrics_dict, print_medical_report
    from AI_Medical.models.transfer_cnn import build_binary_transfer_model

    random.seed(a.graine)
    np.random.seed(a.graine)
    torch.manual_seed(a.graine)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Calcul sur la carte graphique : {torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
    else:
        print(f"ATTENTION : PAS de carte graphique (PyTorch {torch.__version__}, interpréteur {sys.executable}). "
              "L'entraînement serait environ 8 fois plus lent. Choisir l'interpréteur venv_gpu.")
        if not (a.essai or a.autoriser_cpu):
            raise SystemExit("Arrêt : relancer avec venv_gpu, ou ajouter --autoriser-cpu.")
    amp = device == "cuda" and a.degeler > 0  # précision mixte : accélère le ré-entraînement des blocs dégelés
    moy =np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    ect = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    class Radios(Dataset):
        def __init__(self, d: pd.DataFrame, augmenter: bool):
            self.f, self.y, self.aug = list(d["fichier"]), list(d["label"]), augmenter

        def __len__(self):
            return len(self.f)

        def __getitem__(self, i):
            f = self.f[i]
            if "GOGETA_CLEAN" in f:
                img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            else:  # test externe : même prétraitement, appliqué à la volée (fichiers en lecture seule)
                img = pt.pretraiter(f)
            img = cv2.resize(img, (a.taille, a.taille), interpolation=cv2.INTER_AREA)
            if self.aug:
                if random.random() < 0.5:
                    img = img[:, ::-1]
                forte = a.augmentation == "forte"
                ang = random.uniform(-10, 10) if forte else random.uniform(-8, 8)
                echelle = random.uniform(1.0, 1.15) if forte else 1.0  # léger zoom = recadrage aléatoire
                m = cv2.getRotationMatrix2D((a.taille / 2, a.taille / 2), ang, echelle)
                if forte:
                    m[:, 2] += np.random.uniform(-0.05, 0.05, 2) * a.taille  # petit décalage
                img = cv2.warpAffine(np.ascontiguousarray(img), m, (a.taille, a.taille), borderMode=cv2.BORDER_REFLECT)
                if forte:  # imite des appareils et des expositions différentes
                    f32 = img.astype(np.float32) / 255.0
                    f32 = np.clip(f32 * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)  # contraste, luminosité
                    f32 = f32 ** random.uniform(0.7, 1.4)  # gamma
                    if random.random() < 0.3:
                        f32 = cv2.GaussianBlur(f32, (0, 0), random.uniform(0.5, 1.5))
                    if random.random() < 0.3:
                        f32 = np.clip(f32 + np.random.normal(0, random.uniform(0.01, 0.03), f32.shape), 0, 1)
                    img = (f32 * 255).astype(np.uint8)
            x = (np.repeat(img[None].astype(np.float32) / 255.0, 3, axis=0) - moy) / ect
            return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.float32)

    def chargeur(d, aug, melange):
        return DataLoader(Radios(d, aug), batch_size=a.batch_size, shuffle=melange, num_workers=0,
                          pin_memory=device == "cuda")

    def evaluer(model, loader, crit=None, seuil=0.5):
        """Retourne (métriques, y, scores). Ajoute sensibilité, spécificité et perte si crit est fourni."""
        model.eval()
        ys, ss, pertes = [], [], []
        with torch.no_grad():
            for x, y in loader:
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    logit = model(x.to(device)).view(-1).float()
                if crit is not None:
                    pertes.append(float(crit(logit, y.to(device))) * len(y))
                ss.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(y.numpy())
        y, s = np.concatenate(ys).astype(int), np.concatenate(ss)
        pred = (s >= seuil).astype(int)
        met = classification_metrics_dict(y, pred, y_score=s)
        vp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
        vn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
        met.update({"sensibilite": vp / max(vp + fn, 1), "specificite": vn / max(vn + fp, 1),
                    "VP": vp, "FN": fn, "VN": vn, "FP": fp, "n": int(len(y))})
        if crit is not None:
            met["perte"] = sum(pertes) / max(len(y), 1)
        return met, y, s

    def ic_auc(y, s, n_boot=1000):
        """Intervalle de confiance à 95 % de l'AUC par bootstrap (graine fixe)."""
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(a.graine)
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) == 2:
                vals.append(roc_auc_score(y[i], s[i]))
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2

    import json
    import time

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from AI_Medical.evaluation.medical_metrics import plot_confusion_matrix

    def degeler(model, bb: str, n: int) -> list:
        """Rend entraînables les n derniers blocs du réseau ; retourne leurs paramètres."""
        if n <= 0:
            return []
        if bb == "resnet50":
            blocs = [model.layer1, model.layer2, model.layer3, model.layer4][-n:]
        else:  # mobilenetv2 : 19 blocs « features », regroupés par 4 (le dernier groupe = 3 blocs)
            f = list(model.features)
            groupes = [f[0:4], f[4:8], f[8:12], f[12:16], f[16:19]]
            blocs = [b for g in groupes[-n:] for b in g]
        params = []
        for b in blocs:
            for p in b.parameters():
                p.requires_grad = True
                params.append(p)
        return params

    nom = a.experience if a.version == "v1" else f"{a.experience}_{a.version}"
    a.sortie.mkdir(parents=True, exist_ok=True)
    dossier_res = a.resultats / nom
    dossier_res.mkdir(parents=True, exist_ok=True)
    ld = {k: chargeur(t[k], k == "train", k == "train") for k in t}
    lignes = []
    for bb in a.backbones:
        model, opt, crit = build_binary_transfer_model(bb, learning_rate=a.lr)
        params_bb = degeler(model, bb, a.degeler)
        if params_bb:  # deux taux : la tête apprend vite, les blocs dégelés lentement (ne pas « oublier » ImageNet)
            ids = {id(p) for p in params_bb}
            tete = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
            opt = torch.optim.Adam([{"params": tete, "lr": a.lr}, {"params": params_bb, "lr": a.lr_backbone}])
            print(f"[{bb}] {a.degeler} bloc(s) dégelé(s) : {sum(p.numel() for p in params_bb):,} paramètres "
                  f"ré-entraînés (lr {a.lr_backbone}), tête lr {a.lr}")
        planif = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
        echelle = torch.amp.GradScaler("cuda", enabled=amp)
        model = model.to(device)
        crit.pos_weight = torch.tensor([pos_weight], device=device)
        ckpt = a.sortie / f"{nom}_{bb}_best.pt"
        meilleur, attente, histo = -1.0, 0, []
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            model.train()
            somme, n = 0.0, 0
            for x, y in tqdm(ld["train"], desc=f"{bb} époque {ep}", leave=False):
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    loss = crit(model(x.to(device)).view(-1).float(), y.to(device))
                echelle.scale(loss).backward()
                echelle.step(opt)
                echelle.update()
                somme, n = somme + float(loss) * len(y), n + len(y)
            mv, _, _ = evaluer(model, ld["val"], crit)
            auc = mv.get("roc_auc", float("nan"))
            planif.step(auc)
            histo.append({"epoque": ep, "perte_train": somme / n, "perte_val": mv["perte"], "auc_val": auc,
                          "sensibilite_val": mv["sensibilite"], "specificite_val": mv["specificite"],
                          "duree_s": round(time.time() - t0, 1)})
            print(f"[{bb}] époque {ep:02d} perte train {somme / n:.4f} | perte val {mv['perte']:.4f} | "
                  f"AUC val {auc:.4f} | {time.time() - t0:.0f} s", flush=True)
            pd.DataFrame(histo).to_csv(dossier_res / f"historique_{bb}.csv", index=False)
            if auc > meilleur:
                meilleur, attente = auc, 0
                torch.save({"model_state_dict": model.state_dict(), "auc_val": auc, "epoque": ep,
                            "experience": a.experience, "pretraitement": pt.VERSION}, ckpt)
            else:
                attente += 1
                if attente >= a.patience:
                    print(f"[{bb}] arrêt anticipé (pas de progrès depuis {a.patience} époques)")
                    break
        h = pd.DataFrame(histo)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(h.epoque, h.perte_train, label="train")
        ax[0].plot(h.epoque, h.perte_val, label="validation")
        ax[0].set_title(f"{bb} : perte")
        ax[0].set_xlabel("époque")
        ax[0].legend()
        ax[1].plot(h.epoque, h.auc_val, color="tab:green")
        ax[1].set_title(f"{bb} : AUC de validation")
        ax[1].set_xlabel("époque")
        fig.tight_layout()
        fig.savefig(dossier_res / f"courbes_{bb}.png", dpi=150)
        plt.close(fig)

        etat = torch.load(ckpt, map_location=device)
        model.load_state_dict(etat["model_state_dict"])
        # Seuil de décision choisi sur la VALIDATION (indice de Youden : sensibilité + spécificité − 1 maximal),
        # jamais sur le test ni sur le jeu nigérian. Le seuil 0,5 reste rapporté pour comparaison.
        from sklearn.metrics import roc_curve
        _, yv, sv = evaluer(model, ld["val"])
        fpr, tpr, seuils = roc_curve(yv, sv)
        seuil_val = float(seuils[np.argmax(tpr - fpr)])
        print(f"[{bb}] seuil choisi sur la validation (Youden) : {seuil_val:.3f}")
        jeux = ["test"] + (["nigeria"] if a.tester_nigeria else [])
        for jeu in jeux:
            for seuil, etiquette in ((0.5, "0.5"), (seuil_val, "validation")):
                met, y, s = evaluer(model, ld[jeu], seuil=seuil)
                bas, haut = ic_auc(y, s)
                print_medical_report(met, title=f"{bb} : {jeu} ({nom}, seuil {etiquette} = {seuil:.3f})")
                print(f"  AUC IC95 % : [{bas:.3f} ; {haut:.3f}]  sensibilité {met['sensibilite']:.3f}  "
                      f"spécificité {met['specificite']:.3f}")
                suffixe = "" if etiquette == "0.5" else "_seuil_val"
                plot_confusion_matrix(y, (s >= seuil).astype(int), labels=("NORMAL", "PNEUMONIA"),
                                      save_path=str(dossier_res / f"confusion_{bb}_{jeu}{suffixe}.png"), show=False)
                lignes.append({"experience": nom, "modele": bb, "jeu": jeu, "seuil": etiquette,
                               "valeur_seuil": round(seuil, 4), "epoque_retenue": etat["epoque"],
                               "auc_val": round(etat["auc_val"], 4),
                               **{k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()},
                               "auc_ic95_bas": round(bas, 4), "auc_ic95_haut": round(haut, 4)})
            pd.DataFrame({"y": y, "score": s}).to_csv(dossier_res / f"scores_{bb}_{jeu}.csv", index=False)
        pd.DataFrame(lignes).to_csv(dossier_res / "resultats.csv", index=False)
    with open(dossier_res / "parametres.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) for k, v in vars(a).items()} | {"pos_weight": pos_weight, "pretraitement": pt.VERSION,
                   "device": device, "torch": torch.__version__}, f, ensure_ascii=False, indent=2)
    print()
    print(f"Résultats enregistrés dans {dossier_res}")


if __name__ == "__main__":
    main()
