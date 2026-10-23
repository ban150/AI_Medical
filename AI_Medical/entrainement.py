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
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    t = charger_tables(a.racine, a.experience)
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
    moy = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
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
                ang = random.uniform(-8, 8)
                m = cv2.getRotationMatrix2D((a.taille / 2, a.taille / 2), ang, 1.0)
                img = cv2.warpAffine(np.ascontiguousarray(img), m, (a.taille, a.taille), borderMode=cv2.BORDER_REFLECT)
            x = (np.repeat(img[None].astype(np.float32) / 255.0, 3, axis=0) - moy) / ect
            return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.float32)

    def chargeur(d, aug, melange):
        return DataLoader(Radios(d, aug), batch_size=a.batch_size, shuffle=melange, num_workers=0,
                          pin_memory=device == "cuda")

    def evaluer(model, loader):
        model.eval()
        ys, ss = [], []
        with torch.no_grad():
            for x, y in loader:
                ss.append(torch.sigmoid(model(x.to(device)).view(-1)).cpu().numpy())
                ys.append(y.numpy())
        y, s = np.concatenate(ys).astype(int), np.concatenate(ss)
        return classification_metrics_dict(y, (s >= 0.5).astype(int), y_score=s)

    a.sortie.mkdir(parents=True, exist_ok=True)
    ld = {k: chargeur(t[k], k == "train", k == "train") for k in t}
    for bb in a.backbones:
        model, opt, crit = build_binary_transfer_model(bb, learning_rate=a.lr)
        model = model.to(device)
        crit.pos_weight = torch.tensor([pos_weight], device=device)
        ckpt = a.sortie / f"{a.experience}_{bb}_best.pt"
        meilleur, attente = -1.0, 0
        for ep in range(1, a.epochs + 1):
            model.train()
            for x, y in tqdm(ld["train"], desc=f"{bb} époque {ep}", leave=False):
                opt.zero_grad()
                loss = crit(model(x.to(device)).view(-1), y.to(device))
                loss.backward()
                opt.step()
            auc = evaluer(model, ld["val"]).get("roc_auc", float("nan"))
            print(f"[{bb}] époque {ep:02d} AUC val = {auc:.4f}")
            if auc > meilleur:
                meilleur, attente = auc, 0
                torch.save({"model_state_dict": model.state_dict(), "auc_val": auc,
                            "experience": a.experience, "pretraitement": pt.VERSION}, ckpt)
            else:
                attente += 1
                if attente >= a.patience:
                    break
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        print_medical_report(evaluer(model, ld["test"]), title=f"{bb} : test ({a.experience})")
        if a.tester_nigeria:
            print_medical_report(evaluer(model, ld["nigeria"]), title=f"{bb} : test externe nigérian")


if __name__ == "__main__":
    main()
