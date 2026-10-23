import nbformat

nb = nbformat.v4.new_notebook()
cells = []

def md(text):
    return nbformat.v4.new_markdown_cell(text)

def code(text):
    return nbformat.v4.new_code_cell(text)

# ═══════════════════════════════════════════════════════
# CONTENU EXACT DES CELLULES — DANS L'ORDRE
# ═══════════════════════════════════════════════════════

cells.append(md("""
# 🫁 Détection de Pneumonie par CNN — Notebook Optimisé
**Mémoire de Master — ESGIS Lomé**

Pipeline complet : ResNet50 + MobileNetV2 en 2 phases de fine-tuning.
Comparaison finale pour le contexte africain (déployabilité mobile).

**Corrections appliquées :**
- ✅ debug EXPLICITE — jamais lié au device
- ✅ shuffle=True sur train/val/test
- ✅ Fine-tuning 2 phases (backbone gelé → unfreeze partiel)
- ✅ ReduceLROnPlateau + EarlyStopping sur val_loss
- ✅ Seuil optimal calculé sur courbe ROC
- ✅ GPU RTX 5060 supporté (PyTorch nightly cu128)
"""))

# ── CELLULE 1 : Imports ───────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 1 : Imports et vérification environnement ────────────────
import os, json, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA dispo : {torch.cuda.is_available()}")
"""))

cells.append(md("## ⚙️ Configuration centrale\nTous les paramètres du projet sont ici. Modifier uniquement ce bloc."))

# ── CELLULE 2 : CFG ───────────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 2 : Configuration centrale CFG ───────────────────────────
CFG = {
    'splits_dir'         : r'C:\\Project_D\\Z_Soutenance\\DATA\\SPLITS',
    'out_dir'            : r'C:\\Project_D\\Z_Soutenance\\outputs',
    'backbones'          : ['resnet50', 'mobilenet_v2'],
    'dropout'            : 0.5,
    'pretrained'         : True,
    'img_size'           : 224,
    'epochs_phase1'      : 15,
    'epochs_phase2'      : 30,
    'lr_phase1'          : 1e-3,
    'lr_phase2_head'     : 1e-4,
    'lr_phase2_backbone' : 1e-5,
    'n_unfreeze_layers'  : 20,
    'patience'           : 7,
    'grad_clip'          : 1.0,
    'threshold_strategy' : 'f1',
    # DEBUG : False = entraînement complet
    # JAMAIS lié au device — c'était le bug original
    'debug'              : False,
}
os.makedirs(CFG['out_dir'], exist_ok=True)
print("CFG chargé ✅")
"""))

cells.append(md("## 🖥️ Détection GPU\nVérification automatique du GPU et adaptation du batch_size."))

# ── CELLULE 3 : GPU ───────────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 3 : Détection et configuration GPU ───────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU détecté  : {gpu_name}")
    print(f"   VRAM totale  : {gpu_mem:.1f} GB")
    CFG['batch_size']  = 64 if gpu_mem >= 6 else 32
    CFG['num_workers'] = 4
    CFG['pin_memory']  = True
    print(f"   batch_size   : {CFG['batch_size']}")
    print(f"   num_workers  : {CFG['num_workers']}")
else:
    print("⚠️  CPU uniquement — entraînement lent")
    print("   → Installer PyTorch nightly cu128 pour RTX 5060 :")
    print("   pip install --pre torch torchvision torchaudio")
    print("   --index-url https://download.pytorch.org/whl/nightly/cu128")
    CFG['batch_size']  = 32
    CFG['num_workers'] = 0
    CFG['pin_memory']  = False

print(f"\\nDevice final : {DEVICE}")
print(f"batch_size   : {CFG['batch_size']}")
"""))

cells.append(md("## 🔬 Diagnostic GPU complet\nTest rapide pour confirmer que les tenseurs tournent bien sur GPU."))

# ── CELLULE 4 : Diagnostic GPU ────────────────────────────────────────
cells.append(code("""
# ── CELLULE 4 : Diagnostic GPU complet ───────────────────────────────
print("="*55)
print("  DIAGNOSTIC GPU")
print("="*55)
print(f"PyTorch version  : {torch.__version__}")
print(f"CUDA disponible  : {torch.cuda.is_available()}")
print(f"Device utilisé   : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU              : {torch.cuda.get_device_name(0)}")
    print(f"CUDA version     : {torch.version.cuda}")
    x = torch.randn(3, 3).to(DEVICE)
    print(f"Test tensor GPU  : ✅ ({x.device})")
    del x
    torch.cuda.empty_cache()
    print(f"VRAM utilisée    : {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"VRAM réservée    : {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
else:
    print("\\n⚠️  GPU non utilisé")
print("="*55)
"""))

cells.append(md("## 🔄 Transforms\nAugmentation radiologique modérée sur le train uniquement. Aucune augmentation sur val/test."))

# ── CELLULE 5 : Transforms ────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 5 : Définition des transforms ────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# TRAIN : augmentation modérée adaptée aux RX thoraciques
# ⛔ Inversion verticale exclue — cliniquement incohérente
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0,
                            translate=(0.05, 0.05),
                            scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# VAL / TEST : aucune augmentation — évaluation reproductible
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

print(f"Train transforms : {len(train_transform.transforms)} étapes")
print(f"Eval  transforms : {len(eval_transform.transforms)} étapes")
"""))

cells.append(md("## 📂 Chargement des données\n**Correction critique** : `shuffle=True` sur tous les splits pour éviter que l'évaluation ne voie qu'une seule classe."))

# ── CELLULE 6 : Dataset ───────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 6 : Chargement datasets + comptage des classes ────────────
# ⚠️ CORRECTION CRITIQUE : shuffle=True sur TOUS les splits
# ImageFolder trie alphabétiquement → NORMAL avant PNEUMONIA
# Avec shuffle=False → seule la classe NORMAL était vue → recall=0

loaders, class_counts = {}, {}
for split in ['train', 'val', 'test']:
    path = os.path.join(CFG['splits_dir'], split)
    tf   = train_transform if split == 'train' else eval_transform
    ds   = datasets.ImageFolder(path, transform=tf)
    class_counts[split] = {
        cls: sum(1 for _, lb in ds.samples if lb == i)
        for i, cls in enumerate(ds.classes)
    }
    loaders[split] = DataLoader(
        ds,
        batch_size  = CFG['batch_size'],
        shuffle     = True,           # ← CORRECTION CRITIQUE
        num_workers = CFG['num_workers'],
        pin_memory  = CFG.get('pin_memory', False),
        drop_last   = (split == 'train'),
    )
    print(f"[{split:5s}] {len(ds):5d} images | {class_counts[split]}")
"""))

cells.append(md("## 📊 Distribution des classes\nVisualisation du déséquilibre NORMAL / PNEUMONIA par split."))

# ── CELLULE 7 : Distribution ──────────────────────────────────────────
cells.append(code("""
# ── CELLULE 7 : Visualisation distribution des classes ───────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, split in zip(axes, ['train', 'val', 'test']):
    counts = class_counts[split]
    bars = ax.bar(counts.keys(), counts.values(),
                  color=['steelblue', 'coral'])
    ax.bar_label(bars)
    total = sum(counts.values())
    ax.set_title(f'{split.capitalize()} ({total} images)')
    ax.set_ylabel("Nombre d'images")
plt.suptitle('Distribution des classes par split', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(CFG['out_dir'], 'class_distribution.png'), dpi=150)
plt.show()
print("Figure sauvegardée ✅")
"""))

cells.append(md("## 🖼️ Exemples d'images\nAperçu visuel des radiographies NORMAL et PNEUMONIA du dataset."))

# ── CELLULE 8 : Exemples images ───────────────────────────────────────
cells.append(code("""
# ── CELLULE 8 : Visualisation exemples d'images (grille 2×5) ─────────
mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
std  = torch.tensor(IMAGENET_STD).view(3,1,1)

imgs_batch, labels_batch = next(iter(loaders['train']))
imgs_denorm = (imgs_batch[:10] * std + mean).clamp(0, 1)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
classes = ['NORMAL', 'PNEUMONIA']
for i, ax in enumerate(axes.flat):
    img = imgs_denorm[i].permute(1, 2, 0).numpy()
    lbl = labels_batch[i].item()
    ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
    ax.set_title(classes[lbl], fontsize=10,
                 color='red' if lbl == 1 else 'blue')
    ax.axis('off')
plt.suptitle('Exemples radiographies (rouge=Pneumonie, bleu=Normal)',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(CFG['out_dir'], 'sample_images.png'), dpi=150)
plt.show()
"""))

cells.append(md("## 🏗️ Architecture CNN\nRéseau basé sur ResNet50 ou MobileNetV2 avec tête améliorée (BN + Dropout + Dense)."))

# ── CELLULE 9 : Modèle ────────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 9 : Classe PneumoniaCNN ──────────────────────────────────
class PneumoniaCNN(nn.Module):
    def __init__(self, backbone='resnet50', dropout=0.5, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        w = 'IMAGENET1K_V1' if pretrained else None

        if backbone == 'resnet50':
            base = models.resnet50(weights=w)
            in_f = base.fc.in_features                 # 2048
            self.features = nn.Sequential(
                *list(base.children())[:-1])

        elif backbone == 'mobilenet_v2':
            base = models.mobilenet_v2(weights=w)
            in_f = base.classifier[1].in_features      # 1280
            self.features = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Backbone inconnu : {backbone}")

        # Tête : BN → Dropout → Dense(256) → BN → Dropout → logit
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_f),
            nn.Dropout(dropout),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),   # logit brut — sigmoid géré par loss
        )
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_n(self, n=20):
        params = list(self.features.parameters())
        for p in params:      p.requires_grad = False
        for p in params[-n:]: p.requires_grad = True

    def forward(self, x):
        f = self.features(x)
        if self.backbone_name == 'mobilenet_v2':
            f = self.pool(f)
        f = f.view(f.size(0), -1)
        return self.head(f).squeeze(1)   # (B,) logit
"""))

cells.append(md("## 📐 Instanciation et paramètres\nAffichage du nombre de paramètres entraînables pour chaque backbone."))

# ── CELLULE 10 : Instanciation ────────────────────────────────────────
cells.append(code("""
# ── CELLULE 10 : Instanciation + affichage paramètres ────────────────
for bk in CFG['backbones']:
    m = PneumoniaCNN(bk, CFG['dropout'], CFG['pretrained'])
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters()
                    if p.requires_grad)
    size_mb   = total * 4 / 1e6
    print(f"[{bk:15s}] total={total:,} | "
          f"entraînables={trainable:,} | "
          f"taille≈{size_mb:.1f} MB")
del m
"""))

cells.append(md("## ⚖️ Loss et pos_weight\nCompensation du déséquilibre NORMAL/PNEUMONIA via `BCEWithLogitsLoss(pos_weight=...)`."))

# ── CELLULE 11 : Loss ─────────────────────────────────────────────────
cells.append(code("""
# ── CELLULE 11 : Calcul pos_weight + définition de la loss ───────────
n_neg = class_counts['train'].get('NORMAL', 1)
n_pos = class_counts['train'].get('PNEUMONIA', 1)
pos_w = torch.tensor([n_neg / n_pos],
                      dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

print(f"NORMAL          : {n_neg}")
print(f"PNEUMONIA       : {n_pos}")
print(f"pos_weight      : {pos_w.item():.4f}")
print(f"criterion sur   : {pos_w.device}")
"""))

cells.append(md("## 🔧 Fonctions utilitaires\n`evaluate()` parcourt **TOUS** les batches sans limite — correction du bug max_batches original."))

# ── CELLULE 12 : evaluate() ───────────────────────────────────────────
cells.append(code("""
# ── CELLULE 12 : Fonction evaluate() — SANS max_batches ──────────────
# ⚠️ CORRECTION : on parcourt TOUS les batches
# max_batches supprimé — c'était la cause de recall=0

@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    for imgs, lbls in loader:    # ← TOUS les batches
        imgs = imgs.to(DEVICE)
        lbls = lbls.float().to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, lbls).item() * len(lbls)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    y_true   = np.array(all_labels, dtype=int)
    y_scores = np.array(all_probs)
    y_pred   = (y_scores >= threshold).astype(int)

    rep = classification_report(
        y_true, y_pred,
        target_names=['NORMAL', 'PNEUMONIA'],
        output_dict=True, zero_division=0
    )
    auc = roc_auc_score(y_true, y_scores) \
          if len(np.unique(y_true)) == 2 else None

    return {
        'loss'     : total_loss / len(y_true),
        'accuracy' : rep['accuracy'],
        'precision': rep['PNEUMONIA']['precision'],
        'recall'   : rep['PNEUMONIA']['recall'],
        'f1'       : rep['PNEUMONIA']['f1-score'],
        'auc'      : auc,
        'y_true'   : y_true,
        'y_scores' : y_scores,
        'y_pred'   : y_pred,
    }

print("evaluate() définie ✅")
"""))

cells.append(md("## 🎯 Seuil optimal\nCalcul du seuil de décision optimal sur la courbe ROC (stratégie F1, Youden ou Recall)."))

# ── CELLULE 13 : find_optimal_threshold() ────────────────────────────
cells.append(code("""
# ── CELLULE 13 : Fonction find_optimal_threshold() ───────────────────
def find_optimal_threshold(y_true, y_scores, strategy='f1'):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    if strategy == 'youden':
        idx = np.argmax(tpr - fpr)

    elif strategy == 'recall':
        valid = np.where(tpr >= 0.90)[0]
        idx   = valid[0] if len(valid) > 0 \
                else np.argmax(tpr - fpr)

    else:  # f1
        best_f1, idx = 0.0, 0
        for k, thr in enumerate(thresholds):
            yp    = (y_scores >= thr).astype(int)
            score = f1_score(y_true, yp, zero_division=0)
            if score > best_f1:
                best_f1, idx = score, k

    return float(thresholds[idx])

print("find_optimal_threshold() définie ✅")
print(f"Stratégie active : {CFG['threshold_strategy']}")
"""))

cells.append(md("## 🏋️ Historique et stockage\nStructures pour sauvegarder les métriques des deux modèles."))

# ── CELLULE 14 : Init historique ──────────────────────────────────────
cells.append(code("""
# ── CELLULE 14 : Initialisation historiques et résultats ─────────────
summary_metrics = {}   # métriques finales test par backbone
histories       = {}   # courbes loss/recall/f1/auc par backbone
ckpt_paths      = {}   # chemins des meilleurs checkpoints

KEYS_HISTORY = ['train_loss', 'val_loss',
                'val_precision', 'val_recall',
                'val_f1', 'val_auc']

print("Structures d'historique initialisées ✅")
"""))

# ────────────────────────────────────────────────────────────
#  RESNET50
# ────────────────────────────────────────────────────────────
cells.append(md("""---
## ═══ ENTRAÎNEMENT RESNET50 ═══
**Phase 1** : backbone entièrement gelé — seule la tête est entraînée (15 époques).
**Phase 2** : unfreeze des 20 dernières couches — fine-tuning profond (30 époques).
"""))

# ── CELLULE 15 : ResNet50 instanciation ───────────────────────────────
cells.append(code("""
# ── CELLULE 15 : Instanciation ResNet50 ──────────────────────────────
model_r50 = PneumoniaCNN(
    backbone   = 'resnet50',
    dropout    = CFG['dropout'],
    pretrained = CFG['pretrained']
).to(DEVICE)

histories['resnet50'] = {k: [] for k in KEYS_HISTORY}

total_r50     = sum(p.numel() for p in model_r50.parameters())
trainable_r50 = sum(p.numel() for p in model_r50.parameters()
                    if p.requires_grad)
print(f"ResNet50 sur : {next(model_r50.parameters()).device}")
print(f"Paramètres   : {total_r50:,} total | {trainable_r50:,} entraînables")
"""))

cells.append(md("### Phase 1 — ResNet50 (backbone gelé)\nOptimizer Adam lr=1e-3 + ReduceLROnPlateau."))

# ── CELLULE 16 : ResNet50 Phase1 optimizer ────────────────────────────
cells.append(code("""
# ── CELLULE 16 : ResNet50 — Phase 1 optimizer + scheduler ────────────
opt_r50_p1 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_r50.parameters()),
    lr = CFG['lr_phase1']
)
sched_r50_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_r50_p1, mode='min', factor=0.5,
    patience=3, min_lr=1e-6, verbose=True
)
print("Phase 1 ResNet50 — optimizer prêt ✅")
print(f"lr initial : {CFG['lr_phase1']}")
"""))

# ── CELLULE 17 : ResNet50 Phase1 boucle ───────────────────────────────
cells.append(code("""
# ── CELLULE 17 : ResNet50 — Boucle Phase 1 (backbone gelé) ───────────
best_val_loss_r50  = float('inf')
best_threshold_r50 = 0.5
patience_ctr_r50   = 0

print(f"Phase 1 ResNet50 — {CFG['epochs_phase1']} époques | backbone gelé")
print("─" * 75)

for epoch in range(1, CFG['epochs_phase1'] + 1):
    model_r50.train()
    total_loss = 0.0
    for i, (imgs, lbls) in enumerate(loaders['train']):
        if CFG['debug'] and i >= 2: break
        imgs = imgs.to(DEVICE)
        lbls = lbls.float().to(DEVICE)
        opt_r50_p1.zero_grad()
        loss = criterion(model_r50(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_r50.parameters(), CFG['grad_clip'])
        opt_r50_p1.step()
        total_loss += loss.item() * len(lbls)

    train_loss = total_loss / len(loaders['train'].dataset)
    val_m      = evaluate(model_r50, loaders['val'])
    sched_r50_p1.step(val_m['loss'])

    histories['resnet50']['train_loss'].append(train_loss)
    histories['resnet50']['val_loss'].append(val_m['loss'])
    histories['resnet50']['val_precision'].append(val_m['precision'])
    histories['resnet50']['val_recall'].append(val_m['recall'])
    histories['resnet50']['val_f1'].append(val_m['f1'])
    histories['resnet50']['val_auc'].append(
        val_m['auc'] or float('nan'))

    auc_str = f"{val_m['auc']:.4f}" if val_m['auc'] else "N/A"
    print(f"  [phase1][ResNet50] ep {epoch:02d}/{CFG['epochs_phase1']} | "
          f"train={train_loss:.4f} | val={val_m['loss']:.4f} | "
          f"prec={val_m['precision']:.4f} | "
          f"recall={val_m['recall']:.4f} | "
          f"f1={val_m['f1']:.4f} | auc={auc_str}")

    if val_m['loss'] < best_val_loss_r50:
        best_val_loss_r50 = val_m['loss']
        patience_ctr_r50  = 0
        ckpt = os.path.join(CFG['out_dir'], 'best_resnet50.pt')
        torch.save(model_r50.state_dict(), ckpt)
        ckpt_paths['resnet50'] = ckpt
    else:
        patience_ctr_r50 += 1
        if patience_ctr_r50 >= CFG['patience']:
            print(f"  [EarlyStopping Phase1 ResNet50] ep {epoch}")
            break

print("✅ Phase 1 ResNet50 terminée.")
"""))

cells.append(md("### Phase 2 — ResNet50 (unfreeze partiel)\nDégel des 20 dernières couches. LR backbone=1e-5, LR tête=1e-4."))

# ── CELLULE 18 : ResNet50 Phase2 unfreeze + optimizer ─────────────────
cells.append(code("""
# ── CELLULE 18 : ResNet50 — Unfreeze + Phase 2 optimizer ─────────────
model_r50.unfreeze_last_n(CFG['n_unfreeze_layers'])

trainable_p2 = sum(p.numel() for p in model_r50.parameters()
                   if p.requires_grad)
print(f"Après unfreeze({CFG['n_unfreeze_layers']}) : "
      f"{trainable_p2:,} paramètres entraînables")

opt_r50_p2 = torch.optim.Adam([
    {'params': [p for p in model_r50.features.parameters()
                if p.requires_grad],
     'lr': CFG['lr_phase2_backbone']},
    {'params': model_r50.head.parameters(),
     'lr': CFG['lr_phase2_head']},
])
sched_r50_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_r50_p2, mode='min', factor=0.5,
    patience=3, min_lr=1e-7, verbose=True
)
patience_ctr_r50 = 0
print("Phase 2 ResNet50 — optimizer prêt ✅")
"""))

# ── CELLULE 19 : ResNet50 Phase2 boucle ───────────────────────────────
cells.append(code("""
# ── CELLULE 19 : ResNet50 — Boucle Phase 2 + EarlyStopping ───────────
print(f"Phase 2 ResNet50 — {CFG['epochs_phase2']} époques | unfreeze actif")
print("─" * 75)

for epoch in range(1, CFG['epochs_phase2'] + 1):
    model_r50.train()
    total_loss = 0.0
    for i, (imgs, lbls) in enumerate(loaders['train']):
        if CFG['debug'] and i >= 2: break
        imgs = imgs.to(DEVICE)
        lbls = lbls.float().to(DEVICE)
        opt_r50_p2.zero_grad()
        loss = criterion(model_r50(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_r50.parameters(), CFG['grad_clip'])
        opt_r50_p2.step()
        total_loss += loss.item() * len(lbls)

    train_loss = total_loss / len(loaders['train'].dataset)
    val_m      = evaluate(model_r50, loaders['val'])
    sched_r50_p2.step(val_m['loss'])

    histories['resnet50']['train_loss'].append(train_loss)
    histories['resnet50']['val_loss'].append(val_m['loss'])
    histories['resnet50']['val_precision'].append(val_m['precision'])
    histories['resnet50']['val_recall'].append(val_m['recall'])
    histories['resnet50']['val_f1'].append(val_m['f1'])
    histories['resnet50']['val_auc'].append(
        val_m['auc'] or float('nan'))

    auc_str = f"{val_m['auc']:.4f}" if val_m['auc'] else "N/A"
    print(f"  [phase2][ResNet50] ep {epoch:02d}/{CFG['epochs_phase2']} | "
          f"train={train_loss:.4f} | val={val_m['loss']:.4f} | "
          f"prec={val_m['precision']:.4f} | "
          f"recall={val_m['recall']:.4f} | "
          f"f1={val_m['f1']:.4f} | auc={auc_str}")

    if val_m['loss'] < best_val_loss_r50:
        best_val_loss_r50 = val_m['loss']
        patience_ctr_r50  = 0
        ckpt = os.path.join(CFG['out_dir'], 'best_resnet50.pt')
        torch.save(model_r50.state_dict(), ckpt)
        ckpt_paths['resnet50'] = ckpt
        if val_m['auc']:
            best_threshold_r50 = find_optimal_threshold(
                val_m['y_true'], val_m['y_scores'],
                CFG['threshold_strategy'])
            print(f"    → checkpoint sauvegardé | "
                  f"seuil={best_threshold_r50:.4f}")
    else:
        patience_ctr_r50 += 1
        if patience_ctr_r50 >= CFG['patience']:
            print(f"  [EarlyStopping Phase2 ResNet50] ep {epoch}")
            break

print("✅ Phase 2 ResNet50 terminée.")
"""))

cells.append(md("### Courbes ResNet50\nLoss, Recall, F1 et AUC sur les 2 phases."))

# ── CELLULE 20 : Courbes ResNet50 ────────────────────────────────────
cells.append(code("""
# ── CELLULE 20 : Courbes d'entraînement ResNet50 ─────────────────────
h = histories['resnet50']
sep = CFG['epochs_phase1']   # ligne de séparation Phase1/Phase2

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Courbes entraînement — ResNet50', fontsize=14)

plots = [
    ('train_loss', 'val_loss',      'Loss',      'val_loss'),
    ('val_recall', 'val_f1',        'Recall & F1', None),
    ('val_precision', None,         'Précision',  None),
    ('val_auc', None,               'AUC-ROC',    None),
]

for ax, (k1, k2, title, _) in zip(axes.flat, plots):
    ax.plot(h[k1], label=k1)
    if k2 and k2 in h:
        ax.plot(h[k2], label=k2)
    ax.axvline(sep, color='gray', linestyle='--',
               label='Début Phase 2', alpha=0.7)
    ax.set_title(title); ax.legend(); ax.grid(True)
    ax.set_xlabel('Époque')

plt.tight_layout()
path = os.path.join(CFG['out_dir'], 'curves_resnet50.png')
plt.savefig(path, dpi=150)
plt.show()
print(f"Courbes sauvegardées → {path}")
"""))

# ────────────────────────────────────────────────────────────
#  MOBILENETV2
# ────────────────────────────────────────────────────────────
cells.append(md("""---
## ═══ ENTRAÎNEMENT MOBILENETV2 ═══
MobileNetV2 est prioritaire pour le **contexte africain** :
léger (~14 MB), convertible TensorFlow Lite, fonctionne **hors ligne sur Android**.
"""))

# ── CELLULE 21 : MobileNetV2 instanciation ────────────────────────────
cells.append(code("""
# ── CELLULE 21 : Instanciation MobileNetV2 ───────────────────────────
model_mv2 = PneumoniaCNN(
    backbone   = 'mobilenet_v2',
    dropout    = CFG['dropout'],
    pretrained = CFG['pretrained']
).to(DEVICE)

histories['mobilenet_v2'] = {k: [] for k in KEYS_HISTORY}

total_mv2     = sum(p.numel() for p in model_mv2.parameters())
trainable_mv2 = sum(p.numel() for p in model_mv2.parameters()
                    if p.requires_grad)
print(f"MobileNetV2 sur : {next(model_mv2.parameters()).device}")
print(f"Paramètres      : {total_mv2:,} total | "
      f"{trainable_mv2:,} entraînables")
"""))

cells.append(md("### Phase 1 — MobileNetV2 (backbone gelé)"))

# ── CELLULE 22 : MobileNetV2 Phase1 optimizer ─────────────────────────
cells.append(code("""
# ── CELLULE 22 : MobileNetV2 — Phase 1 optimizer + scheduler ─────────
best_val_loss_mv2  = float('inf')
best_threshold_mv2 = 0.5
patience_ctr_mv2   = 0

opt_mv2_p1 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_mv2.parameters()),
    lr = CFG['lr_phase1']
)
sched_mv2_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_mv2_p1, mode='min', factor=0.5,
    patience=3, min_lr=1e-6, verbose=True
)
print("Phase 1 MobileNetV2 — optimizer prêt ✅")
"""))

# ── CELLULE 23 : MobileNetV2 Phase1 boucle ───────────────────────────
cells.append(code("""
# ── CELLULE 23 : MobileNetV2 — Boucle Phase 1 ────────────────────────
print(f"Phase 1 MobileNetV2 — {CFG['epochs_phase1']} époques")
print("─" * 75)

for epoch in range(1, CFG['epochs_phase1'] + 1):
    model_mv2.train()
    total_loss = 0.0
    for i, (imgs, lbls) in enumerate(loaders['train']):
        if CFG['debug'] and i >= 2: break
        imgs = imgs.to(DEVICE)
        lbls = lbls.float().to(DEVICE)
        opt_mv2_p1.zero_grad()
        loss = criterion(model_mv2(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_mv2.parameters(), CFG['grad_clip'])
        opt_mv2_p1.step()
        total_loss += loss.item() * len(lbls)

    train_loss = total_loss / len(loaders['train'].dataset)
    val_m      = evaluate(model_mv2, loaders['val'])
    sched_mv2_p1.step(val_m['loss'])

    histories['mobilenet_v2']['train_loss'].append(train_loss)
    histories['mobilenet_v2']['val_loss'].append(val_m['loss'])
    histories['mobilenet_v2']['val_precision'].append(val_m['precision'])
    histories['mobilenet_v2']['val_recall'].append(val_m['recall'])
    histories['mobilenet_v2']['val_f1'].append(val_m['f1'])
    histories['mobilenet_v2']['val_auc'].append(
        val_m['auc'] or float('nan'))

    auc_str = f"{val_m['auc']:.4f}" if val_m['auc'] else "N/A"
    print(f"  [phase1][MobileNetV2] ep {epoch:02d}/{CFG['epochs_phase1']} | "
          f"train={train_loss:.4f} | val={val_m['loss']:.4f} | "
          f"prec={val_m['precision']:.4f} | "
          f"recall={val_m['recall']:.4f} | "
          f"f1={val_m['f1']:.4f} | auc={auc_str}")

    if val_m['loss'] < best_val_loss_mv2:
        best_val_loss_mv2 = val_m['loss']
        patience_ctr_mv2  = 0
        ckpt = os.path.join(CFG['out_dir'], 'best_mobilenet_v2.pt')
        torch.save(model_mv2.state_dict(), ckpt)
        ckpt_paths['mobilenet_v2'] = ckpt
    else:
        patience_ctr_mv2 += 1
        if patience_ctr_mv2 >= CFG['patience']:
            print(f"  [EarlyStopping Phase1 MobileNetV2] ep {epoch}")
            break

print("✅ Phase 1 MobileNetV2 terminée.")
"""))

cells.append(md("### Phase 2 — MobileNetV2 (unfreeze partiel)"))

# ── CELLULE 24 : MobileNetV2 Phase2 unfreeze ──────────────────────────
cells.append(code("""
# ── CELLULE 24 : MobileNetV2 — Unfreeze + Phase 2 optimizer ──────────
model_mv2.unfreeze_last_n(CFG['n_unfreeze_layers'])

trainable_p2 = sum(p.numel() for p in model_mv2.parameters()
                   if p.requires_grad)
print(f"Après unfreeze({CFG['n_unfreeze_layers']}) : "
      f"{trainable_p2:,} paramètres entraînables")

opt_mv2_p2 = torch.optim.Adam([
    {'params': [p for p in model_mv2.features.parameters()
                if p.requires_grad],
     'lr': CFG['lr_phase2_backbone']},
    {'params': model_mv2.head.parameters(),
     'lr': CFG['lr_phase2_head']},
])
sched_mv2_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt_mv2_p2, mode='min', factor=0.5,
    patience=3, min_lr=1e-7, verbose=True
)
patience_ctr_mv2 = 0
print("Phase 2 MobileNetV2 — optimizer prêt ✅")
"""))

# ── CELLULE 25 : MobileNetV2 Phase2 boucle ───────────────────────────
cells.append(code("""
# ── CELLULE 25 : MobileNetV2 — Boucle Phase 2 + EarlyStopping ────────
print(f"Phase 2 MobileNetV2 — {CFG['epochs_phase2']} époques")
print("─" * 75)

for epoch in range(1, CFG['epochs_phase2'] + 1):
    model_mv2.train()
    total_loss = 0.0
    for i, (imgs, lbls) in enumerate(loaders['train']):
        if CFG['debug'] and i >= 2: break
        imgs = imgs.to(DEVICE)
        lbls = lbls.float().to(DEVICE)
        opt_mv2_p2.zero_grad()
        loss = criterion(model_mv2(imgs), lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_mv2.parameters(), CFG['grad_clip'])
        opt_mv2_p2.step()
        total_loss += loss.item() * len(lbls)

    train_loss = total_loss / len(loaders['train'].dataset)
    val_m      = evaluate(model_mv2, loaders['val'])
    sched_mv2_p2.step(val_m['loss'])

    histories['mobilenet_v2']['train_loss'].append(train_loss)
    histories['mobilenet_v2']['val_loss'].append(val_m['loss'])
    histories['mobilenet_v2']['val_precision'].append(val_m['precision'])
    histories['mobilenet_v2']['val_recall'].append(val_m['recall'])
    histories['mobilenet_v2']['val_f1'].append(val_m['f1'])
    histories['mobilenet_v2']['val_auc'].append(
        val_m['auc'] or float('nan'))

    auc_str = f"{val_m['auc']:.4f}" if val_m['auc'] else "N/A"
    print(f"  [phase2][MobileNetV2] ep {epoch:02d}/{CFG['epochs_phase2']} | "
          f"train={train_loss:.4f} | val={val_m['loss']:.4f} | "
          f"prec={val_m['precision']:.4f} | "
          f"recall={val_m['recall']:.4f} | "
          f"f1={val_m['f1']:.4f} | auc={auc_str}")

    if val_m['loss'] < best_val_loss_mv2:
        best_val_loss_mv2 = val_m['loss']
        patience_ctr_mv2  = 0
        ckpt = os.path.join(CFG['out_dir'], 'best_mobilenet_v2.pt')
        torch.save(model_mv2.state_dict(), ckpt)
        ckpt_paths['mobilenet_v2'] = ckpt
        if val_m['auc']:
            best_threshold_mv2 = find_optimal_threshold(
                val_m['y_true'], val_m['y_scores'],
                CFG['threshold_strategy'])
            print(f"    → checkpoint sauvegardé | "
                  f"seuil={best_threshold_mv2:.4f}")
    else:
        patience_ctr_mv2 += 1
        if patience_ctr_mv2 >= CFG['patience']:
            print(f"  [EarlyStopping Phase2 MobileNetV2] ep {epoch}")
            break

print("✅ Phase 2 MobileNetV2 terminée.")
"""))

# ── CELLULE 26 : Courbes MobileNetV2 ─────────────────────────────────
cells.append(md("### Courbes MobileNetV2"))
cells.append(code("""
# ── CELLULE 26 : Courbes d'entraînement MobileNetV2 ──────────────────
h   = histories['mobilenet_v2']
sep = CFG['epochs_phase1']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Courbes entraînement — MobileNetV2', fontsize=14)

for ax, (k1, k2, title) in zip(axes.flat, [
    ('train_loss',    'val_loss',  'Loss'),
    ('val_recall',    'val_f1',    'Recall & F1'),
    ('val_precision', None,        'Précision'),
    ('val_auc',       None,        'AUC-ROC'),
]):
    ax.plot(h[k1], label=k1)
    if k2: ax.plot(h[k2], label=k2)
    ax.axvline(sep, color='gray', linestyle='--',
               label='Début Phase 2', alpha=0.7)
    ax.set_title(title); ax.legend(); ax.grid(True)

plt.tight_layout()
path = os.path.join(CFG['out_dir'], 'curves_mobilenet_v2.png')
plt.savefig(path, dpi=150)
plt.show()
print(f"Courbes sauvegardées → {path}")
"""))

# ────────────────────────────────────────────────────────────
#  ÉVALUATION FINALE
# ────────────────────────────────────────────────────────────
cells.append(md("""---
## ═══ ÉVALUATION FINALE — Test Set ═══
Chargement des meilleurs checkpoints et évaluation **complète** sur le test set.
"""))

# ── CELLULE 27 : Chargement meilleurs modèles ─────────────────────────
cells.append(code("""
# ── CELLULE 27 : Chargement meilleurs checkpoints ────────────────────
model_r50.load_state_dict(
    torch.load(ckpt_paths['resnet50'], map_location=DEVICE))
model_mv2.load_state_dict(
    torch.load(ckpt_paths['mobilenet_v2'], map_location=DEVICE))
print("✅ Meilleurs checkpoints chargés")
print(f"   ResNet50    : {ckpt_paths['resnet50']}")
print(f"   MobileNetV2 : {ckpt_paths['mobilenet_v2']}")
"""))

# ── CELLULE 28 : Éval ResNet50 ────────────────────────────────────────
cells.append(md("### Évaluation ResNet50 — Test Set"))
cells.append(code("""
# ── CELLULE 28 : Évaluation complète ResNet50 sur Test Set ───────────
test_r50 = evaluate(model_r50, loaders['test'],
                    threshold=best_threshold_r50)
print("\\n" + "="*58)
print(f"  RESNET50 | seuil={best_threshold_r50:.4f}")
print("="*58)
print(f"  Accuracy  : {test_r50['accuracy']:.4f}")
print(f"  Precision : {test_r50['precision']:.4f}")
print(f"  Recall    : {test_r50['recall']:.4f}  ← priorité clinique")
print(f"  F1-Score  : {test_r50['f1']:.4f}")
print(f"  AUC-ROC   : {test_r50['auc']:.4f}")
print("="*58)
print(classification_report(
    test_r50['y_true'], test_r50['y_pred'],
    target_names=['NORMAL','PNEUMONIA'], zero_division=0))
"""))

# ── CELLULE 29 : Matrice confusion ResNet50 ───────────────────────────
cells.append(code("""
# ── CELLULE 29 : Matrice de confusion ResNet50 ───────────────────────
cm = confusion_matrix(test_r50['y_true'], test_r50['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                fontsize=14,
                color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['NORMAL','PNEUMONIA'], fontsize=11)
ax.set_yticklabels(['NORMAL','PNEUMONIA'], fontsize=11)
ax.set_xlabel('Prédit', fontsize=12)
ax.set_ylabel('Réel',   fontsize=12)
ax.set_title(f'Matrice confusion — ResNet50\\n'
             f'(seuil={best_threshold_r50:.2f})', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(CFG['out_dir'],
            'confusion_resnet50.png'), dpi=150)
plt.show()
"""))

# ── CELLULE 30 : ROC ResNet50 ─────────────────────────────────────────
cells.append(code("""
# ── CELLULE 30 : Courbe ROC ResNet50 avec seuil optimal marqué ───────
fpr, tpr, thrs = roc_curve(test_r50['y_true'], test_r50['y_scores'])
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f"AUC = {test_r50['auc']:.4f}")
ax.plot([0,1],[0,1], 'k--', lw=1)

# Marquer le seuil optimal
idx  = np.argmin(np.abs(thrs - best_threshold_r50))
ax.scatter(fpr[idx], tpr[idx], color='red', s=100, zorder=5,
           label=f"Seuil optimal = {best_threshold_r50:.2f}")
ax.annotate(f"  seuil={best_threshold_r50:.2f}",
            (fpr[idx], tpr[idx]), fontsize=10)

ax.set_xlabel('Taux Faux Positifs', fontsize=12)
ax.set_ylabel('Sensibilité (Recall)', fontsize=12)
ax.set_title('Courbe ROC — ResNet50', fontsize=13)
ax.legend(fontsize=11); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CFG['out_dir'],
            'roc_resnet50.png'), dpi=150)
plt.show()
"""))

# ── CELLULE 31 : Éval MobileNetV2 ────────────────────────────────────
cells.append(md("### Évaluation MobileNetV2 — Test Set"))
cells.append(code("""
# ── CELLULE 31 : Évaluation complète MobileNetV2 sur Test Set ────────
test_mv2 = evaluate(model_mv2, loaders['test'],
                    threshold=best_threshold_mv2)
print("\\n" + "="*58)
print(f"  MOBILENETV2 | seuil={best_threshold_mv2:.4f}")
print("="*58)
print(f"  Accuracy  : {test_mv2['accuracy']:.4f}")
print(f"  Precision : {test_mv2['precision']:.4f}")
print(f"  Recall    : {test_mv2['recall']:.4f}  ← priorité clinique")
print(f"  F1-Score  : {test_mv2['f1']:.4f}")
print(f"  AUC-ROC   : {test_mv2['auc']:.4f}")
print("="*58)
print(classification_report(
    test_mv2['y_true'], test_mv2['y_pred'],
    target_names=['NORMAL','PNEUMONIA'], zero_division=0))
"""))

# ── CELLULE 32 : Matrice confusion MobileNetV2 ───────────────────────
cells.append(code("""
# ── CELLULE 32 : Matrice de confusion MobileNetV2 ────────────────────
cm = confusion_matrix(test_mv2['y_true'], test_mv2['y_pred'])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Greens')
plt.colorbar(im, ax=ax)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                fontsize=14,
                color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['NORMAL','PNEUMONIA'], fontsize=11)
ax.set_yticklabels(['NORMAL','PNEUMONIA'], fontsize=11)
ax.set_xlabel('Prédit', fontsize=12)
ax.set_ylabel('Réel',   fontsize=12)
ax.set_title(f'Matrice confusion — MobileNetV2\\n'
             f'(seuil={best_threshold_mv2:.2f})', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(CFG['out_dir'],
            'confusion_mobilenet_v2.png'), dpi=150)
plt.show()
"""))

# ── CELLULE 33 : ROC MobileNetV2 ─────────────────────────────────────
cells.append(code("""
# ── CELLULE 33 : Courbe ROC MobileNetV2 avec seuil optimal marqué ────
fpr, tpr, thrs = roc_curve(test_mv2['y_true'], test_mv2['y_scores'])
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color='green', lw=2,
        label=f"AUC = {test_mv2['auc']:.4f}")
ax.plot([0,1],[0,1], 'k--', lw=1)

idx = np.argmin(np.abs(thrs - best_threshold_mv2))
ax.scatter(fpr[idx], tpr[idx], color='red', s=100, zorder=5,
           label=f"Seuil optimal = {best_threshold_mv2:.2f}")
ax.annotate(f"  seuil={best_threshold_mv2:.2f}",
            (fpr[idx], tpr[idx]), fontsize=10)

ax.set_xlabel('Taux Faux Positifs', fontsize=12)
ax.set_ylabel('Sensibilité (Recall)', fontsize=12)
ax.set_title('Courbe ROC — MobileNetV2', fontsize=13)
ax.legend(fontsize=11); ax.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(CFG['out_dir'],
            'roc_mobilenet_v2.png'), dpi=150)
plt.show()
"""))

cells.append(md("""---
## 📊 Tableau comparatif final
Comparaison ResNet50 vs MobileNetV2 pour le **Chapitre 3 du mémoire**.
"""))

# ── CELLULE 34 : Tableau comparatif ──────────────────────────────────
cells.append(code("""
# ── CELLULE 34 : Tableau comparatif ResNet50 vs MobileNetV2 ──────────
def model_size_mb(m):
    return sum(p.numel() for p in m.parameters()) * 4 / 1e6

def infer_time_ms(m, n=100):
    m.eval()
    x = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        t0 = time.time()
        for _ in range(n): m(x)
    return (time.time() - t0) / n * 1000

size_r50  = model_size_mb(model_r50)
size_mv2  = model_size_mb(model_mv2)
time_r50  = infer_time_ms(model_r50)
time_mv2  = infer_time_ms(model_mv2)

rows = [
    ('Accuracy',                  f"{test_r50['accuracy']:.4f}",
                                  f"{test_mv2['accuracy']:.4f}"),
    ('Precision (Pneumonie)',      f"{test_r50['precision']:.4f}",
                                  f"{test_mv2['precision']:.4f}"),
    ('Recall (Pneumonie)',         f"{test_r50['recall']:.4f}",
                                  f"{test_mv2['recall']:.4f}"),
    ('F1-Score',                   f"{test_r50['f1']:.4f}",
                                  f"{test_mv2['f1']:.4f}"),
    ('AUC-ROC',                    f"{test_r50['auc']:.4f}",
                                  f"{test_mv2['auc']:.4f}"),
    ('Seuil optimal',              f"{best_threshold_r50:.4f}",
                                  f"{best_threshold_mv2:.4f}"),
    ('Taille modèle (MB)',         f"{size_r50:.1f}",
                                  f"{size_mv2:.1f}"),
    ('Temps inférence CPU (ms)',   f"{time_r50:.1f}",
                                  f"{time_mv2:.1f}"),
    ('Déployable mobile (TF Lite)','❌ Non (>90 MB)',
                                  '✅ Oui (<15 MB)'),
]

df = pd.DataFrame(rows, columns=['Métrique','ResNet50','MobileNetV2'])
print(df.to_string(index=False))
print()
print("="*58)
print("  RECOMMANDATION — CONTEXTE AFRICAIN")
print("="*58)
print("  ResNet50    → performance maximale, GPU requis")
print("               hôpitaux équipés, télémédecine")
print()
print("  MobileNetV2 → déployabilité terrain prioritaire")
print("               cliniques rurales sans GPU")
print("               Android hors ligne via TF Lite")
print("               → RECOMMANDÉ contexte africain ✅")
print("="*58)
"""))

# ── CELLULE 35 : Sauvegarde JSON ──────────────────────────────────────
cells.append(md("## 💾 Sauvegarde des résultats\nExport JSON pour le mémoire et la reproductibilité."))
cells.append(code("""
# ── CELLULE 35 : Sauvegarde résultats JSON ───────────────────────────
results_export = {}
for name, test_m, thr in [
    ('resnet50',     test_r50, best_threshold_r50),
    ('mobilenet_v2', test_mv2, best_threshold_mv2),
]:
    results_export[name] = {
        'accuracy'        : float(test_m['accuracy']),
        'precision'       : float(test_m['precision']),
        'recall'          : float(test_m['recall']),
        'f1'              : float(test_m['f1']),
        'auc_roc'         : float(test_m['auc']),
        'best_threshold'  : float(thr),
        'threshold_strategy': CFG['threshold_strategy'],
    }

path = os.path.join(CFG['out_dir'], 'results_final.json')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(results_export, f, indent=2, ensure_ascii=False)

print(f"Résultats sauvegardés → {path}")
print(json.dumps(results_export, indent=2))
"""))

cells.append(md("""---
## ✅ Résumé final

| Métrique | ResNet50 | MobileNetV2 |
|---|---|---|
| Recall (Pneumonie) | voir résultats | voir résultats |
| F1-Score | voir résultats | voir résultats |
| AUC-ROC | voir résultats | voir résultats |
| Taille modèle | ~98 MB | ~14 MB |
| Déployable mobile | ❌ | ✅ |

**Conclusion pour le mémoire :**
MobileNetV2 est recommandé pour le contexte africain grâce à sa légèreté
et sa compatibilité TensorFlow Lite pour déploiement hors ligne sur Android.
ResNet50 reste la référence en termes de performance pure.
"""))

# ── Écriture du fichier ───────────────────────────────────────────────
nb.cells = cells
output_path = "notebook_pneumonia_v2.ipynb"
with open(output_path, "w", encoding="utf-8", newline="\n") as f:
    nbformat.write(nb, f)

print(f"\nNotebook cree : {output_path}")
print(f"Nombre de cellules : {len(nb.cells)}")
