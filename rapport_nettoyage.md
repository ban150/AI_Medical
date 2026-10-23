# Rapport de nettoyage : projet AI_Medical et données de soutenance

Date : 22/10/2026. Phase **lecture seule** : aucune donnée supprimée, déplacée, renommée ni modifiée. Aucune commande Git n'a été exécutée en écriture.

**Fichiers écrits pendant cette phase, et uniquement ceux-là :**
- `rapport_nettoyage.md` (ce fichier) ;
- `nettoyage_propose.ps1` ;
- le contenu de `audit_out/`, où `audit_donnees.py` a **réécrit** les 9 fichiers du 05/10. Comme ils sont suivis par Git, `git status` les montre modifiés ;
- les ajouts dans `audit_out/` : `audit_complements.py` et `resultats_complements.md`, ainsi que 5 CSV (`paul_doublons_internes.csv`, `splits_identites.csv`, `splits_fuite_identites.csv`, `tb_vs_rsna_phash.csv` et `ng_recouvrement_phash.csv`).

## Étape 0 : accès

| Élément | Résultat |
|---|---|
| `C:\Z_Soutenance\DATA` | **N'existe pas.** Sur ta consigne, j'ai utilisé `C:\Project_D\Z_Soutenance\DATA`. |
| Projet `C:\Users\User\PycharmProjects\AI_Medical` | Lisible. |
| Zip Zenodo | `C:\Project_D\Z_Soutenance\NG Pneum Dataset.zip` (190 247 603 octets). Son **MD5 `a04ce100…028c` est identique à celui publié sur Zenodo.** |
| Copie extraite | `DATA\NG Pneum Dataset\NG Dataset` : ses 190 fichiers sont identiques octet par octet à ceux du zip. |

### Arborescence du projet (2 niveaux)

| Chemin | Fichiers | Taille | Modifié | Remarque |
|---|---:|---:|---|---|
| `AI_Medical/` | 300 | 774,5 Mo | 10/06 | **Dépôt Git imbriqué** (son propre `.git`, 372,6 Mo), enregistré comme lien de sous-module sans `.gitmodules` |
| `AI_Medical/models/checkpoints` | 11 | 387 Mo | 10/06 | 7 fichiers `.pt` (ResNet50 et MobileNetV2) et 3 figures |
| `AI_Medical/data` | 51 | 5,3 Mo | 10/06 | Code de fusion et de split, plus 36 PNG factices (`mock_*`) |
| `AI_Medical/config`, `eda`, `evaluation`, `training`, `utils` | 4 à 6 chacun | < 0,1 Mo | 30/03 à 10/06 | Code |
| `AI_Medical/temp.h5` | 1 | 10,2 Mo | 30/03 | Voir étape 2 |
| `venv_gpu/` | 21 503 | 4 611 Mo | 07/10 | Python 3.10 avec PyTorch CUDA, **suivi par Git** |
| `.venv/` | 46 688 | 1 752 Mo | 07/04 | Python 3.14, non suivi (son propre `.gitignore`) |
| `audit_out/` | 16 | ≈ 26 Mo | 22/10 | Sorties de l'audit |
| `.idea/`, `.claude/`, `__pycache__/` | 7, 1, 1 | < 0,1 Mo | | Configuration de l'IDE |
| Racine | 9 fichiers | ≈ 1,1 Mo | 14/04 à 05/10 | `audit_donnees.py`, `create_notebook*.py`, 2 notebooks, `main.py`, `pyproject.toml`, `rapport_audit.md` |
| `.git/` | | 2 524 Mo | | Le pack fait 2,29 Gio |

## Étape 1 : Git (lecture seule)

### État

| Élément | Dépôt externe (`AI_Medical`) | Dépôt imbriqué (`AI_Medical/AI_Medical`) |
|---|---|---|
| Branche | `main` (propre) | `fix-leak-clahe` (propre) |
| Commits | `3a174a3d`, `57a19775`, `c3b222fa`, `921bd35d` (racine) | `95f9aed`, `8c4f0de`, `2151f6d`, `29d8d16`, puis `ac7c0a0` |
| Distante | `origin = github.com/ban150/AI_Medical`, `origin/master = ac7c0a0` | Même distante, `origin/master = ac7c0a0` |
| Non poussé | 4 commits. **`main` n'a aucun ancêtre commun avec `origin/master`.** | 4 commits |
| Autres | 1 stash (`epitaxy: pre-switch from main`). Branches distantes `claude/elegant-shannon-2fh9t7` et `claude/kind-bell-xsh23y`. | Aucun |
| `.gitignore` | **Absent** | **Absent** |

### Le commit `3a174a3d` n'est pas la cause des 2,29 Go

- **Ce qu'il contient :** `git show --stat` donne 13 fichiers et 476 lignes. Ce sont `audit_donnees.py`, `rapport_audit.md`, `audit_out/*` (alors vides), le notebook v2 et le lien vers le dépôt imbriqué. Le commit est léger.
- **D'où viennent les 2,29 Gio :** du commit racine **`921bd35d` (10/04)**. Il ajoute les **21 503 fichiers de `venv_gpu/`**, soit 4 561 Mo non compressés.
- **Pourquoi le push échoue :** pousser `main` envoie tout l'historique, donc aussi ce commit. GitHub refuse les fichiers de plus de 100 Mo.

Les plus gros fichiers de l'historique, tous dans `venv_gpu/Lib/site-packages/torch/lib/` :

| Fichier | Taille |
|---|---:|
| `torch_cuda.dll` | 896 Mo |
| `dnnl.lib` | 623 Mo |
| `cudnn_engines_precompiled64_9.dll` | 562 Mo |
| `cublasLt64_12.dll` | 514 Mo |
| `cusparse64_12.dll` | 250 Mo |
| `torch_cpu.dll` | 240 Mo |
| `cudnn_adv64_9.dll` | 230 Mo |
| `cufft64_11.dll` | 182 Mo |

**Aucune donnée d'image n'est dans le dépôt externe.** Le dépôt imbriqué, lui, suit :
- 7 fichiers `.pt` de 9 à 90 Mo, sous la limite de GitHub mais lourds ;
- `temp.h5` (10 Mo) ;
- 36 PNG factices ;
- des `__pycache__`.

**Le code de `AI_Medical/AI_Medical` n'est pas dans le dépôt externe.** Celui-ci ne stocke qu'un pointeur vers le commit `95f9aed`, et ce commit n'existe que sur ta machine.

### `.gitignore` à créer

Les deux dépôts n'en ont pas. Il faut donc tout ajouter, y compris ce que tu demandes : `data/`, `*.h5`, `*.zip`, `*.jpeg`, `*.jpg`, `*.png`, `*.dcm` et `audit_out/`.

```gitignore
# Environnements
venv_gpu/
.venv/
__pycache__/
*.pyc
# Données et modèles
data/
DATA/
*.dcm
*.jpeg
*.jpg
*.png
*.zip
*.h5
*.pt
models/checkpoints/
# Sorties d'audit et d'exécution
audit_out/
reports/
outputs/
# IDE
.idea/workspace.xml
.claude/settings.local.json
```

Ajouter `*.png` exclut aussi les figures du mémoire. Si tu veux les versionner, ajoute une exception du type `!figures/*.png`.

### Plan pour sortir `venv_gpu` sans perdre le code (à ne pas exécuter sans ton accord)

1. **Sauvegarde.** Créer un `git bundle --all` de chacun des deux dépôts. Copier aussi le dossier projet sans les environnements, avec `robocopy`, vers `C:\Project_D\Z_Soutenance\BACKUP\`.
2. **`.gitignore`.** Créer le fichier ci-dessus à la racine, avec `venv_gpu/`.
3. **Branche neuve, sans réécrire l'historique.**
   - Lancer `git checkout --orphan propre`. Cette commande garde le dossier de travail tel quel. Ne pas utiliser `git switch --orphan`, qui efface les fichiers suivis du disque.
   - Retirer de l'index `venv_gpu` et le lien `AI_Medical`, avec `git rm -r --cached` : les fichiers restent sur le disque.
   - Intégrer le code du dépôt imbriqué comme fichiers normaux. Pour cela, il faut d'abord mettre de côté son dossier `.git`, une fois le bundle de l'étape 1 fait.
   - Faire un commit, puis `git push -u origin propre`. C'est un push normal, sans `--force`.
   
   L'ancienne `main` reste intacte en local et n'est jamais poussée.
4. **Option à éviter.** Réécrire `main` avec `git filter-repo --path venv_gpu --invert-paths` réécrit l'historique. Je ne le propose pas.

Les commandes sont dans `nettoyage_propose.ps1`, section 1, toutes commentées.

## Étape 2 : tri du projet

### Les problèmes suspectés, vérifiés dans le code

| Suspicion | Verdict | Preuve |
|---|---|---|
| Doublons dans GOGETA : `Fusion.py:37` écrit `fichier.jpeg`, `test_1.py:80` écrit `kermany_fichier.jpeg` | **Exact dans la version d'origine** (`git show ac7c0a0:Fusion.py`, lignes 37 et 80). Aujourd'hui `Fusion.py` et `test_1.py` ne font que 10 à 12 lignes et appellent `data/dataset_fusion.py`. | GOGETA ne contient plus de noms sans préfixe. Il reste 38 copies exactes et environ 3 500 jumelles TB↔RSNA (voir l'étape 3). |
| GOGETA jamais vidé | **Confirmé.** | `dataset_fusion.py:302-304` crée le dossier avec `exist_ok` et ne le purge jamais. |
| Tuberculose étiquetée PNEUMONIA | **Confirmé.** | `dataset_fusion.py:268` aujourd'hui ; `test_1.py:250` et `Fusion.py:119` à l'origine. Résultat : 700 images TB dans `GOGETA/PNEUMONIA`. |
| Fuite entre train, val et test, SPLITS jamais purgé | **Confirmé et mesuré** (voir l'étape 3). | `create_splits.py` ne purge pas. `run_pipeline.py:31-45` purge, mais SPLITS contient deux exécutions superposées. |
| Fenêtrage de TDM sur des radios (`Fusion.py:81-82`) | **Exact dans l'original** (`np.clip(img, -1000, 600)`) et dans `data/legacy_fusion_v1_original.py`. **Absent du pipeline actuel**, qui utilise une normalisation min-max. | Sans effet sur les données présentes : les chemins RSNA de la v1 étaient faux et `cv2.imshow` bloquait. |
| CLAHE appliqué seulement à RSNA (`test_1.py:111`) | **Exact dans l'original.** Corrigé dans `dataset_fusion.py` (commit `2151f6d`). | **Mais SPLITS mélange** des PNG avec CLAHE (2e exécution) et des JPEG bruts de PAUL (1re exécution). |
| Effectifs en dur (`entrainement.py:59-60`) | **Exact dans l'original** (22463 / 20808). Corrigé : `training/train.py:142` calcule `pos_weight` à partir des données. | |
| Push échoué, commit `3a174a3d` de 2,29 Go | **Inexact sur la cause.** | Voir l'étape 1 : c'est `venv_gpu` dans `921bd35d`. |

### Les scripts qui écrivent dans GOGETA ou SPLITS, dans l'ordre reconstitué

Ordre établi d'après le code, l'historique Git et les dates des fichiers. `copy2` conserve la date d'origine des fichiers.

| Ordre | Script | Effet |
|---:|---|---|
| 1 | `Fusion.py` v1 (`ac7c0a0`) | Écrit dans GOGETA sans préfixe. Ne pouvait pas traiter RSNA. |
| 2 | `test_1.py` v2 (`ac7c0a0`) | Écrit dans GOGETA avec les préfixes `kermany_`, `rsna_` et `tb_`. CLAHE sur RSNA seulement. |
| 3 | `pretriatement.py` v1 (`ac7c0a0`) | Écrit SPLITS par tirage aléatoire 70/15/15. Pas de purge, pas de groupement par patient. |
| 4 | `data/rebuild_gogeta_splits.py` (`.pyc` du 08/04 18:13) | **`rmtree` de GOGETA et de SPLITS**, puis copie brute : `paul_<n>.jpeg` et `rsna_<n>.dcm`. Split aléatoire sans patient. **C'est la 1re exécution encore présente dans SPLITS.** |
| 5 | `data/dataset_fusion.py` (via `Fusion.py` ou `test_1.py`, le 10/06 de 17:18 à 17:42) | Réécrit GOGETA en PNG avec CLAHE, **sans purge**. |
| 6 | `data/create_splits.py` (via `pretriatement.py`, le 10/06 vers 17:43) | Split groupé par patient, **ajouté** dans SPLITS sans purge. **C'est la 2e exécution.** |
| | `run_pipeline.py` | Purge SPLITS puis lance les étapes 6 et l'entraînement. La 1re exécution étant encore présente, **cette version n'a pas tourné sur ce dossier**. |

### Tableau des fichiers

| Fichier | Rôle | Importé par | Modifié | Verdict |
|---|---|---|---|---|
| `audit_donnees.py` | Audit en lecture seule | Ligne de commande | 05/10 | **GARDER.** Limites : il ignore les `.dcm` de SPLITS, et son affichage final plante en cp1252 après l'écriture des fichiers. |
| `audit_out/audit_complements.py` | Compléments d'audit (ce jour) | Ligne de commande | 22/10 | **GARDER** |
| `create_notebook_v2.py` | Génère `notebook_pneumonia_v2.ipynb` | — | 04/05 | **GARDER** |
| `create_notebook.py` | Ancien générateur. Écrit **le même** `notebook_pneumonia_v2.ipynb`. | — | 04/05 | **ARCHIVER.** Il écraserait le notebook v2. |
| `notebook_pneumonia_v2.ipynb` | Notebook actuel (63 cellules) | — | 04/09 | **GARDER** |
| `notebook_pneumonia.ipynb` | Ancien notebook (29 cellules, 22 avec sorties, 987 Ko) | — | 04/05 | **ARCHIVER** |
| `main.py` | Lance `training.train` en mode rapide de débogage | — | 14/04 | **ARCHIVER.** Fait doublon avec `entrainement.py`. |
| `pyproject.toml` | Empaquetage | — | 14/04 | **GARDER** |
| `rapport_audit.md` | Rapport de la mission 1 (chiffres à 0, mauvais chemin) | — | 05/10 | **ARCHIVER.** Remplacé par ce rapport. |
| `AI_Medical/Fusion.py` | Alias de `run_fusion` | — | 14/04 | **GARDER**, comme unique point d'entrée de la fusion |
| `AI_Medical/test_1.py` | Alias de `run_fusion`, en double | — | 14/04 | **ARCHIVER.** Redondant et mal nommé. |
| `AI_Medical/test_2.py` | Alias de l'analyse exploratoire | — | 14/04 | **ARCHIVER.** `from eda.…` ne résout pas le paquet `AI_Medical`. |
| `AI_Medical/entrainement.py` | Alias de `training.train` | — | 14/04 | **GARDER** |
| `AI_Medical/pretriatement.py` | Alias de `create_splits` | — | 14/04 | **GARDER** |
| `AI_Medical/run_pipeline.py` | Purge SPLITS, découpe puis entraîne | — | 10/06 | **GARDER.** À corriger : il ne fusionne pas et ne purge pas GOGETA. |
| `AI_Medical/config/settings.py` | Chemins et hyperparamètres | Tous | 14/04 | **GARDER** |
| `AI_Medical/data/dataset_fusion.py` | Fusion vers GOGETA | `Fusion.py`, `test_1.py` | 10/06 | **GARDER, à corriger** : retirer TB, retirer les RSNA « Not Normal », purger, dédoublonner |
| `AI_Medical/data/create_splits.py` | Split groupé par patient | `pretriatement.py`, `run_pipeline.py` | 10/06 | **GARDER, à corriger** : les images NORMAL de Kermany sont traitées comme des patients isolés, ce qui laisse fuir 47 patients |
| `AI_Medical/data/rebuild_gogeta_splits.py` | Reconstruction avec `rmtree`, split sans patient | Aucun | 14/04 | **ARCHIVER.** Code mort et dangereux. |
| `AI_Medical/data/legacy_fusion_v1_original.py` | Ancienne fusion de référence | Aucun | 14/04 | **ARCHIVER** |
| `AI_Medical/data/mock_dataset.py` | Génère un jeu factice | Aucun | 14/04 | **ARCHIVER** (code mort) |
| `AI_Medical/data/SPLITS/**` (36 PNG `mock_*`) et `data/GOGETA/**/.gitkeep` | Données factices | — | 10/04 | **ARCHIVER** (6 Mo) |
| `AI_Medical/eda/homogeneity_check.py` | Analyse exploratoire de GOGETA, écrit dans `reports/eda/` | `test_2.py` | 14/04 | **GARDER** (analyse pour le mémoire) |
| `AI_Medical/evaluation/medical_metrics.py` | Métriques | `train.py` | 14/04 | **GARDER** |
| `AI_Medical/models/transfer_cnn.py` | ResNet50 et MobileNetV2 (torchvision) | `train.py` | 14/04 | **GARDER** |
| `AI_Medical/training/train.py` | Entraînement PyTorch (`ImageFolder` sur SPLITS) | `entrainement.py`, `run_pipeline.py`, `main.py` | 14/04 | **GARDER** |
| `AI_Medical/utils/seeds.py` | Graine aléatoire | `train.py` | 14/04 | **GARDER** |
| `AI_Medical/utils/device.py` | Choix du périphérique (PyTorch), fonctions TensorFlow inutilisées | `utils/__init__` | 14/04 | **GARDER.** Le code TensorFlow est mort. |
| `AI_Medical/models/checkpoints/*.pt` (7 fichiers, 387 Mo) | Modèles entraînés le 10/04, le 29/04 et le 10/06 | — | | **ARCHIVER.** Entraînés sur le SPLITS contaminé : leurs scores de test ne sont pas valides. |
| `AI_Medical/temp.h5` | Voir ci-dessous | Aucun | 30/03 | **SUPPRIMER proposé** (10 Mo, conservé dans la sauvegarde) |
| `__pycache__/` (projet, 149 Ko) | Cache Python | — | | **SUPPRIMER proposé** |
| `venv_gpu/` (4,6 Go) | Environnement GPU | — | | **GARDER sur le disque, SORTIR de Git** |
| `.venv/` (1,75 Go) | Environnement Python 3.14 (sans `pydicom` ni `h5py`) | — | | **GARDER.** À toi de voir s'il faut garder deux environnements. |

### `temp.h5`

- **Contenu :** un fichier Keras HDF5 contenant un modèle **`MobileNetV3Small`**, avec une entrée `[None, None, None, 3]`.
- **Pas un modèle entraîné :**
  - il n'a **ni couche `Dense` ni configuration d'entraînement** ;
  - c'est la structure du réseau de base, sans tête de classification ;
  - aucun script actuel ou passé n'en parle.
- **Origine :** il est présent depuis le **premier commit `ac7c0a0` (21/01/2026)**, et sa date sur le disque est le 30/03.
- **Données utilisées :** aucune. Il ne contient que les poids ImageNet, ou des poids tirés au hasard.

### `MODELS\ResNet50_best.h5`

- **Ce que c'est :** un modèle Keras ResNet50 avec une sortie `Dense(1, sigmoid)`, la perte `binary_crossentropy`, l'optimiseur Adam et les métriques précision et AUC, daté du 16/01/2026.
- **Ce qui l'a produit :** cela correspond à l'ancien `entrainement.py` TensorFlow (`MODELS_DIR = C:\Z_Soutenance\MODELS`). Il a donc été entraîné sur l'ancien SPLITS, qui contenait TB.

## Étape 3 : vérification des données

### Ce qui a été exécuté

`audit_donnees.py --root C:\Project_D\Z_Soutenance\DATA --out audit_out` a tourné jusqu'au bout. L'affichage final a planté sur un caractère `−` (encodage cp1252), mais **après** l'écriture de tous les fichiers.

`audit_complements.py` a ensuite couvert ce que l'audit principal ne mesure pas :
- les fichiers `.dcm` de SPLITS ;
- la fuite au niveau du patient ;
- la comparaison pHash entre TB et RSNA ;
- le jeu nigérian.

Le seuil pHash est une distance de 6 ou moins. Dans ces données, les distances observées sont toujours paires.

### Attendu, trouvé, écart

| Jeu | Attendu | Trouvé | Écart |
|---|---|---|---|
| **PAUL** | 5 856 à 5 863 images, 2 classes | **5 856** fichiers réels : NORMAL 1 583, PNEUMONIA 4 273. **5 824 uniques** en SHA-256 (1 579 / 4 245). | ✅ en nombre de fichiers. **32 doublons exacts internes** (`paul_doublons_internes.csv`), sans étiquette contradictoire. En plus : une copie imbriquée identique `chest_xray/chest_xray` (5 856 fichiers, 1 193 Mo) et 5 856 fichiers `._*.jpeg` de macOS dans `__MACOSX` (17 Mo), **illisibles comme images**. |
| **RSNA** | 26 684 images d'entraînement, dont 6 012 pneumonies | **26 684** `.dcm` d'entraînement, 26 684 patients avec étiquette, sans manque dans un sens ni dans l'autre. Target=1 : **6 012** (« Lung Opacity »). | ✅ Mais **Target=0 ne veut pas dire normal** : il regroupe 8 851 « Normal » et **11 821 « No Lung Opacity / Not Normal »**. |
| **Jeu nigérian** | 190 images de test externe : 93 normales, 97 pneumonies | **190** : NORMAL 93 (75 jpg et 18 png), PNEUMONIA 97 (73 jpg et 24 png). 0 doublon. | ✅ **Le zip ne contient pas de README.** Ce qui suit vient de la fiche Zenodo (DOI 10.5281/zenodo.14185822, CC BY 4.0, v1 du 19/11/2024) : enfants **de moins de 5 ans**, UCH Ibadan et Rainbow Scans (Nigeria), collecte de 2023-11 à 2024-08, images **confirmées par des radiologues**, identifiants retirés. Les formats jpg et png sont mélangés. |
| **TB_RAHMAN dans GOGETA** | 0 dans le jeu final | Par nom (`tb_`) : **4 200** (NORMAL 3 500, PNEUMONIA 700). Par SHA-256 : **0**, car le CLAHE réécrit les octets. Par pHash : **3 060 / 4 200**, car le CLAHE déplace le hash. | **4 200 fichiers à retirer** (4 194 uniques). Le nom est le critère fiable. |
| **GOGETA** | Somme des uniques des sources retenues | Attendu : PAUL 5 783 (après le filtre qualité) + RSNA 13 226 (sous-échantillon équilibré) + TB 4 194 = **23 203**. Réel : **23 241** fichiers (12 297 / 10 944), dont **23 203 uniques**. | **+38 copies exactes** : 32 venant de PAUL et 6 venant de TB. Aucune image n'a d'étiquette contradictoire. |

### Comment RSNA est converti en deux classes

**Problème de méthode.** Les deux scripts font `groupby(patientId).Target.max()`, puis Target 0 donne NORMAL et Target 1 donne PNEUMONIA. Résultat :
- dans `GOGETA/NORMAL`, les 7 214 images RSNA sont un tirage équilibré de Target=0. **4 148 d'entre elles (57,5 %) sont « No Lung Opacity / Not Normal »**, c'est-à-dire des radios **anormales** étiquetées normales ;
- dans SPLITS, sur les 27 886 fichiers RSNA rangés NORMAL, 15 969 sont « Not Normal ». La 1re exécution a pris les 20 672 images Target=0.

**Correction proposée :** NORMAL = classe détaillée « Normal » (8 851), PNEUMONIA = « Lung Opacity » (6 012). On exclut les 11 821 images « Not Normal ».

### Doublons et quasi-doublons

**Dans GOGETA (SHA-256) :**
- 23 203 hash uniques pour 23 241 fichiers : 34 hash présents 2 fois, 2 hash présents 3 fois ;
- **0 hash présent à la fois dans NORMAL et dans PNEUMONIA.**

**Quasi-doublons dans GOGETA (pHash, distance 4 ou moins) :** 1 463 groupes, 3 535 images (`quasi_doublons_phash.csv`). Une partie vient des jumelles TB↔RSNA décrites ci-dessous.

**TB_RAHMAN contre RSNA :**
- **Les 3 500 images TB Normal sont toutes des images RSNA « Normal »** : 3 336 à distance 0 et 164 à distance 2 (`tb_vs_rsna_phash.csv`). Dans GOGETA, une même radio est donc présente deux fois, une fois en `tb_` et une fois en `rsna_`. **2 256 images GOGETA sans préfixe `tb_` ont une jumelle TB.** Retirer TB règle le problème.
- TB Tuberculosis contre RSNA : 84 correspondances à distance 4 à 6, aucune à distance 0 ou 2. C'est **douteux**, probablement des faux positifs.

**Jeu nigérian contre les sources d'entraînement :**
- 21 correspondances au seuil de 6 : PAUL 10, RSNA 9, TB 2 ;
- une seule à distance 4, toutes les autres à 6 ;
- les étiquettes se croisent souvent : par exemple, une image NORMAL du jeu nigérian correspond à une pneumonie de PAUL.

Ce sont **très probablement des faux positifs.** Aucune n'est un doublon exact. À vérifier à l'œil dans `ng_recouvrement_phash.csv` avant la soutenance.

### SPLITS : deux exécutions superposées et fuite

| Exécution | Contenu | train | val | test | Total |
|---|---|---:|---:|---:|---:|
| 1re : `rebuild_gogeta_splits.py` | `paul_<n>.jpeg` (5 824) et `rsna_<n>.dcm` (26 684), bruts | 22 786 | 4 872 | 4 850 | 32 508 |
| 2e : `dataset_fusion.py` puis `create_splits.py` | PNG avec CLAHE (`kermany_`, `rsna_`, `tb_`), copie complète de GOGETA | 16 600 | 3 320 | 3 321 | 23 241 |
| **Total** | | | | | **55 749** |

La 1re exécution contient 26 684 fichiers `.dcm`, et `ImageFolder` ignore ce format. **L'entraînement a donc vu 29 065 images** : les 23 241 PNG et les 5 824 JPEG de PAUL. `audit_donnees.py` ne trouve que 7 hash communs entre splits, parce qu'un JPEG brut et son PNG CLAHE n'ont pas les mêmes octets. L'analyse par **identité source** donne :

| Mesure (images vues par `ImageFolder`) | Valeur |
|---|---|
| Images sources distinctes | 23 282 pour 29 065 fichiers |
| Images présentes dans plusieurs splits | **2 664** : train+val 1 222, test+train 1 197, test+val 245 |
| **Images de test aussi présentes en train** | **1 197 sur 4 069, soit 29 % du test, toutes de PAUL** |
| Patients présents dans plusieurs splits | **1 775** (dont 254 dans les trois) |
| Fuite de patients à l'intérieur de la 2e exécution | 47 patients : images NORMAL de Kermany (`IM-xxxx`) traitées comme des patients isolés |

Sur l'ensemble des 55 749 fichiers, `.dcm` compris :
- 8 632 images et 7 743 patients sont présents dans plusieurs splits ;
- la 1re exécution laisse fuir, à elle seule, 711 patients.

Le détail est dans `splits_identites.csv` et `splits_fuite_identites.csv`.

**Conséquence : les scores de test des modèles actuels sont surestimés et ne doivent pas être utilisés dans le mémoire.** Cela concerne les `.pt` et `ResNet50_best.h5`.

### Images uniques par source et par classe dans GOGETA (actuel)

| Source | NORMAL | PNEUMONIA | % PNEUMONIA |
|---|---:|---:|---:|
| PAUL | 1 579 | 4 204 | 72,7 |
| RSNA | 7 214 (dont 4 148 « Not Normal ») | 6 012 | 45,5 |
| TB_RAHMAN | 3 500 (= images RSNA) | 694 (tuberculose) | 16,5 |

**GOGETA_CLEAN estimé, avant le filtre qualité :**
- PAUL : 5 824 (1 579 / 4 245) ;
- RSNA « Normal » et « Lung Opacity » : 8 851 / 6 012 ;
- total **≈ 20 687 images**.

Je ne l'ai pas construit et j'attends ton feu vert.

## Étape 4 : verdict par dossier

| Dossier | Taille | Verdict |
|---|---:|---|
| `DATA\PAUL` | 2 369 Mo | **GARDER intact.** Il contient une copie imbriquée identique `chest_xray\chest_xray` (1 193 Mo) : je ne propose pas de la supprimer, c'est à ta décision. |
| `DATA\RSNA` | 3 773 Mo | **GARDER intact.** Les 3 000 images de `stage_2_test_images` n'ont pas d'étiquette et ne servent pas. |
| `Z_Soutenance\NG Pneum Dataset.zip` | 181 Mo | **GARDER intact.** MD5 vérifié. |
| `DATA\NG Pneum Dataset` | 183 Mo | **GARDER.** Copie extraite identique, à utiliser comme test externe uniquement. |
| `DATA\TB_RAHMAN` | 663 Mo | **GARDER intact, sorti du projet**, par exemple vers `Z_Soutenance\HORS_MEMOIRE\TB_RAHMAN`. Déplacement proposé, commenté. |
| `DATA\GOGETA` | 11 952 Mo | **ARCHIVER** une fois GOGETA_CLEAN validé |
| `DATA\SPLITS` | 16 585 Mo | **ARCHIVER** une fois GOGETA_CLEAN et le nouveau split validés |
| `DATA\PEDICXR_TEST` | — | **À CRÉER vide.** PediCXR n'est pas encore téléchargé. |
| `Z_Soutenance\MODELS\ResNet50_best.h5` | 91 Mo | **ARCHIVER** (ancien modèle Keras, données avec TB) |
| `Z_Soutenance\outputs\` (3 PNG du 30/04) | 0,9 Mo | **ARCHIVER** (figures faites sur les données contaminées) |
| Projet `AI_Medical\models\checkpoints` | 387 Mo | **ARCHIVER** |
| Projet `AI_Medical\data\SPLITS` et `GOGETA` (factices) | 6 Mo | **ARCHIVER** |

### Suppressions proposées

Ce sont uniquement des résidus d'essais, pas des sources brutes.

| Élément | Nombre | Espace |
|---|---:|---:|
| `DATA\PAUL\chest_xray\__MACOSX\` (métadonnées macOS `._*`) | 5 871 fichiers | 17 Mo |
| `.DS_Store` et `._*` hors `__MACOSX` dans PAUL | 8 fichiers | < 0,1 Mo |
| `AI_Medical\temp.h5` | 1 | 10 Mo |
| `__pycache__\` du projet (hors environnements) | 9 dossiers | 0,15 Mo |
| **Total libéré** | | **≈ 27 Mo** |

L'archivage, lui, **déplace** environ 29 000 Mo (28,3 Gio) (GOGETA, SPLITS, anciens modèles et figures) vers `Z_Soutenance\ARCHIVE_2026-10\`, sur le même disque. Pour libérer cet espace, il faudrait supprimer l'archive plus tard, ce qui reste ta décision.

## Plan

1. Sauvegarder (bundles Git et copie du code), créer le `.gitignore`, puis pousser une branche orpheline `propre` sans `venv_gpu`, sans réécrire l'historique.
2. Corriger `dataset_fusion.py` : PAUL et RSNA seulement, RSNA « Normal » contre « Lung Opacity », purge de la sortie, dédoublonnage par SHA-256, même prétraitement pour toutes les sources. Produire ainsi `GOGETA_CLEAN`.
3. Découper `GOGETA_CLEAN` par patient vers un dossier neuf `SPLITS_CLEAN`, avec contrôle de fuite par identité source. Pour Kermany NORMAL, le patient est `IM-xxxx`.
4. Réentraîner ResNet50 et MobileNetV2, puis tester une seule fois sur le jeu nigérian (190 images). `PEDICXR_TEST` reste vide.
5. Après validation, archiver GOGETA, SPLITS et les anciens modèles et figures, puis supprimer les résidus (environ 27 Mo).

**En attente de ton accord. Rien n'a été exécuté de ce plan.**
