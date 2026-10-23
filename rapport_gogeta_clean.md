# Rapport : GOGETA_CLEAN, split, test externe et code

Date : 22/10/2026. Données : `C:\Project_D\Z_Soutenance\DATA`. **GOGETA, SPLITS, PAUL, RSNA et TB_RAHMAN n'ont pas été modifiés.** Rien n'a été réentraîné. Aucune commande Git n'a été lancée en écriture.

## 1. Sauvegarde (section 0) : simulation seulement, en attente de ton « ok »

| Vérification | Résultat |
|---|---|
| Espace libre sur C: | **99,5 Go** (disque de 951,6 Go) |
| `BACKUP\` | N'existe pas encore. Le simulateur indique qu'il serait créé (`-WhatIf`). |
| Bundle du dépôt externe (`--all` : `main`, `origin/*`, stash) | ≈ 2,3 Go (pack de 2,29 Gio). `git bundle` n'a pas de mode simulation, donc pas exécuté. |
| Bundle du dépôt imbriqué (`--all` : `fix-leak-clahe`, `master`, `origin/*`) | ≈ 0,37 Go (363 Mio d'objets non compressés et un pack de 9 Mio). Pas exécuté. |
| Copie du code (`robocopy /L`, sans `venv_gpu`, `.venv` ni `*.pyc`) | 437 fichiers, **3 488 Mo**. 415 seraient copiés et 22 exclus (`*.pyc`). |
| **Total estimé** | **≈ 6,2 Go**. Il resterait environ 93 Go libres. |

La copie `robocopy` inclut les deux dossiers `.git` (2,5 Go et 0,37 Go). Cela fait doublon avec les bundles, mais c'est volontaire, par prudence. Si tu préfères, j'ajoute `/XD .git` et la sauvegarde passe à environ 3,6 Go.

## 2. Ce qui a été créé ou déplacé

| Élément | Action |
|---|---|
| `DATA\GOGETA_CLEAN\` (2 340 Mo) | **Créé.** `NORMAL\`, `PNEUMONIA\`, `manifest.csv`, `controle_quasi_doublons.csv`, `resume_fusion.md`, `resume_split.md` et `_cache\` |
| `DATA\NIGERIA_TEST\` (184 Mo) | **Créé.** Le zip a été extrait après vérification du MD5 Zenodo. 190 images en **lecture seule** (attribut R), plus `manifest_nigeria.csv`. |
| `DATA\PEDICXR_TEST\` | **Créé vide** |
| `AI_Medical\data\pretraitement.py` | Nouveau module, utilisé par tous les scripts : même prétraitement partout |
| `AI_Medical\fusion_clean.py`, `split_clean.py`, `entrainement.py` | **Les trois scripts** : fusion, split, entraînement |
| `AI_Medical\biais_source.py` | Test de biais de source (écrit, pas exécuté) |
| `AI_Medical\config\exclusions_manuelles.csv` | Liste d'exclusions manuelles, vide pour l'instant |
| `AI_Medical\archive\` | **Déplacés** : `Fusion.py`, `test_1.py`, `test_2.py`, `pretriatement.py`, `run_pipeline.py`, `entrainement.py` (ancien), `data\dataset_fusion.py`, `data\create_splits.py`, `data\rebuild_gogeta_splits.py`, `data\legacy_fusion_v1_original.py`, `data\mock_dataset.py`, `training\train.py` |
| `archive\` (racine) | **Déplacés** : `main.py`, `create_notebook.py`, `notebook_pneumonia.ipynb` |
| `audit_out\` | Planches de contact, CSV de contrôle, journaux, `planche_contact.py` |

Rien n'a été supprimé. Le dépôt imbriqué verra ces déplacements comme des suppressions et des fichiers non suivis, jusqu'à ce que tu décides quoi commiter.

## 3. Résultat : GOGETA_CLEAN

### Images retenues par source, classe et split

Graine 42, 70/15/15, découpage par patient, strates source × classe.

| Source | Classe | train | val | test | Total |
|---|---|---:|---:|---:|---:|
| PAUL | NORMAL | 1 109 | 236 | 234 | 1 579 |
| PAUL | PNEUMONIA | 2 972 | 636 | 637 | 4 245 |
| RSNA | NORMAL | 3 746 | 801 | 804 | 5 351 |
| RSNA | PNEUMONIA | 4 208 | 903 | 901 | 6 012 |
| **Total** | | **12 035** | **2 576** | **2 576** | **17 187** |

Proportion de PNEUMONIA en train : PAUL 72,8 %, RSNA 52,9 %. La pondération est lue dans le manifeste : `pos_weight` = 4 855 / 7 180 = **0,676**. Elle remplace les effectifs en dur 22 463 / 20 808.

### Images exclues, toutes listées dans `manifest.csv`

| Motif (colonne `classe`) | PAUL | RSNA | Total |
|---|---:|---:|---:|
| `exclu_not_normal` (« No Lung Opacity / Not Normal ») | — | 11 821 | 11 821 |
| `exclu_jumeau_tb` (image RSNA identique à une image TB_RAHMAN) | 0 | **3 500** | 3 500 |
| `exclu_doublon_sha` (copie exacte) | **32** | 0 | 32 |
| `exclu_doublon_phash` (quasi-doublon confirmé) | 0 | 0 | 0 |
| `exclu_doublon_nigeria` | 0 | 0 | 0 |
| `exclu_conflit_etiquette`, `exclu_illisible`, `exclu_manuel` | 0 | 0 | 0 |
| **Sources** : PAUL 5 856, RSNA 26 684 | | | **15 353 exclues, 17 187 retenues** |

Ignorés dès l'inventaire : la copie imbriquée `PAUL\chest_xray\chest_xray` (5 856), `__MACOSX` (5 856 `._*`) et les `.DS_Store`.

**Impact de la règle TB :** RSNA « Normal » passe de 8 851 à 5 351 images. Les 3 500 images normales de TB_RAHMAN sont toutes des radios RSNA, que j'ai retirées comme tu l'as demandé. C'est un choix à confirmer : ces images RSNA sont légitimes en soi. Pour les garder, il suffirait de retirer le contrôle TB des références d'exclusion. Ce serait une ligne à changer, je ne l'ai pas fait.

### Comment un doublon est décidé (point de méthode important)

1. **SHA-256** des fichiers sources : 32 copies exactes dans PAUL, toutes de même classe, retirées.
2. **Le pHash seul ne suffit pas.** J'ai vérifié à l'œil (`audit_out/planche_exclus_phash_1.png`, `planche_distance4_1.png`) : des radios **distinctes** ont un pHash à distance 2, voire 0. Un premier passage avec « pHash ≤ 2 = doublon » aurait supprimé 27 images à tort.
3. Règle retenue :
   - pHash ≤ 6 **propose** une paire candidate ;
   - la **corrélation des vignettes 64×64** (images prétraitées) **confirme** si elle atteint au moins 0,99.
4. Calibrage sur données réelles :

| Paires | Corrélation |
|---|---|
| Les 3 500 jumelles TB↔RSNA | ≥ 0,99 |
| Radios distinctes, même à pHash ≤ 2 | ≤ 0,946 |
| Cas limites, entre 0,95 et 0,99 | 48 paires (`planche_limites_*.png`) : à l'œil, deux examens **différents** du même patient, pas des copies |

Les 16 559 paires candidates examinées et la décision prise pour chacune sont dans `GOGETA_CLEAN\controle_quasi_doublons.csv`.

### Prétraitement, identique pour toutes les sources

- **DICOM** : table de modalité (pente et ordonnée), puis inversion si MONOCHROME1. **Pas de fenêtrage de TDM.** Vérifié sur les 26 684 fichiers : **tous en MONOCHROME2**, 8 bits, sans pente ni ordonnée, 1024×1024, modalité CR, `PatientID` = nom de fichier.
- **Normalisation** : écrêtage aux percentiles 0,5 et 99,5, puis mise à l'échelle 0 à 255. C'est plus robuste que le min-max face aux textes incrustés.
- **Taille** : redimensionnement direct à **512×512** (`INTER_AREA`). Pas de bandes noires, qui coderaient la source à travers le rapport largeur/hauteur.
- **CLAHE : aucun, pour toutes les sources.** Il amplifie le bruit et les textures propres à chaque appareil, donc le risque de raccourci « source = classe ». Il est irréversible et ne doit être appliqué ni à moitié ni sur le test externe. On pourra le tester plus tard comme variante d'entraînement, appliqué à la volée.
- **Noms** : `source_classe_sha256[:16].png`, par exemple `paul_normal_f17a003185299f17.png`.
- **Manifeste** : `chemin`, `source`, `classe`, `hash`, `split`, `patient`, plus `classe_source`, `chemin_source`, `phash` et `doublon_de`.

### Idempotence (vérifiée)

J'ai lancé un `--reset` (reconstruction complète), puis **10 relances** de fusion et de split. L'empreinte des 17 187 PNG (SHA-256 de chaque fichier) est la même à chaque passage : `699f614db9db7f0c`. Celle du manifeste aussi : `3d71dd7833d80a08`. Chaque relance affiche « Écriture de 0 images ». La reconstruction après `--reset` produit les mêmes octets que la construction précédente.

## 4. Split : identifiants patients

- **RSNA** : `PatientID` lu dans le DICOM, égal au nom de fichier pour les 26 684 images.
  - **Limite :** cet identifiant désigne une **image**, pas une personne. RSNA vient de NIH ChestX-ray14, qui contient des examens de suivi : les planches montrent des paires d'examens d'un même patient sous deux identifiants.
  - **Parade :** deux images d'une même source sont mises dans le même split si r ≥ 0,92 ou si leur pHash est à 2 ou moins. Cela fait 346 regroupements ; le plus gros groupe compte 17 images à 0,92, alors qu'à 0,90 il en compterait 118. 38 groupes mêlent les deux classes et forment une strate à part.
  - Cette heuristique **réduit** la fuite sans la **garantir nulle**. La vraie solution serait la correspondance entre RSNA et les identifiants patients NIH (`Data_Entry_2017.csv`). Il faudrait un téléchargement, que je n'ai pas fait.
- **PAUL** : le motif des noms le permet pour **5 856 fichiers sur 5 856**.
  - `person<N>_bacteria|virus_<M>[_k].jpeg` (4 273) correspond au patient `person<N>`.
  - `[NORMAL2-]IM-<NNNN>-<SSSS>[-<k>].jpeg` (1 583) correspond à `IM-<NNNN>`. Il y a des examens multiples, par exemple `IM-0429-0001-0001` et `-0002`.
  - 1 185 patients ont plusieurs images, jusqu'à 31.
  - **Le découpage d'origine de Kermany laissait déjà fuir 170 patients** entre son train, son val et son test. Il est ignoré ici.
  - Choix prudent : `IM-0427` et `NORMAL2-IM-0427` sont regroupés (328 numéros en commun), au cas où ce serait la même personne.
- **Contrôles bloquants, tous passés :**
  - 0 hash source, 0 patient (après regroupement) et 0 fichier PNG en commun entre train, val et test ;
  - 0 paire pHash à distance 2 ou moins entre deux splits.
  - Il reste 234 paires à distance 4, vérifiées : ce sont des images distinctes.

### Évaluation croisée entre sources (`entrainement.py --experience`)

| Expérience | train | val | test | test NORMAL/PNEUMONIA |
|---|---:|---:|---:|---|
| `principal` (PAUL+RSNA) | 12 035 | 2 576 | 2 576 | 1 038 / 1 538 |
| `paul_vers_rsna` | 4 081 (PAUL) | 872 (PAUL) | 1 705 (RSNA) | 804 / 901 |
| `rsna_vers_paul` | 7 954 (RSNA) | 1 704 (RSNA) | 871 (PAUL) | 234 / 637 |

Le jeu nigérian (190 images) n'est évalué qu'avec `--tester-nigeria`. Il ne sert jamais à l'entraînement ni à la validation. Il reçoit le même prétraitement, appliqué à la volée, et ses fichiers restent en lecture seule.

## 5. Jeu nigérian : les 21 correspondances pHash

**Les planches sont `audit_out/planche_ng_1.png` à `planche_ng_3.png`. Ces paires sont à vérifier à l'œil.**

**Mon avis : aucune des 21 paires n'est un vrai doublon.**
- Les deux images de chaque paire sont manifestement différentes : autre patient, autres marqueurs. Les 11 paires avec RSNA ou TB opposent même un enfant à un adulte.
- La corrélation des vignettes va de 0,41 à 0,95 : aucune paire n'atteint le seuil de 0,99.
- La fusion n'a donc **rien retiré** (`exclu_doublon_nigeria` = 0).

Si tu juges une paire douteuse, ajoute son hash ou son `chemin_source` dans `AI_Medical\config\exclusions_manuelles.csv`, puis relance la fusion et le split.

## 6. Code

| Script | Rôle | Testé |
|---|---|---|
| `python -m AI_Medical.fusion_clean [--dry-run] [--reset] [--preparer-tests]` | Construit GOGETA_CLEAN et le manifeste | Oui (`--reset` et 10 relances) |
| `python -m AI_Medical.split_clean [--dry-run]` | Écrit la colonne `split`, avec contrôles bloquants | Oui |
| `python -m AI_Medical.entrainement [--dry-run] [--experience …] [--tester-nigeria]` | ResNet50 et MobileNetV2, effectifs lus dans le manifeste, sélection sur l'AUC de validation | **`--dry-run` seulement** (pas d'entraînement) |
| `python -m AI_Medical.biais_source [--dry-run]` | Régression logistique qui devine PAUL ou RSNA, sur toutes les classes puis à classe fixée | **`--dry-run` seulement** : le lancer revient à entraîner un petit classifieur, ce que je n'ai pas fait sans ton accord |

Point bloquant pour l'entraînement : `venv_gpu` (Python 3.10, PyTorch CUDA) **n'a pas pandas, opencv, scikit-learn, tqdm ni matplotlib**. Il faudra les y installer. Le Python système 3.14 a tout ce qu'il faut, mais en version CPU de PyTorch.

## 7. Git : plan « nouveau dépôt propre » (proposé, rien d'exécuté)

1. Faire la sauvegarde (section 0) **après ton « ok »**.
2. Créer un dossier neuf `C:\Users\User\PycharmProjects\AI_Medical_propre`, avec `git init`. Les deux anciens dépôts ne sont pas touchés.
3. Y copier uniquement :
   - le paquet `AI_Medical\` (sans `archive\`, `models\checkpoints\`, `data\SPLITS\`, `data\GOGETA\` ni `temp.h5`) ;
   - `pyproject.toml`, `audit_donnees.py`, `create_notebook_v2.py` et les rapports `.md`.
4. `.gitignore` proposé :
   ```gitignore
   venv*/
   .venv/
   __pycache__/
   *.pyc
   data/
   DATA/
   *.dcm
   *.jpeg
   *.jpg
   *.png
   *.zip
   *.h5
   *.pt
   audit_out/
   outputs/
   reports/
   archive/
   .idea/workspace.xml
   .claude/settings.local.json
   ```
5. `requirements.txt` proposé, avec les versions testées ici :
   ```text
   numpy>=2.2
   pandas>=2.2
   opencv-python>=4.10
   pydicom>=3.0
   Pillow>=11
   scikit-learn>=1.5
   tqdm>=4.66
   matplotlib>=3.9
   torch>=2.4        # version CUDA : index https://download.pytorch.org/whl/cu121 ou cu128
   torchvision>=0.19
   ```
6. Faire un premier commit, créer un **nouveau** dépôt GitHub (vide), puis `git push -u origin main`. C'est un push normal, sans réécriture.

## 8. Ce qui attend ton accord

1. **Lancer la sauvegarde réelle** (section 0), avec ou sans `/XD .git`.
2. **Valider les 21 paires du jeu nigérian** (planches) : mon avis est qu'aucune n'est un doublon.
3. **Règle TB** : confirmer le retrait des 3 500 images RSNA « Normal » identiques aux images normales de TB_RAHMAN, ou les réintégrer.
4. **Identifiants patients RSNA** : accepter l'heuristique, ou télécharger la correspondance NIH pour un découpage par vrai patient.
5. **Lancer `biais_source.py`** : il entraîne seulement une régression logistique, en quelques minutes.
6. **Plan Git** de la section 7.
