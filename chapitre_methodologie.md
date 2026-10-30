# Chapitre : Matériel et méthodes

*Brouillon à adapter à ton style et à compléter avec tes références bibliographiques. Les chiffres viennent de `manifest.csv` et de `resultats/`.*

## 1. Objectif

Ce travail évalue la détection automatique de la pneumonie sur des radiographies thoraciques par deux réseaux de neurones convolutifs, ResNet50 et MobileNetV2. Il s'intéresse surtout à leur capacité de généralisation, mesurée sur une autre population que celle de l'entraînement et sur un jeu externe nigérian.

## 2. Données

### 2.1 Sources

| Jeu | Rôle | Population | Images disponibles |
|---|---|---|---|
| Chest X-Ray Images (Kermany), noté PAUL | Entraînement, validation, test interne | Enfants | 5 856 |
| RSNA Pneumonia Detection Challenge | Entraînement, validation, test interne | Adultes | 26 684 (`stage_2_train_images`) |
| Under5-Children-Pneumonia-CXR-Nigerian-Dataset (Zenodo, DOI 10.5281/zenodo.14185822) | Test externe uniquement | Enfants de moins de 5 ans, Ibadan | 190 (93 normales, 97 pneumonies) |

Le jeu TB_RAHMAN (tuberculose) a été écarté, car la tuberculose n'est pas l'objet de l'étude.

### 2.2 Définition des classes

Pour RSNA, NORMAL correspond à la classe détaillée « Normal » et PNEUMONIA à « Lung Opacity ». Les 11 821 images « No Lung Opacity / Not Normal » sont exclues : elles sont anormales et ne peuvent pas servir d'exemples de radios normales.

### 2.3 Sélection des images

Les exclusions sont appliquées dans un ordre fixe et toutes consignées dans un manifeste (`manifest.csv`).

| Motif | Images exclues |
|---|---:|
| RSNA « No Lung Opacity / Not Normal » | 11 821 |
| Images RSNA identiques aux images « Normal » de TB_RAHMAN | 3 500 |
| Copies exactes dans PAUL (SHA-256) | 32 |
| **Total** | **15 353** |

Les doublons sont recherchés en deux temps. Le SHA-256 détecte les copies exactes. Pour les quasi-doublons, un hachage perceptuel (pHash, distance ≤ 6) propose des paires candidates, puis la corrélation de vignettes 64×64 les confirme (≥ 0,99). Le pHash seul a été écarté, car un contrôle visuel a montré que des radios distinctes peuvent avoir un pHash très proche. Le seuil a été calibré sur les 3 500 images jumelles TB_RAHMAN/RSNA.

Sur les 190 images nigérianes, 21 correspondances pHash avec les jeux d'entraînement ont été examinées à l'œil et par corrélation : aucune n'est un doublon.

### 2.4 Prétraitement

Le même prétraitement s'applique à toutes les sources :

1. lecture en niveaux de gris ; pour les DICOM, application de la table de modalité et inversion si MONOCHROME1 (les 26 684 fichiers RSNA sont en MONOCHROME2) ;
2. écrêtage aux percentiles 0,5 et 99,5, puis mise à l'échelle sur [0, 255] ;
3. redimensionnement direct en 512×512, sans bandes noires ;
4. aucun CLAHE, pour ne pas amplifier les textures propres à chaque appareil.

Les images sont ensuite réduites à 224×224, dupliquées sur trois canaux et normalisées avec les moyennes et écarts-types d'ImageNet.

### 2.5 Découpage

Le découpage 70/15/15 est réalisé **par patient**, séparément pour chaque combinaison source × classe, avec une graine fixe (42).

- **PAUL** : le patient est extrait du nom de fichier (`person<N>`, `IM-<NNNN>`), ce qui est possible pour tous les fichiers. Le découpage d'origine du jeu n'est pas utilisé, car 170 patients s'y retrouvent dans plusieurs splits.
- **RSNA** : l'identifiant du DICOM désigne une image et non une personne. Le vrai patient est retrouvé grâce à la correspondance officielle RSNA ↔ NIH ChestX-ray14. Les 11 363 images retenues appartiennent à 6 334 patients.
- Par précaution, deux images d'une même source très semblables (corrélation ≥ 0,92 ou pHash ≤ 2) sont placées dans le même split.

Contrôles : aucun hash, aucun patient et aucun fichier commun entre les splits.

| Source | Classe | Train | Val | Test | Total |
|---|---|---:|---:|---:|---:|
| PAUL | NORMAL | 1 109 | 236 | 234 | 1 579 |
| PAUL | PNEUMONIA | 2 972 | 636 | 637 | 4 245 |
| RSNA | NORMAL | 3 704 | 825 | 822 | 5 351 |
| RSNA | PNEUMONIA | 4 254 | 877 | 881 | 6 012 |
| **Total** | | **12 039** | **2 574** | **2 574** | **17 187** |

## 3. Modèles et entraînement

ResNet50 et MobileNetV2, pré-entraînés sur ImageNet, reçoivent une tête de classification binaire (dropout 0,5 et une sortie). La perte est une entropie croisée binaire pondérée (`pos_weight` = NORMAL/PNEUMONIA en entraînement, soit 0,666 pour l'expérience principale). L'optimiseur est Adam, avec des lots de 32 images. Le meilleur modèle est choisi sur l'AUC de validation.

| Paramètre | Version 1 | Version 2 |
|---|---|---|
| Réseau de base | Gelé | 2 derniers blocs dégelés (taux 1e-5) |
| Taux de la tête | 1e-4 | 1e-4 |
| Époques max. / patience | 20 / 6 | 40 / 8 |
| Augmentation | Retournement, rotation ±8° | En plus : rotation ±10°, zoom, décalage, contraste, luminosité, gamma, flou, bruit |
| Seuil de décision | 0,5 | 0,5 et seuil de Youden choisi sur la validation |

Le taux d'apprentissage est divisé par deux quand l'AUC de validation stagne pendant 2 époques.

## 4. Protocole d'évaluation

Trois expériences sont menées pour chaque version :

- **Principale** : entraînement et validation sur PAUL + RSNA, test interne sur les deux sources.
- **PAUL → RSNA** : entraînement et validation sur PAUL, test sur RSNA.
- **RSNA → PAUL** : entraînement et validation sur RSNA, test sur PAUL.

Chaque modèle est aussi évalué sur le jeu nigérian. Les métriques sont l'AUC-ROC (intervalle de confiance à 95 % par bootstrap, 1 000 tirages), la sensibilité, la spécificité et l'exactitude.

**Biais de source.** Une régression logistique sur des vignettes 32×32 et des histogrammes d'intensité tente de prédire la source d'une image. **Interprétabilité.** Grad-CAM est appliqué à la dernière couche convolutive, sur quelques images de chaque catégorie (vrais/faux positifs et négatifs).

## 5. Limites de la méthode

- **Usage du jeu externe.** Le jeu nigérian n'a servi ni à l'entraînement ni à la validation, et aucun hyperparamètre n'a été réglé directement sur lui. Mais il a été mesuré **pour chaque expérience**, et la décision de construire la version 2 a été prise après avoir vu les résultats de la version 1. Le score nigérian de la version 2 peut donc être légèrement optimiste. Une évaluation vraiment indépendante demanderait un nouveau jeu externe, par exemple PediCXR.
- **Taille du test externe.** Avec 190 images, les intervalles de confiance sont larges.
- **Identifiants patients.** Pour PAUL, l'identifiant `IM-<NNNN>` est une approximation déduite du nom de fichier.
- **Étiquettes.** L'étiquette RSNA repose sur une opacité radiologique, ce qui ne recouvre pas exactement le diagnostic clinique de pneumonie.
- **Sélection RSNA.** Retirer les 3 500 images communes avec TB_RAHMAN réduit les radios normales disponibles (8 851 à 5 351).
- **Reproductibilité.** Le jeu de données et le découpage sont reproductibles à l'identique (vérifié par empreintes). L'entraînement sur GPU utilise une graine fixe, sans garantie d'identité exacte des poids.
