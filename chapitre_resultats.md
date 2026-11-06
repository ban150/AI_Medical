# Chapitre 3 : Résultats

*Brouillon à adapter. Les chiffres viennent de `resultats/memoire/tableau_recapitulatif.csv`. Les figures indiquées entre crochets sont à insérer depuis le dossier `resultats/`. Seules les AUC disposent d'un intervalle de confiance à 95 % (bootstrap, 1 000 tirages).*

## 3.1 Jeu de données final

Après les exclusions décrites au chapitre 2, le jeu compte **17 187 images**, réparties par patient en 12 039 images d'entraînement, 2 574 de validation et 2 574 de test. La proportion de pneumonies est de 72,8 % pour PAUL et de 53,4 % pour RSNA (ensemble d'entraînement). Les contrôles ne trouvent aucun hash, aucun patient et aucun fichier commun entre les trois ensembles.

## 3.2 Biais de source

Une régression logistique sur des vignettes 32×32 et des histogrammes d'intensité prédit la source d'une image (PAUL ou RSNA) avec une **AUC de 0,999** (exactitude de 99,3 %). Le résultat ne change pas à classe fixée : 1,000 pour les radios normales et 0,998 pour les pneumonies. Les deux sources sont donc distinguables par des caractéristiques simples, sans lien avec la maladie. Un réseau peut ainsi apprendre l'association « source = classe » au lieu des signes radiologiques.

*Note : ce test a été réalisé sur le découpage d'avant la correction des identifiants patients RSNA. Il porte sur les propriétés des images, et cette correction n'en change pas la nature.*

## 3.3 Test interne (expérience principale)

**Tableau 3.1 – Test interne, PAUL + RSNA (2 574 images)**

| Modèle | Version | AUC [IC95 %] | Sensibilité | Spécificité | Exactitude |
|---|---|---|---|---|---|
| ResNet50 | v1 | 0,921 [0,910 ; 0,931] | 0,823 | 0,886 | 0,849 |
| ResNet50 | v2 | **0,980** [0,975 ; 0,984] | 0,910 | 0,955 | 0,928 |
| ResNet50 | v2, seuil de validation (0,284) | 0,980 | 0,935 | 0,926 | 0,932 |
| MobileNetV2 | v1 | 0,917 [0,906 ; 0,928] | 0,799 | 0,900 | 0,840 |
| MobileNetV2 | v2 | **0,973** [0,968 ; 0,978] | 0,844 | 0,965 | 0,894 |
| MobileNetV2 | v2, seuil de validation (0,171) | 0,973 | 0,933 | 0,922 | 0,928 |

Sauf mention contraire, le seuil de décision est 0,5.

La version 2 (deux blocs dégelés, augmentation forte) améliore l'AUC de 0,06 environ pour les deux réseaux. Les intervalles de confiance des deux versions ne se chevauchent pas. Le seuil choisi sur la validation (indice de Youden) équilibre sensibilité et spécificité autour de 0,93.

Par source, en version 2, l'AUC vaut 0,991 sur PAUL et 0,972 sur RSNA pour ResNet50, contre 0,987 et 0,962 pour MobileNetV2. En version 1, elle valait 0,934 / 0,906 (ResNet50) et 0,952 / 0,898 (MobileNetV2).

Les courbes d'apprentissage ne montrent pas de sur-apprentissage : la perte de validation suit la perte d'entraînement. Le modèle ResNet50 v2 est retenu à l'époque 22, avec un arrêt anticipé à l'époque 30. Le modèle MobileNetV2 v2 est retenu à l'époque 39.

[Figure 3.1 – `resultats/memoire/roc_test_interne.png` : courbes ROC du test interne, v1 et v2]
[Figure 3.2 – `resultats/principal_v2/courbes_resnet50.png` : perte et AUC de validation par époque]
[Figure 3.3 – `resultats/principal_v2/confusion_resnet50_test_seuil_val.png` : matrice de confusion]

## 3.4 Évaluations croisées entre sources

**Tableau 3.2 – Entraînement sur une source, test sur l'autre (seuil 0,5)**

| Expérience | Modèle | Version | AUC [IC95 %] | Sensibilité | Spécificité |
|---|---|---|---|---|---|
| PAUL → RSNA (1 703) | ResNet50 | v1 | 0,713 [0,689 ; 0,737] | 0,983 | 0,114 |
| | | v2 | **0,910** [0,897 ; 0,923] | 0,919 | 0,676 |
| | MobileNetV2 | v1 | 0,705 [0,679 ; 0,729] | 0,980 | 0,112 |
| | | v2 | **0,920** [0,907 ; 0,932] | 0,900 | 0,772 |
| RSNA → PAUL (871) | ResNet50 | v1 | 0,732 [0,693 ; 0,766] | 0,570 | 0,778 |
| | | v2 | **0,872** [0,848 ; 0,895] | 0,716 | 0,842 |
| | MobileNetV2 | v1 | 0,534 [0,488 ; 0,575] | 0,198 | 0,876 |
| | | v2 | **0,822** [0,794 ; 0,850] | 0,350 | 0,983 |

**En version 1**, la performance s'effondre quand on change de population. Entraîné sur des enfants, le modèle classe presque tous les adultes en pneumonie : sensibilité de 0,98, mais spécificité de 0,11. Dans l'autre sens, MobileNetV2 tombe au niveau du hasard (0,534).

**En version 2**, l'AUC sur l'autre source atteint 0,82 à 0,92. Elle reste toutefois en dessous de la validation dans la source d'origine, comprise entre 0,966 et 0,995.

Au seuil 0,5, les modèles croisés restent mal calibrés : par exemple, RSNA → PAUL avec MobileNetV2 donne une sensibilité de 0,350 et une spécificité de 0,983. Le seuil choisi sur la validation de la source d'origine déplace ce compromis sans le corriger : pour RSNA → PAUL avec ResNet50, il donne 0,950 de sensibilité et 0,419 de spécificité.

[Figure 3.4 – `resultats/memoire/roc_croise.png` : courbes ROC des évaluations croisées, v1 et v2]

## 3.5 Test externe nigérian

**Tableau 3.3 – Jeu nigérian (190 images : 93 normales, 97 pneumonies)**

| Expérience d'entraînement | Modèle | AUC v1 [IC95 %] | AUC v2 [IC95 %] | Sens. / Spéc. v2 (seuil 0,5) | Sens. / Spéc. v2 (seuil de validation) |
|---|---|---|---|---|---|
| Principale (PAUL + RSNA) | ResNet50 | 0,586 [0,502 ; 0,663] | **0,709** [0,628 ; 0,784] | 0,546 / 0,720 | 0,670 / 0,634 |
| Principale (PAUL + RSNA) | MobileNetV2 | 0,603 [0,515 ; 0,680] | **0,705** [0,622 ; 0,776] | 0,515 / 0,753 | 0,722 / 0,548 |
| PAUL seul | ResNet50 | 0,593 [0,509 ; 0,677] | 0,637 [0,557 ; 0,715] | 0,515 / 0,699 | 0,701 / 0,452 |
| PAUL seul | MobileNetV2 | 0,627 [0,544 ; 0,706] | 0,657 [0,576 ; 0,735] | 0,464 / 0,753 | 0,608 / 0,645 |
| RSNA seul | ResNet50 | 0,532 [0,450 ; 0,614] | **0,722** [0,648 ; 0,789] | 0,371 / 0,892 | 0,742 / 0,538 |
| RSNA seul | MobileNetV2 | 0,546 [0,467 ; 0,633] | 0,689 [0,609 ; 0,759] | 0,278 / 0,989 | 0,691 / 0,538 |

- **En version 1**, les modèles sont proches du hasard sur le jeu nigérian : AUC de 0,53 à 0,63, et la borne basse de l'intervalle touche ou passe sous 0,5.
- **En version 2**, l'expérience principale atteint 0,71, et la borne basse (environ 0,62) dépasse 0,5. L'écart avec le test interne reste toutefois d'environ 0,27 point d'AUC.
- **Les intervalles sont larges** (environ ±0,08) en raison de la taille du jeu. Ceux de la v1 et de la v2 se chevauchent légèrement pour l'expérience principale.
- **Les différences entre expériences d'entraînement ne sont pas significatives** (0,64 à 0,72 en v2) : leurs intervalles se chevauchent largement.
- **Le seuil choisi sur la validation** augmente la sensibilité sur le jeu nigérian (par exemple de 0,546 à 0,670 pour ResNet50, expérience principale), au prix de la spécificité.

[Figure 3.5 – `resultats/memoire/roc_nigeria.png` : courbes ROC du jeu nigérian, v1 et v2]
[Figure 3.6 – `resultats/principal_v2/confusion_resnet50_nigeria_seuil_val.png` : matrice de confusion]

## 3.6 Interprétabilité (Grad-CAM)

Grad-CAM a été appliqué aux modèles v2 de l'expérience principale, sur deux images par catégorie (vrais positifs, faux négatifs, faux positifs, vrais négatifs) et par source.

- **Pneumonies correctement détectées :** les zones d'activation se situent le plus souvent dans le thorax (champs pulmonaires, région hilaire et cardiaque).
- **Erreurs :** l'activation se porte souvent hors des poumons (abdomen sous le diaphragme, bords et coins de l'image, épaule). Sur un faux négatif nigérian, elle se situe près du marqueur de latéralité « L ».

Ces observations sont qualitatives. Elles portent sur un petit nombre d'images, avec une carte de faible résolution (grille 7×7).

[Figure 3.7 – `resultats/gradcam/gradcam_principal_v2_resnet50_test.png`]
[Figure 3.8 – `resultats/gradcam/gradcam_principal_v2_resnet50_nigeria.png`]

## 3.7 Synthèse des résultats

1. **Le test interne est très bon :** AUC de 0,98 en version 2, mesurée sans fuite de patients ni d'images.
2. **Les deux sources sont presque parfaitement distinguables** (AUC 0,999). En version 1, ce biais de source se traduit par un effondrement des performances d'une population à l'autre.
3. **La version 2 réduit nettement l'écart entre sources** (0,53–0,73 → 0,82–0,92) **et améliore le test nigérian** (environ 0,59–0,60 → 0,71), sans le combler.
4. **Sur des radios nigérianes, la performance reste nettement inférieure à la performance interne.** Ce résultat est au cœur de la discussion (chapitre 4).
