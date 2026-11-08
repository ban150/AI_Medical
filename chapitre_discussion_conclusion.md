# Chapitre 4 : Discussion

*Brouillon à adapter. Deux références de la littérature sont proposées : vérifie-les et complète-les avant de les citer. Ailleurs, le marqueur [RÉFÉRENCE À AJOUTER] signale une citation à compléter.*

## 4.1 Synthèse des résultats principaux

Ce travail montre qu'un modèle de détection de la pneumonie peut atteindre d'excellentes performances sur des données issues des mêmes sources que l'entraînement (AUC de 0,98), tout en restant nettement moins performant sur des radiographies d'une autre origine (AUC de 0,71 sur le jeu nigérian). Cet écart d'environ 0,27 point d'AUC est le résultat central de l'étude. La performance mesurée en interne ne suffit pas à juger l'utilité d'un modèle dans un autre contexte clinique, en particulier en Afrique subsaharienne.

## 4.2 Le biais de source : un raccourci d'apprentissage

Trois observations convergent :

1. **Les deux sources sont distinguables presque parfaitement** par un classifieur très simple (AUC de 0,999), même à classe fixée.
2. **En version 1, la performance s'effondre quand on change de population.** Un modèle entraîné sur des enfants (PAUL) classe en pneumonie presque toutes les radiographies d'adultes (spécificité de 0,11). Dans l'autre sens, MobileNetV2 tombe au niveau du hasard (AUC 0,534).
3. **Grad-CAM montre que les erreurs s'accompagnent souvent d'une attention hors des poumons**, sur les bords, l'abdomen ou les marqueurs de latéralité.

Ces éléments indiquent que les modèles exploitent en partie des caractéristiques liées à l'appareil, au protocole d'acquisition ou à la population (âge, morphologie), plutôt que des signes radiologiques de pneumonie. Ce phénomène est décrit dans la littérature. Zech et al. (2018, *PLOS Medicine*) ont montré qu'un modèle de détection de la pneumonie pouvait reconnaître l'hôpital d'origine et se généraliser moins bien à d'autres établissements. DeGrave et al. (2021, *Nature Machine Intelligence*) ont observé des raccourcis comparables pour la COVID-19. *[Références à vérifier et à compléter.]*

Le déséquilibre entre sources aggrave le risque. La proportion de pneumonies est de 72,8 % dans PAUL contre 53,4 % dans RSNA : reconnaître la source donne donc déjà une information sur la classe.

## 4.3 Apport de la version 2

Dégeler les deux derniers blocs et utiliser une augmentation plus forte, qui imite la variabilité des appareils (contraste, exposition, flou, bruit, cadrage), a eu un effet marqué :

- **test interne :** 0,92 → 0,98 ;
- **évaluations croisées :** 0,53–0,73 → 0,82–0,92 ;
- **jeu nigérian (expérience principale) :** environ 0,59–0,60 → 0,71, avec une borne basse de l'intervalle désormais au-dessus de 0,5.

Une interprétation plausible : les couches profondes, adaptées aux radiographies plutôt qu'aux photos d'ImageNet, et l'augmentation photométrique rendent le modèle moins dépendant des caractéristiques propres à chaque appareil. Le biais n'est pourtant pas éliminé. La performance sur l'autre source (0,82–0,92) reste inférieure à celle de la validation dans la source d'origine (0,97–0,995), et l'écart avec le jeu nigérian reste important.

**Précaution méthodologique.** La version 2 a été retenue sur la base de la validation (AUC 0,983 contre 0,920), et non du jeu nigérian. En revanche, la décision de la construire a été prise après avoir vu les résultats nigérians de la version 1. Le score nigérian de la version 2 peut donc être légèrement optimiste.

## 4.4 Seuil de décision et calibration

Le seuil de 0,5 n'est pas adapté quand le modèle change de population. En évaluation croisée, il produit des compromis extrêmes, par exemple une sensibilité de 0,35 pour une spécificité de 0,98. Un seuil choisi sur la validation équilibre le test interne (sensibilité et spécificité autour de 0,93). Sur le jeu nigérian, il augmente la sensibilité (0,55 → 0,67 pour ResNet50) au prix de la spécificité (0,72 → 0,63).

Dans un contexte de dépistage, rater une pneumonie chez un jeune enfant est plus grave qu'une fausse alerte : un seuil favorisant la sensibilité serait préférable. Mais **un seuil ne se transfère pas d'une population à l'autre**. Avant tout déploiement, il devrait être recalibré sur des données locales.

## 4.5 ResNet50 ou MobileNetV2 ?

Les deux réseaux obtiennent des performances très proches : AUC de 0,980 contre 0,973 en interne, et de 0,709 contre 0,705 sur le jeu nigérian, avec des intervalles qui se chevauchent. MobileNetV2 est beaucoup plus léger : environ 9 Mo contre 94 Mo pour les fichiers de modèle, et environ 0,7 ms contre 1,9 ms par image en entraînement sur notre carte graphique. Pour un déploiement dans des structures à ressources limitées, il offre donc un meilleur compromis, à performance quasi égale. *[Mesurer le temps d'inférence sur un processeur ordinaire renforcerait cet argument.]*

## 4.6 Rigueur du protocole

Une partie importante du travail a porté sur la qualité des données. L'audit de la première version du projet a révélé :
- 29 % des images de test présentes aussi dans l'entraînement ;
- la tuberculose étiquetée comme pneumonie ;
- des radiographies RSNA anormales étiquetées normales ;
- 3 500 images communes à deux jeux de données ;
- un découpage par image et non par patient (les 11 363 images RSNA retenues appartiennent à 6 334 patients).

Ces défauts auraient produit des scores surestimés. Les corrections apportées (découpage par vrai patient grâce à la correspondance RSNA ↔ NIH, déduplication, définition stricte des classes, manifeste traçable) sont une contribution méthodologique de ce mémoire, transposable à d'autres projets d'imagerie médicale.

## 4.7 Limites

- **Taille du test externe.** 190 images donnent des intervalles larges : les différences entre configurations sur le jeu nigérian ne sont pas significatives.
- **Usage répété du jeu externe** (voir 4.3) : un second jeu externe indépendant serait nécessaire pour confirmer.
- **Définition des étiquettes.** L'étiquette RSNA repose sur une opacité radiologique et non sur un diagnostic clinique. Les étiquettes de PAUL et du jeu nigérian proviennent d'autres protocoles d'annotation.
- **Identifiants patients.** Pour PAUL, ils sont déduits des noms de fichiers. Pour RSNA, ils dépendent de la correspondance publiée.
- **Image seule.** Aucune donnée clinique (âge, symptômes, saturation) n'est utilisée.
- **Interprétabilité.** Grad-CAM est qualitatif, de faible résolution, et appliqué à peu d'images.
- **Reproductibilité de l'entraînement.** La graine est fixe, mais les calculs sur carte graphique ne garantissent pas des poids identiques d'une exécution à l'autre.

# Conclusion et perspectives

Ce mémoire visait à évaluer des réseaux de neurones convolutifs pour la détection de la pneumonie sur radiographie thoracique, en vue d'un usage en contexte africain.

Après un audit approfondi des données et la reconstruction d'un jeu sans fuite (17 187 images, découpage par patient), ResNet50 et MobileNetV2 atteignent une AUC de 0,98 et 0,97 sur le test interne. Les évaluations croisées et le test externe sur 190 radiographies d'enfants nigérians montrent toutefois une généralisation limitée : AUC de 0,71 dans la meilleure configuration. Un fort biais de source en est en grande partie responsable. Le ré-entraînement partiel du réseau et une augmentation de données imitant la diversité des appareils réduisent nettement cet écart, sans le combler.

**Le principal enseignement est qu'une bonne performance interne ne garantit pas une performance en conditions réelles.** La validation externe sur des données locales est une étape indispensable avant tout usage clinique en Afrique.

**Perspectives :**
1. **Valider sur un second jeu externe pédiatrique** (par exemple PediCXR) et sur des données collectées localement, de taille suffisante pour des conclusions statistiques.
2. **Adapter le modèle au domaine** : affiner le modèle (fine-tuning) sur une petite partie de données locales et recalibrer le seuil sur place.
3. **Réduire le biais de source** : normalisation plus poussée entre appareils, apprentissage pénalisant la reconnaissance de la source, segmentation pulmonaire pour concentrer l'attention sur les poumons.
4. **Déployer MobileNetV2** sur matériel modeste, avec une évaluation du temps de calcul et une étude prospective en conditions réelles.
5. **Combiner image et données cliniques** (âge, symptômes, saturation en oxygène).
