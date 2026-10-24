# Argumentaire de soutenance : questions possibles et éléments de réponse

Document vivant, à **mettre à jour à chaque étape** jusqu'aux résultats finaux.
Légende : ✅ réponse appuyée sur des chiffres vérifiés ; ⏳ à compléter après l'entraînement.
Dernière mise à jour : 23/10/2026 (données et split terminés, entraînement pas encore lancé).

---

## 1. Données : choix des sources

**Q1. Pourquoi PAUL (Kermany) et RSNA ?** ✅
- Ce sont deux jeux publics de référence, avec des étiquettes posées par des médecins.
  - PAUL : enfants de 1 à 5 ans, Guangzhou.
  - RSNA : adultes, sous-ensemble du jeu NIH ChestX-ray14 annoté par des radiologues.
- Les combiner couvre deux populations et deux types d'appareils. C'est aussi ce qui rend nécessaire le contrôle du biais de source (Q15).

**Q2. Pourquoi avoir exclu la tuberculose (TB_RAHMAN) ?** ✅
- La tuberculose n'est pas une pneumonie. L'étiqueter « PNEUMONIA » faussait la tâche. C'était le cas dans l'ancien pipeline : 700 images.
- Découverte importante : **les 3 500 images « Normal » de TB_RAHMAN sont des radios RSNA** (corrélation ≥ 0,99 avec une image RSNA).
  - Les garder créait des doublons cachés entre les deux sources.
  - Ces 3 500 images ont été retirées de RSNA pour qu'aucune image de TB_RAHMAN n'entre dans le jeu.

**Q3. Pourquoi avoir retiré 11 821 images RSNA ?** ✅
- Dans RSNA, « Target = 0 » regroupe deux catégories :
  - « Normal » (8 851) ;
  - « No Lung Opacity / Not Normal » (11 821), c'est-à-dire des radios **anormales** sans opacité.
- Les ranger en NORMAL aurait appris au modèle que des poumons malades sont normaux. Dans l'ancien GOGETA, 57,5 % des RSNA « NORMAL » étaient anormales.
- Elles restent listées dans `manifest.csv` avec la classe `exclu_not_normal`, donc tout est traçable.

**Q4. Combien d'images au final ?** ✅
- **17 187 images** : PAUL 1 579 NORMAL et 4 245 PNEUMONIA ; RSNA 5 351 NORMAL et 6 012 PNEUMONIA.
- Plus **190 images nigérianes** (93 / 97) en test externe.

**Q5. Pourquoi pas PediCXR ?** ⏳
- Le jeu n'était pas encore téléchargé (accès PhysioNet). Un emplacement `PEDICXR_TEST` est prévu, vide.
- Il servira de second test externe, pédiatrique, s'il est obtenu à temps.

## 2. Qualité des données : doublons et fuites

**Q6. Comment avez-vous traité les doublons ?** ✅
1. **Copies exactes** (même SHA-256) : 32 dans PAUL, retirées.
2. **Quasi-doublons** : le pHash propose des paires candidates (distance ≤ 6), et la **corrélation des vignettes 64×64 ≥ 0,99** les confirme.
3. **Pourquoi pas le pHash seul ?** Les vérifications à l'œil (planches de contact) ont montré que des radios **différentes** ont souvent un pHash très proche, voire identique : les radios thoraciques se ressemblent toutes. Le pHash seul aurait supprimé 27 images à tort.
4. **Calibrage** :
   - vraies copies : corrélation ≥ 0,99 ;
   - images différentes : ≤ 0,946.
   - 16 559 paires ont été examinées, et toutes sont tracées dans `controle_quasi_doublons.csv`.

**Q7. Votre test est-il vraiment indépendant de l'entraînement ?** ✅ *(question la plus probable)*
- **Oui, le découpage se fait par patient**, pas par image.
  - PAUL : identifiant dans le nom de fichier (`person123`, `IM-0427`), retrouvé pour 100 % des fichiers.
  - RSNA : **vrais identifiants patients NIH**, tirés du fichier officiel de correspondance publié par la RSNA.
- **Pourquoi c'est crucial** : les 11 363 radios RSNA viennent de **6 334 personnes**, l'une ayant 47 radios. Sans cette correspondance, 1 038 personnes auraient eu des radios à la fois en apprentissage et en test.
- **Contrôles bloquants** : 0 hash, 0 patient et 0 fichier en commun entre train, val et test.
- Le **découpage d'origine de Kermany** contenait lui-même 170 patients à cheval entre ses splits. Nous ne l'avons pas utilisé.

**Q8. Quelle était l'erreur dans la première version du projet ?** ✅ *(à présenter honnêtement : cela montre la rigueur)*
- L'ancien dossier SPLITS mélangeait deux exécutions jamais purgées : **29 % des images de test étaient aussi dans l'entraînement**.
- La tuberculose était étiquetée « pneumonie », et les RSNA anormales étiquetées « normales ».
- **Les anciens scores sont donc invalides.** Tout a été reconstruit, et ce travail d'audit fait partie de la contribution du mémoire.

**Q9. Le split est-il reproductible ?** ✅
- Oui : graine 42 et algorithme déterministe.
- 10 relances successives donnent exactement les mêmes fichiers et le même manifeste (vérifié par empreinte SHA-256).

## 3. Prétraitement

**Q10. Pourquoi ce prétraitement ?** ✅
- Il est **identique pour toutes les sources**, test externe compris.
- **Normalisation par percentiles** (0,5 et 99,5) : plus robuste aux textes et marqueurs incrustés que le min-max.
- **Taille 512×512** sans bandes noires : les bandes trahiraient la source à travers le rapport largeur/hauteur.
- **DICOM** : conversion vérifiée. Les 26 684 fichiers RSNA sont en MONOCHROME2, 8 bits, sans pente ni ordonnée. **Pas de fenêtrage de scanner (TDM)** : l'ancien code l'appliquait à tort sur des radios.

**Q11. Pourquoi pas de CLAHE ?** ✅
- Le CLAHE renforce le bruit et la texture propres à chaque appareil, ce qui augmente le risque que le modèle reconnaisse la source plutôt que la maladie.
- L'ancien code ne l'appliquait **qu'à RSNA**, ce qui créait un biais.
- La règle est donc : tout ou rien. Le CLAHE pourra être testé comme variante, appliqué à la volée. ⏳ *Résultat de la variante à ajouter si elle est testée.*

**Q12. Pourquoi 224×224 en entrée des réseaux ?** ✅
- C'est la taille native de ResNet50 et de MobileNetV2, pré-entraînés sur ImageNet. Les images sont stockées en 512 pour garder de la marge.

## 4. Modèles et entraînement

**Q13. Pourquoi ResNet50 et MobileNetV2 ?** ✅ / ⏳
- ResNet50 est une référence solide en imagerie médicale.
- MobileNetV2 est léger, donc compatible avec un **déploiement en contexte à ressources limitées** (lien avec le test nigérian).
- ⏳ À ajouter : temps d'inférence, taille des modèles, scores comparés.

**Q14. Comment gérez-vous le déséquilibre des classes ?** ✅
- Pondération de la perte (`pos_weight`) **calculée à partir du manifeste** : NORMAL / PNEUMONIA en train = 4 813 / 7 226 = 0,666. Ce n'est plus codé en dur.
- Le meilleur modèle est choisi sur **l'AUC de validation**, pas sur le rappel : le rappel seul récompenserait un modèle qui prédit « pneumonie » partout.

**Q15. Le modèle reconnaît-il la pneumonie ou la source ?** ✅ / ⏳ *(question piège majeure)*
- **Test de biais réalisé** : un simple classifieur sur des vignettes 32×32 devine la source (PAUL ou RSNA) avec une **AUC de 0,999**, même à classe fixée (NORMAL : 1,000 ; PNEUMONIA : 0,998). Un CNN pourrait donc tricher.
- **Parades mises en place :**
  1. **Évaluation croisée** : apprendre sur PAUL et tester sur RSNA, puis l'inverse. ⏳ *Scores à ajouter.*
  2. **Test externe nigérian**, jamais vu. ⏳ *Scores à ajouter.*
  3. ✅ Grad-CAM réalisé (section 6) : les bonnes détections regardent le thorax ; les erreurs regardent souvent hors des poumons (abdomen, bords, marqueurs).

**Q16. Pourquoi geler le backbone, puis le dégeler en version 2 ?** ✅ / ⏳
- **v1 (gelé)** : seul le classifieur final apprend. C'est rapide et peu sujet au sur-apprentissage, mais les caractéristiques restent celles d'ImageNet (photos), pas des radios.
- **v2** : on dégèle les **2 derniers blocs**, avec un taux d'apprentissage 10 fois plus faible (1e-5) pour ne pas effacer ImageNet. On ajoute :
  - une **augmentation forte** (contraste, luminosité, gamma, flou, bruit, recadrage), pour imiter la diversité des appareils ;
  - 40 époques maximum, avec réduction automatique du taux si l'AUC de validation stagne ;
  - un **seuil de décision choisi sur la validation** (indice de Youden), en plus du seuil 0,5.
- Les exemples d'augmentation sont dans `resultats/exemples_augmentation_forte.png`.
- ⏳ Comparer v1 et v2 (test interne, croisé, Nigeria). **Règle :** le choix entre v1 et v2 se fait sur la **validation**, jamais sur le jeu nigérian.
- C'est l'apprentissage par transfert classique pour un jeu de taille moyenne. ⏳ *Si un fine-tuning partiel est testé, comparer.*

## 5. Évaluation

**Q17. Quelles métriques et pourquoi ?** ✅ / ⏳
- AUC-ROC (indépendante du seuil), sensibilité (rappel : ne pas rater une pneumonie), spécificité, précision, F1, matrice de confusion.
- ⏳ Ajouter des intervalles de confiance (bootstrap), surtout pour le jeu nigérian de 190 images.

**Q18. Pourquoi un test externe nigérian ?** ✅
- Pour vérifier la **généralisation** à une autre population (enfants de moins de 5 ans), d'autres appareils et un autre pays, ce qui correspond au cadre du mémoire.
- Source : Zenodo (DOI 10.5281/zenodo.14185822), UCH Ibadan, images confirmées par des radiologues, licence CC BY 4.0. MD5 vérifié.
- Il n'est **jamais** utilisé en entraînement ni en validation, et il n'est évalué qu'une seule fois, à la fin.
- 21 ressemblances possibles avec l'entraînement ont été vérifiées à l'œil : **aucune copie**.

**Q19. 190 images, est-ce suffisant ?** ⏳
- C'est un échantillon modeste. On le dira honnêtement, avec des intervalles de confiance. ⏳ *Ajouter la largeur des IC.*

## 6. Résultats (session 1 du 23/10/2026 ; sessions croisées ⏳)

Backbone gelé, 20 époques, Adam (lr 1e-4), lots de 32, RTX 5060. Le seuil de décision est 0,5. Entre crochets : IC95 % de l'AUC (bootstrap, 1 000 tirages). Fichiers : `resultats/principal/`.

| Expérience | Modèle | AUC [IC95 %] | Sensibilité | Spécificité | Exactitude |
|---|---|---|---|---|---|
| principal, test interne (2 574) | ResNet50 | **0,921** [0,910 ; 0,931] | 0,823 | 0,886 | 0,849 |
| principal, test interne (2 574) | MobileNetV2 | **0,917** [0,906 ; 0,928] | 0,799 | 0,900 | 0,840 |
| … dont PAUL (871) | ResNet50 / MobileNetV2 | 0,934 / 0,952 | 0,890 / 0,892 | 0,868 / 0,897 | |
| … dont RSNA (1 703) | ResNet50 / MobileNetV2 | 0,906 / 0,898 | 0,774 / 0,732 | 0,892 / 0,900 | |
| **Nigeria, externe (190)** | ResNet50 | **0,586** [0,502 ; 0,663] | 0,546 | 0,538 | 0,542 |
| **Nigeria, externe (190)** | MobileNetV2 | **0,603** [0,515 ; 0,680] | 0,557 | 0,591 | 0,574 |
| PAUL → RSNA, test RSNA (1 703) | ResNet50 | **0,713** [0,689 ; 0,737] | 0,983 | **0,114** | 0,564 |
| PAUL → RSNA, test RSNA (1 703) | MobileNetV2 | **0,705** [0,679 ; 0,729] | 0,980 | **0,112** | 0,561 |
| PAUL → RSNA, Nigeria (190) | ResNet50 / MobileNetV2 | 0,593 / **0,627** [0,544 ; 0,706] | 0,485 / 0,639 | 0,667 / 0,484 | |
| RSNA → PAUL, test PAUL (871) | ResNet50 | **0,732** [0,693 ; 0,766] | 0,570 | 0,778 | 0,626 |
| RSNA → PAUL, test PAUL (871) | MobileNetV2 | **0,534** [0,488 ; 0,575] | 0,198 | 0,876 | 0,380 |
| RSNA → PAUL, Nigeria (190) | ResNet50 / MobileNetV2 | 0,532 / 0,546 | 0,330 / 0,144 | 0,774 / 0,946 | |

### Version 2 (24/10/2026) : 2 blocs dégelés, augmentation forte, 40 époques max., seuil choisi sur la validation

Fichiers : `resultats/principal_v2/`. ResNet50 retenu à l'époque 22 (arrêt anticipé à 30), MobileNetV2 à l'époque 39. Durée ≈ 39 min et 38 min sur la RTX 5060.

| Expérience | Modèle | AUC [IC95 %] | Sens. / Spéc. (seuil 0,5) | Sens. / Spéc. (seuil de validation) |
|---|---|---|---|---|
| test interne (2 574) | ResNet50 v2 | **0,980** [0,975 ; 0,984] (v1 : 0,921) | 0,910 / 0,955 | 0,935 / 0,926 (seuil 0,284) |
| test interne (2 574) | MobileNetV2 v2 | **0,973** [0,968 ; 0,978] (v1 : 0,917) | 0,844 / 0,965 | 0,933 / 0,922 (seuil 0,171) |
| … dont PAUL / RSNA | ResNet50 v2 | 0,991 / 0,972 | | |
| … dont PAUL / RSNA | MobileNetV2 v2 | 0,987 / 0,962 | | |
| **Nigeria (190)** | ResNet50 v2 | **0,709** [0,628 ; 0,784] (v1 : 0,586) | 0,546 / 0,720 | 0,670 / 0,634 |
| **Nigeria (190)** | MobileNetV2 v2 | **0,705** [0,622 ; 0,776] (v1 : 0,603) | 0,516 / 0,753 | 0,722 / 0,548 |

**Évaluations croisées en v2 (24/10/2026)** : `resultats/paul_vers_rsna_v2/` et `resultats/rsna_vers_paul_v2/`.

| Expérience | Modèle | AUC autre source v1 → **v2** [IC95 % v2] | AUC Nigeria v1 → **v2** [IC95 % v2] |
|---|---|---|---|
| PAUL → RSNA | ResNet50 | 0,713 → **0,911** [0,897 ; 0,923] | 0,593 → **0,637** [0,557 ; 0,715] |
| PAUL → RSNA | MobileNetV2 | 0,705 → **0,920** [0,908 ; 0,932] | 0,627 → **0,657** [0,576 ; 0,735] |
| RSNA → PAUL | ResNet50 | 0,732 → **0,872** [0,848 ; 0,895] | 0,532 → **0,722** [0,648 ; 0,789] |
| RSNA → PAUL | MobileNetV2 | 0,534 → **0,822** [0,794 ; 0,850] | 0,546 → **0,689** [0,609 ; 0,760] |

- **Le dégel et l'augmentation forte réduisent fortement le biais de source.** Entre enfants et adultes, l'AUC passe d'environ 0,53–0,73 à 0,82–0,92. C'est l'effet recherché : des caractéristiques moins liées à l'appareil.
- **L'écart subsiste néanmoins** : en v2, la validation dans la même source est à 0,97–0,995, contre 0,82–0,92 sur l'autre source.
- **Correction d'une intuition de la v1 :** en v2, le meilleur modèle sur le jeu nigérian vient de **RSNA → PAUL** (ResNet50 : 0,722), pas du modèle appris sur les enfants (0,637–0,657). L'« indice pédiatrique » de la v1 **n'est pas confirmé** : avec 190 images, ces différences ne sont pas significatives (les intervalles se chevauchent).
- **Au seuil 0,5, les modèles croisés sont mal calibrés** sur l'autre source (par exemple RSNA → PAUL MobileNetV2 : sensibilité 0,35, spécificité 0,98). Le seuil choisi sur la validation de la source d'origine ne corrige pas cela : un seuil ne se transfère pas d'une population à l'autre.

**Grad-CAM (24/10/2026)** : `resultats/gradcam/` (ResNet50 et MobileNetV2 v2 ; test interne et Nigeria ; catégories VP, FN, FP, VN).
- **Pneumonies bien détectées (VP)** : la zone chaude est le plus souvent **dans le thorax** (champs pulmonaires, région hilaire et cardiaque). Le modèle s'appuie sur la bonne région.
- **Erreurs (FN, FP)** : l'attention sort souvent des poumons, vers l'abdomen sous le diaphragme, les bords ou les coins de l'image, l'épaule. Sur un faux négatif nigérian, elle se pose **près du marqueur « L » et du haut de l'image**. C'est le signe d'un **raccourci partiel** : quand le modèle se trompe, c'est souvent qu'il ne regarde pas les poumons.
- **Limites à dire au jury :** Grad-CAM sur la dernière couche est **grossier** (grille 7×7), **qualitatif**, et porte sur peu d'images (2 par catégorie). C'est une illustration, pas une preuve statistique.

**Lecture de la v2 :**
- **Le choix de la v2 est justifié par la VALIDATION** (AUC 0,983 contre 0,920 en v1), pas par le jeu nigérian. Le test nigérian a seulement été mesuré après coup, ce qui reste conforme à la règle.
- **Le dégel et l'augmentation améliorent nettement le test interne** (0,92 → 0,98) **et le test externe** (0,59–0,60 → 0,70–0,71).
- **Sur le jeu nigérian, la borne basse de l'intervalle (≈ 0,62) est maintenant au-dessus de 0,5** : le modèle fait clairement mieux que le hasard. Il reste cependant **loin** du score interne : l'écart de généralisation est réduit, pas éliminé.
- Les intervalles de v1 et v2 sur le jeu nigérian se chevauchent légèrement (v1 jusqu'à 0,66–0,68, v2 à partir de 0,62). Avec 190 images, on parlera d'une **amélioration probable**, pas démontrée au sens statistique strict.
- Le seuil choisi sur la validation équilibre sensibilité et spécificité sur le test interne. Sur le jeu nigérian, il augmente la sensibilité (0,55 → 0,67 pour ResNet50), au prix d'une spécificité plus basse.

**Lecture (à dire honnêtement au jury) :**
- **Le test interne est bon** (AUC ≈ 0,92). Il est mesuré sans fuite : par patient, sans doublon, contrôlé.
- **Sur le jeu nigérian, le modèle est à peine meilleur que le hasard.** L'AUC vaut 0,59 à 0,60, et la borne basse de l'intervalle touche 0,50. Le score moyen donné aux pneumonies (≈ 0,50 à 0,53) est à peine plus élevé que celui des radios normales (≈ 0,43 à 0,46).
- **C'est précisément ce que le test de biais laissait craindre (Q15).** Ce que le modèle apprend reste lié aux appareils et aux populations de PAUL et RSNA, et ne se transfère pas à des radios nigérianes d'autres hôpitaux et d'autres appareils.
- **C'est un résultat en soi, et un argument pour le mémoire :** la validation externe est indispensable, et un bon score interne ne garantit pas une utilisation en Afrique.
- **Évaluations croisées (sessions 2 et 3) : la preuve du biais de source.**
  - Un modèle entraîné sur PAUL atteint 0,96 d'AUC en validation sur PAUL, mais **0,71 sur RSNA**. Sur les adultes, il prédit « pneumonie » presque partout : sensibilité 0,98, mais spécificité 0,11.
  - Dans l'autre sens, en passant de RSNA à PAUL, l'AUC tombe de 0,90 à **0,73** (ResNet50), voire à **0,53** (MobileNetV2, quasiment le hasard).
  - Conclusion : ce qu'un modèle apprend sur une population ne se transfère pas à l'autre. La chute observée sur le jeu nigérian n'est donc pas une anomalie.
- **Indice (à confirmer) :** le jeu nigérian, composé d'enfants, est le mieux prédit par le modèle entraîné **sur des enfants** (PAUL → Nigeria, MobileNetV2 : 0,627), et le moins bien par celui entraîné sur des adultes (RSNA → Nigeria : 0,53 à 0,55). Mais les intervalles de confiance se chevauchent, donc ce n'est **pas significatif** avec 190 images.
- Les deux modèles sont **encore en progrès à l'époque 20** (AUC de validation 0,920 et 0,907) : l'entraînement n'a pas convergé.

**Pistes d'amélioration ⏳ (à tester, puis à noter ici) :**
1. Entraîner plus longtemps et **dégeler les dernières couches** du réseau (fine-tuning).
2. Augmentation de données plus forte : contraste, luminosité, flou, bruit. Le but est d'imiter la diversité des appareils.
3. Ajuster le seuil de décision sur la validation, plutôt que 0,5.
4. ✅ Évaluations croisées faites (voir ci-dessus).
5. ✅ Grad-CAM fait (voir ci-dessus).
6. ⚠ Ne **jamais** régler le modèle en regardant les scores nigérians : sinon, ce n'est plus un test externe.

## 7. Limites à reconnaître spontanément ✅

- Deux sources très différentes (enfants et adultes), donc un biais de source fort et mesuré.
- Étiquettes RSNA définies par une « opacité », ce qui ne recouvre pas exactement la pneumonie clinique.
- Test externe de petite taille (190 images).
- Pas de données cliniques (âge précis, symptômes) : classification sur l'image seule.

## 8. Reproductibilité ✅

- Code sur GitHub, branche `propre`.
- Les trois scripts (fusion, split, entraînement) et le manifeste (`manifest.csv`) retracent chaque image : sa source, sa classe, son patient, son split et le motif d'exclusion éventuel.
- Données brutes jamais modifiées, sauvegardes Git faites.

---

### Journal des mises à jour
- 24/10/2026 : évaluations croisées v2 et Grad-CAM terminés ; biais de source fortement réduit mais pas éliminé.
- 24/10/2026 : version 2 (principal) terminée : test interne 0,98, Nigeria 0,71. Un premier lancement depuis PyCharm tournait sur le processeur (interpréteur .venv) ; relancé sur la carte graphique, avec une protection ajoutée.
- 23/10/2026 : version 2 préparée (dégel, augmentation forte, seuil de validation) ; à lancer depuis PyCharm (configurations 6, 7, 8).
- 23/10/2026 : sessions 2 et 3 (croisées) terminées ; biais de source confirmé.
- 23/10/2026 : session 1 (principal) terminée ; résultats internes et nigérians ajoutés en section 6.
- 23/10/2026 : création. Données, split et test de biais faits ; entraînement à venir.
