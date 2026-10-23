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
  3. ⏳ *Si besoin* : Grad-CAM, pour vérifier que le modèle regarde les poumons et pas les marqueurs ou les bords.

**Q16. Pourquoi geler le backbone ?** ⏳
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

## 6. Résultats ⏳ *(à remplir après l'entraînement)*

| Expérience | Modèle | AUC | Sensibilité | Spécificité | F1 |
|---|---|---|---|---|---|
| principal (test interne) | ResNet50 | | | | |
| principal (test interne) | MobileNetV2 | | | | |
| PAUL → RSNA | ResNet50 | | | | |
| RSNA → PAUL | ResNet50 | | | | |
| Nigeria (externe) | ResNet50 | | | | |
| Nigeria (externe) | MobileNetV2 | | | | |

**Lecture attendue** : si le score s'effondre en évaluation croisée ou sur le jeu nigérian, c'est le **biais de source** (Q15). Il faut l'assumer et l'expliquer plutôt que le cacher.

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
- 23/10/2026 : création. Données, split et test de biais faits ; entraînement à venir.
