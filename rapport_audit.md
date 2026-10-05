# Rapport d'audit – Mission 1 (lecture seule)

## 0. Limite importante : ce que j'ai pu et n'ai pas pu vérifier

Cette session s'exécute dans un conteneur cloud. Il contient **le dépôt (6 fichiers de code + `temp.h5`) mais aucune image** : pas de `C:\Z_Soutenance\DATA`, ni PAUL, RSNA, TB_RAHMAN, GOGETA ou SPLITS.

| Point demandé | Statut |
|---|---|
| 1. Scripts qui construisent GOGETA / SPLITS | **Fait** (analyse de code, lignes citées) |
| 4. Origine des doublons (lignes de code) | **Fait**, sous forme d'hypothèses fondées sur le code. Le script d'audit les confirme ou les infirme |
| 2. Comptages, 3. SHA-256 / pHash, 5. tableau source × classe | **Non exécutés.** Aucun chiffre n'est inventé ici. Le script `audit_donnees.py` les calcule sur votre PC |
| 6. `doublons.csv` | **Non généré** : il sort du script, sinon il serait faux |

Aucun fichier de données n'a été touché. Le script d'audit n'écrit que dans son dossier `--out`. Il a été testé sur un jeu synthétique : doublons exacts, étiquette contradictoire, fuite train/test, quasi-doublon pHash et origine par préfixe ont tous été détectés.

**À lancer chez vous :**
```
pip install numpy pillow
python audit_donnees.py --root C:\Z_Soutenance\DATA --out audit_out
```
Il produit dans `audit_out/` : `resultats_audit.md` (tous les comptages et tableaux), `doublons.csv` (chemin, hash, nombre de copies, dossier), `etiquettes_contradictoires.csv`, `fuite_splits.csv`, `quasi_doublons_phash.csv`, `origine_gogeta.csv`, `collisions_noms_sources.csv`, `comptages.csv`, `hashes.csv`. Collez-moi `resultats_audit.md` et je complète les sections 2, 3 et 5.

Le dépôt ne contient aucun notebook (`.ipynb`) et un seul commit, donc pas d'historique des exécutions. Si vos notebooks ou d'autres scripts écrivent dans GOGETA, ils sont ailleurs sur votre disque. `temp.h5` (10 Mo) n'a pas été ouvert (pas de `h5py`) ; il ne semble pas lié aux données.

## 1. Scripts qui construisent GOGETA / SPLITS

Ordre logique d'exécution déduit des chemins et des commentaires (« fusion numero 2 », « PROCHAINE ÉTAPE : CRÉER LES SPLITS » dans `test_2.py:412`) :

`Fusion.py` (v1) → `test_1.py` (v2, malgré son nom) → `test_2.py` (EDA) → `pretriatement.py` (SPLITS) → `entrainement.py`

| Script | Rôle | Entrées | Sorties |
|---|---|---|---|
| `Fusion.py` | Fusion v1 | `PAUL\chest_xray\{train,val,test}\{NORMAL,PNEUMONIA}` (L44-49) ; RSNA `.dcm` + CSV dans `DATA\stage_2_train_images` (L55-56) ; `TB_RAHMAN\TB_Chest_Radiography_Database\{Normal,Tuberculosis}` (L118-119) | `GOGETA\NORMAL`, `GOGETA\PNEUMONIA`, **nom d'origine conservé** (L37) |
| `test_1.py` | Fusion v2 : préfixe de source, RSNA dédupliqué par patient, CLAHE, équilibrage RSNA | PAUL (L127-137), `DATA\RSNA\stage_2_train_images` (L148), TB (L240-250) | mêmes dossiers GOGETA, noms `kermany_…`, `rsna_<id>.png`, `tb_…` (L80, L219) |
| `test_2.py` | EDA en lecture, sauf graphiques écrits dans `GOGETA\EDA_report` (L16, L232, L271, L337) | GOGETA | PNG de rapport (non-récursif, donc ignorés par les autres scripts) |
| `pretriatement.py` | Découpage 70/15/15 stratifié, graine 42 (L10, L35-36), puis `shutil.copy` | `os.listdir` de GOGETA (L21) | `SPLITS\{train,val,test}\{NORMAL,PNEUMONIA}` (L44-45) |
| `entrainement.py` | ResNet50 / MobileNetV2 | SPLITS | modèles. `class_weight` codé en dur : 22 463 NORMAL / 20 808 PNEUMONIA (L59-60) |

## 4. D'où peuvent venir les doublons (d'après le code)

**Constat principal : relancer un seul de ces scripts ne duplique pas, d'après le code.** Les noms de fichiers sont déterministes et `shutil.copy` / `cv2.imwrite` écrasent. Votre soupçon n'est donc pas confirmé pour une relance simple. Les mécanismes plausibles sont les suivants.

1. **Deux fusions avec deux conventions de nommage dans le même GOGETA (cause la plus probable).**
   - `Fusion.py:37` écrit `fichier.jpeg`, `test_1.py:80` écrit `kermany_fichier.jpeg`. Si les deux ont tourné, chaque image PAUL et TB existe deux fois sous deux noms. Les deux copient les octets tels quels, ce sont donc des **doublons SHA-256 exacts**.
   - Idem RSNA : `Fusion.py:97` écrit `<id>.png`, `test_1.py:219` écrit `rsna_<id>.png`. Les traitements diffèrent (fenêtrage `-1000..600` en L81-82, contre min-max + CLAHE en L107-111), donc le SHA-256 est différent mais **le pHash est quasi identique**.
2. **GOGETA n'est jamais vidé.** `os.makedirs(..., exist_ok=True)` (`Fusion.py:12-13`, `test_1.py:16-17`) et aucun nettoyage. Un essai partiel (`max_images`, `balance_classes`, `test_1.py:143`) puis une exécution complète laissent des résidus. Le compteur final (`test_1.py:261-262`) compte tout le dossier, donc il inclut ces résidus.
3. **Aucune vérification d'existence ni de hash avant copie** (`Fusion.py:36-38`, `test_1.py:79-83`). Deux fichiers de même nom venant de dossiers différents (par exemple `PAUL\train` et `PAUL\val`) s'écrasent silencieusement : c'est de la **perte**, pas du doublon, et c'est invisible.
4. **Pas de suffixes `_1` / `(1)` dans le code.** S'il y en a dans GOGETA, ils viennent d'un copier-coller Windows, d'une extraction d'archive faite deux fois ou d'un autre script/notebook. À vérifier avec `doublons.csv`. Le zip Kaggle de Paul Mooney contient en général un `chest_xray\chest_xray` imbriqué et un `__MACOSX` (à vérifier chez vous). Les scripts lisent seulement `PAUL\chest_xray\{train,val,test}` en non-récursif, donc ils les évitent. Mon audit parcourt PAUL en récursif, et `comptages.csv` permet de voir si ce dossier imbriqué double le total.
5. **`Fusion.py:67` boucle sur les lignes du CSV RSNA sans `groupby`.** Si le CSV a une ligne par boîte englobante, un même patient est retraité plusieurs fois (écrasé, donc sans doublon de fichier, mais du temps perdu). `test_1.py:161` corrige avec `groupby('patientId').max()`.

**Fuite SPLITS (le risque le plus grave pour le mémoire) :**
- `pretriatement.py:13-15` et `:45` : SPLITS n'est jamais purgé. Si GOGETA change entre deux exécutions, la répartition aléatoire change aussi, tandis que les anciens fichiers restent. Une même image peut alors se retrouver dans `train` (ancienne exécution) et dans `test` (nouvelle).
- `pretriatement.py:35-36` : le découpage se fait par fichier. Il n'y a ni déduplication ni regroupement par patient. Les doublons de GOGETA se répartissent donc au hasard entre train, val et test. Les images PAUL du même patient (`personN_bacteria_k`) aussi.

**Autres anomalies relevées (hors doublons) :**
- `Fusion.py:84-86, 99-100` : `cv2.imshow` + `waitKey(0)` bloquent à chaque image, et `test_rsna.png` est réécrit dans le dossier courant à chaque tour.
- `Fusion.py:81-82` : fenêtrage `-1000..600` (valeurs de TDM) appliqué à des radios, donc contraste détruit. Le filtre de contraste est commenté (`:89`). Les filtres qualité diffèrent entre `Fusion.py:22` (couleur) et `test_1.py:53` (niveaux de gris).
- `Fusion.py:119` et `test_1.py:250` : **tuberculose étiquetée PNEUMONIA**. Cela change la signification de la classe et est à justifier dans le mémoire.
- `entrainement.py:59-60` : effectifs en dur, à comparer au comptage réel de GOGETA.

## 5. Origine et biais de source (analyse du code, à confirmer par les chiffres)

D'après le code, **aucune source n'est limitée à une seule classe** : PAUL fournit NORMAL et PNEUMONIA, RSNA aussi (Target 0/1), et TB_RAHMAN fournit Normal → NORMAL et Tuberculosis → PNEUMONIA. Le script produit le tableau source × classe en images **uniques** pour vérifier que GOGETA reflète bien cela. L'origine est retrouvée par hash identique à la source, puis par préfixe, puis par motif de nom (UUID = RSNA, `Normal-N`/`Tuberculosis-N` = TB, `IM-`/`person` = PAUL).

Le risque de raccourci « source → classe » reste **élevé** même sans source mono-classe :
- Les proportions de classes diffèrent beaucoup d'une source à l'autre. D'après les tailles habituelles de ces jeux publics (de mémoire, à vérifier avec votre tableau) : PAUL ≈ 70 % pneumonie, TB ≈ 17 % « pneumonie », RSNA ≈ 45 % après l'équilibrage de `test_1.py:171`.
- Les populations et les formats diffèrent : enfants (PAUL, JPEG), adultes (RSNA, DICOM converti), TB.
- **Seul RSNA reçoit un CLAHE** (`test_1.py:111`). C'est une signature de source facile à apprendre.

Le test de biais de source (classifieur qui devine la source) est prévu pour la mission 2.

## 2-3. Résultats chiffrés

**En attente de l'exécution de `audit_donnees.py`** sur votre machine (voir section 0). Ils porteront sur les comptages, l'écart sources/GOGETA/SPLITS, les doublons exacts, les étiquettes contradictoires, les fuites entre splits et les quasi-doublons.

## Verdict provisoire et suite

Les chiffres décideront de la suite. D'après le code, les deux hypothèses à tester en premier sont : (a) `Fusion.py` et `test_1.py` ont tous deux écrit dans GOGETA ; (b) SPLITS a été régénéré sans purge. **La mission 2 n'est pas commencée et ne le sera qu'après votre accord.**
