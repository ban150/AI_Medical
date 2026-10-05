# Comment exécuter l'audit en local

Cette session (cloud, Linux) ne voit pas `C:\Z_Soutenance\DATA` : les scripts
doivent être exécutés **sur ta machine Windows**, dans un clone local de ce
dépôt (branche `claude/elegant-shannon-2fh9t7`).

Les deux scripts sont **en lecture seule** : ils ne modifient, ne déplacent,
ne renomment et ne suppriment rien sous `C:\Z_Soutenance`. Ils n'écrivent que
dans `--out` (par défaut `audit_out\`, créé automatiquement).

## 1. Dépendances

```
pip install opencv-python pydicom pandas numpy tqdm
```

## 2. Lancer l'audit principal

```
python audit_donnees.py --root C:\Z_Soutenance\DATA --out audit_out
```

Ajoute `--phash` pour activer la recherche de quasi-doublons par pHash dans
GOGETA (plus lent — peut prendre plusieurs minutes de plus selon la taille de
GOGETA). Sans `--phash`, tout le reste (présence, comptages, doublons exacts
SHA-256, étiquettes contradictoires, fuite train/val/test, tableau
source×classe) est quand même produit.

Si le dossier dure longtemps, lance-le en arrière-plan (ex. dans un terminal
dédié, ou avec `start /B` sous cmd) et laisse-le tourner.

## 3. Lancer les compléments

```
python audit_complements.py --root C:\Z_Soutenance\DATA --out audit_out --rsna-sample 5000
```

- `--rsna-sample 5000` limite le nombre de DICOM RSNA utilisés pour le pHash
  contre TB_RAHMAN (sinon il traite les ~26 000 fichiers, ce qui est plus
  lent). Retire l'option pour un passage exhaustif si tu as le temps.
- Si le zip Zenodo n'est pas trouvé automatiquement (recherché sous `DATA` et
  dans ton dossier Téléchargements), indique son chemin exact :
  `--zenodo-zip "C:\chemin\vers\fichier.zip"`.

## 4. Ce qu'il faut me renvoyer

Une fois les deux scripts terminés, envoie-moi le contenu du dossier
`audit_out\` (le plus simple : zippe-le et transmets-le, ou colle le contenu
des fichiers CSV/JSON si c'est petit). Je m'appuierai uniquement sur ces
sorties réelles pour rédiger `rapport_audit.md` et les CSV finaux demandés —
je n'inventerai aucun chiffre.

Si les noms des dossiers sous `DATA` diffèrent de `PAUL`, `RSNA`,
`TB_RAHMAN`, `GOGETA`, `SPLITS`, le script l'affichera clairement au début
(« dossiers attendus non trouvés » / « dossiers présents non reconnus ») —
dis-le-moi et j'adapterai `EXPECTED_TOPDIRS` dans `audit_donnees.py`.
