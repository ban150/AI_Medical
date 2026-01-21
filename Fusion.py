import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydicom

### Chemins locaux
base_dir = r"C:\Z_Soutenance\DATA"
unified_dir = r"C:\Z_Soutenance\DATA\GOGETA"
os.makedirs(os.path.join(unified_dir, 'NORMAL'), exist_ok=True)
os.makedirs(os.path.join(unified_dir, 'PNEUMONIA'), exist_ok=True)

def good_image(img_path):
    """COntrole qualite simple"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        if h < 224 or w < 224 or img.std() < 10:       # contraste trop faible
            return False
        return True
    except:
        return False

def copy_images(source_folder, target_class):
    if not os.path.exists(source_folder):
        print(f"Warning: Dossier non trouvé : {source_folder}")
        return
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Copie de {len(files)} images depuis {source_folder} vers {target_class}")
    for file in tqdm(files):
        src = os.path.join(source_folder, file)
        if good_image(src):
            dst = os.path.join(unified_dir, target_class, file)
            shutil.copy(src, dst)

print ("--- Fusion DES TROIS SAYENS ---\n")

# Mooney (Kermany) S_principal
print ("- Money (enfants)")
copy_images(os.path.join(base_dir, 'PAUL', 'chest_xray', 'train', 'NORMAL'), 'NORMAL')
copy_images(os.path.join(base_dir, 'PAUL', 'chest_xray', 'train', 'PNEUMONIA'), 'PNEUMONIA')
copy_images(os.path.join(base_dir, 'PAUL', 'chest_xray', 'val', 'NORMAL'), 'NORMAL')
copy_images(os.path.join(base_dir, 'PAUL', 'chest_xray', 'val', 'PNEUMONIA'), 'PNEUMONIA')
copy_images(os.path.join(base_dir, 'PAUL', 'chest_xray', 'test', 'NORMAL'), 'NORMAL')
copy_images(os.path.join(base_dir, 'PAUL', 'chest_xray', 'test', 'PNEUMONIA'), 'PNEUMONIA')

## RSNA (adultes)
##print ("\n-- RSNA (adultes) ")
##copy_images(os.path.join(base_dir, 'RSNA', 'NORMAL'), 'NORMAL')
##copy_images(os.path.join(base_dir, 'RSNA', 'PNEUMONIA'), 'PNEUMONIA')
rsna_images_dir = os.path.join(base_dir, 'stage_2_train_images')  # dossier avec les .dcm
rsna_labels_csv = os.path.join(base_dir, 'stage_2_train_labels.csv')

if not os.path.exists(rsna_images_dir) or not os.path.exists(rsna_labels_csv):
    print("Warning : Dossier RSNA ou CSV manquant – RSNA ignoré")
else:

    df = pd.read_csv(rsna_labels_csv)
    print(df.head())

    print(f"RSNA : {len(df)} images à traiter...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row['patientId']
        target = row['Target']  # 0 = Normal, 1 = Pneumonia

        dcm_path = os.path.join(rsna_images_dir, patient_id + '.dcm')
        if not os.path.exists(dcm_path):
            continue

        try:
            # Lecture DICOM
            ds = pydicom.dcmread(dcm_path)
            img = ds.pixel_array

            # Fenêtrage pulmonaire simple
            img = np.clip(img, -1000, 600)
            img = (img + 1000) / 1600 * 255
            img = img.astype(np.uint8)
            cv2.imshow("RSNA", img)
            cv2.waitKey(0)
            cv2.imwrite("test_rsna.png", img)

            # Contrôle qualité
            #if img.shape[0] < 224 or img.shape[1] < 224 or img.std() < 10:
            if img.shape[0] < 224 or img.shape[1] < 224:
                  continue

            # Classe cible
            target_class = 'NORMAL' if target == 0 else 'PNEUMONIA'

            # Sauvegarde en PNG
            dst = os.path.join(unified_dir, target_class, patient_id + '.png')
            # VISUALISATION TEMPORAIRE
            cv2.imshow("RSNA sample", img)
            cv2.waitKey(0)  # Appuie sur une touche pour passer à l'image suivante

            #  SAUVEGARDE

            cv2.imwrite(dst, img)

            #  LOG DE SAUVEGARDE
            print(f"Sauvegardée : {dst}")


        except Exception as e:
            print(f"Erreur sur {patient_id} : {e}")

    print("RSNA traité et ajouté à GOGETA !")


### TB_RAHMAN (tuberculose → pathologie grave)
print ("\n--- TB_RAHMAN")
copy_images(os.path.join(base_dir, 'TB_RAHMAN', 'TB_Chest_Radiography_Database', 'Normal'), 'NORMAL')
copy_images(os.path.join(base_dir, 'TB_RAHMAN','TB_Chest_Radiography_Database', 'Tuberculosis'), 'PNEUMONIA')

#### Resultat Final
normal_count = len(os.listdir(os.path.join(unified_dir, 'NORMAL')))
pneumonia_count = len(os.listdir(os.path.join(unified_dir, 'PNEUMONIA')))

print("\n--- FUSION TERMINÉE ---")
print(f"Images NORMAL : {normal_count}")
print(f"Images PNEUMONIA : {pneumonia_count}")
print(f"Total : {normal_count + pneumonia_count}")
print(f"Dossier final : {unified_dir}")