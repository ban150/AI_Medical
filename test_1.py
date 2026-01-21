############### fusion numero 2

import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import pydicom
from pathlib import Path

BASE_DIR = r"C:\Z_Soutenance\DATA"
UNIFIED_DIR = r"C:\Z_Soutenance\DATA\GOGETA"

# Créer les dossiers de sortie
os.makedirs(os.path.join(UNIFIED_DIR, 'NORMAL'), exist_ok=True)
os.makedirs(os.path.join(UNIFIED_DIR, 'PNEUMONIA'), exist_ok=True)

# Compteurs globaux (clés en minuscule pour cohérence)
stats = {
    'kermany_normal': 0,
    'kermany_pneumonia': 0,
    'rsna_normal': 0,
    'rsna_pneumonia': 0,
    'tb_normal': 0,
    'tb_pneumonia': 0,
    'skipped_quality': 0,
    'skipped_error': 0
}

# FONCTIONS UTILITAIRES

def good_image(img_array, min_size=224, min_contrast=10):
    """Contrôle qualité d'une image (array numpy)"""
    if img_array is None or img_array.size == 0:
        return False

    h, w = img_array.shape[:2]

    checks = {
        'sufficient_size': h >= min_size and w >= min_size,
        'not_blank': np.sum(img_array) > 1000,
        'sufficient_contrast': np.std(img_array) >= min_contrast,
        'valid_range': np.min(img_array) >= 0 and np.max(img_array) <= 255
    }

    return all(checks.values())


def good_image_from_path(img_path):
    """Contrôle qualité depuis un chemin de fichier"""
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return good_image(img)
    except Exception as e:
        print(f"  Erreur lecture {img_path}: {e}")
        return False


def copy_images(source_folder, target_class, source_name):
    """
    Copie des images d'un dossier vers la classe cible
    source_name doit correspondre aux clés dans stats (ex: 'kermany', 'tb')
    """
    if not os.path.exists(source_folder):
        print(f" Dossier introuvable : {source_folder}")
        return

    files = [f for f in os.listdir(source_folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"   {source_folder}")
    print(f"     Fichiers trouvés : {len(files)}")

    copied = 0
    for file in tqdm(files, desc=f"  Copie {target_class}", leave=False):
        src = os.path.join(source_folder, file)

        if good_image_from_path(src):
            dst_filename = f"{source_name}_{file}"
            dst = os.path.join(UNIFIED_DIR, target_class, dst_filename)
            shutil.copy(src, dst)
            copied += 1
        else:
            stats['skipped_quality'] += 1

    print(f"     Copiées : {copied}/{len(files)}")

    # Mise à jour des stats (clé cohérente : source_name_class)
    stat_key = f"{source_name.lower()}_{target_class.lower()}"
    if stat_key in stats:
        stats[stat_key] += copied


def process_dicom_rsna(dcm_path):
    """Traite un fichier DICOM RSNA et retourne une image normalisée + CLAHE"""
    try:
        ds = pydicom.dcmread(dcm_path)
        img = ds.pixel_array.astype(np.float32)

        img_min = np.min(img)
        img_max = np.max(img)

        if img_max - img_min < 1:
            return None

        img_normalized = (img - img_min) / (img_max - img_min) * 255
        img_uint8 = img_normalized.astype(np.uint8)

        # CLAHE pour améliorer le contraste
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_uint8 = clahe.apply(img_uint8)

        return img_uint8

    except Exception as e:
        print(f"  Erreur traitement DICOM {dcm_path}: {e}")
        return None


# FUSION DATASET 1 : KERMANY
def process_kermany():
    print("\n" + "=" * 70)
    print(" DATASET 1 : KERMANY (Paul Mooney - Enfants)")
    print("=" * 70)

    kermany_base = os.path.join(BASE_DIR, 'PAUL', 'chest_xray')

    if not os.path.exists(kermany_base):
        print(f" Dossier Kermany introuvable : {kermany_base}")
        return

    for split in ['train', 'val', 'test']:
        print(f"\n Split : {split}")
        for class_name in ['NORMAL', 'PNEUMONIA']:
            folder = os.path.join(kermany_base, split, class_name)
            copy_images(folder, class_name, 'kermany')  # source_name fixe = 'kermany'

    print(f"\n Kermany traité avec succès")


# FUSION DATASET 2 : RSNA
def process_rsna(max_images=None, visualize_samples=False, balance_classes=True):
    print("\n" + "=" * 70)
    print(" DATASET 2 : RSNA Pneumonia Detection Challenge")
    print("=" * 70)

    rsna_images_dir = os.path.join(BASE_DIR, 'RSNA', 'stage_2_train_images')
    rsna_labels_csv = os.path.join(BASE_DIR, 'RSNA', 'stage_2_train_labels.csv')

    if not os.path.exists(rsna_images_dir):
        print(f" Dossier RSNA introuvable : {rsna_images_dir}")
        return

    if not os.path.exists(rsna_labels_csv):
        print(f" Fichier labels RSNA introuvable : {rsna_labels_csv}")
        return

    print(" Chargement des labels...")
    df = pd.read_csv(rsna_labels_csv)
    df_grouped = df.groupby('patientId')['Target'].max().reset_index()

    print(f"   Total patients : {len(df_grouped)}")
    print(f"   Normal (Target=0) : {(df_grouped['Target'] == 0).sum()}")
    print(f"   Pneumonia (Target=1) : {(df_grouped['Target'] == 1).sum()}")

    # Équilibrage optionnel des classes
    if balance_classes:
        normal_count = (df_grouped['Target'] == 0).sum()
        pneumonia_count = (df_grouped['Target'] == 1).sum()
        target_count = int(min(normal_count, pneumonia_count) * 1.2)

        if normal_count > target_count:
            normal_df = df_grouped[df_grouped['Target'] == 0].sample(n=target_count, random_state=42)
            pneumonia_df = df_grouped[df_grouped['Target'] == 1]
        elif pneumonia_count > target_count:
            normal_df = df_grouped[df_grouped['Target'] == 0]
            pneumonia_df = df_grouped[df_grouped['Target'] == 1].sample(n=target_count, random_state=42)
        else:
            normal_df = df_grouped[df_grouped['Target'] == 0]
            pneumonia_df = df_grouped[df_grouped['Target'] == 1]

        df_grouped = pd.concat([normal_df, pneumonia_df]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   Classes équilibrées → Normal: {(df_grouped['Target']==0).sum()}, Pneumonia: {(df_grouped['Target']==1).sum()}")

    # Limitation pour tests
    if max_images:
        df_grouped = df_grouped.sample(n=min(max_images, len(df_grouped)), random_state=42)
        print(f"    Limitation à {len(df_grouped)} images")

    print("\n Traitement des fichiers DICOM...")
    processed = 0
    visualized = 0

    for _, row in tqdm(df_grouped.iterrows(), total=len(df_grouped), desc="RSNA"):
        patient_id = row['patientId']
        target = row['Target']
        dcm_path = os.path.join(rsna_images_dir, patient_id + '.dcm')

        if not os.path.exists(dcm_path):
            stats['skipped_error'] += 1
            continue

        img = process_dicom_rsna(dcm_path)
        if img is None:
            stats['skipped_error'] += 1
            continue

        if not good_image(img):
            stats['skipped_quality'] += 1
            continue

        if visualize_samples and visualized < 5:
            cv2.imshow("RSNA Sample", img)
            cv2.waitKey(500)
            visualized += 1

        target_class = 'NORMAL' if target == 0 else 'PNEUMONIA'
        dst_filename = f"rsna_{patient_id}.png"
        dst = os.path.join(UNIFIED_DIR, target_class, dst_filename)
        cv2.imwrite(dst, img)

        if target == 0:
            stats['rsna_normal'] += 1
        else:
            stats['rsna_pneumonia'] += 1

        processed += 1

    cv2.destroyAllWindows()
    print(f"\n RSNA traité : {processed} images sauvegardées")


# FUSION DATASET 3 : TB RAHMAN
def process_tb_rahman():
    print("\n" + "=" * 70)
    print(" DATASET 3 : TB Rahman (Tuberculose)")
    print("=" * 70)

    tb_base = os.path.join(BASE_DIR, 'TB_RAHMAN', 'TB_Chest_Radiography_Database')

    if not os.path.exists(tb_base):
        print(f" Dossier TB Rahman introuvable : {tb_base}")
        return

    print("\n Classe : NORMAL")
    copy_images(os.path.join(tb_base, 'Normal'), 'NORMAL', 'tb')

    print("\n Classe : TUBERCULOSIS → PNEUMONIA")
    copy_images(os.path.join(tb_base, 'Tuberculosis'), 'PNEUMONIA', 'tb')

    print(f"\n TB Rahman traité avec succès")


# RAPPORT FINAL
def print_final_report():
    print("\n" + "=" * 70)
    print(" RAPPORT FINAL DE FUSION")
    print("=" * 70)

    normal_count = len(os.listdir(os.path.join(UNIFIED_DIR, 'NORMAL')))
    pneumonia_count = len(os.listdir(os.path.join(UNIFIED_DIR, 'PNEUMONIA')))
    total = normal_count + pneumonia_count

    print(f"\n Dossier final : {UNIFIED_DIR}")
    print(f"\n RÉSUMÉ GLOBAL :")
    print(f"   NORMAL       : {normal_count:6,} images ({normal_count / total * 100:.1f}%)")
    print(f"   PNEUMONIA    : {pneumonia_count:6,} images ({pneumonia_count / total * 100:.1f}%)")
    print(f"   {'─' * 50}")
    print(f"   TOTAL        : {total:6,} images")

    print(f"\n DÉTAILS PAR SOURCE :")
    print(f"   Kermany (enfants)     : Normal {stats['kermany_normal']:6,} | Pneumonia {stats['kermany_pneumonia']:6,}")
    print(f"   RSNA (adultes)        : Normal {stats['rsna_normal']:6,} | Pneumonia {stats['rsna_pneumonia']:6,}")
    print(f"   TB Rahman             : Normal {stats['tb_normal']:6,} | Pneumonia {stats['tb_pneumonia']:6,}")

    print(f"\n IMAGES IGNORÉES :")
    print(f"   Qualité insuffisante : {stats['skipped_quality']:6,}")
    print(f"   Erreurs traitement   : {stats['skipped_error']:6,}")

    ratio = normal_count / pneumonia_count if pneumonia_count > 0 else 0
    print(f"\n ÉQUILIBRE DES CLASSES : Ratio Normal/Pneumonia = {ratio:.2f}")
    if 0.8 <= ratio <= 1.2:
        print("    Classes bien équilibrées")
    else:
        print("    Déséquilibre détecté (à corriger avec class_weight ou sous-échantillonnage)")

    print("\n" + "=" * 70)
    print(" FUSION TERMINÉE AVEC SUCCÈS !")
    print("=" * 70)


# SCRIPT PRINCIPAL
def main():
    print("\n" + "=" * 70)
    print(" FUSION DE DATASETS - DÉTECTION PNEUMONIE")
    print("   Master IA-BD - ESGIS 2024-2025")
    print("=" * 70)

    process_kermany()
    process_rsna(max_images=None, visualize_samples=False, balance_classes=True)  # Équilibrage activé par défaut
    process_tb_rahman()
    print_final_report()


if __name__ == "__main__":
    main()