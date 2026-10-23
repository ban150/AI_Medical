"""
Vérification de l'homogénéité des radiographies (tailles, canaux, contraste).

Les graphiques sont écrits sous ``reports/eda/`` (hors du dossier GOGETA pour ne pas mélanger
données brutes et artefacts d'analyse). Chemins pilotés par ``config.settings``.
"""

import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm

from AI_Medical.config.settings import load_training_config

_cfg = load_training_config()
DATASET_DIR = str(_cfg.gogeta_dir) if _cfg.gogeta_dir else str(_cfg.data_root / "GOGETA")
CLASSES = ["NORMAL", "PNEUMONIA"]

_project_root = Path(__file__).resolve().parent.parent
REPORT_DIR = str(_project_root / "reports" / "eda")
os.makedirs(REPORT_DIR, exist_ok=True)

# FONCTION D'ANALYSE PAR CLASSE

def check_images(class_name, max_samples=None):
    """
    Analyse les images d'une classe

    Args:
        class_name: 'NORMAL' ou 'PNEUMONIA'
        max_samples: Nombre max d'images à analyser (None = toutes)
    """

    print("\n" + "=" * 70)
    print(f" ANALYSE DE LA CLASSE : {class_name}")
    print("=" * 70)

    folder = os.path.join(DATASET_DIR, class_name)

    if not os.path.exists(folder):
        print(f" Dossier introuvable : {folder}")
        return None

    # Lister les fichiers
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f" Dossier : {folder}")
    print(f" Nombre total d'images : {len(files)}")

    if len(files) == 0:
        print(" Aucune image trouvée !")
        return None

    # Limiter si nécessaire
    if max_samples and len(files) > max_samples:
        files = files[:max_samples]
        print(f" Analyse limitée à {max_samples} images (pour rapidité)")

    # Initialisation des statistiques
    stats = {
        'sizes': [],
        'means': [],
        'stds': [],
        'formats': [],
        'channels': [],
        'corrupted': [],
        'low_contrast': [],
        'extreme_brightness': [],
        'min_size': float('inf'),
        'max_size': 0
    }

    # Analyse de chaque image
    print(f"\nAnalyse en cours...")
    for f in tqdm(files, desc=f"  Scan {class_name}"):
        path = os.path.join(folder, f)

        try:
            img = cv2.imread(path)

            if img is None:
                stats['corrupted'].append(f)
                continue

            # Dimensions
            h, w = img.shape[:2]
            stats['sizes'].append((h, w))
            stats['min_size'] = min(stats['min_size'], min(h, w))
            stats['max_size'] = max(stats['max_size'], max(h, w))

            # Nombre de canaux
            if len(img.shape) == 3:
                stats['channels'].append(img.shape[2])
            else:
                stats['channels'].append(1)

            # Format
            stats['formats'].append(str(img.dtype))

            # Convertir en niveaux de gris si nécessaire
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = img

            # Statistiques d'intensité
            mean_val = np.mean(img_gray)
            std_val = np.std(img_gray)

            stats['means'].append(mean_val)
            stats['stds'].append(std_val)

            # Detection extreme luminosité (seuils plus stricts : <20 ou >235)
            if mean_val < 20 or mean_val > 235:  # trop sombre ou trop clair
                stats['extreme_brightness'].append(f)

            # Contraste faible
            if std_val < 10:
                stats['low_contrast'].append(f)

        except Exception as e:
            print(f" Erreur sur {f}: {e}")
            stats['corrupted'].append(f)

    # Affichage des résultats
    print_statistics(class_name, stats, len(files))

    # Visualisations
    plot_distributions(class_name, stats)
    plot_samples(folder, files[:10], class_name)

    return stats

### AFFICHAGE DES STATISTIQUES


def print_statistics(class_name, stats, total_files):
    """Affiche les statistiques calculées"""

    print(f"\n RÉSULTATS POUR {class_name} :")
    print(f"   Images analysées : {len(stats['sizes'])} / {total_files}")

    # Images corrompues
    if stats['corrupted']:
        print(f"\n  IMAGES CORROMPUES : {len(stats['corrupted'])}")
        if len(stats['corrupted']) <= 10:
            for f in stats['corrupted']:
                print(f"      - {f}")
        else:
            print(f"      (trop nombreuses pour afficher)")
    else:
        print(f"\n Aucune image corrompue")

    # Contraste faible
    if stats['low_contrast']:
        print(f"\n  CONTRASTE FAIBLE (<10) : {len(stats['low_contrast'])}")
        if len(stats['low_contrast']) <= 10:
            for f in stats['low_contrast']:
                print(f"      - {f}")
    else:
        print(f"\n Toutes les images ont un contraste suffisant")

    # Luminosité extreme
    if stats['extreme_brightness']:
        print(f"\n  LUMINOSITÉ EXTRÊME (<20 ou >235) : {len(stats['extreme_brightness'])}")
    else:
        print(f"\n Pas d'images à luminosité extrême")

    # Dimensions
    print(f"\n DIMENSIONS :")
    size_counts = Counter(stats['sizes'])
    print(f"   Tailles uniques : {len(size_counts)}")
    print(f"   Top 5 tailles les plus fréquentes :")
    for size, count in size_counts.most_common(5):
        print(f"      {size[0]}×{size[1]} : {count} images ({count / len(stats['sizes']) * 100:.1f}%)")
    print(f"   Taille min : {stats['min_size']}px")
    print(f"   Taille max : {stats['max_size']}px")

    # Canaux
    print(f"\n CANAUX :")
    channel_counts = Counter(stats['channels'])
    for ch, count in channel_counts.items():
        ch_name = "Niveaux de gris" if ch == 1 else f"RGB ({ch} canaux)"
        print(f"   {ch_name} : {count} images ({count / len(stats['channels']) * 100:.1f}%)")

    # Types de données
    print(f"\n TYPES DE DONNÉES :")
    format_counts = Counter(stats['formats'])
    for fmt, count in format_counts.items():
        print(f"   {fmt} : {count} images ({count / len(stats['formats']) * 100:.1f}%)")

    # Intensités
    if stats['means']:
        print(f"\n INTENSITÉ DES PIXELS :")
        print(f"   Moyenne globale : {np.mean(stats['means']):.2f} ± {np.std(stats['means']):.2f}")
        print(f"   Min/Max : {min(stats['means']):.2f} / {max(stats['means']):.2f}")
        print(f"   Écart-type moyen : {np.mean(stats['stds']):.2f} ± {np.std(stats['stds']):.2f}")
        print(f"   Min/Max écart-type : {min(stats['stds']):.2f} / {max(stats['stds']):.2f}")


### VISUALISATIONS


def plot_distributions(class_name, stats):
    """Affiche les distributions d'intensité"""

    if not stats['means']:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme des moyennes
    axes[0].hist(stats['means'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(stats['means']), color='red', linestyle='--',
                    label=f'Moyenne : {np.mean(stats["means"]):.1f}')
    axes[0].set_xlabel('Intensité moyenne', fontsize=12)
    axes[0].set_ylabel('Nombre d\'images', fontsize=12)
    axes[0].set_title(f'{class_name} - Distribution des intensités moyennes',
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Histogramme des écarts-types
    axes[1].hist(stats['stds'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(stats['stds']), color='darkred', linestyle='--',
                    label=f'Moyenne : {np.mean(stats["stds"]):.1f}')
    axes[1].set_xlabel('Écart-type', fontsize=12)
    axes[1].set_ylabel('Nombre d\'images', fontsize=12)
    axes[1].set_title(f'{class_name} - Distribution du contraste (std)',
                      fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'{class_name}_distributions.png')
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"  Graphique sauvegardé : reports/eda/{class_name}_distributions.png")


def plot_samples(folder, files, class_name, n_samples=10):
    """Affiche des échantillons d'images"""

    n_samples = min(n_samples, len(files))

    if n_samples == 0:
        return

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()

    for i, f in enumerate(files[:n_samples]):
        img_path = os.path.join(folder, f)
        img = cv2.imread(img_path)

        if img is None:
            axes[i].text(0.5, 0.5, 'Image corrompue', ha='center', va='center')
            axes[i].axis('off')
            continue

        # Convertir BGR → RGB
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img

        axes[i].imshow(img_rgb, cmap='gray' if len(img.shape) == 2 else None)
        axes[i].set_title(f"{f[:20]}...\n{img.shape[0]}×{img.shape[1]}", fontsize=8)
        axes[i].axis('off')

    plt.suptitle(f'Échantillons - {class_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, f'{class_name}_samples.png')
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"  Échantillons sauvegardés : reports/eda/{class_name}_samples.png")


### COMPARAISON ENTRE CLASSES


def compare_classes(stats_normal, stats_pneumonia):
    """Compare les statistiques entre les deux classes"""

    if stats_normal is None or stats_pneumonia is None:
        print("\n  Impossible de comparer : une classe manque de données")
        return

    print("\n" + "=" * 70)
    print("  COMPARAISON ENTRE CLASSES")
    print("=" * 70)

    # Nombre d'images
    n_normal = len(stats_normal['sizes'])
    n_pneumonia = len(stats_pneumonia['sizes'])
    print(f"\n ÉQUILIBRE DES CLASSES :")
    print(f"   NORMAL    : {n_normal:5,} images ({n_normal / (n_normal + n_pneumonia) * 100:.1f}%)")
    print(f"   PNEUMONIA : {n_pneumonia:5,} images ({n_pneumonia / (n_normal + n_pneumonia) * 100:.1f}%)")
    print(f"   Ratio     : {n_normal / n_pneumonia:.2f}")

    if 0.8 <= n_normal / n_pneumonia <= 1.2:
        print(f"   Classes bien équilibrées")
    else:
        print(f"    Déséquilibre détecté")

    # Intensités
    print(f"\n INTENSITÉ MOYENNE :")
    print(f"   NORMAL    : {np.mean(stats_normal['means']):.2f} ± {np.std(stats_normal['means']):.2f}")
    print(f"   PNEUMONIA : {np.mean(stats_pneumonia['means']):.2f} ± {np.std(stats_pneumonia['means']):.2f}")

    # Contraste
    print(f"\n CONTRASTE (std) :")
    print(f"   NORMAL    : {np.mean(stats_normal['stds']):.2f} ± {np.std(stats_normal['stds']):.2f}")
    print(f"   PNEUMONIA : {np.mean(stats_pneumonia['stds']):.2f} ± {np.std(stats_pneumonia['stds']):.2f}")

    # Graphique comparatif
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Moyennes
    axes[0].hist(stats_normal['means'], bins=50, alpha=0.6, label='NORMAL', color='green')
    axes[0].hist(stats_pneumonia['means'], bins=50, alpha=0.6, label='PNEUMONIA', color='red')
    axes[0].set_xlabel('Intensité moyenne')
    axes[0].set_ylabel('Nombre d\'images')
    axes[0].set_title('Comparaison des intensités moyennes')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Écarts-types
    axes[1].hist(stats_normal['stds'], bins=50, alpha=0.6, label='NORMAL', color='green')
    axes[1].hist(stats_pneumonia['stds'], bins=50, alpha=0.6, label='PNEUMONIA', color='red')
    axes[1].set_xlabel('Écart-type (contraste)')
    axes[1].set_ylabel('Nombre d\'images')
    axes[1].set_title('Comparaison du contraste')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(REPORT_DIR, 'comparison_classes.png')
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\n  Graphique sauvegardé : reports/eda/comparison_classes.png")

# RECOMMANDATIONS


def give_recommendations(stats_normal, stats_pneumonia):
    """Donne des recommandations basées sur l'analyse"""

    print("\n" + "=" * 70)
    print("💡 RECOMMANDATIONS POUR L'ENTRAÎNEMENT")
    print("=" * 70)

    recommendations = []

    # Vérifier les dimensions
    all_sizes_normal = set(stats_normal['sizes'])
    all_sizes_pneumonia = set(stats_pneumonia['sizes'])

    if len(all_sizes_normal) > 100 or len(all_sizes_pneumonia) > 100:
        recommendations.append(
            "️  Grande variabilité de dimensions détectée\n"
            "   → OBLIGATOIRE : Redimensionner toutes les images à 224×224"
        )

    # Vérifier le contraste
    low_contrast_ratio_normal = len(stats_normal['low_contrast']) / len(stats_normal['sizes'])
    low_contrast_ratio_pneumonia = len(stats_pneumonia['low_contrast']) / len(stats_pneumonia['sizes'])

    if low_contrast_ratio_normal > 0.05 or low_contrast_ratio_pneumonia > 0.05:
        recommendations.append(
            "️  >5% d'images à faible contraste détectées\n"
            "   → Appliquer CLAHE (égalisation adaptative d'histogramme)"
        )

    # Vérifier l'équilibre
    n_normal = len(stats_normal['sizes'])
    n_pneumonia = len(stats_pneumonia['sizes'])
    ratio = n_normal / n_pneumonia

    if ratio > 1.5:
        recommendations.append(
            f"  Déséquilibre : {ratio:.2f}× plus de NORMAL que PNEUMONIA\n"
            "   → Appliquer class_weight lors de l'entraînement\n"
            f"   → Poids recommandés : Normal={1 / ratio:.2f}, Pneumonia=1.0"
        )
    elif ratio < 0.7:
        recommendations.append(
            f"  Déséquilibre : {1 / ratio:.2f}× plus de PNEUMONIA que NORMAL\n"
            "   → Appliquer class_weight lors de l'entraînement\n"
            f"   → Poids recommandés : Normal=1.0, Pneumonia={ratio:.2f}"
        )

    # Contraste moyen global
    global_std = np.mean(stats_normal['stds'] + stats_pneumonia['stds'])
    if global_std < 35:
        recommendations.append(
            "️  Contraste moyen faible détecté (std global < 35)\n"
            "   → Fortement recommandé : Appliquer CLAHE lors du prétraitement"
        )

    # Afficher les recommandations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec}")
    else:
        print("\n Aucune action corrective nécessaire")
        print("   Le dataset est prêt pour l'entraînement !")

    print("\n" + "=" * 70)
    print(" Tous les graphiques et échantillons sont sauvegardés dans :")
    print(f" {REPORT_DIR}")
    print(" PROCHAINE ÉTAPE : CRÉER LES SPLITS TRAIN/VAL/TEST")
    print("=" * 70)

# SCRIPT PRINCIPAL


def main():
    """Point d'entrée"""
    print("\n" + "=" * 70)
    print(" VÉRIFICATION DE L'HOMOGÉNÉITÉ DU DATASET")
    print("   Master IA-BD - ESGIS 2024-2025")
    print("=" * 70)

    # Vérifier que le dossier existe
    if not os.path.exists(DATASET_DIR):
        print(f"\n Dossier introuvable : {DATASET_DIR}")
        return

    # Analyser chaque classe
    stats_results = {}

    for class_name in CLASSES:
        stats = check_images(class_name, max_samples=None)  # Analyse TOUTES les images
        stats_results[class_name] = stats

    # Comparaison entre classes
    if len(stats_results) == 2:
        compare_classes(stats_results['NORMAL'], stats_results['PNEUMONIA'])
        give_recommendations(stats_results['NORMAL'], stats_results['PNEUMONIA'])


if __name__ == "__main__":
    main()