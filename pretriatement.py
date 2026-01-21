import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Configuration
GOGETA_DIR = r"C:\Z_Soutenance\DATA\GOGETA"
SPLITS_DIR = r"C:\Z_Soutenance\DATA\SPLITS"  # Nouveau dossier pour train/val/test
RANDOM_SEED = 42

# Créer structure
for split in ['train', 'val', 'test']:
    for class_name in ['NORMAL', 'PNEUMONIA']:
        os.makedirs(os.path.join(SPLITS_DIR, split, class_name), exist_ok=True)

def load_dataset_paths():
    data = []
    for class_idx, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = os.path.join(GOGETA_DIR, class_name)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for f in files:
            data.append({
                'path': os.path.join(folder, f),
                'label': class_name
            })
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"Total images : {len(df)}")
    print(f"NORMAL : {(df['label'] == 'NORMAL').sum()}")
    print(f"PNEUMONIA : {(df['label'] == 'PNEUMONIA').sum()}")
    return df

def create_splits(df):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=RANDOM_SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=RANDOM_SEED)
    print(f"Train : {len(train_df)}, Val : {len(val_df)}, Test : {len(test_df)}")
    return train_df, val_df, test_df

def copy_split(split_df, split_name):
    print(f"\nCopie {split_name}...")
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        src = row['path']
        dst = os.path.join(SPLITS_DIR, split_name, row['label'], os.path.basename(src))
        shutil.copy(src, dst)

# Exécution
print("CRÉATION DES DOSSIERS SPLITS (sans .npy)")
df = load_dataset_paths()
train_df, val_df, test_df = create_splits(df)
copy_split(train_df, 'train')
copy_split(val_df, 'val')
copy_split(test_df, 'test')

print("\nTERMINE ! Structure prête dans :", SPLITS_DIR)
print("train/NORMAL + train/PNEUMONIA etc.")
print("Prêt pour entraînement avec generators !")