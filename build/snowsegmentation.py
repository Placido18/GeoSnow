#!/usr/bin/env python3

import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from .HRNet import HRNetWithDeepSupervision


def get_sliding_windows(dimension_length, window_size, stride):
    """
    Renvoie la liste d
    """
    steps = list(range(0, dimension_length - window_size + 1, stride))
    if not steps or steps[-1] + window_size < dimension_length:
        steps.append(dimension_length - window_size)
    return steps



def hrnetpredict(input_dir, output_dir, weights_path, window_size=512, stride=256, threshold=0.4, device="mps"):
    """
    Parcourt un dossier d'images, applique le modèle HRNet par fenêtre glissante,
    et sauvegarde les masques de neige générés dans un dossier de sortie.
    """
    
    # vérification et création des dossiers
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le dossier d'entrée n'existe pas : {input_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Lister les images valides
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"Aucune image trouvée dans {input_dir}")
        return

    # Initialisation du modèle
    DEVICE = torch.device(device)
    print(f"Chargement du modèle sur {DEVICE}...")
    model = HRNetWithDeepSupervision(model_name='hrnet_w18', num_classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # normalisation, pour pytorch (les vaeurs sont celles de l'entraînement)
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    print(f"{len(image_files)} images à traiter. Début de la prédiction...")

    # Boucle sur toutes les images du dossier
    for img_name in tqdm(image_files, desc="Progression globale"):
        img_path = os.path.join(input_dir, img_name)
        
        # Préparation du nom de sortie (ex: photo_mask.png)
        name_without_ext, _ = os.path.splitext(img_name)
        output_path = os.path.join(output_dir, f"{name_without_ext}_mask.png")

        # Chargement de l'image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Erreur de lecture pour {img_name}, image ignorée.")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H_orig, W_orig, _ = img_rgb.shape

        # Sécurité : Padding si l'image est plus petite que WINDOW_SIZE
        pad_h = max(0, window_size - H_orig)
        pad_w = max(0, window_size - W_orig)
        if pad_h > 0 or pad_w > 0:
            img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        H, W, _ = img_rgb.shape

        # Matrices d'assemblage (Soft Voting)
        prob_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        y_steps = get_sliding_windows(H, window_size, stride)
        x_steps = get_sliding_windows(W, window_size, stride)

        # Inférence par fenêtre glissante sur l'image en cours
        with torch.no_grad():
            for y in y_steps:
                for x in x_steps:
                    patch = img_rgb[y:y+window_size, x:x+window_size]
                    
                    augmented = transform(image=patch)
                    patch_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
                    
                    logits = model(patch_tensor)
                    probs = torch.sigmoid(logits)
                    probs_np = probs.squeeze().cpu().numpy()
                    
                    prob_map[y:y+window_size, x:x+window_size] += probs_np
                    count_map[y:y+window_size, x:x+window_size] += 1

        # Fusion des carrés (moyenne)
        final_prob_map = prob_map / count_map

        # Retrait du padding éventuel pour retrouver la taille d'origine
        if pad_h > 0 or pad_w > 0:
            final_prob_map = final_prob_map[:H_orig, :W_orig]

        # Binarisation
        final_mask = (final_prob_map > threshold).astype(np.uint8) * 255

        # Sauvegarde
        cv2.imwrite(output_path, final_mask)

    print(f"\nTraitement terminé ! Masques sauvegardés dans : {output_dir}")
