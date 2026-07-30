#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob

def verif(H, img_width, img_height):
    """
    Vérifie que la transformation appliquée à l'image d'entrée n'est pas absurde, ie pas
    une trop grosse translation ni rotation ni un changement d'échelle trop important.
    Prend en entrée une matrice d'homographie H, et les dimensions de l'image.
    """

    # le déterminant de la matrice de rotation contenue dans H représente l'échelle au carré
    det = H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]
    if det <= 0:
        return False
    scale = np.sqrt(det)

    # un changement d'échelle trop important est refusé
    if scale > 1.25 or scale < 0.8:
        return False

    # H[0, 2] représente la translation horizontale, H[1, 2] la translation verticale
    # on ne veut pas plus de 10% de l'image de translatée
    t_max_x = 0.1 * img_width
    t_max_y = 0.1 * img_height
    if abs(H[0, 2]) > t_max_x or abs(H[1, 2]) > t_max_y:
        return False

    # H[0, 0] représente le cosinus et H[1, 0] le sinus
    # un sinus loin de 0 est refusé, un cos loin de 1 aussi
    cos = H[0, 0]/scale
    sin = H[1, 0]/scale
    if cos > 1.1 or cos < 0.9:
        return False
    if abs(sin) > 0.1:
        return False

    return True

def recalage_image_sift(img_master_path, file_input_path, file_output_path):
    """
    Recale (aligne) un dossier par rapport à une image de référence en utilisant SIFT et RANSAC,
    tel que décrit par Portenier et al. (2020). Rajout d'une vérification géométrique pour pas 
    que l'algorithme ne fasse une transformation absurde.
    """

    img_master = cv2.imread(img_master_path, cv2.IMREAD_GRAYSCALE)
    height, width = img_master.shape

    # détection des keypoints sur l'image d'entrée
    sift = cv2.SIFT_create()
    kp_master, des_master = sift.detectAndCompute(img_master, None)

    # ça servira à déterminer les points de correspondance
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # on recherche seulement les images (en .jpg)
    file_jpg = []
    file_jpg.extend(glob.glob(os.path.join(file_input_path, "*.jpg")))

    for img_input_path in file_jpg:

        nom_fichier = os.path.basename(img_input_path)
        chemin_sauvegarde = os.path.join(file_output_path, nom_fichier)

        img_input_color = cv2.imread(img_input_path)
        img_input = cv2.cvtColor(img_input_color, cv2.COLOR_BGR2GRAY)

        kp_input, des_input = sift.detectAndCompute(img_input, None)
        matches = flann.knnMatch(des_input, des_master, k=2)

        # filtrage des bons matchs (test de ratio de Lowe)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # il faut au moins 4 points pour calculer notre matrice d'homographie
        min_match = 4
        if len(good_matches) >= min_match:
            # Extraire les coordonnées des points correspondants
            src_pts = np.float32([kp_input[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_master[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # on trouve la matrice homographique avec ransac
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

            if H is not None:
                if verif(H, width, height):

                    # transformation de l'image d'entrée
                    img_aligned = cv2.warpPerspective(img_input_color, H, (width, height))
                    print(f"{nom_fichier} recalée avec succès")

                    cv2.imwrite(chemin_sauvegarde, img_aligned)
                    continue
                else:
                    print(f"{nom_fichier} non alignée, transformation absurde")
                    cv2.imwrite(chemin_sauvegarde, img_input_color)
                    continue
            else:
                print("L'homographie n'a pas pu être calculée.")
                cv2.imwrite(chemin_sauvegarde, img_input_color)
                continue
        else:
            print(f"Pas assez de points de correspondance trouvé pour {nom_fichier} - {len(good_matches)}/{min_match}")
            cv2.imwrite(chemin_sauvegarde, img_input_color)
