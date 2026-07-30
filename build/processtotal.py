#!/usr/bin/env python3

import os
import cv2
import numpy as np
import yaml
import subprocess
import tempfile
import glob
from tqdm import tqdm
import sys

# Géoréférencement
from numpy import linalg as LA
from osgeo import gdal

# Segmentation
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from build.HRNet import HRNetWithDeepSupervision

gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

# 1. Fonctions utilitaires

def verif(H, img_width, img_height):
    """Vérifie que la transformation homographique n'est pas absurde."""
    det = H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]
    if det <= 0: return False
    scale = np.sqrt(det)
    if scale > 1.25 or scale < 0.8: return False
    
    t_max_x = 0.1 * img_width
    t_max_y = 0.1 * img_height
    if abs(H[0, 2]) > t_max_x or abs(H[1, 2]) > t_max_y: return False
    
    cos = H[0, 0] / scale
    sin = H[1, 0] / scale
    if cos > 1.1 or cos < 0.9: return False
    if abs(sin) > 0.1: return False
    return True

def get_sliding_windows(dimension_length, window_size, stride):
    steps = list(range(0, dimension_length - window_size + 1, stride))
    if not steps or steps[-1] + window_size < dimension_length:
        steps.append(dimension_length - window_size)
    return steps

def writeTIFFMask(filename, geotransform, geoprojection, data):
    """Écrit le masque final en tiff"""

    y, x = data.shape
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(filename, x, y, 1, gdal.GDT_Byte)
    dst_ds.GetRasterBand(1).WriteArray(data)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(geoprojection)
    dst_ds.FlushCache()

def get_projmatrices(obscoords, tgtcoords, upperleftx, y0, focallength, fwidth, fheight, rolldeg):
    d = focallength
    w = fwidth / 2.0
    h = fheight / 2.0
    vectorc = np.copy(obscoords)
    vectorc[0] = vectorc[0] - upperleftx
    vectorc[1] = vectorc[1] - y0

    vectorview = tgtcoords - obscoords
    vectorn = vectorview / LA.norm(vectorview)
    vectornp = np.copy(vectorn)
    vectornp[2] = 0
    vectorncross = np.copy(vectorn)
    if vectorncross[2] == 0: vectorncross[2] = 1

    if vectorn[2] > 0:
        vectoru = np.cross(vectornp, vectorncross)
    else:
        vectoru = np.cross(vectorncross, vectornp)

    vectoru = vectoru / LA.norm(vectoru)
    rolladd = np.tan(np.radians(-rolldeg))
    vectoru[2] = vectoru[2] + rolladd
    vectoru = vectoru / LA.norm(vectoru)
    vectorv = np.cross(vectoru, vectorn)

    ttrans = np.matrix([[1, 0, 0, -vectorc[0]],
                        [0, 1, 0, -vectorc[1]],
                        [0, 0, 1, -vectorc[2]],
                        [0, 0, 0, 1]])

    tview = np.matrix([[vectoru[0], vectoru[1], vectoru[2], 0],
                       [vectorv[0], vectorv[1], vectorv[2], 0],
                       [vectorn[0], vectorn[1], vectorn[2], 0],
                       [0, 0, 1 / d, 1]])
    return ttrans, tview

def init_georef_matrices(yaml_georef, realsizex, realsizey):
    """Calcule les matrices de projection géographiques."""
    with open(yaml_georef, 'r') as geoset:
        grfyml = yaml.safe_load(geoset)
    
    dsdem = gdal.Open(grfyml['demfname'])
    elevation = dsdem.GetRasterBand(1).ReadAsArray()[::-1, ...]
    nrows, ncols = elevation.shape
    geotransform = dsdem.GetGeoTransform()
    geoproj = dsdem.GetProjection()
    x0, dx, dxdy, y1, dydx, dy = geotransform
    
    y0 = y1 + dy * nrows
    dl = dx

    ds_vis = gdal.Open(grfyml['visfname']) 
    vis = ds_vis.GetRasterBand(1).ReadAsArray()[::-1, ...]
    
    fwidth, fheight = grfyml['fwidth'], grfyml['fheight']
    focallength = grfyml['focallength']
    
    ttrans, tview = get_projmatrices(
        np.array(grfyml['obscoords']), np.array(grfyml['tgtcoords']), 
        x0, y0, focallength, fwidth, fheight, grfyml['roll']
    )

    layerx = np.outer(np.ones(nrows), np.r_[0:ncols]) * dl + ttrans[0, 3]
    layery = np.outer(np.r_[0:nrows], np.ones(ncols)) * dl + ttrans[1, 3]
    layerz = elevation + ttrans[2, 3]

    vectorview = np.array(grfyml['tgtcoords']) - np.array(grfyml['obscoords'])
    visproj = layerx * vectorview[0] + layery * vectorview[1] + layerz * vectorview[2]
    vis[visproj < 0] = 0
    layerx, layery, layerz = layerx * vis, layery * vis, layerz * vis

    drop = (layerx**2 + layery**2) / (2 * 6371000)
    layerz = layerz - drop
    layerw = np.ones([nrows, ncols])

    coordsc = np.zeros([nrows, ncols, 4])
    coordsc[:, :, 0] = tview[0,0]*layerx + tview[0,1]*layery + tview[0,2]*layerz + tview[0,3]*layerw
    coordsc[:, :, 1] = tview[1,0]*layerx + tview[1,1]*layery + tview[1,2]*layerz + tview[1,3]*layerw
    coordsc[:, :, 2] = tview[2,0]*layerx + tview[2,1]*layery + tview[2,2]*layerz + tview[2,3]*layerw

    imgz = coordsc[:, :, 2]
    imgw = (fwidth / 2.0) * imgz / focallength
    
    with np.errstate(invalid='ignore'):
        imgx = coordsc[:, :, 0] / imgw
        imgy = coordsc[:, :, 1] / imgw

    scale = 100.0 * (fwidth / 2.0) * (realsizex / fwidth / 100.0)
    imgx = np.nan_to_num(np.rint(scale * imgx + np.rint(realsizex / 2.0)))
    imgy = np.nan_to_num(np.rint(scale * imgy + np.rint(realsizey / 2.0)))
    
    imgx, imgy = np.clip(imgx, 0, realsizex - 1), np.clip(imgy, 0, realsizey - 1)
    imgx, imgy = imgx * vis, imgy * vis
    
    imgx[imgy <= 0] = 0
    imgy[imgx <= 0] = 0
    
    return imgx.astype(int), imgy.astype(int), geotransform, geoproj


def segment_patch(img_bgr, model, transform, device, window_size=1024, stride=512, threshold=0.5):
    """Applique la prédiction HRNet par fenêtre glissante."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H_orig, W_orig, _ = img_rgb.shape
    
    pad_h = max(0, window_size - H_orig)
    pad_w = max(0, window_size - W_orig)
    if pad_h > 0 or pad_w > 0:
        img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    H, W, _ = img_rgb.shape

    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in get_sliding_windows(H, window_size, stride):
            for x in get_sliding_windows(W, window_size, stride):
                patch = img_rgb[y:y+window_size, x:x+window_size]
                patch_tensor = transform(image=patch)['image'].unsqueeze(0).to(device)
                probs = torch.sigmoid(model(patch_tensor)).squeeze().cpu().numpy()
                prob_map[y:y+window_size, x:x+window_size] += probs
                count_map[y:y+window_size, x:x+window_size] += 1

    final_prob_map = (prob_map / count_map)
    if pad_h > 0 or pad_w > 0:
        final_prob_map = final_prob_map[:H_orig, :W_orig]
        
    return (final_prob_map > threshold).astype(np.uint8) * 255


# 2. Fonction principale

def all_process(input_dir, output_dir, yaml_file, weights_path, device="mps", 
                do_recalage=True, do_decoupage=True, 
                keep_intermediate=False, recal_dir=None, decoup_dir=None, seg_dir=None):
    """
    Fait toutes les étapes demandées du processus en ouvrant d'abord l'interface de géoréférencement.
    Permet de conserver les fichiers intermédiaires si keep_intermediate=True.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Création des dossiers intermédiaires si demandée
    if keep_intermediate:
        if do_recalage and recal_dir: os.makedirs(recal_dir, exist_ok=True)
        if do_decoupage and decoup_dir: os.makedirs(decoup_dir, exist_ok=True)
        if seg_dir: os.makedirs(seg_dir, exist_ok=True)
        
    # Lecture du fichier yaml
    with open(yaml_file, "r") as f:
        config_yaml = yaml.safe_load(f)

    img_master_path = config_yaml.get("masterimage")
    if not img_master_path or not os.path.exists(img_master_path):
        raise FileNotFoundError(f"L'image de référence '{img_master_path}' est introuvable ou non définie dans le YAML.")

    # =========================================================================
    # ÉTAPE 1 : PRÉPARATION ET LANCEMENT DE L'INTERFACE DE GÉORÉFÉRENCEMENT
    # =========================================================================
    gui_img_path = img_master_path
    
    if do_decoupage:
        print("Découpage de l'image de référence pour l'interface...")
        cfg = config_yaml["decoupage"]
        vf_string = f"v360=input=cylindrical:output=rectilinear:ih_fov={cfg['inputhfov']}:yaw={cfg['yaw']}:h_fov={cfg['outputhfov']}:v_fov={cfg['outputvfov']}"
        
        # On s'assure que le dossier de découpe existe et on y enregistre l'image
        if decoup_dir:
            os.makedirs(decoup_dir, exist_ok=True)
            master_base_name = os.path.splitext(os.path.basename(img_master_path))[0]
            gui_img_path = os.path.join(decoup_dir, f"{master_base_name}-decoup.jpg")
            
            cmd = ["ffmpeg", "-y", "-i", img_master_path, "-vf", vf_string, "-frames:v", "1", gui_img_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    # On met à jour le YAML pour que l'interface ouvre bien cette image
    config_yaml['imgfname'] = gui_img_path
    with open(yaml_file, "w") as f:
        yaml.dump(config_yaml, f, default_flow_style=False, sort_keys=False)
        
    print("\n--- Ouverture de l'interface de géoréférencement ---")
    subprocess.run([sys.executable, "./build/interface_georef.py", "-s", yaml_file])
    print("--- Calibration terminée ---\n")

    # On recharge le YAML car l'utilisateur vient potentiellement de le modifier via l'interface
    with open(yaml_file, "r") as f:
        config_yaml = yaml.safe_load(f)
    # =========================================================================

    # ÉTAPE 2 : Chargement SIFT conditionnel
    if do_recalage:
        print("Initialisation SIFT...")
        sift = cv2.SIFT_create()
        img_master = cv2.imread(img_master_path, cv2.IMREAD_GRAYSCALE)
        height, width = img_master.shape
        kp_master, des_master = sift.detectAndCompute(img_master, None)
        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        
    # ÉTAPE 3 : Chargement HRNet
    print(f"Chargement HRNet sur {device}...")
    DEVICE = torch.device(device)
    model = HRNetWithDeepSupervision(model_name='hrnet_w18', num_classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    transform = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])

    # ÉTAPE 4 : Variables globales pour le géoréférencement
    if do_decoupage:
        cfg = config_yaml["decoupage"]
        vf_string = f"v360=input=cylindrical:output=rectilinear:ih_fov={cfg['inputhfov']}:yaw={cfg['yaw']}:h_fov={cfg['outputhfov']}:v_fov={cfg['outputvfov']}"

    georef_data = None 
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"Lancement du pipeline pour {len(image_files)} images...")

    with tempfile.TemporaryDirectory() as temp_dir:
        for img_path in tqdm(image_files, desc="Pipeline en cours"):
            nom_fichier = os.path.basename(img_path)
            base_name = os.path.splitext(nom_fichier)[0]
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            # RECALAGE
            img_aligned = img_bgr 
            if do_recalage:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                kp_input, des_input = sift.detectAndCompute(img_gray, None)
                matches = flann.knnMatch(des_input, des_master, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
                if len(good_matches) >= 4:
                    src_pts = np.float32([kp_input[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_master[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
                    if H is not None and verif(H, width, height):
                        img_aligned = cv2.warpPerspective(img_bgr, H, (width, height))
                        
                # Sauvegarde intermédiaire si demandée
                if keep_intermediate and recal_dir:
                    cv2.imwrite(os.path.join(recal_dir, nom_fichier), img_aligned)
                    
            # DECOUPAGE
            img_ready_for_seg = img_aligned
            if do_decoupage:
                temp_aligned = os.path.join(temp_dir, f"align_{nom_fichier}")
                cv2.imwrite(temp_aligned, img_aligned) 
                
                if keep_intermediate and decoup_dir:
                    out_cut_path = os.path.join(decoup_dir, f"{base_name}-decoup.jpg")
                else:
                    out_cut_path = os.path.join(temp_dir, f"cut_{nom_fichier}")
                    
                cmd = ["ffmpeg", "-y", "-i", temp_aligned, "-vf", vf_string, "-frames:v", "1", out_cut_path]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                img_ready_for_seg = cv2.imread(out_cut_path)
                if img_ready_for_seg is None: continue
                
            # INIT GEOREFERENCEMENT
            if georef_data is None:
                realsizey, realsizex = img_ready_for_seg.shape[:2]
                print(f"\n -> Initialisation de la matrice de géoréférencement ({realsizex}x{realsizey})...")
                georef_data = init_georef_matrices(yaml_file, realsizex, realsizey)
            imgx, imgy, geotransform, geoproj = georef_data

            # SEGMENTATION
            mask = segment_patch(img_ready_for_seg, model, transform, DEVICE)
            
            # Sauvegarde du masque non géoréférencé si demandée
            if keep_intermediate and seg_dir:
                cv2.imwrite(os.path.join(seg_dir, f"{base_name}_mask.png"), mask)
                
            # GEOREFERENCEMENT
            mask[0, :] = 0
            mask[:, 0] = 0
            mask[:, mask.shape[1] - 1] = 0
            mask[mask.shape[0] - 1, :] = 0
            mask[:, mask.shape[1] - 2] = 0
            zalbedo = mask[imgy, imgx]
            out_tif = os.path.join(output_dir, f"{base_name}_mask_geo.tif")
            writeTIFFMask(out_tif, geotransform, geoproj, zalbedo[::-1, ...])
            
    print("\nTraitement terminé ! Tous les masques géoréférencés finaux sont dans :", output_dir)
    if keep_intermediate:
        print("Les fichiers intermédiaires ont été conservés dans leurs dossiers respectifs.")