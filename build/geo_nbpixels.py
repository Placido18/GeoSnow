#!/usr/bin/env python3
import os
import yaml
import numpy as np
from numpy import linalg as LA
import cv2
from osgeo import gdal

def writeTIFF_Density(filename, geotransform, geoprojection, data):
    """Sauvegarde un TIF à une seule bande (Float32) pour la carte de densité."""
    (y, x) = data.shape
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_datatype = gdal.GDT_Float32
    dst_ds = driver.Create(filename, x, y, 1, dst_datatype)
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
    if vectorncross[2] == 0:
        vectorncross[2] = 1

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

def generate_density_map(yaml_file, image_path, output_tif):
    """
    Génère une carte de densité brute (.tif) et un aperçu visuel en couleurs (.png)
    """
    # 1. Chargement des paramètres du fichier YAML
    print(f"Chargement des paramètres depuis : {yaml_file}")
    with open(yaml_file, 'r') as geoset:
        grfyml = yaml.safe_load(geoset)

    demfname = grfyml['demfname']
    visfname = grfyml['visfname']
    obscoords = np.array(grfyml['obscoords'])
    tgtcoords = np.array(grfyml['tgtcoords'])
    fwidth = grfyml['fwidth']
    fheight = grfyml['fheight']
    focallength = grfyml['focallength']
    rolldeg = grfyml['roll']

    # 2. Chargement du MNT (DEM) et Vueshed
    dsdem = gdal.Open(demfname)
    elevation = dsdem.GetRasterBand(1).ReadAsArray()[::-1, ...]
    nrows, ncols = elevation.shape
    geotransform = dsdem.GetGeoTransform()
    geoproj = dsdem.GetProjection()
    x0, dx, dxdy, y1, dydx, dy = geotransform

    y0 = y1 + dy * nrows
    upperleftx = x0
    dl = dx

    visibility = gdal.Open(visfname)
    vis = visibility.GetRasterBand(1).ReadAsArray()[::-1, ...]

    # 3. Récupération des dimensions de l'image
    print(f"Lecture de l'image : {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur : Impossible de lire l'image à l'emplacement {image_path}")
        return

    realsizey, realsizex = img.shape[:2]
    print(f"Dimensions trouvées : {realsizex}x{realsizey}")

    # 4. CALCUL DE LA MATRICE DE PROJECTION
    print("Calcul des géométries de projection...")
    d = focallength
    w = fwidth / 2.0
    resolution = realsizex / fwidth / 100.0
    scale = 100.0 * w * resolution

    ttrans, tview = get_projmatrices(obscoords, tgtcoords, upperleftx, y0, focallength, fwidth, fheight, rolldeg)

    onenrows = np.ones(nrows)
    seqnrows = np.r_[0:nrows]
    seqncols = np.r_[0:ncols]
    onencols = np.ones(ncols)

    layerx = np.outer(onenrows, seqncols) * dl + ttrans[0, 3]
    layery = np.outer(seqnrows, onencols) * dl + ttrans[1, 3]
    layerz = elevation + ttrans[2, 3]

    vectorview = tgtcoords - obscoords
    visproj = layerx * vectorview[0] + layery * vectorview[1] + layerz * vectorview[2]
    vis[visproj < 0] = 0
    
    R = 6371000  # Rayon terrestre
    disth = np.sqrt(layerx**2 + layery**2)
    drop = disth**2 / (2 * R)
    layerz = layerz - drop
    layerw = np.ones([nrows, ncols])

    coordsc = np.zeros([nrows, ncols, 4])
    coordsc[:, :, 0] = tview[0, 0] * layerx + tview[0, 1] * layery + tview[0, 2] * layerz + tview[0, 3] * layerw
    coordsc[:, :, 1] = tview[1, 0] * layerx + tview[1, 1] * layery + tview[1, 2] * layerz + tview[1, 3] * layerw
    coordsc[:, :, 2] = tview[2, 0] * layerx + tview[2, 1] * layery + tview[2, 2] * layerz + tview[2, 3] * layerw
    coordsc[:, :, 3] = tview[3, 0] * layerx + tview[3, 1] * layery + tview[3, 2] * layerz + tview[3, 3] * layerw

    imgx = coordsc[:, :, 0]
    imgy = coordsc[:, :, 1]
    imgz = coordsc[:, :, 2]

    imgw = w * imgz / d
    with np.errstate(invalid='ignore'):
        imgx = imgx / imgw
        imgy = imgy / imgw

    midx = realsizex / 2.0
    midy = realsizey / 2.0
    
    # Coordonnées continues dans l'image
    imgx_float = scale * imgx + midx
    imgy_float = scale * imgy + midy

    print("Calcul de la densité (nombre de pixels photo par pixel MNT)...")
    
    # Calcul du Jacobien
    dx_dr, dx_dc = np.gradient(imgx_float)
    dy_dr, dy_dc = np.gradient(imgy_float)
    pixel_count = np.abs(dx_dc * dy_dr - dx_dr * dy_dc)

    # Masque
    valid_mask = (vis > 0) & (imgx_float >= 0) & (imgx_float < realsizex) & (imgy_float >= 0) & (imgy_float < realsizey)
    pixel_count = np.nan_to_num(pixel_count) * valid_mask

    # 5. SAUVEGARDE DU FICHIER TIF (Données brutes)
    os.makedirs(os.path.dirname(os.path.abspath(output_tif)), exist_ok=True)
    writeTIFF_Density(output_tif, geotransform, geoproj, pixel_count[::-1, ...].astype(np.float32))
    print(f"Fichier TIF brut sauvegardé sous : {output_tif}")

    # 6. CRÉATION DE L'APERÇU VISUEL (Heatmap en couleurs)
    print("Génération de l'aperçu couleur...")
    valid_pixels = pixel_count[pixel_count > 0]
    
    if len(valid_pixels) > 0:
        # On utilise le 98ème percentile pour éviter qu'une valeur aberrante n'écrase tout le contraste
        vmax = np.percentile(valid_pixels, 98)
        
        # Normalisation des valeurs entre 0 et 255
        pixel_count_norm = np.clip(pixel_count, 0, vmax)
        pixel_count_norm = (pixel_count_norm / vmax * 255).astype(np.uint8)
        
        # Application de la palette de couleurs (JET va du bleu au rouge)
        heatmap = cv2.applyColorMap(pixel_count_norm, cv2.COLORMAP_JET)
        
        # Remettre les zones invisibles/hors image (0) en vrai Noir
        # (Sinon la palette JET les mettrait en bleu marine)
        heatmap[pixel_count == 0] = [0, 0, 0]
        
        # On redresse l'image pour l'enregistrement (comme pour le TIF)
        heatmap = heatmap[::-1, ...]
        
        # Sauvegarde au format PNG
        preview_path = os.path.splitext(output_tif)[0] + "_preview.png"
        cv2.imwrite(preview_path, heatmap)
        print(f"Aperçu couleur sauvegardé avec succès sous : {preview_path}")
    else:
        print("Aucun pixel valide trouvé pour générer l'aperçu visuel.")


# Exemple d'utilisation :
generate_density_map("./settings/georefparam.yml", "./output/decoup/06-14-centre.jpg", "densitepx.tif")