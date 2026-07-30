#!/usr/bin/env python3
import os
import glob
import yaml
import numpy as np
from numpy import linalg as LA
import cv2
from osgeo import gdal

def writeTIFFRGB(filename, geotransform, geoprojection, data):
    (x, y, z) = data.shape
    format = "GTiff"
    driver = gdal.GetDriverByName(format)
    dst_datatype = gdal.GDT_Byte
    dst_ds = driver.Create(filename, y, x, z, dst_datatype)
    for i in range(z):  
        bi = i + 1
        dst_ds.GetRasterBand(bi).WriteArray(data[:, :, i])

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

def georef_batch(yaml_file, image_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

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

    # 2. Chargement du MNT (DEM) et Viewshed
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

    # 3. Récupération des dimensions de l'image, en supposant qu'elles ont toute la même taille, en regardant la première du dossier
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    image_files = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"Aucune image trouvée dans {image_folder}.")
        return

    first_image = cv2.imread(image_files[0])
    realsizey, realsizex = first_image.shape[:2]
    print(f"Dimensions de référence trouvées : {realsizex}x{realsizey}")

    # 4. Calcul de la matrice de projection, une seule fois pour toutes les images
    print("Calcul de la matrice de projection globale...")
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

    layerx = layerx * vis
    layery = layery * vis
    layerz = layerz * vis

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

    midx = np.rint(realsizex / 2.0)
    midy = np.rint(realsizey / 2.0)
    imgx = np.rint(scale * imgx + midx)
    imgy = np.rint(scale * imgy + midy)

    imgx = np.nan_to_num(imgx)
    imgy = np.nan_to_num(imgy)
    imgx = np.clip(imgx, 0, realsizex - 1)
    imgy = np.clip(imgy, 0, realsizey - 1)

    imgx = imgx * vis
    imgy = imgy * vis
    nullx = np.where(imgx <= 0)
    nully = np.where(imgy <= 0)
    imgx[nully] = 0
    imgy[nullx] = 0
    imgx = imgx.astype(int)
    imgy = imgy.astype(int)

    print("Matrice calculée avec succès. Début du géoréférencement des images...")

    # 5. On parcourt toutes les images du dossier
    for img_path in image_files:
        print(f"Géoréférencement de : {os.path.basename(img_path)}...")

        # Ouverture et préparation de l'image (identique à process_all)
        cv_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        cv_img = cv_img[::-1, ...]

        cv_img[0, :, :] = 0
        cv_img[:, 0, :] = 0
        cv_img[:, realsizex - 1, :] = 0
        cv_img[realsizey - 1, :, :] = 0
        cv_img[:, realsizex - 2, :] = 0

        zalbedo = cv_img[imgy, imgx, :]

        # Sauvegarde du TIF
        nom_fichier = os.path.basename(img_path)
        base_name = os.path.splitext(nom_fichier)[0]

        out_tif = os.path.join(output_folder, f"{base_name}_geo.tif")
        writeTIFFRGB(out_tif, geotransform, geoproj, zalbedo[::-1, ...])

    print(f"\n Géoréférencement terminé pour les {len(image_files)} images !")

