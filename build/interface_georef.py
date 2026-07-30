#!/usr/bin/env python3
import os
import sys
import argparse
import copy
import tkinter as tk
import yaml
import numpy as np
from numpy import linalg as LA
import cv2
import urllib.request
from osgeo import gdal
import time
from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')

# read arguments and set defaults
usage = """
tkinter GUI to georeference a photograph to a digital elevation model (DEM).
Simplified version: No GCP, no image writing. Updates YAML and auto-closes.
"""
parser = argparse.ArgumentParser(description=usage)
parser.add_argument("-s", "--settings", default="./photogeoref/georefparam.yml", help="Full path to yaml settings file")
args = parser.parse_args()

yamlfile = args.settings

def load_settings():
    global elevation, vis, cv_img, cv_imgr, dirnameimg, rootbnameimg
    global nrows, ncols, x0, dx, y1, y0
    
    yamlfile = yamlfilevar.get()
    with open(yamlfile, 'r') as geoset:
        grfyml = yaml.safe_load(geoset)

    demfname = grfyml['demfname']
    visfname = grfyml['visfname']
    imgfname = grfyml['imgfname']
    
    demfnametk.set(demfname)
    visfnametk.set(visfname)
    imgfnametk.set(imgfname)
    
    obscoords = np.array(grfyml['obscoords'])
    obscoordsXtk.set(obscoords[0])
    obscoordsYtk.set(obscoords[1])
    obscoordsZtk.set(obscoords[2])
    
    tgtcoords = np.array(grfyml['tgtcoords'])
    tgtcoordsXtk.set(tgtcoords[0])
    tgtcoordsYtk.set(tgtcoords[1])
    tgtcoordsZtk.set(tgtcoords[2])
    
    fwidthtk.set(grfyml['fwidth'])
    fheighttk.set(grfyml['fheight'])
    focallengthtk.set(grfyml['focallength'])
    rolldegtk.set(grfyml['roll'])
    
    dsdem = gdal.Open(demfname)
    band = dsdem.GetRasterBand(1)
    elevation = band.ReadAsArray()[::-1, ...] # inversion axe y
    nrows, ncols = elevation.shape
    
    x0, dx, dxdy, y1, dydx, dy = dsdem.GetGeoTransform()
    y0 = y1 + dy * nrows

    visibility = gdal.Open(visfname)
    bandv = visibility.GetRasterBand(1)
    vis = bandv.ReadAsArray()[::-1, ...]
    
    dirnameimg = os.path.dirname(imgfname)
    bnameimg = os.path.basename(imgfname)
    rootbnameimg = os.path.splitext(bnameimg)[0]
    
    cv_img = cv2.cvtColor(cv2.imread(imgfname), cv2.COLOR_BGR2RGB)
    realsizey, realsizex = cv_img.shape[:2]
    
    canvimgX = canvas.winfo_reqwidth()
    canvimgY = round(realsizey * canvX / realsizex)
    cv_imgr = cv2.resize(cv_img, (canvimgX, canvimgY))
    
    ax.clear()
    ax.imshow(cv_imgr)
    canvasmp.draw()
    
    button_process['state'] = tk.NORMAL
    button_accept['state'] = tk.NORMAL


def get_projmatrices(obscoords, tgtcoords, upperleftx, y0, focallength, fwidth, fheight, rolldeg):
    d = focallength
    vectorc = copy.copy(obscoords)
    vectorc[0] = vectorc[0] - upperleftx
    vectorc[1] = vectorc[1] - y0
    
    vectorview = tgtcoords - obscoords
    vectorn = vectorview / LA.norm(vectorview)

    vectornp = copy.copy(vectorn)
    vectornp[2] = 0
    vectorncross = copy.copy(vectorn)
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


def process_view(display=1):
    global elevation, vis, cv_img, geotransform, geoproj
    global nrows, ncols, x0, dx, y1, y0
    global imgx, imgy
    
    button_accept['state'] = tk.DISABLED
    
    obscoordsX  = obscoordsXtk.get()
    obscoordsY  = obscoordsYtk.get()
    obscoordsZ  = obscoordsZtk.get()
    tgtcoordsX  = tgtcoordsXtk.get()
    tgtcoordsY  = tgtcoordsYtk.get()
    tgtcoordsZ  = tgtcoordsZtk.get()
    fwidth      = fwidthtk.get()
    fheight     = fheighttk.get()
    focallength = focallengthtk.get()
    rolldeg     = rolldegtk.get()
    
    coordsc = np.zeros([nrows, ncols, 4])
    upperleftx = x0
    dl = dx
    d = focallength
    w = fwidth/2.0
    
    realsizey, realsizex = cv_img.shape[:2]
    resolution = realsizex/fwidth/100.0
    scale = 100.0 * w * resolution
    
    widthheight = canvasmp.get_width_height()
    width = widthheight[0]
    sclx = (2./3.) * width / realsizex
    
    obscoords = np.array([obscoordsX, obscoordsY, obscoordsZ])
    tgtcoords = np.array([tgtcoordsX, tgtcoordsY, tgtcoordsZ])
    ttrans, tview = get_projmatrices(obscoords, tgtcoords, upperleftx, y0, focallength, fwidth, fheight, rolldeg)
    
    onenrows = np.ones(nrows)
    seqnrows = np.r_[0:nrows]
    seqncols = np.r_[0:ncols]
    onencols = np.ones(ncols)
    
    layerx = np.outer(onenrows, seqncols) * dl + ttrans[0, 3]
    layery = np.outer(seqnrows, onencols) * dl + ttrans[1, 3]
    layerz = elevation + ttrans[2, 3]
    
    vectorview = tgtcoords - obscoords
    visproj = layerx*vectorview[0] + layery*vectorview[1] + layerz*vectorview[2]
    vis[visproj < 0] = 0
    
    layerx = layerx * vis
    layery = layery * vis
    layerz = layerz * vis 
    
    R = 6371000  # Rayon terrestre
    disth = np.sqrt(layerx**2 + layery**2)
    drop = disth**2/(2*R)
    layerz = layerz - drop
    layerw = np.ones([nrows, ncols])
    
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
    imgx = np.clip(imgx, 0, realsizex-1)
    imgy = np.clip(imgy, 0, realsizey-1)
    
    imgx = imgx * vis
    imgy = imgy * vis
    nullx = np.where(imgx <= 0)
    nully = np.where(imgy <= 0)
    imgx[nully] = 0
    imgy[nullx] = 0
    imgx = imgx.astype(int)
    imgy = imgy.astype(int)
    
    # Dessin des points bleus
    coimg = copy.copy(cv_img)[::-1, ...]
    epaisseur = 5
    demi_ep = epaisseur // 2
    for dx_ in range(-demi_ep, demi_ep + 1):
        for dy_ in range(-demi_ep, demi_ep + 1):
            y_coords = np.clip(imgy + dy_, 0, realsizey - 1)
            x_coords = np.clip(imgx + dx_, 0, realsizex - 1)
            coimg[y_coords, x_coords] = [0, 0, 255]

    coimg = coimg[::-1, ...]

    if display:
        ax.axis("on")
        ax.clear()
        # Affichage direct de l'image avec les points bleus
        ax.imshow(cv2.resize(coimg, (0,0), fx=sclx, fy=sclx))
        canvasmp.draw()
        button_accept['state'] = tk.NORMAL


def write_settings():
    outyaml = yamlfilevar.get()
    print('Sauvegarde et mise à jour des paramètres dans le fichier :', outyaml)
    
    # 1. on charge les paramètres.
    try:
        with open(outyaml, 'r') as infile:
            datayml = yaml.safe_load(infile) or {}
    except FileNotFoundError:
        datayml = {} # sécurité s le ficier n'existe pas encore
    
    # 2. on met à jour les différents paramètres de géoréférencement.
    datayml['demfname'] = demfnametk.get()
    datayml['visfname'] = visfnametk.get()
    datayml['imgfname'] = imgfnametk.get()
        
    datayml['obscoords'] = [obscoordsXtk.get(), obscoordsYtk.get(), obscoordsZtk.get()]
    datayml['tgtcoords'] = [tgtcoordsXtk.get(), tgtcoordsYtk.get(), tgtcoordsZtk.get()]
    datayml['fwidth'] = fwidthtk.get()
    datayml['fheight'] = fheighttk.get()
    datayml['focallength'] = focallengthtk.get()
    datayml['roll'] = rolldegtk.get()        
    
    # 3. réécrire le fichier avec le dictionnaire
    with open(outyaml, 'w') as outfilesett:
        yaml.dump(datayml, outfilesett, default_flow_style=False, sort_keys=False)


def process_all():
    global nrows, ncols, elevation, cv_img, imgx, imgy
    
    button_accept['state'] = tk.DISABLED
    button_process['state'] = tk.DISABLED
    
    try:
        imgx
    except NameError:
        process_view(display=0)
    
    write_settings()
    
    # Préparation de l'image pour les contours oranges
    cv_img_temp = cv_img[::-1, ...]
    realsizey, realsizex = cv_img_temp.shape[:2]
    
    cv_img_temp[0,:,:] = 0
    cv_img_temp[:,0,:] = 0
    cv_img_temp[:,realsizex-1,:] = 0
    cv_img_temp[realsizey-1,:,:] = 0
    cv_img_temp[:,realsizex-2,:] = 0
    
    zalbedo = cv_img_temp[imgy, imgx, :]
    
    ax.axis("on")
    ax.clear()
    ax.imshow(zalbedo, cmap='gray', origin='lower')
    
    # Ajout des contours oranges
    levels = np.arange(np.floor(np.amin(elevation)/100)*100, np.ceil(np.amax(elevation)/100)*100, 100)
    ax.contour(elevation, colors='#A0522D', levels=levels)
    levels = np.arange(np.floor(np.amin(elevation)/100)*100, np.ceil(np.amax(elevation)/100)*100, 20)
    ax.contour(elevation, colors='#A0522D', levels=levels, linewidths=0.75)
    
    canvasmp.draw()
    print("Mise à jour terminée. Fermeture automatique dans 2.5 secondes...")
    
    # Attend 2500 millisecondes (2.5s) puis ferme la fenêtre proprement
    window.after(2500, quitall)


def quitall():
    window.destroy()
    sys.exit(0)


# GUI SETUP
window = tk.Tk()
window.title("Photogeoref (Simplified)")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
dpi = window.winfo_fpixels('1i')

canvX = screen_width * 0.6
canvY = screen_height * 0.6
figXinches = canvX / dpi
figYinches = canvY / dpi
canvas = tk.Canvas(window, width=canvX, height=canvY, background='white')
canvas.grid(row=0, column=0)

frame = tk.Frame(window)
frame.grid(row=0, column=1, sticky="n")

yamlfilevar = tk.StringVar()
demfnametk = tk.StringVar()
visfnametk = tk.StringVar()
imgfnametk = tk.StringVar()
obscoordsXtk = tk.DoubleVar()
obscoordsYtk = tk.DoubleVar()
obscoordsZtk = tk.DoubleVar()
tgtcoordsXtk = tk.DoubleVar()
tgtcoordsYtk = tk.DoubleVar()
tgtcoordsZtk = tk.DoubleVar()
fwidthtk = tk.DoubleVar()
fheighttk = tk.DoubleVar()
focallengthtk = tk.DoubleVar()
rolldegtk = tk.DoubleVar()

# Labels
tk.Label(frame, text="Georeferencing Oblique Photography").grid(row=0, column=0, columnspan=2, sticky="nw", padx=5, pady=4)
tk.Label(frame, text="YAML settings:").grid(row=1, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="DEM full path:").grid(row=3, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="Viewshed full path:").grid(row=4, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="Image full path:").grid(row=5, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="Observer Coordinates:").grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=4)
tk.Label(frame, text="X").grid(row=7, column=0, sticky="e", padx=5, pady=4)
tk.Label(frame, text="Y").grid(row=8, column=0, sticky="e", padx=5, pady=4)
tk.Label(frame, text="Z").grid(row=9, column=0, sticky="e", padx=5, pady=4)
tk.Label(frame, text="Target Coordinates:").grid(row=10, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="X").grid(row=11, column=0, sticky="e", padx=5, pady=4)
tk.Label(frame, text="Y").grid(row=12, column=0, sticky="e", padx=5, pady=4)
tk.Label(frame, text="Z").grid(row=13, column=0, sticky="e", padx=5, pady=4)
tk.Label(frame, text="Sensor width [m]:").grid(row=14, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="Sensor height [m]:").grid(row=15, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="Focal length [m]:").grid(row=16, column=0, sticky="w", padx=5, pady=4)
tk.Label(frame, text="roll [\u00B0]:").grid(row=17, column=0, sticky="w", padx=5, pady=4)

# Entries
tk.Entry(frame, textvariable=yamlfilevar, width=50).grid(row=1, column=1, sticky='w', padx=5, pady=4)
yamlfilevar.set(yamlfile)
tk.Button(frame, text="Load Settings", command=load_settings).grid(row=2, column=1, sticky="we", padx=5, pady=4)

tk.Entry(frame, textvariable=demfnametk, width=50, bg="lightgrey").grid(row=3, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=visfnametk, width=50, bg="lightgrey").grid(row=4, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=imgfnametk, width=50, bg="lightgrey").grid(row=5, column=1, sticky='w', padx=5, pady=4)

tk.Entry(frame, textvariable=obscoordsXtk, width=50).grid(row=7, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=obscoordsYtk, width=50).grid(row=8, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=obscoordsZtk, width=50).grid(row=9, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=tgtcoordsXtk, width=50).grid(row=11, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=tgtcoordsYtk, width=50).grid(row=12, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=tgtcoordsZtk, width=50).grid(row=13, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=fwidthtk, width=50).grid(row=14, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=fheighttk, width=50).grid(row=15, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=focallengthtk, width=50).grid(row=16, column=1, sticky='w', padx=5, pady=4)
tk.Entry(frame, textvariable=rolldegtk, width=50).grid(row=17, column=1, sticky='w', padx=5, pady=4)

# Figure Setup
fig = Figure(figsize=(figXinches, figYinches))
ax = fig.add_subplot()
ax.axis("off")
logofname = "https://meteoexploration.com/static/assets/img/meteoexplorationtr.jpg"
try:
    req = urllib.request.urlopen(logofname)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    cv_logobgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    cv_logo = cv2.cvtColor(cv_logobgr, cv2.COLOR_BGR2RGB)
    ax.imshow(cv_logo)
except:
    pass

canvasmp = FigureCanvasTkAgg(fig, master=canvas)
canvasmp.draw()
toolbar = NavigationToolbar2Tk(canvasmp, window, pack_toolbar=False)
toolbar.update()

canvasmp.mpl_connect("key_press_event", key_press_handler)

# Action Buttons
button_process = tk.Button(frame, text="Process Projection", command=process_view, state=tk.DISABLED)
button_process.grid(row=18, column=1, sticky="we", padx=5, pady=4)

button_accept = tk.Button(frame, text="Accept & Save", command=process_all, state=tk.DISABLED)
button_accept.grid(row=19, column=1, sticky="we", padx=5, pady=4)

tk.Button(frame, text="Quit", command=quitall).grid(row=20, column=1, sticky="we", padx=5, pady=4)

toolbar.grid(row=1, column=0, padx=5, pady=4)
canvasmp.get_tk_widget().grid(row=0, column=0, sticky="nw", padx=5, pady=4)

window.mainloop()