#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import tkinter as tk
import yaml
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

parser = argparse.ArgumentParser(description="Interface de paramétrage du découpage FFmpeg")
# Les arguments ne sont plus required=True, ils ont maintenant des valeurs par défaut
parser.add_argument("-s", "--settings", default="settings/georefparam.yml", help="Chemin vers le fichier YAML")
parser.add_argument("-i", "--input", default=None, help="Dossier contenant les panoramas bruts (optionnel)")
args = parser.parse_args()

yaml_file = args.settings
input_folder = args.input
temp_preview = "temp_preview.jpg" # Fichier temporaire pour l'aperçu

# --- LECTURE DU YAML ---
try:
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"Erreur : Le fichier de configuration '{yaml_file}' est introuvable.")
    sys.exit(1)

master_image_path = config.get("masterimage")
if not master_image_path:
    print(f"Erreur : Aucune image de référence ('masterimage') trouvée dans {yaml_file}")
    sys.exit(1)

img_path = None

# --- RECHERCHE DE L'IMAGE ---
if input_folder:
    # Si on a fourni un dossier d'entrée, on essaie de retrouver le panorama brut original
    master_basename = os.path.basename(master_image_path)
    search_name = master_basename.replace("-centre", "").split('.')[0]
    valid_exts = ('.jpg', '.jpeg', '.png')
    
    try:
        for f in os.listdir(input_folder):
            if not f.startswith('.') and f.lower().endswith(valid_exts):
                if search_name in f:
                    img_path = os.path.join(input_folder, f)
                    break
    except FileNotFoundError:
        print(f"Erreur : Le dossier {input_folder} est introuvable.")
        sys.exit(1)
        
    if not img_path:
        if os.path.exists(os.path.join(input_folder, master_basename)):
            img_path = os.path.join(input_folder, master_basename)
else:
    # Si aucun dossier d'entrée n'est fourni, on utilise directement l'image du YAML
    img_path = master_image_path

# Dernière vérification de sécurité
if not img_path or not os.path.exists(img_path):
    print(f"Erreur : Impossible de trouver l'image à traiter. Chemin testé : {img_path}")
    sys.exit(1)
# --------------------------------------------------

def load_settings():
    """Charge les paramètres depuis le YAML (ou met des valeurs par défaut)"""
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)
    
    dec = config_data.get("decoupage", {})
    ih_fov_var.set(dec.get("inputhfov", 210))
    yaw_var.set(dec.get("yaw", 0))
    h_fov_var.set(dec.get("h_fov", 80))
    v_fov_var.set(dec.get("v_fov", 40))

def update_preview(event=None):
    """Génère l'aperçu avec FFmpeg et met à jour l'affichage"""
    ih = ih_fov_var.get()
    yaw = yaw_var.get()
    hf = h_fov_var.get()
    vf = v_fov_var.get()
    
    vf_string = f"v360=input=cylindrical:output=rectilinear:ih_fov={ih}:yaw={yaw}:h_fov={hf}:v_fov={vf}"
    
    cmd = [
        "ffmpeg", "-y",
        "-i", img_path,
        "-vf", vf_string,
        "-frames:v", "1",
        temp_preview
    ]
    
    # Exécute FFmpeg silencieusement
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Met à jour l'image dans l'interface
    if os.path.exists(temp_preview):
        ax.clear()
        ax.axis("off")
        img = mpimg.imread(temp_preview)
        ax.imshow(img)
        canvasmp.draw()

def save_and_quit():
    """Sauvegarde dans le YAML, nettoie et ferme"""
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)
    
    if "decoupage" not in config_data:
        config_data["decoupage"] = {}
        
    config_data["decoupage"]["inputhfov"] = ih_fov_var.get()
    config_data["decoupage"]["yaw"] = yaw_var.get()
    config_data["decoupage"]["outputhfov"] = h_fov_var.get()
    config_data["decoupage"]["outputvfov"] = v_fov_var.get()
    
    with open(yaml_file, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False)
        
    if os.path.exists(temp_preview):
        os.remove(temp_preview) # Supprime l'image temporaire
        
    print("Paramètres de découpage sauvegardés avec succès !")
    window.destroy()
    sys.exit(0)

# Lors d'une fermeture manuelle par la croix rouge
def on_closing():
    if os.path.exists(temp_preview):
        os.remove(temp_preview)
    window.destroy()
    sys.exit(0)


# --- CONFIGURATION DE LA FENÊTRE TKINTER ---
window = tk.Tk()
window.title(f"Réglage du Découpage - {os.path.basename(img_path)}")
window.protocol("WM_DELETE_WINDOW", on_closing)

frame_controls = tk.Frame(window)
frame_controls.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

ih_fov_var = tk.IntVar()
yaw_var = tk.IntVar()
h_fov_var = tk.IntVar()
v_fov_var = tk.IntVar()

load_settings()

# Création des curseurs (Sliders)
def create_slider(parent, label, variable, from_, to):
    tk.Label(parent, text=label, font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 0))
    slider = tk.Scale(parent, variable=variable, from_=from_, to=to, orient=tk.HORIZONTAL, length=250)
    slider.bind("<ButtonRelease-1>", update_preview)
    slider.pack(anchor="w")

create_slider(frame_controls, "Input H_FOV (Champ de vision initial) :", ih_fov_var, 100, 360)
create_slider(frame_controls, "Yaw (Panoramique Gauche/Droite) :", yaw_var, -180, 180)
create_slider(frame_controls, "Output H_FOV (Zoom horizontal) :", h_fov_var, 10, 180)
create_slider(frame_controls, "Output V_FOV (Zoom vertical) :", v_fov_var, 10, 180)

tk.Button(frame_controls, text="Accepter & Sauvegarder", command=save_and_quit, bg="green", fg="white", font=("Arial", 12)).pack(pady=30, fill=tk.X)

# --- CONFIGURATION DE L'AFFICHAGE MATPLOTLIB ---
frame_canvas = tk.Frame(window)
frame_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

fig = Figure(figsize=(8, 6))
ax = fig.add_subplot()
ax.axis("off")
ax.text(0.5, 0.5, "Génération de l'aperçu...", ha='center', va='center')

canvasmp = FigureCanvasTkAgg(fig, master=frame_canvas)
canvasmp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Lancer un premier aperçu au démarrage
update_preview()

window.mainloop()