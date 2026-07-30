#!/usr/bin/env python3

import os
import subprocess
import yaml

def decoupage(input_rep, output_path, yaml_file):
    """Extrait avec projection rectiligne une partie de tous les panoramas du
    dossier input_rep, en appelant le logiciel ffmpeg. Renvoie toutes les photos 
    extraites dans output_path. Les paramètres du découpage sont enregistrés dans
    le fichier yaml_file.
    """

    os.makedirs(output_path, exist_ok=True) #crée output_path s'il n'existe pas

    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)

    ih_fov = config["decoupage"]["inputhfov"]
    yaw = config["decoupage"]["yaw"]
    h_fov = config["decoupage"]["outputhfov"]
    v_fov = config["decoupage"]["outputvfov"]

    vf_string = f"v360=input=cylindrical:output=rectilinear:ih_fov={ih_fov}:yaw={yaw}:h_fov={h_fov}:v_fov={v_fov}"

    for file in os.listdir(input_rep):

        file_path = os.path.join(input_rep, file)
        name_output = f"{file[0:13]}-decoup.jpg"
        output = os.path.join(output_path, name_output)

        commande = [
            "ffmpeg", "-y",
            "-i", file_path, 
            "-vf", vf_string,
            "-frames:v", "1", 
            output
        ]

        subprocess.run(commande, check=False)

