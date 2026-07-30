import argparse
import torch
import yaml
import subprocess
import sys
from build.decoup_batch import decoupage
from build.photogeo_batch import georef_batch
from build.recalage import recalage_image_sift
from build.snowsegmentation import hrnetpredict
from build.processtotal import all_process

class Parameters:
    """
    Définit les paramètres par défaut du programme
    """

    def __init__(self, args=None):
        
        # Paramètres du device, pour la segmentation, personnalisé mac 
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_workers = 4
        self.batch_size = 8

        # Paramètres pour le géoréférencement
        self.paramgeoref = args.paramgeoref if args and hasattr(args, "paramgeoref") else "./settings/georefparam.yml"

        # Chemins par défaut
        self.input = "./input_images/"
        self.decoup = "./output/decoup/"
        self.recal = "./output/recal/"
        self.georef = "./output/georef/"
        self.segment = "./output/segmentation/"

        # Chemin du modèle de segmentation
        self.segmodel = "./segmentmodels/best_snow_hrnetv2.pth"


def get_parser():
    """
    Définit les arguments entrés par l'utilisateur
    """

    parser = argparse.ArgumentParser(
        prog="GeoSnow",
        description="Géoréférence et segmente la " \
        "neige sur une ou plusieurs photos obliques de montagnes"
    )

    parser.add_argument("-a", "--all",
                        help="Fait tout le processus (recalage-géoréférencement-segmentation). Rajouter -d si photos panoramiques.",
                        action="store_true")

    parser.add_argument("-d", "--decoup", nargs="*",
                        help="Découpe et projette rectilignement les images panoramiques.")
    
    parser.add_argument("-u","--unique",
                        help="Indique si une seule photo est traitée",
                        action="store_true")
    
    parser.add_argument("-r", "--recalage", nargs="*",
                        help="Recale les images par rapport à une image de référence, définie dans le fichier .yaml.")
    
    parser.add_argument("-s", "--segmentation", nargs="*",
                        help="Segmente la/les photo(s) du dossier ./images/.")
    
    parser.add_argument("-g", "--georef", nargs="*",
                        help="Géoréférence la/les photo(s) du dossier ./images/ à partir des paramètres du fichier .yaml.")

    parser.add_argument("-ig", "--interfacegeoref",
                        help="Ouvre l'interface de géoréférencement pour ajuster les paramètres du fichier yaml pour le géoréférencement.",
                        action="store_true")

    parser.add_argument("-id", "--interfacedecoup",
                        help="Ouvre l'interface de découpage pour ajuster les paramètres du fichier yaml pour le découpage.",
                        action="store_true")

    parser.add_argument("-i", "--intermediaire",
                        help="Ajouter cette option pour que chaque étape du processus soit conservée dans les dossiers dédiés.",
                        action="store_true")
        
    return parser

def start_interface_georef(yaml_file):
    """
    Lance l'interface de photogeoref pour ajuster les paramètres du fichier yaml_file en entrée
    """
    print("\n--- Ouverture de l'interface de géoréférencement ---")
    subprocess.run([sys.executable, "./build/interface_georef.py", "-s", yaml_file])
    print("--- Calibration terminée ---\n")

def start_interface_decoup(yaml_file):
    """
    Lance l'interface de découpage pour ajuster les paramètres du fichier yaml_file en entrée
    """
    print("\n--- Ouverture de l'interface de découpage ---")
    subprocess.run([sys.executable, "./build/interface_decoup.py", "-s", yaml_file])
    print("--- Calibration terminée ---\n")

def main():
    parser = get_parser()
    args = parser.parse_args()
    params = Parameters(args)
    images = params.input
    paramsyaml = params.paramgeoref
    decoup_folder = params.decoup
    seg_folder = params.segment
    recal_folder = params.recal
    georef_folder = params.georef
    seg_model = params.segmodel
    device = params.device

    # booléens pour les tests ensuite
    flag_decoup = args.decoup is not None
    flag_recalage = args.recalage is not None
    flag_segmentation = args.segmentation is not None
    flag_georef = args.georef is not None

    # pour obtenir l'image de référence
    with open(params.paramgeoref, "r") as file:
        georefconfig = yaml.safe_load(file)
    master_image = georefconfig["masterimage"]


    if args.all:

        # Configurations impossibles
        if flag_recalage or flag_segmentation or flag_georef or args.interfacegeoref:
            parser.error("L'argument -a ne peut pas être combiné avec les arguments -r, -s, -g, -ig.")

        else:
            all_process(images, georef_folder, paramsyaml,
                        seg_model, device=device,
                        do_recalage=not args.unique,
                        do_decoupage=flag_decoup, 
                        keep_intermediate=args.intermediaire,
                        recal_dir=recal_folder,
                        decoup_dir=decoup_folder,
                        seg_dir=seg_folder)

    else:

        # fonction pour récupérer les dossiers d'entrée et de sortie
        def get_paths(arg_list, default_in, default_out, flag_name):
            if len(arg_list) == 0:
                return default_in, default_out
            elif len(arg_list) == 2:
                return arg_list[0], arg_list[1]
            else:
                parser.error(f"L'argument {flag_name} nécessite soit 0 soit 2 chemins (ex: {flag_name} dossier_in/ dossier_out/)")

        if flag_decoup:
            in_folder, out_folder = get_paths(args.decoup, images, decoup_folder, "-d")
            decoupage(in_folder, out_folder, paramsyaml)

        if flag_recalage:
            in_folder, out_folder = get_paths(args.recalage, images, recal_folder, "-r")
            recalage_image_sift(master_image, in_folder, out_folder)

        if flag_segmentation:
            in_folder, out_folder = get_paths(args.segmentation, images, seg_folder, "-s")
            hrnetpredict(in_folder, out_folder, seg_model, device=device)

        if flag_georef:
            in_folder, out_folder = get_paths(args.georef, images, georef_folder, "-g")       
            georef_batch(paramsyaml, in_folder, out_folder)

        if args.interfacegeoref:
            start_interface_georef(paramsyaml)

        if args.interfacedecoup:
            start_interface_decoup(paramsyaml)


if __name__=="__main__":
    main()
