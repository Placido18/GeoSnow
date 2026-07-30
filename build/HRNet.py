import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

cv2.setNumThreads(0)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

# Configuration des paramètres
DATASET_ROOT = "/Users/placideneuilly/Desktop/stage-neige/snowseg/dataset"

TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train", "images")
TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, "train", "masks")
VAL_IMG_DIR = os.path.join(DATASET_ROOT, "val", "images")
VAL_MASK_DIR = os.path.join(DATASET_ROOT, "val", "masks")

BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 0.01
EPOCHS = 100
LAMBDA_WEIGHT = 0.3  
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")



# Définition du dataset
class SnowDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith(valid_extensions)])
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(valid_extensions)]
        self.mask_dict = { os.path.splitext(f)[0]: f for f in mask_files }


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
            img_name = self.img_names[idx]
            img_path = os.path.join(self.img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            mask_name = self.mask_dict.get(base_name + "_mask")
            
            # sécurité, on regarde que le masque existe bien
            if mask_name is None:
                mask_path = ""
            else:
                mask_path = os.path.join(self.mask_dir, mask_name)
            
            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # sécurité, si prioblème sur un patch
            if image is None or mask is None:
                print(f"\nERREUR LECTURE : {img_name} ou son masque est corrompu/introuvable.")
                print("Remplacement par une autre image aléatoire pour sauver l'époque...")
                import random
                # on prend une autre image aléatoirement
                return self.__getitem__(random.randint(0, len(self.img_names) - 1))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = (mask > 127).astype(np.float32)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            mask = mask.unsqueeze(0)
            return image, mask

# data augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Architecture HRNet, avec deepsupervision, comme expliqué dans le rapport
class HRNetWithDeepSupervision(nn.Module):
    def __init__(self, model_name='hrnet_w18', num_classes=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, features_only=True)
        encoder_channels = self.backbone.feature_info.channels()
        
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(encoder_channels[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, 128, kernel_size=1) for ch in encoder_channels
        ])
        total_channels = 128 * len(encoder_channels)
        self.main_classifier = nn.Sequential(
            nn.Conv2d(total_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_shape = x.shape[2:] 
        features = self.backbone(x)
        
        aux_out = self.aux_classifier(features[0])
        aux_out = nn.functional.interpolate(aux_out, size=input_shape, mode='nearest')
        
        target_size = features[0].shape[2:] 
        aligned_features = []
        for i, f in enumerate(features):
            proj = self.convs[i](f)
            if proj.shape[2:] != target_size:
                proj = nn.functional.interpolate(proj, size=target_size, mode='nearest')
            aligned_features.append(proj)
            
        x_fused = torch.cat(aligned_features, dim=1)
        main_out = self.main_classifier(x_fused)
        main_out = nn.functional.interpolate(main_out, size=input_shape, mode='nearest')
        
        if self.training:
            return {"main": main_out, "aux": aux_out}
        else:
            return main_out

# métrique IoU
def compute_iou(preds, masks, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


# processur d'entraînement, pour MacOS
if __name__ == '__main__':
    print("Démarrage du script principal pour M4")

    train_dataset = SnowDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_transform)
    val_dataset = SnowDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True, 
        pin_memory=False,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=False,
        num_workers=4
    )

    model = HRNetWithDeepSupervision(model_name='hrnet_w18', num_classes=1).to(DEVICE)

    model.load_state_dict(torch.load("best_snow_hrnet.pth"))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=5)

    best_val_iou = 0.0

    for epoch in range(EPOCHS):
        # phase d'entraînement
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{EPOCHS} [Train]")
        
        for images, masks in train_pbar:
            # non_blocking=True pour optimiser
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            
            # set_to_none=True est plus rapide pour libérer la mémoire des gradients
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(images)
            loss_main = criterion(outputs["main"], masks)
            loss_aux = criterion(outputs["aux"], masks)
            loss_total = (1.0 - LAMBDA_WEIGHT) * loss_main + LAMBDA_WEIGHT * loss_aux
            
            loss_total.backward()
            optimizer.step()
            
            train_loss += loss_total.item()
            train_pbar.set_postfix({"Loss": f"{(train_loss / (train_pbar.n + 1)):.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # phase de validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Époque {epoch+1}/{EPOCHS} [Val]")
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                
                main_out = model(images)
                loss = criterion(main_out, masks)
                
                val_loss += loss.item()
                val_iou += compute_iou(main_out, masks) 
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Bilan Époque {epoch+1} | LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

        scheduler.step(avg_val_iou)

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), "best_snow_hrnetv2.pth")
            print(f" => Nouveau record ! Modèle sauvegardé (IoU: {best_val_iou:.4f})")