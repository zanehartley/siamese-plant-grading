import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random

class MaskedPlantDataset(Dataset):
    def __init__(self, csv_file, root_dir, test=False):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.test = test

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        img1_path = self.root_dir + self.data.iloc[idx, 0]
        mask1_path = img1_path.replace("phytotox", "masks_all").replace("RGB", "mask")
        img2_path = self.root_dir + self.data.iloc[idx, 1]
        mask2_path = img2_path.replace("phytotox", "masks_all").replace("RGB", "mask")
        img1 = Image.open(img1_path).convert('RGB')
        mask1 = Image.open(mask1_path).convert('L').point(lambda x: 255 if x > 128 else 0)
        img2 = Image.open(img2_path).convert('RGB')
        mask2 = Image.open(mask2_path).convert('L').point(lambda x: 255 if x > 128 else 0)
        
        # Apply the transformations to the images     
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)
        mask1 = TF.to_tensor(mask1) / 255
        mask2 = TF.to_tensor(mask2) / 255

        #if (img1.shape == mask1.shape):
        #    return

        img1 *= (1-mask1)
        img2 *= (1-mask2)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        angles = []
        for filename in [img1_path, img2_path]:
            filename = filename.replace("Top_0", "Angle_0")
            angle_str = filename.split("Angle_", 1)[1].split("_", 1)[0]
            angle = int(angle_str)
            angles.append(angle)
        
        label1 = self.data.iloc[idx, 2]
        label2 = self.data.iloc[idx, 3]
        
        if self.test:
            label1 = torch.tensor(label1)
            label2 = torch.tensor(label2)
            return img1, img2, label1, label2, angles
        else:
            label = (label1+label2)/2
            label = torch.tensor(label)
            return img1, img2, label