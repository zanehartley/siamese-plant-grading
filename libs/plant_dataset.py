import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as transforms
import random

class PlantDataset(Dataset):
    def __init__(self, csv_file, root_dir, test=False):
        self.data = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.test = test

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        img1_path = self.root_dir + self.data.iloc[idx, 0]
        img2_path = self.root_dir + self.data.iloc[idx, 1]

        fn1 = self.data.iloc[idx, 0]
        fn2 = self.data.iloc[idx, 1]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Apply the transformations to the images
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
            return img1, img2, label1, label2, angles, fn1, fn2
        else:
            label = (label1+label2)/2
            label = torch.tensor(label)
            return img1, img2, label