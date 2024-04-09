import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from siamese import SiameseNetwork
from libs.plant_dataset import PlantDataset
from libs.masked_plant_dataset import MaskedPlantDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name',type=str,help="Name of this experiment.",required=True)
    parser.add_argument('--train_csv_file',type=str,help="Name of csv file containing training dataset.",required=True)
    parser.add_argument('--val_csv_file',type=str,help="Name of csv file containing validation dataset.",required=True)
    parser.add_argument('--root_dir',type=str,help="Path to train image and csv files.",required=True)
    parser.add_argument('-o','--out_path',type=str,help="Path for outputting model weights and tensorboard summary.",required=True)
    parser.add_argument('-b','--backbone',type=str,help="Network backbone from torchvision.models to be used in the siamese network.",default="resnet18")
    parser.add_argument('-lr','--learning_rate',type=float,help="Learning Rate",default=1e-4)
    parser.add_argument('-bs','--batch_size',type=int,help="Batch size",default=12)
    parser.add_argument('-e','--epochs',type=int,help="Number of epochs to train",default=120)
    parser.add_argument('-op', '--optimizer',type=str,help="Optimizer to use.",default="SGD")
    parser.add_argument('-s','--save_after',type=int,help="Model checkpoint is saved after each specified number of epochs.",default=5)

    args = parser.parse_args()

    print(f"Starting experiment: {args.name}\n")

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, args.name), exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Creating Datasets\n")

    print(args.train_csv_file)

    train_dataset = PlantDataset(args.train_csv_file, args.root_dir)
    val_dataset = PlantDataset(args.val_csv_file, args.root_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True)

    print(f"Train Dataset Size: {len(train_dataset)}\n")
    print(f"Validation Dataset Size: {len(val_dataset)}\n")
    print("Datasets Created\n\n")

    print("Creating Model\n")

    model = SiameseNetwork(backbone=args.backbone)
    model.to(device)

    print("Model Created\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()

    writer = SummaryWriter(os.path.join(args.out_path, args.name, "summary"))

    print("Starting Training\n")

    best_val_loss = 10000.0
    best_checkpoint = None
    loss_history = []
    val_history = []
    
    for epoch in range(args.epochs):
        losses = []
        val_losses = []

        #Training Phase
        model.train()
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for i, data in enumerate(pbar, 0):
            img1, img2, labels = data
            img1, img2, labels = map(lambda x: x.to(device), [img1, img2, labels])

            optimizer.zero_grad()

            output = model(img1, img2)

            loss = criterion(output, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            writer.add_scalar('train_loss', sum(losses)/len(losses), epoch)
            pbar.set_postfix({'output': output.tolist()[0][0], 'label': labels.tolist()[0], 'total loss': sum(losses)/len(losses)})
            

            if i % 100 == 0:
                vutils.save_image(img1[0], os.path.join(args.out_path, args.name, f"image_{i}.png"))


        loss_history.append(sum(losses)/len(losses))

        #Validation Phase
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')
            for i, data in enumerate(val_pbar, 0):
                img1, img2, labels = data
                img1, img2, labels = map(lambda x: x.to(device), [img1, img2, labels])


                output = model(img1, img2)
                loss = criterion(output, labels.float().unsqueeze(1))
                val_losses.append(loss.item())
                val_loss = sum(val_losses)/len(val_losses)
                writer.add_scalar('val_loss', val_loss, epoch)
                val_pbar.set_postfix({'validation loss': val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    "backbone": args.backbone,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss
                }
                torch.save(best_checkpoint, os.path.join(args.out_path, args.name, "best_model.pth"))
                print(f"Saving best at epoch {epoch}\n")

        val_history.append(sum(val_losses)/len(val_losses))

        if (epoch + 1) % args.save_after == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "optimizer_state_dict": optimizer.state_dict()
                },
                os.path.join(args.out_path, args.name, "epoch_{}.pth".format(epoch + 1))
            )
            print(f"Saving Epoch {epoch}\n")
    
        # Plot the training loss
        plt.plot(loss_history, label='train loss', color="blue")
        plt.plot(val_history, label='val loss', color="orange")
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.savefig(os.path.join(args.out_path, args.name, 'loss_plot_epoch'+str(epoch)+'.png'))
        plt.close()


    print("Concluded Training\n")