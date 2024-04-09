import os
import argparse

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from siamese import SiameseNetwork
from libs.plant_dataset import PlantDataset
import warnings


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--name',type=str,help="Name of this experiment.",required=True)
    parser.add_argument('-t','--test_csv_path',type=str,help="Path to directory containing validation dataset.",required=True)
    parser.add_argument('-o','--out_path',type=str,help="Path for saving prediction images.",required=True)
    parser.add_argument('-c','--checkpoint',type=int,help="Path of model checkpoint to be used for inference.",required=True)
    parser.add_argument('-r', '--root_dir',type=str,help="Path to train image and csv files.",required=True)

    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, args.name), exist_ok=True)

    # Set device to CUDA if a CUDA device is available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = PlantDataset(args.test_csv_path, args.root_dir, test=True)
    test_dataloader   = DataLoader(test_dataset, batch_size=1)

    print(args.name)

    criterion = torch.nn.MSELoss()

    #checkpoint = torch.load( "./out/{}/best_model.pth".format(args.name, args.checkpoint))
    checkpoint = torch.load( "./out/{}/epoch_{}.pth".format(args.name, args.checkpoint))
    model = SiameseNetwork(backbone=checkpoint['backbone'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    losses = []
    correct = 0
    total = 0

    inv_transform = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                         std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                         std = [ 1., 1., 1. ]),
                                   ])

    grades = []
    preds = []
    errors = []
    angles1 = []
    angles2 = []

    header = ["File1","File2", "gt", "pred"]
    rows = []

    pbar = tqdm(test_dataloader, desc='Testing')
    for i, data in enumerate(pbar, 0):

        img1, img2, label1, label2, angles, fn1, fn2 = data
        img1, img2, label1, label2 = map(lambda x: x.to(device), [img1, img2, label1, label2])
        
        grade1 = label1[0]
        grade2 = label2[0]

        prob = model(img1, img2)
        prob = prob.item()
        
        grade1 = grade1.item()
        grade2 = grade2.item()
        angle1 = angles[0]
        angle2 = angles[1]

        diff = abs(prob-(grade1 + grade2)/2)

        grades.append(abs(grade1+grade2)/2)
        preds.append(prob)
        errors.append(diff)
        angles1.append(angle1)
        angles2.append(angle2)
        
        total += len(errors)
        avg_error = sum(errors) / len(errors)
        mse = sum([(error ** 2) for error in errors]) / len(errors)

        pbar.set_postfix({"Avg error: ": avg_error})

        row = [fn1, fn2, (grade1 + grade2)/2, prob]
        rows.append(row)

    print(f"{args.name} has Final mean average error: {avg_error:.2f}")
    print(f"{args.name} has Final mean squared error: {mse:.2f}")

    score_ranges = [(0, 19), (20, 39), (40, 59), (60, 79), (80, 100)]
    score_counts = [0] * len(score_ranges)
    mae_scores = [0] * len(score_ranges)
    mse_scores = [0] * len(score_ranges)

    for grade, error in zip(grades, errors):
        for i, (start, end) in enumerate(score_ranges):
            if start <= grade <= end:
                score_counts[i] += 1
                mae_scores[i] += abs(error)
                mse_scores[i] += error ** 2
                break

    for i, (start, end) in enumerate(score_ranges):
        print(f"Score Range: {start}-{end}")
        print(f"MAE: {mae_scores[i] / score_counts[i]:.2f}")
        print(f"MSE: {mse_scores[i] / score_counts[i]:.2f}")
        print("----------")
    
    # Create a histogram
    plt.hist(errors, bins=100)
    plt.xlabel('Number')
    plt.ylabel('Frequency')
    plt.title('Histogram of Errors')
    plt.savefig('{}{}/e{}_histogram.png'.format(args.out_path, args.name, args.checkpoint))

    plt.clf()

    # Create a scatter plot
    plt.scatter(grades, preds)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.savefig('{}{}/e{}_true_vs_predicted.png'.format(args.out_path, args.name, args.checkpoint))

    plt.clf()

    # Create a scatter plot
    #angle_diff = [abs(a + b) for a, b in zip(angles1, angles2)]

    #plt.scatter(angle_diff, errors)
    #plt.xlabel('Angle Difference')
    #plt.ylabel('Error')
    #plt.title('Difference vs Errors')
    #plt.savefig('{}{}/e{}_diff_vs_error.png'.format(args.out_path, args.name, args.checkpoint))

    #print("Validation: Loss={:.2f}\t Accuracy={:.2f}\t".format(sum(losses)/len(losses), correct / total))

    # Save the rows to a CSV file
    with open('{}{}/e{}_results.csv'.format(args.out_path, args.name, args.checkpoint), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(header)
        
        # Write each row of data
        for row in rows:
            writer.writerow(row)
            #print(row)