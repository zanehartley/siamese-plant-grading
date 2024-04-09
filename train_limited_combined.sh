#!/bin/bash
#SBATCH -J Balanced
#SBATCH -q ampere -p ampere --gres gpu --mem 16g -c 12

source /usr2/share/gpu.sbatch

python train.py \
  --name "faked_labels_196_combined"\
  --train_csv_file "combined_train_data.csv" \
  --val_csv_file "validation_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/pseudo/phytotox/" \
  --out_path "./out"