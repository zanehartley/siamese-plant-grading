#!/bin/bash
#SBATCH -J true
#SBATCH -q ampere -p ampere --gres gpu --mem 32g -c 18

module load nvidia/cuda-11.7

source /usr2/share/gpu.sbatch

python train.py \
  --name "old_train_pairs_245_matches"\
  --train_csv_file "old_train_pairs_245_matches.csv" \
  --val_csv_file "validation_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/pseudo/phytotox/" \
  --out_path "./out" \
  --backbone "resnet50"