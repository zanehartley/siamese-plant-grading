#!/bin/bash
#SBATCH -J syngenta_test
#SBATCH --gres gpu --mem 16g -c 12

source /usr2/share/gpu.sbatch

python eval.py --name "05-15-train_245_combined_matches_adam" --test_csv_path "test_pairs_49_matches.csv" --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/pseudo/phytotox/"  --out_path "./results/"  --checkpoint 100
python eval.py --name "05-15-true_train_adam" --test_csv_path "test_pairs_49_matches.csv" --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/pseudo/phytotox/"  --out_path "./results/"  --checkpoint 100