#!/bin/bash
#SBATCH -J eval
#SBATCH -p ampere -q ampere --gres gpu --mem 24g -c 8

source /usr2/share/gpu.sbatch

python eval.py \
  --name "human"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 100

python eval.py \
  --name "05-12-245_matches_adam"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "05-12-245_combined_matches_adam"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "paper_true"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "final500"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "final500combined"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "final1000"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75
  
python eval.py \
  --name "final1000combined"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "final5000"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "final5000combined"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 75

python eval.py \
  --name "final10000"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 70

python eval.py \
  --name "final10000combined"\
  --test_csv_path "test_data.csv" \
  --root_dir "/db/psyzh/syngenta/control_comparisons/syngenta/Balanced/phytotox/" \
  --out_path "./results/" \
  --checkpoint 100