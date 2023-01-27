#!/bin/bash

for data in "bert_english"; do
python train_linear_weight.py -d ${data} | tee models/train_${data}.log 2>&1
done
python apply_linear_weight.py -d bert_english -e bert_greek | tee models/apply_bert_english_to_greek.log 2>&1

# run:
# sbatch -p cpu -c 10 --mem=16G -t 1-00:00:00 run.sh