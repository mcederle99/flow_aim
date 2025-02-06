#!/bin/bash

# Script to reproduce results
for ((s=2;s<11;s+=1))
do
  python main.py \
  --seed $s \
  --save_model \
  --file_name continuous_smart_fixom \
  --nn_architecture smart
done
