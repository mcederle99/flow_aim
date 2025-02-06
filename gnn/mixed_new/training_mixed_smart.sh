#!/bin/bash

# Script to reproduce results
for ((s=2;s<11;s+=1))
do
  python main_fair.py \
  --seed $s \
  --save_model \
  --file_name continuous_smart \
  --nn_architecture smart
done
