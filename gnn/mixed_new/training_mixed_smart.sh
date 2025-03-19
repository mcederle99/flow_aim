#!/bin/bash

# Script to reproduce results
for ((s=1;s<8;s+=6))
do
  CUDA_VISIBLE_DEVICES=2 python main_fair.py \
  --seed $s \
  --save_model \
  --file_name continuous_smart \
  --nn_architecture smart
done
