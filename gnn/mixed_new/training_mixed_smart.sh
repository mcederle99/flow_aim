#!/bin/bash

# Script to reproduce results
for ((s=8;s<11;s+=1))
do
  CUDA_VISIBLE_DEVICES=0 python main_fair.py \
  --seed $s \
  --save_model \
  --file_name continuous_smart \
  --nn_architecture smart
done
