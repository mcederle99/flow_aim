#!/bin/bash

# Script to reproduce results
for ((s=2;s<11;s+=1))
do
  CUDA_VISIBLE_DEVICES=2 python main.py \
  --seed $s \
  --save_model \
  --file_name continuous_base_fixom \
  --nn_architecture base
done
