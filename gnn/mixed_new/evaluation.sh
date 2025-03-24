#!/bin/bash

# Script to reproduce results
for ((s=1;s<8;s+=6))
do
  CUDA_VISIBLE_DEVICES=2 python evaluation_run.py \
  --seed $s \
  --file_name continuous_smart \
  --nn_architecture smart
done
