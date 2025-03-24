#!/bin/bash

# Script to reproduce results
for ((s=1;s<10;s+=8))
do
  CUDA_VISIBLE_DEVICES=2 python evaluation_run.py \
  --seed $s \
  --file_name continuous_smart_fixom \
  --nn_architecture smart
done
