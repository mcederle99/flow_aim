#!/bin/bash

# Script to reproduce results
for ((s=1;s<11;s+=1))
do
  python evaluation_run.py \
  --seed $s \
  --file_name continuous_smart_fixom \
  --nn_architecture smart
done
