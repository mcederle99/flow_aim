#!/bin/bash

# Script to reproduce results
for (( i=0; i<10; i++ ))
do
	CUDA_VISIBLE_DEVICES=0 python main.py \
	--seed $i
done
