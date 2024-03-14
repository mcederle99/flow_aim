#!/bin/bash

# Script to reproduce results
for (( i=3; i<10; i++ ))
do
	CUDA_VISIBLE_DEVICES=2 python main.py \
	--seed $i
done
