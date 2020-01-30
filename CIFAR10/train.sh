#!/bin/bash

GPU=1

CUDA_VISIBLE_DEVICES=$GPU python SAT_REG_1_training.py --network wrn_28_10 --exp_name exp_sat_r1_run1 --max_epochs 100
rm -r models
mkdir models
CUDA_VISIBLE_DEVICES=$GPU python SAT_REG_2_training.py --network wrn_28_10 --exp_name exp_sat_r2_run1 --max_epochs 100 
rm -r models
mkdir models
CUDA_VISIBLE_DEVICES=$GPU python SAT_REG_3_training.py --network wrn_28_10 --exp_name exp_sat_r3_run1 --max_epochs 100 
rm -r models
mkdir models

