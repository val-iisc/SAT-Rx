#!/bin/bash

GPU=1


CUDA_VISIBLE_DEVICES=$GPU python SAT_REG_1_training.py --exp_name exp_sat_r1 --max_epochs 20 
CUDA_VISIBLE_DEVICES=$GPU python SAT_REG_2_training.py --exp_name exp_sat_r2 --max_epochs 20 
CUDA_VISIBLE_DEVICES=$GPU python SAT_REG_3_training.py --exp_name exp_sat_r3 --max_epochs 20 

