# SAT-Rx
Repository for pytorch implementation of single-step adversarial training with regularizers.

# Prerequisites
Python 2.7  
Pytorch 0.3.1  
Torchvision 0.2.1

# Datasets
MNIST and CIFAR-10 datasets would be downloaded on running the code for the first time. 

# Usage
Folders `MNIST/`, and `CIFAR10/`  contains training codes for SAT-R1, SAT-R2, and SAT-R3. To train LeNet+ on MNIST dataset using the proposed regularizers, run the following commands from `MNIST/` folder:
## SAT-R1
`CUDA_VISIBLE_DEVICES=0 python SAT_REG_1_training.py --exp_name exp_sat_r1 --max_epochs 20`
## SAT-R2
`CUDA_VISIBLE_DEVICES=0 python SAT_REG_2_training.py --exp_name exp_sat_r2 --max_epochs 20` 
## SAT-R3
`CUDA_VISIBLE_DEVICES=0 python SAT_REG_3_training.py --exp_name exp_sat_r3 --max_epochs 20` 

