#torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.utils.data.sampler import SubsetRandomSampler

# torch dependencies for data load 
import torchvision
from torchvision import datasets, transforms
# numpy
import numpy as np
# time
import time
import random
#utils
import sys
sys.path.append( "../tools/")
sys.path.append( "../tools/networks")
from utils import   get_data_loader
from utils import get_network
from reg_utils import  train_reg2
from reg_utils import  evaluate
###########################parse inputs#######################################
import os
import argparse
##############################################################################
parser = argparse.ArgumentParser(description='Training LeNet+ using SAT-R2')
parser.add_argument('--exp_name',default='sat_r2_mnist', type=str, help='Name of the experiment')
parser.add_argument('--dataset',default='mnist',type=str, help='Dataset')
parser.add_argument('--dataset_path',default='../dataset/mnist/',type=str, help='Path to the dataset')
parser.add_argument('--network',default='lenet_plus',type=str, help='Network')
parser.add_argument('--max_epochs', default=20, type=int, help='Maximum training epochs')
parser.add_argument('--train_batch_size', default=32, type=int, help='Train mini-batch size')
parser.add_argument('--val_batch_size',   default=1000, type=int, help='Val. mini-batch size')
parser.add_argument('--test_batch_size',  default=1000, type=int, help='Test  mini-batch size')
parser.add_argument('--model_weight_prefix',default='models/mnist_lenet_plus_reg2_epoch_',type=str, help='Path to save models')
parser.add_argument('--train_log_fname',default='log/train_loss_sat_reg2.txt',type=str, help='Train loss log file name')
parser.add_argument('--results_dir',default='results/',type=str, help='Results directory')
parser.add_argument('--base_lr', default =1e-2, type=float, help='Initial learning rate')
parser.add_argument('--lambda',default=5.0, type=float, help='lambda')
parser.add_argument('--epsilon', default =0.3, type=float, help='Epsilon')
parser.add_argument('--train_ifgsm_steps', default =10, type=int, help='Train IFGSM steps')
parser.add_argument('--pgd_epsilon_step', default =0.01, type=float, help='PGD epsilon step')
parser.add_argument('--pgd_steps', default =100, type=int, help='PGD steps')
parser.add_argument('--ifgsm_steps', default =100, type=int, help='IFGSM steps')
parser.add_argument('--mifgsm_steps', default =100, type=int, help='MI-FGSM steps')
parser.add_argument('--rfgsm_alpha', default =0.01, type=float, help='RFGSM random noise l_infty norm')
###################################################################################################
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True 
torch.set_default_tensor_type('torch.cuda.FloatTensor')    
print 'Cudnn status:',torch.backends.cudnn.enabled
#Comment out below for multi-run
torch.manual_seed(0)
np.random.seed(0)
#######################################Set tensor to CUDA#########################################

def main():
    args = parser.parse_args()
    config = vars(args)
    train_loader,val_loader,test_loader = get_data_loader(dataset_name=config['dataset'],\
                                                          data_path=config['dataset_path'],\
                                                          TRAIN_BATCH_SIZE=config['train_batch_size'],\
                                                          VAL_BATCH_SIZE=config['val_batch_size'],\
                                                          TEST_BATCH_SIZE=config['test_batch_size'])
    model = get_network(config['network'])
    model.train()
    model.cuda()
    train_reg2(model,train_loader,config)
    evaluate(model,val_loader,test_loader,config)

if __name__ == '__main__':
    main()
