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


from resnet import *
from wideresnet import *
from MNIST_Network import *
from attacks import *


def get_dataset_classcount(dataset_name):
    dataset_classcount = {'mnist':10,'cifar10':10}
    count = dataset_classcount[dataset_name]
    return count


DATASET=''
def get_data_loader(dataset_name,data_path,TRAIN_BATCH_SIZE=64,VAL_BATCH_SIZE=100,TEST_BATCH_SIZE=100):
    global DATASET
    DATASET = dataset_name 
    if dataset_name=='cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0., 0., 0.), (1, 1, 1)),])
        val_transform= transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0., 0., 0.), (1, 1, 1)),])
        test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0., 0., 0.), (1, 1, 1)),])
        train_set  = torchvision.datasets.CIFAR10(root=data_path, train=True,download=True, transform = train_transform)
        val_set    = torchvision.datasets.CIFAR10(root=data_path, train=True,download=False, transform = val_transform)
        test_set   = torchvision.datasets.CIFAR10(root=data_path, train=False,download=True,transform = test_transform)

        # Split training into train and validation
        train_size = 40000
        valid_size = 10000
        test_size  = 10000

        #get indices seed
        seed_file = file(data_path+'train_val_seed.npy','rb')
        indices   = np.load(seed_file)
        train_indices = indices[0:train_size]
        val_indices   = indices[train_size:]

        #get data loader ofr train val and test
        train_loader=torch.utils.data.DataLoader(train_set,batch_size=TRAIN_BATCH_SIZE,sampler=SubsetRandomSampler(train_indices))
        val_loader = torch.utils.data.DataLoader(val_set,sampler=SubsetRandomSampler(val_indices),batch_size=VAL_BATCH_SIZE)
        test_loader=torch.utils.data.DataLoader(test_set,batch_size=TEST_BATCH_SIZE,shuffle=False)
    elif dataset_name=='mnist':
        transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0., 0., 0.], std = [1., 1., 1.]),])

        transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0., 0., 0.], std = [1., 1., 1.]),])

        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean = [0., 0., 0.], std = [1., 1., 1.]),])

        train_set  = torchvision.datasets.MNIST(root=data_path, train=True , download=True, transform=transform_train )
        val_set    = torchvision.datasets.MNIST(root=data_path, train=True , download=False, transform=transform_val)
        test_set   = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform_test)

        # Split training into train and validation
        train_size = 50000
        valid_size = 10000
        test_size  = 10000
        #get indices seed
        seed_file  = file(data_path+'train_val_seed.npy','rb')
        indices    = np.load(seed_file)
        train_indices = indices[0:train_size]
        val_indices   = indices[train_size:]
        #get data loader ofr train val and test
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=TRAIN_BATCH_SIZE ,sampler=SubsetRandomSampler(train_indices))
        val_loader   = torch.utils.data.DataLoader(val_set,sampler = SubsetRandomSampler(val_indices),batch_size=VAL_BATCH_SIZE)
        test_loader   = torch.utils.data.DataLoader(test_set,batch_size=TEST_BATCH_SIZE)
        print 'MNIST dataloder: Done'
    return train_loader,val_loader,test_loader    

def get_network(net_name):
    if net_name=='resnet18':
        model = ResNet18()
    elif net_name=='resnet34':
        model = ResNet34()
    elif net_name=='wrn_28_10':
        model = Wide_ResNet(28,10,0,10)
    elif net_name=='lenet_plus':
        model = LeNet_Plus()
    elif net_name=='model_a':
        model = ModelA()
    elif net_name=='model_b':
        model = ModelB()
    elif net_name=='model_c':
        model = ModelC()
    elif net_name=='model_d':
        model = ModelD() 
    else:
        print 'network arch. not found'
        exit()
    return model


def write2file(filename,msg):
    log_file = open(filename,'a')
    log_file.write(msg)
    log_file.close()
    
    
def evaluate_model(data_loader,source_model,target_model,attack_type,attack_eps,attack_eps_step,attack_steps):
    global DATASET
    accuracy = 0
    cost = 0
    sample_count=0
    loss = nn.CrossEntropyLoss()
    loss_sum = nn.CrossEntropyLoss(size_average=False)
    for data, target in data_loader:
        if attack_type=='FGSM':
            data = FGSM_Attack_step(source_model,loss,data,target,eps=attack_eps,steps=1)
        elif attack_type=='IFGSM':
            data = FGSM_Attack_step(source_model,loss,data,target,eps=attack_eps,steps=attack_steps)
        elif attack_type=='MI_FGSM':
            data = MI_FGSM(source_model,loss,data,target,eps=attack_eps,steps=attack_steps)
        elif attack_type=='PGD':
            data = PGD(source_model,loss,data,target,eps=attack_eps,eps_iter=attack_eps_step,steps=attack_steps)
        elif attack_type=='PGD_CW':
            data = PGD_CW(source_model,data,target,eps=attack_eps,eps_iter=attack_eps_step,steps=attack_steps,\
                          num_classes=get_dataset_classcount(DATASET))    
        elif attack_type=='R_FGSM':
            data = RFGSM_Attack_step(source_model,loss,data,target,eps=attack_eps,eps_iter=attack_eps_step,steps=1)
        elif attack_type=='no_attack':
            no_attack = 1
        data   = Variable(data).cuda()
        target = Variable(target).cuda()
        out = target_model(data)
        ce_loss = loss_sum(out,target)
        prediction = out.data.max(1)[1] 
        accuracy = accuracy + prediction.eq(target.data).sum()
        cost = cost + ce_loss.data.cpu().numpy()[0] 
        sample_count += target.size(0) 
    acc = (accuracy*1.0) / (sample_count) * 100
    cost = (cost*1.0) / (sample_count)
    return acc, cost
