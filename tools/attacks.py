#torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


# numpy
import numpy as np
# dependeny for deepfool
import copy
# time
import time
import random


def L2Norm_per_sample(per):
    B,C,H,W = per.size()
    per = per.view(B,-1)
    per = per * per
    per = per.sum(1)
    per = torch.pow(per,0.5)
    return per

def L1_Norm(grad):
    B,C,H,W = grad.size()
    grad = grad.view(B,-1)
    grad = torch.abs(grad)
    grad = grad.sum(1)
    grad = grad.view(B,1,1,1).repeat(1,C,H,W)
    return grad

def MI_FGSM(model,loss,image,target,eps=0.1,bounds=[0,1],steps=30):
    assert (not(model.training)), 'Model should be in eval mode'
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps
    # momentum
    B,C,H,W = image.size()
    g = torch.zeros(B,C,H,W).cuda()
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        #get gradient of loss wrt data
        grad = img.grad.data
        grad_L1 = L1_Norm(grad)
        g  = g + grad/grad_L1 
        per = eps * torch.sign(g) 
        #per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img

def RFGSM_Attack_step(model,loss,image,target,eps=0.1,eps_iter=0.01,bounds=[0,1],steps=30):
    assert (not(model.training)), 'Model should be in eval mode'
    tar = Variable(target.cuda())
    B,C,H,W = image.size()
    eps = eps - eps_iter
    eps = eps/steps
    noise  = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0,size=(B,C,H,W)))
    img = torch.clamp(image.cuda()+eps_iter*torch.sign(noise.cuda()),0,1)
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img

def FGSM_Attack_step(model,loss,image,target,eps=0.1,bounds=[0,1], steps=30):
    assert (not(model.training)), 'Model should be in eval mode'    
    tar = Variable(target.cuda())
    img = image.cuda()
    eps = eps/steps
    for step in range(steps):
        img = Variable(img,requires_grad=True)
        zero_gradients(img) 
        out  = model(img)
        cost = loss(out,tar)
        cost.backward()
        per = eps * torch.sign(img.grad.data)
        adv = img.data + per.cuda() 
        img = torch.clamp(adv,bounds[0],bounds[1])
    return img

def PGD(model,loss,data,target,eps=0.1,eps_iter=0.01,bounds=[0,1],steps=30):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps_iter,eps_iter,(B,C,H,W))).cuda()
    for step in range(steps):
        # convert data and corresponding into cuda variable
        temp_img = data + noise
        temp_img = torch.clamp(temp_img,bounds[0],bounds[1])
        img = Variable(temp_img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        #compute loss using true label
        cost = loss(out,tar)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  eps_iter * torch.sign(img.grad.data)
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        noise = adv - data
        noise  = torch.clamp(noise,-eps,eps)
    img = data + noise
    img = torch.clamp(img,bounds[0],bounds[1])
    return img



def one_hot(batch,depth):
    ones = (torch.sparse.torch.eye(depth)).cuda()
    tar  = ones.index_select(0,batch.data)
    tar  = Variable(tar.float(),requires_grad=False)
    return tar

def CW_loss(logits,target,depth=10):
    label_mask    = one_hot(target,depth=depth)
    correct_logit = (label_mask * logits).sum(1)
    wrong_logit   = (1-label_mask) * logits - 1e4*label_mask 
    wrong_logit   = wrong_logit.max(1)[0]
    loss          = F.relu(correct_logit - wrong_logit + 50)
    loss          =  -loss.sum()
    return loss

def PGD_CW(model,data,target,eps=0.1,eps_iter=0.01,bounds=[0,1],steps=30,num_classes=10):
    """
    model
    loss : loss used for training
    data : input to network
    target : ground truth label corresponding to data
    eps : perturbation srength added to image
    eps_iter
    """
    #Raise error if in training mode
    if model.training:
        assert 'Model is in  training mode'
    tar = Variable(target.cuda())
    data = data.cuda()
    B,C,H,W = data.size()
    noise  = torch.FloatTensor(np.random.uniform(-eps_iter,eps_iter,(B,C,H,W))).cuda()
    for step in range(steps):
        # convert data and corresponding into cuda variable
        temp_img = data + noise
        temp_img = torch.clamp(temp_img,bounds[0],bounds[1])
        img = Variable(temp_img,requires_grad=True)
        # make gradient of img to zeros
        zero_gradients(img) 
        # forward pass
        out  = model(img)
        #compute loss using true label
        cost = CW_loss(out,tar,depth=num_classes)
        #backward pass
        cost.backward()
        #get gradient of loss wrt data
        per =  eps_iter * torch.sign(img.grad.data)
        #  ascent
        adv = img.data + per.cuda()
        #clip per channel data out of the range
        noise = adv - data
        noise  = torch.clamp(noise,-eps,eps)
    img = data + noise
    img = torch.clamp(img,bounds[0],bounds[1])
    return img

