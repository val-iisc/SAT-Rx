#torch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
# time
import time
import random
import sys
# Utils and attacks
from utils import *
from attacks import *



#Returns Euclidean distance (**2) between 2 vectors
def l2_square(x,y):
    diff = x-y
    diff = diff*diff
    diff = diff.sum(1)
    diff = diff.mean(0)
    return diff

#Implements step learning rate scheduling
def step_lr_scheduling(optimizer,epoch,max_epoch,base_lr):
    if epoch == (3*max_epoch/4):
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr/125
    elif epoch == (2*max_epoch/4): 
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr/25
    elif epoch == (max_epoch/4):
        for param_group in optimizer.param_groups:
            param_group['lr'] = base_lr/5

# Training of the model
def train_reg1(model,train_loader,config):
    base_lr           = config['base_lr']
    max_epochs        = config['max_epochs']
    train_batch_size  = config['train_batch_size']
    epsilon           = config['epsilon']
    Lambda            = config['lambda'] 
    train_ifgsm_steps = config['train_ifgsm_steps']
    
    # log file settings
    model_save_prefix  = config['model_weight_prefix']
    train_log_fname    = config['train_log_fname']

    # iterations counter
    iteration        = 0 
    
    #define loss and optimizer
    loss      = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr=base_lr,weight_decay=5e-4)
    optimizer.zero_grad()
    #start of training
    for epoch in range(max_epochs):
        start = time.time()
        counter =0 
        for data, target in train_loader:
            model.eval()
            B = target.size(0)
            #FGSM samples for m samples in the mini-batch
            adv1   = FGSM_Attack_step(model,loss,data,target,eps=epsilon,steps=1)
            #I-FGSM adversarial sample corresponding to last sample in the mini-batch
            adv2   = FGSM_Attack_step(model,loss,data[B-1:,:,:,:],target[B-1:],eps=epsilon,steps=train_ifgsm_steps)
            adv    = torch.cat((adv1,adv2),0)
            #target = torch.cat((target,target[B-1:]),0)
            model.train()

            # Forward
            data   = Variable(adv)
            target = Variable(target).cuda()
            optimizer.zero_grad()
            out  = model(data)
            
            # COMPUTE LOSS 
            B = out.size(0) 
            CE_loss  =  loss(out[:B-1,:],target)
            #CE_loss  =  loss(out,target)
            REG_loss = l2_square(out[B-2:B-1,:],out[B-1:,:])
            cost =  CE_loss + Lambda*REG_loss
            
            # BACKWARD PASS AND UPDATE MODEL'S PARAMETERS 
            cost.backward()
            optimizer.step()
            
            #log
            if iteration%100==0:
                msg = 'iter,'+str(iteration)+',loss,'+str(CE_loss.data.cpu().numpy()[0])+','+str(REG_loss.data.cpu().numpy()[0])+'\n'
                write2file(train_log_fname,msg)
                
            iteration = iteration + 1
            ##console log
            counter = counter + 1
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d]  : Loss:%f \t\t '
                    %(epoch, max_epochs, cost.data.cpu().numpy()[0]))
        end = time.time()
        print 'Epoch:',epoch,' Time taken:',(end-start)
        # Save model
        model_name = model_save_prefix+str(epoch)+'.pkl'
        torch.save(model.state_dict(),model_name)
        # Update model lr
        step_lr_scheduling(optimizer,epoch,max_epochs,base_lr)
    print 'Training done'

    
# SAT-R2 training
def GenerateAdvSamples_sat_r2(model,loss,img,tar,eps=0.3,eps_iter=0.01,bounds=[0,1]): 
    B,C,H,W = img.size()
    noise   = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0,size=(B,C,H,W)))
    img     = torch.cat((img,img+eps_iter*torch.sign(noise)),0)
    img     = Variable(img.cuda(),requires_grad=True)
    tar     = Variable(tar.cuda())
    tar     = torch.cat((tar,tar),0)
    
    zero_gradients(img) 
    out  = model(img)
    cost = loss(out,tar)
    cost.backward()
    img_per  = img.data + eps*torch.sign(img.grad.data)
    img = torch.clamp(img_per,bounds[0],bounds[1])
    return img

def train_reg2(model,train_loader,config):
    base_lr           = config['base_lr']
    max_epochs        = config['max_epochs']
    train_batch_size  = config['train_batch_size']
    epsilon           = config['epsilon']
    Lambda            = config['lambda'] 
    rfgsm_alpha       = config['rfgsm_alpha']
    
    # log file settings
    model_save_prefix  = config['model_weight_prefix']
    train_log_fname    = config['train_log_fname']

    # iterations counter
    iteration        = 0 
    
    #define loss and optimizer
    loss      = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr=base_lr,weight_decay=5e-4)
    optimizer.zero_grad()
    #start of training
    for epoch in range(max_epochs):
        start = time.time()
        counter =0 
        for data, target in train_loader:
            model.eval()
            adv = GenerateAdvSamples_sat_r2(model,loss,data,target,eps=epsilon,eps_iter=rfgsm_alpha)
            model.train()

            # Forward
            data   = Variable(adv)
            target = Variable(target).cuda()
            optimizer.zero_grad()
            out  = model(data)
            
            # COMPUTE LOSS 
            B = out.size(0) 
            # Cross-entropy loss on FGSM samples only
            CE_loss  =  loss(out[:B/2,:],target)
            REG_loss = l2_square(out[:B/2,:],out[B/2:,:])
            cost =  CE_loss + Lambda*REG_loss
            
            # BACKWARD PASS AND UPDATE MODEL'S PARAMETERS 
            cost.backward()
            optimizer.step()
            
            # LOG
            if iteration%100==0:
                msg = 'iter,'+str(iteration)+',loss,'+str(CE_loss.data.cpu().numpy()[0])+','+str(REG_loss.data.cpu().numpy()[0])+'\n'
                write2file(train_log_fname,msg)
                
            iteration = iteration + 1
            ##console log
            counter = counter + 1
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d]  : Loss:%f \t\t '
                    %(epoch, max_epochs, cost.data.cpu().numpy()[0]))
        end = time.time()
        print 'Epoch:',epoch,' Time taken:',(end-start)
        # Save model
        model_name = model_save_prefix+str(epoch)+'.pkl'
        torch.save(model.state_dict(),model_name)
        # Update model lr
        step_lr_scheduling(optimizer,epoch,max_epochs,base_lr)
    print 'Training done'

def GenerateAdvSamples_sat_r3(model,loss,image,target,config,bounds=[0,1]):   
    tar = Variable(target.cuda())
    img     = image.cuda()
    eps_low = config['epsilon_low']
    eps_high= config['epsilon_high']
    alpha   = config['rfgsm_alpha']
    B,C,H,W = image.size()
    noise   = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0,size=(B,C,H,W))).cuda()
    img     = torch.clamp(img+alpha*torch.sign(noise),0,1)

    img = Variable(img,requires_grad=True)
    zero_gradients(img)
    out  = model(img)
    cost = loss(out,tar)
    cost.backward()
    loss_grad = torch.sign(img.grad.data)
    adv_low = img.data + eps_low * loss_grad
    adv_high = img.data + eps_high * loss_grad
    adv = torch.cat((adv_low,adv_high),0)
    img = torch.clamp(adv,bounds[0],bounds[1])
    return img 


def train_reg3(model,train_loader,config):
    base_lr           = config['base_lr']
    max_epochs        = config['max_epochs']
    train_batch_size  = config['train_batch_size']
    epsilon           = config['epsilon']
    Lambda            = config['lambda'] 
    Tau               = config['tau']

    # log file settings
    model_save_prefix  = config['model_weight_prefix']
    train_log_fname    = config['train_log_fname']

    # iterations counter
    iteration        = 0 
    
    #define loss and optimizer
    loss      = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr=base_lr,weight_decay=5e-4)
    optimizer.zero_grad()
    #start of training
    for epoch in range(max_epochs):
        start = time.time()
        counter =0 
        for data, target in train_loader:
            model.eval()
            adv = GenerateAdvSamples_sat_r3(model,loss,data,target,config)
            model.train()

            # Forward
            data   = Variable(adv)
            target = Variable(target).cuda()
            optimizer.zero_grad()
            out  = model(data)
            
            # COMPUTE LOSS 
            B,C = out.size()
            loss_eps_low   = loss(out[0:B/2,:],target)
            loss_eps_high  = loss(out[B/2:,:] ,target)
            cost = loss_eps_high + Lambda* F.relu(loss_eps_low - Tau*loss_eps_high)

            # BACKWARD PASS AND UPDATE MODEL'S PARAMETERS 
            cost.backward()
            optimizer.step()
            
            # LOG
            if iteration%100==0:
                msg = 'iter,'+str(iteration)+',loss,'+str(loss_eps_low.data.cpu().numpy()[0])+','+str(loss_eps_high.data.cpu().numpy()[0])+'\n'
                write2file(train_log_fname,msg)
                
            iteration = iteration + 1
            ##console log
            counter = counter + 1
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d]  : Loss:%f \t\t '
                    %(epoch, max_epochs, cost.data.cpu().numpy()[0]))
        end = time.time()
        print 'Epoch:',epoch,' Time taken:',(end-start)
        # Save model
        model_name = model_save_prefix+str(epoch)+'.pkl'
        torch.save(model.state_dict(),model_name)
        # Update model lr
        step_lr_scheduling(optimizer,epoch,max_epochs,base_lr)
    print 'Training done'
    
# Training of the model
def evaluate(model,val_loader,test_loader,config):
    model.eval()
    epsilon           = config['epsilon']
    rfgsm_alpha       = config['rfgsm_alpha']
    pgd_epsilon_step  = config['pgd_epsilon_step']
    pgd_steps         = config['pgd_steps']
    ifgsm_steps       = config['ifgsm_steps']
    mifgsm_steps      = config['mifgsm_steps']
    max_epochs        = config['max_epochs']
    exp_name          = config['exp_name']
    model_save_prefix = config['model_weight_prefix']
    
    eval_log_name      = config['results_dir']+exp_name+'.txt'
    acc_epoch_log_name = config['results_dir']+exp_name+'_acc_epoch.txt'
    msg = '#####################Evaluate####################\n'
    # Get accuracy v/s epoch on val set
    write2file(eval_log_name,msg)
    accuracy_log = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        model_name = model_save_prefix+str(epoch)+'.pkl'
        model.load_state_dict(torch.load(model_name))
        accuracy = 0
        acc,cost = evaluate_model(val_loader,model,model,attack_type='IFGSM',\
                                  attack_eps=epsilon,attack_eps_step=0,attack_steps=3)
        accuracy_log[epoch] = acc
        #log accuracy to file
        msg= str(epoch)+','+str(acc)+'\n'
        write2file(acc_epoch_log_name,msg)
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] : Acc:%f \t\t'
                %(epoch, max_epochs,acc))
        sys.stdout.flush() 

    model_name = model_save_prefix+str(accuracy_log.argmax())+'.pkl'
    model.load_state_dict(torch.load(model_name))

    # Clean accuracy
    acc,_ = evaluate_model(test_loader,model,model,attack_type='no_attack',\
                                  attack_eps=0,attack_eps_step=0,attack_steps=0)
    msg = 'Clean, accuracy,'+str(acc)+'\n'
    write2file(eval_log_name,msg)
    
    # FGSM attack
    acc,_ = evaluate_model(test_loader,model,model,attack_type='FGSM',\
                                  attack_eps=epsilon,attack_eps_step=0,attack_steps=1)
    msg  = 'Attack epsilon='+str(epsilon)+'\n' 
    msg += 'FGSM attack, accuracy,'+str(acc)+'\n'
    write2file(eval_log_name,msg)
    
    # IFGSM attack
    acc,_ = evaluate_model(test_loader,model,model,attack_type='IFGSM',\
                                  attack_eps=epsilon,attack_eps_step=0,attack_steps=ifgsm_steps)
    msg = 'IFGSM attack, accuracy,'+str(acc)+'\n'
    write2file(eval_log_name,msg)
    
    # PGD attack
    acc,_ = evaluate_model(test_loader,model,model,attack_type='PGD',\
                                  attack_eps=epsilon,attack_eps_step=pgd_epsilon_step,attack_steps=pgd_steps)
    msg = 'PGD attack, accuracy,'+str(acc)+'\n'
    write2file(eval_log_name,msg)
    
    # PGD-CW attack
    acc,_ = evaluate_model(test_loader,model,model,attack_type='PGD_CW',\
                                  attack_eps=epsilon,attack_eps_step=pgd_epsilon_step,attack_steps=pgd_steps)
    msg = 'PGD_CW attack, accuracy,'+str(acc)+'\n'
    write2file(eval_log_name,msg)
    
    
    print 'Evaluation done'

    
    
    