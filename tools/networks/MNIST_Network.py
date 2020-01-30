import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class LeNet_Plus(nn.Module):
    def __init__(self):
        super(LeNet_Plus, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32,64,kernel_size=5,dilation=1, stride=1, padding=2,bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(64*7*7,1024)
        self.fc2   = nn.Linear(1024, 10)
    def forward(self, input):
        # conv1 + max pool
        out = F.relu(self.conv1(input))
        out = self.pool1(out)
        # conv2 + max pool
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        # fc-1
        B,C,H,W = out.size()
        out = out.view(B,-1) 
        out = F.relu(self.fc1(out))
        # Logits
        out = self.fc2(out)
        return out

    
   
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        """
        conv(64,5,5)+RELU
        conv(64,5,5)+RELU
        Dropout(0.25)
        FC(128)+Relu
        Dropout(0.5)
        FC+Softmax
        """
        self.conv1 = nn.Conv2d(1,64,kernel_size=5,dilation=1, stride=1, padding=0,bias=True)
        self.conv2 = nn.Conv2d(64,64,kernel_size=5,dilation=1, stride=1, padding=0,bias=True)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1      = nn.Linear(64*20*20,128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2      = nn.Linear(128,10)
    def forward(self, input):
        out = F.relu(self.conv1(input))
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0),-1)
        out = self.dropout1(out)
        out = F.relu(self.fc1(out))
        out = self.dropout2(out)
        out = self.fc2(out)
        return out
    
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(1,64,kernel_size=8)
        self.conv2 = nn.Conv2d(64,128,kernel_size=6)
        self.conv3 = nn.Conv2d(128,128,kernel_size=5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc1  = nn.Linear(128*12*12,10)
    def forward(self, input):
        B,C,H,W = input.size()
        input = input.view(B,-1) 
        out = self.dropout1(input)
        out = out.view(B,C,H,W)
        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0),-1)
        out = self.dropout2(out)
        out = self.fc1(out)
        return out
    
class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.conv1 = nn.Conv2d(1,128,kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(128,64,kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1  = nn.Linear(64*5*5,128)
        self.fc2  = nn.Linear(128,10)
    def forward(self, input):
        out = F.tanh(self.conv1(input))
        out = self.pool1(out)
        out = F.tanh(self.conv2(out))
        out = self.pool1(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
    
class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.fc1  = nn.Linear(28*28,300)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2  = nn.Linear(300,300)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3  = nn.Linear(300,300)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc4  = nn.Linear(300,300)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc5  = nn.Linear(300,10)
    def forward(self, input):
        B,C,H,W = input.size()
        input = input.view(B,-1) 
        out = F.relu(self.fc1(input))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = F.relu(self.fc3(out))
        out = self.dropout3(out)
        out = F.relu(self.fc4(out))
        out = self.dropout4(out)
        out = self.fc5(out)
        return out