from torchvision.transforms import ToTensor,Normalize,Compose,Scale
from torch.nn import TripletMarginLoss
from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear, Dropout, BatchNorm2d
from torch.nn import BatchNorm1d
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch
import numpy as np
import cv2

class Face(Module):
  def __init__(self,n_channels):
    super(Face,self).__init__()
    self.conv1 = Conv2d(n_channels,32,(3,3))
    kaiming_uniform_(self.conv1.weight,nonlinearity='relu')
    self.relu1 = ReLU()
    self.pool1 = MaxPool2d((2,2),(2,2))
    self.conv2 = Conv2d(32,32,(3,3))
    kaiming_uniform_(self.conv2.weight,nonlinearity='relu')
    self.relu2 = ReLU()
    self.pool2 = MaxPool2d((2,2),(2,2))
    self.conv3 = Conv2d(32,64,(3,3))
    kaiming_uniform_(self.conv3.weight,nonlinearity='relu')
    self.relu3 = ReLU()
    self.pool3 = MaxPool2d((2,2),(2,2))
    self.conv4 = Conv2d(64,128,(3,3))
    kaiming_uniform_(self.conv4.weight,nonlinearity='relu')
    self.relu4 = ReLU()
    self.pool4 = MaxPool2d((2,2),(2,2))
    self.conv5 = Conv2d(128,256,(3,3))
    kaiming_uniform_(self.conv5.weight,nonlinearity='relu')
    self.relu5 = ReLU()
    self.pool5 = MaxPool2d((2,2),(2,2))
    self.conv6 = Conv2d(256,512,(3,3))
    kaiming_uniform_(self.conv6.weight,nonlinearity='relu')
    self.relu6 = ReLU()
    self.pool6 = MaxPool2d((2,2),(2,2))
    self.layer1 = Linear(1*1*512,128)
    kaiming_uniform_(self.layer1.weight,nonlinearity='relu')
    self.relu3 = ReLU()
    self.layer2 = Linear(128,64)
    kaiming_uniform_(self.layer2.weight)
    self.relu4 = ReLU()
    self.dropout = Dropout(p=0.3)
    
  def forward(self,x):
    x = self.conv1(x)
    x=self.relu1(x)
    x=self.pool1(x)
    x = self.conv2(x)
    x=self.relu2(x)
    x=self.pool2(x)
    x = self.conv3(x)
    x=self.relu3(x)
    x=self.pool3(x)
    x = self.conv4(x)
    x=self.relu4(x)
    x=self.pool4(x)
    x = self.conv5(x)
    x=self.relu5(x)
    x=self.pool5(x)
    x = self.conv6(x)
    x=self.relu6(x)
    x=self.pool6(x)
    x = x.view(-1,1*1*512)
    x = self.layer1(x)

    x= self.relu3(x)
    x= self.layer2(x)
    return x
