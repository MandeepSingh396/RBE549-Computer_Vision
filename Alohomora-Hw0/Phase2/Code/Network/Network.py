"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    criterion = nn.CrossEntropyLoss()
    loss = criterion(out, labels)
    return loss

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))




##################################
#LeNet
##################################

class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      super().__init__()
      self.LeNet = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size = 5, padding = 0),
        nn.Tanh(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(6, 16, kernel_size = 5, padding = 0),
        nn.Tanh(),
        nn.AvgPool2d(2,2),
        nn.Conv2d(16, 120, kernel_size = 5, padding = 0),
        nn.Tanh(),
        nn.Flatten(),
        nn.Linear(120, 48),
        nn.Tanh(),
        nn.Linear(48, OutputSize)
      )

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      out = self.LeNet(xb)
     
      return out

##################################
#VGGNet
##################################

class VGGNet(ImageClassificationBase):
  def __init__(self):
      """
      Inputs: 
      InputSize - Size of the Input
      OutputSize - Size of the Output
      """
      super().__init__()
      self.VGG = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(2,2),
        nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(2,2),
        nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(2,2),
        nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(2,2),
        nn.Flatten(),
        nn.Linear(2*2*512, 512),
        nn.ReLU(inplace = True),
        nn.Dropout(0.2),
        nn.Linear(512, 10)
      )

      
  def forward(self, xb):
      """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      out = self.VGG(xb)
     
      return out



##################################
#ResNet
##################################


class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv_block_1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.res_block_1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(),
                                nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(128),
                                nn.ReLU())

        self.conv_block_2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.MaxPool2d(2))

        self.res_block_2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU(),
                                nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
                                nn.BatchNorm2d(512),
                                nn.ReLU())
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv_block_1(xb)
        # print(out.shape)
        out = self.res_block_1(out) + out
        # print(out.shape)
        out = self.conv_block_2(out)
        # print(out.shape)
        out = self.res_block_2(out) + out
        # print(out.shape)
        out = self.classifier(out)
        return out



##################################
#DenseNet
##################################

class conv2d_block(nn.Module):
    def __init__(self, in_channels, k):
        super().__init__()
        
        self.layers = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels, 4*k, kernel_size=1, padding=0),          #1x1 convolution
                                    nn.BatchNorm2d(4*k),
                                    nn.ReLU(),
                                    nn.Conv2d(4*k, k, kernel_size=3, padding=1))                    #3x3 convolution
              

    def forward(self, xb):
        # print(x.shape)
        out = self.layers(xb)
        # print(out.shape)
        out = torch.cat([xb, out], 1)            #adding previous layer channels in stack
        return out


class DenseNet(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        k = 12                                                                  #growth rate (no. of channels to be increased within each layer in dense_block)
        nblocks = 12                                                            #no. of layers in each dense_block
        in_channels = 2 * k
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size = 3)                  #initial convolution
        self.dense_block1 = self.dense_block(in_channels, k, nblocks)
        self.transition1 = nn.Sequential(nn.BatchNorm2d(168),                     #transition layer to downsample the data
                                        nn.ReLU(),
                                        nn.Conv2d(168, 84, kernel_size=1, padding=0),
                                        nn.AvgPool2d(2))
        in_channels = 84
        self.dense_block2 = self.dense_block(in_channels, k, nblocks)
        self.transition2 = nn.Sequential(nn.BatchNorm2d(228),
                                        nn.ReLU(),
                                        nn.Conv2d(228, 114, kernel_size=1, padding=0),
                                        nn.AvgPool2d(2))

        in_channels = 114
        self.dense_block3 = self.dense_block(in_channels, k, nblocks)
        in_channels += nblocks * k

        self.classifier = nn.Sequential(nn.BatchNorm2d(in_channels),            #classifier consists of Global pool and linear layer
                                        nn.AvgPool2d(7),
                                        nn.Flatten(),
                                        nn.Linear(258, 10))

    def dense_block(self, in_channels, k, nblocks):                            #adds a dense block with specified number of layers
        layers = []
        for i in range(nblocks):
            layers.append(conv2d_block(in_channels, k))
            in_channels += k
                
        return nn.Sequential(*layers)

    def forward(self, xb):
        out = self.conv1(xb)
        # print(out.shape)
        out = self.dense_block1(out)
        # print(out.shape)
        out = self.transition1(out)
        # print(out.shape)
        out = self.dense_block2(out)
        # print(out.shape)
        out = self.transition2(out)
        # print(out.shape)
        out = self.dense_block3(out)
        # print(out.shape)
        out = self.classifier(out)
        # print(out.shape)

        return out

##################################
#ResNeXT
##################################

class res_block(nn.Module):
    def __init__(self, in_channels, cardinality, basewidth, stride):
        super().__init__()
        gp_width = cardinality*basewidth               # as described in paper
        self.layers = nn.Sequential(nn.Conv2d(in_channels, gp_width, kernel_size = 1),
                                    nn.BatchNorm2d(gp_width),
                                    nn.Conv2d(gp_width, gp_width, kernel_size = 3, stride = stride, padding = 1, groups = cardinality), #2 groups made(as C=2 in our case)
                                    nn.BatchNorm2d(gp_width),
                                    nn.Conv2d(gp_width, 2*gp_width, kernel_size = 1),
                                    nn.BatchNorm2d(2*gp_width))
        self.add = nn.Sequential(nn.Conv2d(in_channels, 2*gp_width, kernel_size = 1, stride = stride),
                                    nn.BatchNorm2d(2*gp_width))
    
    def forward(self, xb):
        out = self.layers(xb)
        out += self.add(xb)
        out = F.relu(out)
        return out

class ResNeXT(ImageClassificationBase):

    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.out_channels = 256
        self.cardinality = 2
        self.basewidth = 64
        self.widenfactor = 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dense_block1 = self.block(1)
        self.dense_block2 = self.block(2)           #stride 2 is given in block 2 and 3 because downsampling is obtained by stride and not pooling.
        self.dense_block3 = self.block(2)
        self.classifier = nn.Sequential(nn.AvgPool2d(8),    #global pooling
                                        nn.Flatten(),
                                        nn.Linear(1024, 10))



    def block(self, stride):
        layers = []
        layers.append(res_block(self.in_channels, self.cardinality, self.basewidth, stride))
        self.in_channels = self.widenfactor*self.cardinality*self.basewidth
        self.basewidth = self.basewidth*2
        return nn.Sequential(*layers)
        
    def forward(self,xb):
        out = self.conv1(xb)
        # print(out.shape)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dense_block1(out)
        # print(out.shape)
        out = self.dense_block2(out)
        # print(out.shape)
        out = self.dense_block3(out)
        # print(out.shape)
        out = self.classifier(out)

        return out



