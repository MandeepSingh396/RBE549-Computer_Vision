"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
# import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    loss = criterion(out, labels)

    return loss

class HomographyModel(nn.Module):
    def training_step(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images, labels = batch 
        images, labels = images.float(), labels.float()
        # print("label", labels.shape)
        images, labels = images.to(device), labels.to(device)
        out = self(images)                  # Generate predictions
        out = out.float()
        # print("label", labels[0])
        # print("output", out)
        # print(out)
        # print("out shape",out.shape)
        loss = loss_fn(out, labels)         # Calculate loss
        return loss
    
    def validation_step(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images, labels = batch 
        images, labels = images.float(), labels.float()
        images, labels = images.to(device), labels.to(device)
        out = self(images)                    # Generate predictions
        out = out.float()
        loss = loss_fn(out, labels)           # Calculate loss
        return {'loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, train_loss: {:.4f}, ".format(epoch, result['val_loss'], result['train_loss']))


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class Net(HomographyModel):
    def __init__(self, channels, xinput, yinput):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        
        #Regression Network
        self.homography_net = nn.Sequential(conv_block(2, 64),     #input is a grayscale image
                                            conv_block(64, 64, pool = True),
                                            conv_block(64, 64),
                                            conv_block(64, 64, pool = True),
                                            conv_block(64, 128),
                                            conv_block(128, 128, pool = True),
                                            conv_block(128, 128),
                                            conv_block(128, 128),
                                            nn.Dropout2d(0.4),
                                            nn.Flatten(),
                                            nn.Linear(16*16*128, 1024),
                                            nn.Dropout(0.4),
                                            nn.Linear(1024, 8))
        #############################
        # You will need to change the input size and output
        # size for your Spatial transformer network layer!
        #############################
        # Spatial transformer localization-network
    #     self.localization = nn.Sequential(
    #         nn.Conv2d(1, 8, kernel_size=7),
    #         nn.MaxPool2d(2, stride=2),
    #         nn.ReLU(True),
    #         nn.Conv2d(8, 10, kernel_size=5),
    #         nn.MaxPool2d(2, stride=2),
    #         nn.ReLU(True),
    #     )

    #     # Regressor for the 3 * 2 affine matrix
    #     self.fc_loc = nn.Sequential(
    #         nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
    #     )

    #     # Initialize the weights/bias with identity transformation
    #     self.fc_loc[2].weight.data.zero_()
    #     self.fc_loc[2].bias.data.copy_(
    #         torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
    #     )

    # #############################
    # # You will need to change the input size and output
    # # size for your Spatial transformer network layer!
    # #############################
    # def stn(self, x):
    #     "Spatial transformer network forward function"
    #     xs = self.localization(x)
    #     xs = xs.view(-1, 10 * 3 * 3)
    #     theta = self.fc_loc(xs)
    #     theta = theta.view(-1, 2, 3)

    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)

    #     return x

    def forward(self, xb):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        out = self.homography_net(xb)

        return out
