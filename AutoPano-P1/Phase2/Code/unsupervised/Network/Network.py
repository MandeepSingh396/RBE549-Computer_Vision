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
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def loss_fn(batch_b_pred, batch_patch_b):
    # print(batch_b_pred[0][0])
    # print(batch_patch_b[0][0])
    criterion = nn.L1Loss()
    loss = criterion(batch_b_pred, batch_patch_b)

    return loss

class HomographyModel(nn.Module):
    def training_step(self, batch_patch_a, batch_patch_b, batch_patch_set, batch_img, batch_corners):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        batch_patch_a= batch_patch_a.to(device)
        batch_patch_b = batch_patch_b.to(device)
        batch_patch_set = batch_patch_set.to(device)
        batch_img = batch_img.to(device)
        batch_corners = batch_corners.to(device)
        patch_b_pred = self(batch_patch_a, batch_patch_b, batch_patch_set, batch_img, batch_corners) 
        loss = loss_fn(patch_b_pred, batch_patch_b.to(device))         # Calculate loss
        return loss
    
    def validation_step(self, batch):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images = batch 
        images = images.float()
        images = images.to(device)
        out = self(images)                    # Generate predictions
        out = out.float()
        loss = loss_fn(out)           # Calculate loss
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
        
        #Regression Network
        self.homography_net = nn.Sequential(conv_block(2, 64),     
                                            conv_block(64, 64, pool = True),
                                            conv_block(64, 64),
                                            conv_block(64, 64, pool = True),
                                            conv_block(64, 128),
                                            conv_block(128, 128, pool = True),
                                            conv_block(128, 128),
                                            conv_block(128, 128),
                                            nn.Dropout2d(0.5),
                                            nn.Flatten(),
                                            nn.Linear(16*16*128, 1024),
                                            nn.Dropout(0.5),
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

    def forward(self, batch_patch_a, batch_patch_b, batch_patch_set, batch_img, batch_corners):
        h4pt_pred = self.homography_net(batch_patch_set)
        print("h4pt_pred", h4pt_pred)
        # while True: pass
        batch_corners_b_pred = (torch.sub(batch_corners,h4pt_pred))

        batch_corners = batch_corners.reshape(-1,4,2)
        batch_corners_b_pred = batch_corners_b_pred.reshape(-1,4,2)
        print("batch_corners_b_pred", batch_corners_b_pred)
        print("batch_corners", batch_corners)

        h_pred = (kornia.geometry.homography.find_homography_dlt(batch_corners,
                                                                 batch_corners_b_pred, weights=None) )
        print(h_pred)
        # while True: pass
        h_pred_inv = torch.inverse(h_pred)
        # h_pred_inv = torch.nn.functional.normalize(h_pred_inv)

        patch_b_pred = kornia.geometry.transform.warp_perspective(batch_img, h_pred_inv, dsize = (128,128),
                                                                     mode='bilinear', padding_mode='zeros', 
                                                                     align_corners=True, fill_value=torch.zeros(3))
        
        return patch_b_pred
