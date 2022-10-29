#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

from ast import Pass
from email.mime import image
from logging import root
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm.notebook import tqdm
# import Misc.ImageUtils as iu
from Network.Network import Net
from Misc.MiscUtils import *
from Misc.DataUtils import *
import csv
from torch.utils.data import Dataset
from torchsummary import summary
# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

def read_csv(filename):
    lines = []
    with open (filename) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

class YourDataset(Dataset):

    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):

        x = self.x_train[idx]
        y = self.y_train[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

def read_data():

    transformImg=tf.Compose([tf.ToTensor(),
                            tf.Normalize((0.5,0.5),(0.5,0.5), inplace = True)])
    image1_directory_path = os.path.join("..","Data_ag")
    image2_directory_path = os.path.join("..","Data_bg")
    if os.path.exists(image1_directory_path): 
        image1_list = os.listdir(image1_directory_path)
    else:
        raise Exception ("Directory Image1 doesn't exist")
    if os.path.exists(image2_directory_path): 
        image2_list = os.listdir(image2_directory_path)
    else:
        raise Exception ("Directory Image2 doesn't exist")

    images1_path, images2_path = [], []
    for i in range(len(image1_list)):
        image1_path = os.path.join(image1_directory_path,image1_list[i])
        image2_path = os.path.join(image2_directory_path,image2_list[i])
        images1_path.append(image1_path)
        images2_path.append(image2_path)

    images1 = [cv2.imread(i,0) for i in images1_path]
    images2 = images1.copy()
    images2 = [cv2.imread(i,0) for i in images2_path]
    trainsetA = np.array(images1)
    trainsetB = np.array(images2)
    X_train = []
    Y_train = []
    count = 0
    for i in range(0,len(trainsetA)):
    # for i in range(0,len(trainsetA)):
        count+=1
        print(count, end = "\r")
        img1 = trainsetA[i]
        img1 = np.expand_dims(img1, 2)
        img2 = trainsetB[i]
        img2 = np.expand_dims(img2, 2)
        img = np.concatenate((img1, img2), axis = 2)
        # print(img.shape)
        Img = transformImg(img)
        X_train.append(Img)
        label = np.genfromtxt('../H_4pointg/' + str(i+1) + '.csv', delimiter=',')
        label = torch.from_numpy(label)
        Y_train.append(label)
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    trainset = YourDataset(X_train, Y_train)
    return trainset

def GenerateBatch(TrainSet, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """

    random_seed = 50
    torch.manual_seed(random_seed)

    val_size = int(0.2*len(TrainSet))
    train_size = len(TrainSet) - val_size

    train_ds, val_ds = random_split(TrainSet, [train_size, val_size])
    batch_size = MiniBatchSize
    train_loader = DataLoader(train_ds, batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size)

    return train_loader, val_loader          

    

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)



def TrainOperation(ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet):
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    model = Net(2, 128, 128)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    # summary(model,(2,320,240))
    Optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum = 0.9)
    # Optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # Tensorboard
    # Create a summary to monitor loss tensor

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile)
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
    
    train_loss = []
    validation_loss = []
    train_loader, val_loader = GenerateBatch(TrainSet, ImageSize, MiniBatchSize)
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        model.train()
        train_losses = []
        PerEpochCounter = 0
        for Batch in train_loader:
            PerEpochCounter += 1
            LossThisBatch = model.training_step(Batch)
            train_losses.append(LossThisBatch)
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(Epochs, result)
        validation_loss.append(result['val_loss'])
        train_loss.append(result['train_loss'])

        if (Epochs+1)%10 == 0:
            plot_train_losses(train_loss)
        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': Optimizer.state_dict(),
                    'loss': LossThisBatch},
                     SaveName)
        print('\n' + SaveName + ' Model Saved...')
    return train_loss, validation_loss

def plot_losses(train, test):
    # train_loss = [x for x in train]
    # test_loss = [x['val_loss'] for x in test]
    plt.plot(train, '-x', label =  'TrainSet')
    plt.plot(test, '-x', label = 'ValSet')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.legend(ncol=2, loc="upper right")
    plt.title('Loss vs. No. of epochs')
    plt.savefig('loss')


def plot_train_losses(train):
    plt.plot(train, '-x', label =  'TrainSet')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(ncol=2, loc="upper right")
    plt.title('Loss vs. No. of epochs')
    plt.savefig("loss.png")


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=10, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='LogsRes/', help='Path to save Logs for Tensorboard, Default=Logs/')

   
    TrainSet = read_data()

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    # Setup all needed parameters including file reading
    SaveCheckPoint, ImageSize = SetupAll(CheckPointPath)
    

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    train, test = TrainOperation(ImageSize,
                                 NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                                 DivTrain, LatestFile, TrainSet)

    plot_train_losses(train)


if __name__ == '__main__':
    main()
 
