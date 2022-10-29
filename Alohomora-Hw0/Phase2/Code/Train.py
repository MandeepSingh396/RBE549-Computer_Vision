#!/usr/bin/env python3

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

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)


from logging import root
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as tf
from torch.optim import AdamW
from torchvision.datasets import CIFAR10
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
from Network.Network import CIFAR10Model, DenseNet, ResNet, VGGNet, ResNeXT
from Misc.MiscUtils import *
from Misc.DataUtils import *



# Don't generate pyc codes
sys.dont_write_bytecode = True

    
def GenerateBatch(TrainSet, TestSet, TrainLabels, ImageSize, MiniBatchSize):
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
    # val_size = 0
    # train_size = len(TrainSet) - val_size
    # train_ds, val_ds = random_split(TrainSet, [train_size, val_size])
    train_loader= DataLoader(TrainSet, MiniBatchSize, shuffle = True)
    test_loader = DataLoader(TestSet, MiniBatchSize)

    return train_loader, test_loader


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)



def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, TestSet, LogsPath, TrainSet1):
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
    # model = CIFAR10Model(InputSize=3*32*32,OutputSize=10) 
    # model = VGGNet()
    # model = ResNetModel() 
    model = ResNet(3, 10)
    # model = DenseNet()
    # model = ResNeXT()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    Optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-4)

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
    
    train = []
    test = []
    train_loader, test_loader = GenerateBatch(TrainSet, TestSet, TrainLabels, ImageSize, MiniBatchSize)
    train_loader1, test_loader1 = GenerateBatch(TrainSet1, TestSet, TrainLabels, ImageSize, MiniBatchSize)
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        # train_loader, test_loader = GenerateBatch(TrainSet, TestSet, TrainLabels, ImageSize, MiniBatchSize)
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
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                
                torch.save({'epoch': Epochs,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': Optimizer.state_dict(),
                            'loss': LossThisBatch},
                             SaveName)
                # print('\n' + SaveName + ' Model Saved...')

        # Save model every epoch
        print("Train loss and accuracy")
        result_train = evaluate(model, train_loader1)
        model.epoch_end(Epochs, result_train)
        train.append(result_train)
        print("test loss and accuracy")
        result_test = evaluate(model, test_loader)
        model.epoch_end(Epochs, result_test)
        test.append(result_test)
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': Optimizer.state_dict(),
                    'loss': LossThisBatch},
                     SaveName)
        print('\n' + SaveName + ' Model Saved...')
    return train, test

def plot_accuracies(train, test):
    train_acc = [x['acc'] for x in train]
    test_acc = [x['acc'] for x in test]
    plt.plot(train_acc, '-x', label = 'TrainSet')
    plt.plot(test_acc, '-x', label = 'TestSet')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(ncol=2, loc="upper left")
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

def plot_losses(train, test):
    train_acc = [x['loss'] for x in train]
    test_acc = [x['loss'] for x in test]
    plt.plot(train_acc, '-x', label =  'TrainSet')
    plt.plot(test_acc, '-x', label = 'TestSet')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.legend(ncol=2, loc="upper right")
    plt.title('Loss vs. No. of epochs')
    plt.show()

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints_dense/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--NumEpochs', type=int, default=25, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=128, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='LogsRes/', help='Path to save Logs for Tensorboard, Default=Logs/')

    transform_train = tf.Compose([tf.RandomCrop(32, padding=4, padding_mode='reflect'),
                                tf.RandomHorizontalFlip(p=0.5),
                                tf.ToTensor(),
                                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])
    transform_test = tf.Compose([tf.ToTensor(),
                                tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ])

    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
                                    
    TrainSet1 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=tf.Compose([tf.ToTensor(),
                                                                            tf.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                                            ]))
    TestSet = torchvision.datasets.CIFAR10(root='data/', train=False, transform=transform_test)

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    # Setup all needed parameters including file reading
    SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(CheckPointPath)
    

    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    train, test = TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet, TestSet, LogsPath, TrainSet1)

    plot_accuracies(train, test)
    plot_losses(train, test)


if __name__ == '__main__':
    main()
 
