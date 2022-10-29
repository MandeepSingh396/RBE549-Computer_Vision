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

    def __init__(self, x_train, transform=None):
        self.patchA = x_train[:,0]
        self.patchB = x_train[:,1]
        self.x_train = x_train[:,2]
        self.original_image = x_train[:,3]
        self.corners = x_train[:,4]
        print(np.shape(self.patchA))
        print(np.shape(self.patchB))
        print(np.shape(self.x_train))
        print(np.shape(self.original_image))
        print(np.shape(self.corners))
        self.transform = transform

    def __len__(self):
        return len(self.patchA), len(self.patchB), len(self.x_train), len(self.original_image), len(self.corners)

    def __getitem__(self, idx):

        patchA = self.patchA[idx]
        patchB = self.patchB[idx]
        x = self.x_train[idx]
        original_image = self.original_image[idx]
        corners = self.corners[idx]
        if self.transform:
            x = self.transform(x)

        return patchA, patchB, x, original_image, corners

def read_data():

    transformImg=tf.Compose([tf.ToTensor()])
    image1_directory_path = os.path.join("Data_a")
    image2_directory_path = os.path.join("Data_b")
    image_directory_path = os.path.join("Data_img")
    ca_corners_directory_path = os.path.join("p_a_corners")
    if os.path.exists(image1_directory_path): 
        image1_list = os.listdir(image1_directory_path)
    else:
        raise Exception ("Directory Image1 doesn't exist")
    if os.path.exists(image2_directory_path): 
        image2_list = os.listdir(image2_directory_path)
    else:
        raise Exception ("Directory Image2 doesn't exist")
    if os.path.exists(image_directory_path): 
        image_list = os.listdir(image_directory_path)
    else:
        raise Exception ("Directory Image A doesn't exist")
    if os.path.exists(ca_corners_directory_path): 
        corners_list = os.listdir(ca_corners_directory_path)
    else:
        raise Exception ("Directory corners_a doesn't exist")

    images1_path, images2_path, images_path, corners_a_path = [], [], [], []

    for i in range(len(image1_list)):
        image1_path = os.path.join(image1_directory_path,image1_list[i])
        image2_path = os.path.join(image2_directory_path,image2_list[i])
        image_path = os.path.join(image_directory_path,image_list[i])
        ca_corners_path = os.path.join(ca_corners_directory_path,corners_list[i])
        images1_path.append(image1_path)
        images2_path.append(image2_path)
        images_path.append(image_path)
        corners_a_path.append(ca_corners_path)


    images1 = np.expand_dims(np.array([cv2.imread(i,0) for i in images1_path]), axis = 1)
    images2 = np.expand_dims(np.array([cv2.imread(i,0) for i in images2_path]), axis = 1)
    images_A = np.expand_dims(np.array([cv2.imread(i,0) for i in images_path]), axis = 1)
    # corners_a = np.rollaxis(np.array([np.genfromtxt(i, delimiter=',') for i in corners_a_path]), 3,1)
    corners_a = [np.genfromtxt(i, delimiter=',') for i in corners_a_path]
    trainset = []
    print("images 1 length", len(images1))
    for i in range(len(images1)):
        img = np.concatenate((images1[i], images2[i]), axis = 0)
        trainset.append((images1[i], images2[i], img, images_A[i],corners_a[i]))
    trainset = np.array(trainset)
    return trainset

def GenerateBatch(TrainSet, ImageSize, MiniBatchSize, original_image_data):
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
    # print("trainset", TrainSet[0].shape)
    random_seed = 50
    torch.manual_seed(random_seed)

    val_size = int(0.2*len(TrainSet))
    train_size = len(TrainSet) - val_size

    train_ds, val_ds = random_split(TrainSet, [train_size, val_size])
    batch_size = MiniBatchSize
    train_loader = DataLoader(train_ds, batch_size)
    val_loader = DataLoader(val_ds, batch_size)
    original_image_loader = DataLoader(original_image_data, batch_size)
    print("trainloader", len(train_loader))

    return train_loader, val_loader , original_image_loader        

    

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Trainset [PatchA, PAtchB, PAtch+PatchB, Image, Corners]
# (5, )
def generate_minibatch(dataset, batchsize = None):
    PatchA = dataset[:,0]
    PatchB = dataset[:,1]
    TrainSet = dataset[:,2]
    Image = dataset[:,3]
    Corners = dataset[:,4]
    # print("Patch A Shape =", np.shape(PatchA)) 
    # print("Patch B Shape =", np.shape(PatchB)) 
    # print("Trainset Shape =", np.shape(TrainSet)) 
    # print("Image Shape =", np.shape(Image)) 
    # print("Corners Shape =", np.shape(Corners))
    # print("Patch A Shape =", np.shape(PatchA[0]))
    # print("Patch B Shape =", np.shape(PatchB[0]))
    # print("TrainSet Shape =", np.shape(TrainSet[0]))
    # print("Image Shape =", np.shape(Image[0]))
    # print("Corners Shape =", np.shape(Corners[0]))

    num_batches = (np.shape(PatchA)[0])//batchsize
    # print("num_batches", num_batches)
    remaining = (np.shape(PatchA)[0]) - (num_batches*batchsize)
    if(remaining > 0):
        # print("remaining", remaining)
        num_batches += 1
    print("Number of Batches :", num_batches)
    count = 0
    PatchA_Batches = []
    PatchB_Batches = []
    TrainSet_Batches = []
    Image_Batches = []
    Corners_Batches = []
    for i in range(num_batches-1):
        # print("Batch Start =", i+1, "-"*50)
        PatchA_Batch = []
        PatchB_Batch = []
        TrainSet_Batch = []
        Image_Batch = []
        Corners_Batch = []
        while(count < (batchsize*(i+1))):
            # print("generating batch ", i+1, "->", count)
            PatchA_Batch.append(PatchA[count])
            PatchB_Batch.append(PatchB[count])
            TrainSet_Batch.append(TrainSet[count])
            Image_Batch.append(Image[count])
            Corners_Batch.append(Corners[count])
            count += 1
        PatchA_Batches.append(PatchA_Batch)
        PatchB_Batches.append(PatchB_Batch)
        TrainSet_Batches.append(TrainSet_Batch)
        Image_Batches.append(Image_Batch)
        Corners_Batches.append(Corners_Batch)
    # Last_PatchA_Batch = []
    # Last_PatchB_Batch = []
    # Last_TrainSet_Batch = []
    # Last_Image_Batch = []
    # Last_Corners_Batch = []

    # while(count < (np.shape(PatchA)[0])):
    #     Last_PatchA_Batch.append(PatchA[count])
    #     Last_PatchB_Batch.append(PatchB[count])
    #     Last_TrainSet_Batch.append(TrainSet[count])
    #     Last_Image_Batch.append(Image[count])
    #     Last_Corners_Batch.append(Corners[count])
    #     count += 1

    # PatchA_Batches.append(Last_PatchA_Batch)
    # PatchB_Batches.append(Last_PatchB_Batch)
    # TrainSet_Batches.append(Last_TrainSet_Batch)
    # Image_Batches.append(Last_Image_Batch)
    # Corners_Batches.append(Last_Corners_Batch)
    PatchA_Batches = np.array(PatchA_Batches)
    PatchB_Batches = np.array(PatchB_Batches)
    TrainSet_Batches = np.array(TrainSet_Batches)
    Image_Batches = np.array(Image_Batches)
    Corners_Batches = np.array(Corners_Batches)


    PatchA_Batches = torch.Tensor(PatchA_Batches)
    PatchB_Batches = torch.Tensor(PatchB_Batches)
    TrainSet_Batches = torch.Tensor(TrainSet_Batches)
    Image_Batches = torch.Tensor(Image_Batches)
    Corners_Batches = torch.Tensor(Corners_Batches)

    print("patch A", PatchA_Batches.size())
    print("patch B", PatchB_Batches.size())
    print("trainset", TrainSet_Batches.size())
    print("images", Image_Batches.size())
    print("corners", Corners_Batches.size())

    return PatchA_Batches, PatchB_Batches, TrainSet_Batches, Image_Batches, Corners_Batches

def TrainOperation(ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile,Trainset):
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
    Optimizer = torch.optim.SGD(model.parameters(), lr = 0.00001, momentum = 0.9)

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
    
    train_loss_list = []
    PatchA_Batches, PatchB_Batches, TrainSet_Batches, Image_Batches, Corners_Batches = generate_minibatch(Trainset, MiniBatchSize)
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        model.train()
        train_losses = []
        PerEpochCounter = 0
        for i in range(len(PatchA_Batches)):
            PerEpochCounter += 1
            LossThisBatch = model.training_step(PatchA_Batches[i], PatchB_Batches[i], TrainSet_Batches[i], Image_Batches[i]
                                                , Corners_Batches[i])
            train_losses.append(LossThisBatch)
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

        train_loss = torch.stack(train_losses).mean().item()
        print("Train loss",train_loss)
        print("\n")
        train_loss_list.append(train_loss)

        if (Epochs+1)%10 == 0:
            plot_train_losses(train_loss)

        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': Optimizer.state_dict(),
                    'loss': LossThisBatch},
                     SaveName)
        print(SaveName + ' Model Saved...')
    return train_loss_list

def plot_losses(train, test):
    plt.plot(train, '-x', label =  'TrainSet')
    plt.plot(test, '-x', label = 'ValSet')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(ncol=2, loc="upper right")
    plt.title('Loss vs. No. of epochs')
    plt.show()

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
    Parser.add_argument('--NumEpochs', type=int, default=5, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=5, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='LogsRes/', help='Path to save Logs for Tensorboard, Default=Logs/')

   
    Trainset = read_data()

    # generate_minibatch(Trainset, batchsize=100)

    # print(Trainset.shape)
    # print(model_trainset.shape)
    # print(Trainset[0])
    # print(len(list(model_trainset)[0]))

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

    train = TrainOperation(ImageSize,NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                                 DivTrain, LatestFile,Trainset)

    plot_train_losses(train)


if __name__ == '__main__':
    main()
 
