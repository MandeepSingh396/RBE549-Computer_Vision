import os
import glob
import imageio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from helper_code.ray import *
from helper_code.render import *
from helper_code.utils import *
from helper_code.Network import *
import argparse


def TrainOperation(images, poses, focal, height, width, lr, N_encode, epochs,\
                     near_threshold, far_threshold, batch_size, Nc, device):

  model = tNerf()
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)
  

  torch.manual_seed(1200)
  random.seed(1200)

  Loss = []
  Epochs = []
  show_iter = 50

  for i in range(epochs):
    img_idx = random.randint(0, images.shape[0]-1)
    # print(device)
    img = images[img_idx].to(device)
    pose = poses[img_idx].to(device)

    rgb_logit = training(height, width, focal, pose, near_threshold, far_threshold, Nc, batch_size, N_encode, model, device)
    loss = F.mse_loss(rgb_logit, img) #photometric loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % show_iter == 0:
      rgb_logit = training(height, width, focal, pose, near_threshold, far_threshold, Nc, batch_size, N_encode, model, device)
      loss = F.mse_loss(rgb_logit, img)
      print("Loss", loss.item())
      Loss.append(loss.item())
      Epochs.append(i+1)
  plot_figures(Epochs, Loss)

  SaveName =  '../model.ckpt'
                
  torch.save({'epoch': Epochs,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()},
            SaveName)  


def main():

  Parser = argparse.ArgumentParser()
  Parser.add_argument('--CheckPointPath', default='../Checkpoints_dense/')
  Parser.add_argument('--NumEpochs', type=int, default=1000)
  Parser.add_argument('--Nc', type=int, default=32)
  Parser.add_argument('--MiniBatchSize', type=int, default=4096, help='Size of the MiniBatch to use, Default:1')
  Parser.add_argument('--Nn', type=int, default=2)
  Parser.add_argument('--Nf',type=int, default=6)

  Args = Parser.parse_args()
  CheckPointPath = Args.CheckPointPath
  epochs = Args.NumEpochs
  Nc = Args.Nc
  batch_size = Args.MiniBatchSize
  near_threshold = Args.Nn
  far_threshold = Args.Nf
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  images, poses, focal = read_data(device)
  height, width = images.shape[1:3]
  N_encode = 6
  lr = 5e-3

  TrainOperation(images, poses, focal, height, width, lr, N_encode, epochs,\
                     near_threshold, far_threshold, batch_size, Nc, device)


if __name__ == "__main__":
  main()





