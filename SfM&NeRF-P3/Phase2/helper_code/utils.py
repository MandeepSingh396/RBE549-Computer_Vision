import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import imageio
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import shutil

def read_data(device):
    data = np.load("Data/TinyNerfData/tiny_nerf_data.npz")
    images = data["images"]
    # images = torch.from_numpy(images)
    im_shape = images.shape
    (num_images, H, W, _) = images.shape
    poses = data["poses"]
    poses = torch.from_numpy(poses).to(device)
    focal =  data["focal"]
    focal = torch.from_numpy(focal)

    # Plot a random image from the dataset for visualization.
    # plt.imshow(images[np.random.randint(low=0, high=num_images)])
    # plt.show()
    images = torch.from_numpy(images).to(device)

    return images, poses, focal


def positional_encoding(x, L):
  gamma = [x]
  for i in range(L):
    gamma.append(torch.sin((2.0**i) * x))
    gamma.append(torch.cos((2.0**i) * x))
  
  gamma = torch.cat(gamma, axis = -1)

  return gamma


def mini_batches(inputs, batch_size):
  
  return [inputs[i:i + batch_size] for i in range(0, inputs.shape[0], batch_size)]

def plot_figures(Epochs, log_loss):
    plt.figure(figsize=(10, 4))
    plt.plot(Epochs, log_loss)
    plt.title("loss")
    # plt.show()0
    plt.savefig("Results/Loss.png")

def make_video(fps, path, video_file):
  print("Creating video {}, FPS={}".format(video_file, fps))
  clip = ImageSequenceClip(path, fps = fps)
  clip.write_videofile(video_file)
  shutil.rmtree(path)