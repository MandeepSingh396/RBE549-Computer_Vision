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
import json

def read_json(jsonPath):
	# open the json file
	with open(jsonPath, "r") as fp:
		# read the json data
		data = json.load(fp)
	
	# return the data
	return data

def get_image_c2w(jsonData, datasetPath):
	# define a list to store the image paths
	imagePaths = []
	
	# define a list to store the camera2world matrices
	c2ws = []
	# iterate over each frame of the data
	for frame in jsonData["frames"]:
		# grab the image file name
		imagePath = frame["file_path"]
		imagePath = imagePath.replace(".", datasetPath)
		imagePaths.append(f"{imagePath}.png")
		# grab the camera2world matrix
		c2ws.append(frame["transform_matrix"])
	
	# return the image file names and the camera2world matrices
	return imagePaths, c2ws
    
def read_images(imagePaths):
    images = []
    for i in range(len(imagePaths)):
        image = plt.imread(imagePaths[i])
        image.resize((100,100,3))
        # plt.imshow(image)
        # plt.show()
        images.append(image)
    images = np.array(images)
    images = torch.from_numpy(images)
    return images

def TestOperation(images, poses, focal, height, width, lr, N_encode, epochs,\
                     near_threshold, far_threshold, batch_size, Nc, device, ModelPath, save_path):
    
    model = tNerf()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.to(torch.float64)
    model = model.to(device)
    count = 10041
    model.eval()
    print(len(images))
    Loss = []
    for i in range(len(images)):
        print(i)
        # img_idx = random.randint(0, images.shape[0]-1)
        # print(device)
        img = images[i].to(device)
        pose = poses[i].to(device)
        # print("in Test Operation")
        rgb_logit = training(height, width, focal, pose, near_threshold, far_threshold, Nc, batch_size, N_encode, model, device)
        
        loss = F.mse_loss(rgb_logit, img)
        # print("Loss", loss.item())
        Loss.append(loss)
        plt.imshow(rgb_logit.detach().cpu().numpy())
        # plt.show()
        plt.savefig(save_path + "/" + str(count) + ".png")
        count += 1
        # torch.cuda.empty_cache()
    print(sum(Loss)/len(Loss))

def main():

    save_path = os.path.join("test_results")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    jsonPath = "Data/transforms_test.json"
    datasetPath = "Data/test"
    data = read_json(jsonPath)
    imagePaths, poses = get_image_c2w(data, datasetPath)
    poses = np.array(poses)
    poses = torch.from_numpy(poses).to(device)
    images = read_images(imagePaths)
    images = images.to(device)
    ModelPath = "model.ckpt"
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints_dense/')
    Parser.add_argument('--NumEpochs', type=int, default=1000)
    Parser.add_argument('--Nc', type=int, default=32)
    Parser.add_argument('--MiniBatchSize', type=int, default=500, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--Nn', type=int, default=2)
    Parser.add_argument('--Nf',type=int, default=6)

    Args = Parser.parse_args()
    CheckPointPath = Args.CheckPointPath
    epochs = Args.NumEpochs
    Nc = Args.Nc
    batch_size = Args.MiniBatchSize
    near_threshold = Args.Nn
    far_threshold = Args.Nf
    focal = np.array([138.8889])
    focal = torch.from_numpy(focal).to(device)
    height, width = images.shape[1:3]
    # print(height,width)
    N_encode = 6
    lr = 5e-3
    print("reached_here")
    TestOperation(images[1:6], poses[1:6], focal, height, width, lr, N_encode, epochs,\
                     near_threshold, far_threshold, batch_size, Nc, device, ModelPath, save_path)

    video_file = 'NerF.mp4'
    make_video(5, save_path, video_file)

if __name__ == '__main__':
    main()


