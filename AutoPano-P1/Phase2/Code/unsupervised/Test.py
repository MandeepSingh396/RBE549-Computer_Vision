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


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)

import cv2
import os
import sys
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision.transforms as tf
import argparse
from Network.Network import Net
import shutil
import string
import math as m
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch



# Don't generate pyc codes
sys.dont_write_bytecode = True


def data_gen():
    flag = True
    i = 1
    while flag == True:
        img = cv2.imread('../Data/Train/' + str(i) + '.jpg')
        img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA)
        x,y,_ = img.shape
        patch_size = 128
        pert = 16
        if x-2*pert < 128 or y-2*pert < 128:
            continue
        random_x = random.randint(0+pert, x-patch_size-pert)
        random_y = random.randint(0+pert, y-patch_size-pert)
        patch_A = img[random_x : random_x + patch_size, random_y : random_y + patch_size]
        plt.imshow(patch_A)
        plt.show()
        patch_a_corners = np.array([[random_y, random_x],
                                    [random_y, random_x + patch_size],
                                    [random_y + patch_size, random_x + patch_size],
                                    [random_y + patch_size, random_x]])
        

        pts1 = np.array([[0, 0],
                        [0, x],
                        [y, x],
                        [y, 0]]).reshape(-1,1,2)

        randomlist_x = np.array(random.sample(range(-pert, pert), 4)).reshape(4,1)
        randomlist_y = np.array(random.sample(range(-pert, pert), 4)).reshape(4,1)
        random_mat = np.hstack([randomlist_x, randomlist_y])
        patch_b_corners = patch_a_corners + random_mat

        h_ab = cv2.getPerspectiveTransform(np.float32(patch_a_corners), np.float32(patch_b_corners))
        h_ba = np.linalg.inv(h_ab)
        pts2  = cv2.perspectiveTransform(np.float32(pts1), h_ba)
        [xmin, ymin] = np.int32(pts2.min(axis=0).ravel())
        [xmax, ymax] = np.int32(pts2.max(axis=0).ravel())
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        warp_img = cv2.warpPerspective(img, Ht.dot(h_ba), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
        plt.imshow(warp_img)
        plt.show()
        patch_B = warp_img[random_x + t[1] : random_x + patch_size + t[1], random_y + t[0] : random_y + patch_size + t[0]]
        plt.imshow(patch_B)
        plt.show()
        flag = False
    return patch_A, patch_B, patch_a_corners, patch_b_corners, warp_img, t

def input(img1, img2):
    transformImg=tf.Compose([tf.ToTensor()])
    img = np.concatenate((img1, img2), axis = 2)
    Img = transformImg(img)
    Img = Img.unsqueeze(0)
    return Img

def Test_operation(Img, ModelPath):
    model = Net(2, 128, 128)
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.eval()
    h4pt_out = model(Img)
    h4pt_out = h4pt_out.squeeze(0)
    return h4pt_out

def predicted_patch(h4pt_pred, patch_a_corners):
    pred_patch_b_corners = patch_a_corners - h4pt_pred
    return pred_patch_b_corners

def plot_corners(img, patch_b_corners, pred_patch_b_corners, t):
    cv2.polylines(img, [patch_b_corners + t], isClosed = True, color = (255,0,0), thickness = 2)
    cv2.polylines(img, [pred_patch_b_corners + t], isClosed = True, color = (0,255,0), thickness = 2)
    cv2.imshow('img', img)
    cv2.imwrite("unsuper", img)
    # plt.imshow(img)
    # plt.savefig("unsuper")
    # plt.show()


def main():

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--ModelPath",
        dest="ModelPath",
        default="../Checkpoints/4model.ckpt",
        help="Path to load latest model from, Default:ModelPath",
    )
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath

    patch_A, patch_B, patch_a_corners, patch_b_corners, warped_img, t = data_gen()
    input_img = input(patch_A, patch_B)
    h4pt_pred = Test_operation(input_img, ModelPath)
    #transform to 4x2 matrix
    h4pt_pred = h4pt_pred.detach().numpy()
    h4pt_pred = np.reshape(h4pt_pred, (4,2), order = 'F')
    print('predicted h4pt', h4pt_pred)
    pred_patch_b_corners = predicted_patch(h4pt_pred, patch_a_corners)
    pred_patch_b_corners = pred_patch_b_corners.astype(int)
    print('predicted patch b corners', pred_patch_b_corners)
    print('patch b corners', patch_b_corners)
    plot_corners(warped_img, patch_b_corners, pred_patch_b_corners, t)


if __name__ == "__main__":
    main()
