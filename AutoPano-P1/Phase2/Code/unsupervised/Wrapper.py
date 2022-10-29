#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import copy
import math
import os
import shutil
# Add any python libraries here

def plot_images(fig_size, filters, x_len, y_len, name):
	fig = plt.figure(figsize = fig_size)
	length = len(filters)
	for idx in np.arange(length):
		ax = fig.add_subplot(y_len, x_len, idx+1, xticks = [], yticks = [])
		plt.imshow(filters[idx], cmap = 'gray')
	plt.savefig(name)
	plt.close()


def data_generation():
    count = 0
    folder_img = os.path.join("Data_img")
    if os.path.exists(folder_img):
        shutil.rmtree(folder_img)
        os.mkdir(folder_img)
    else:
        os.mkdir(folder_img)
    folder_a = os.path.join("Data_a")
    if os.path.exists(folder_a):
        shutil.rmtree(folder_a)
        os.mkdir(folder_a)
    else:
        os.mkdir(folder_a)

    ca = os.path.join("p_a_corners")
    if os.path.exists(ca):
        shutil.rmtree(ca)
        os.mkdir(ca)
    else:
        os.mkdir(ca)

    folder_b = os.path.join("Data_b")
    if os.path.exists(folder_b):
        shutil.rmtree(folder_b)
        os.mkdir(folder_b)
    else:
        os.mkdir(folder_b)

    for i in range(1, 5000):
        count += 1
        print("Image Count =",count, end = "\r")
        img = cv2.imread('Data/Train/' + str(i) + '.jpg')
        img = cv2.resize(img, (320, 240), interpolation = cv2.INTER_AREA)
        filename_img = folder_img + "/" + str(count) + ".png"
        cv2.imwrite(filename_img, img)
        x,y,_ = img.shape
        patch_size = 128
        pert = 16
        if x-2*pert < 128 or y-2*pert < 128:
            continue
        random_x = random.randint(0+pert, x-patch_size-pert)
        random_y = random.randint(0+pert, y-patch_size-pert)
        patch_A = img[random_x : random_x + patch_size, random_y : random_y + patch_size]
        patch_a_corners = np.array([[random_y, random_x],
                                    [random_y, random_x + patch_size],
                                    [random_y + patch_size, random_x + patch_size],
                                    [random_y + patch_size, random_x]])
        

        pts1 = np.array([[0, 0],
                        [0, x],
                        [y, x],
                        [y, 0]]).reshape(-1,1,2)
        filename_a = folder_a + "/" + str(count) + ".png"
        cv2.imwrite(filename_a, patch_A)

        randomlist_x = np.array(random.sample(range(-pert, pert), 4)).reshape(4,1)
        randomlist_y = np.array(random.sample(range(-pert, pert), 4)).reshape(4,1)
        random_mat = np.hstack([randomlist_x, randomlist_y])
        patch_b_corners = patch_a_corners + random_mat
        H_4point = (np.array(patch_a_corners - patch_b_corners))
        H_4point = H_4point.flatten(order = 'F')

        ######################________________________________________##################
        patch_aa_corners = patch_a_corners.flatten(order = 'F')
        ca_filename = ca + "/" + str(count) + ".csv"

        np.savetxt(ca_filename, patch_aa_corners, delimiter = ",")

        h_ab = cv2.getPerspectiveTransform(np.float32(patch_a_corners), np.float32(patch_b_corners))
        h_ba = np.linalg.inv(h_ab)
        pts2  = cv2.perspectiveTransform(np.float32(pts1), h_ba)
        [xmin, ymin] = np.int32(pts2.min(axis=0).ravel())
        [xmax, ymax] = np.int32(pts2.max(axis=0).ravel())
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        warp_img = cv2.warpPerspective(img, Ht.dot(h_ba), (xmax-xmin, ymax-ymin),flags = cv2.INTER_LINEAR)
        patch_C = warp_img[random_x + t[1] : random_x + patch_size + t[1], random_y + t[0] : random_y + patch_size + t[0]]


        #################_________________________####################
        
        filename_b = folder_b + "/" + str(count) + ".png"
        cv2.imwrite(filename_b, patch_C)

def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    data_generation()


    """
    Read a set of images for Panorama stitching
    """

    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
