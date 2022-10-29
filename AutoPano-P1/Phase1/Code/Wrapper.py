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
# Add any python libraries here
from ast import Pass
import os
import shutil
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from matplotlib import image
import scipy
import math
import copy
import random
from numba import jit
import argparse



def harriscorners(img, img_g):
	img_g = np.float32(img_g)
	dst = cv2.cornerHarris(img_g, 2,3, 0.04)
	img_h = cv2.dilate(dst, None, iterations = 2)

	# img[img_h > 0.001 * img_h.max()] = [255,0,0]
	# plt.imshow(dst)
	# plt.show()
	lm = scipy.ndimage.maximum_filter(img_h, 10)
	msk = (img_h == lm)
	ls = scipy.ndimage.minimum_filter(img_h, 10)
	diff = ((lm-ls) > 20000)
	msk[diff == 0] = 0
	img[img_h>0.01*img_h.max()] = [255,0,0]
	# plt.imshow(img)
	# plt.savefig("harris.png")
	# plt.show()
	coords = []
	for i in range(0, msk.shape[0]):
		for j in range(0, msk.shape[1]):
			if msk[i][j] == True:
				coords.append((i,j))
	# plt.imshow(diff)
	# plt.show()
	# print(len(coords))
	return coords, dst

@jit
def ANMS(img, img_h, n_best, coords):
	num = len(coords)
	inf = sys.maxsize
	r = inf * np.ones((num,3))
	ED = 0
	for i in range(num):
		for j in range(num):
			x_i = coords[i][1]    #We take x_cordinate of one corner point 
			y_i = coords[i][0]   ##We take x_cordinate of one corner point 
			neighbours_x = coords[j][1]  #x_cordinate of Other points
			neighbours_y = coords[j][0]

			if img_h[y_i,x_i] > img_h[neighbours_y,neighbours_x]:
				ED = (neighbours_x - x_i)**2 + (neighbours_y - y_i)**2

			if ED < r[i,0]:
				r[i,0] = ED
				r[i,1] = x_i
				r[i,2] = y_i

	arr = r[:,0]
	feature_sorting = np.argsort(-arr)  #We get the index of biggest that is the reason of -ve sign(Descending order index)
	feature_cord = r[feature_sorting]
	Nbest_corners = feature_cord[:n_best,:]   #We also can is find min of(n_best, num_of_feature_cordinates we got)
	# print(len(Nbest_corners))
	for i in range(len(Nbest_corners)):
		cv2.circle(img, (int(Nbest_corners[i][1]),int(Nbest_corners[i][2])), 3, 255, -1)
	# plt.imshow(img)
	# plt.savefig("anms.png")
	return Nbest_corners

def feature_descriptors(img, img_g,  Nbest_corners, patch_size):
	n_descriptors = []
	x = Nbest_corners[:,1]
	y = Nbest_corners[:,2]

	for i in range(len(Nbest_corners)):
		y_i = x[i]         #reverse the co-ordinates again
		x_i = y[i]
		gray = copy.deepcopy(img_g)
		gray = np.pad(img_g, ((patch_size,patch_size), (patch_size,patch_size)), mode='constant', constant_values=0)     #pad the image by 40 on all sides
		x_start = int(x_i + patch_size/2)
		y_start = int(y_i + patch_size/2)
		descriptor = gray[x_start:x_start+patch_size, y_start:y_start+patch_size]            #40x40 descriptor of one point
		descriptor = cv2.GaussianBlur(descriptor, (7,7), cv2.BORDER_DEFAULT)                  #apply gaussian blur
		descriptor = descriptor[::5,::5]														#sub sampling to 8x8
		descriptor_1 = descriptor.reshape((64,1))
		descriptor_standard = (descriptor_1 - descriptor_1.mean())/descriptor_1.std()
		n_descriptors.append(descriptor_standard)
	
	return n_descriptors

def feature_matching(images, gray_images, img_desc, best_corners, match_ratio):
	f1 = img_desc[0]        #image1 feature vectors
	f2 = img_desc[1]        #image2 feature vectors
	corners1 = best_corners[0]       #image1 feature coordinates
	corners2 = best_corners[1]       #image2 feature coordinates
	matched_pairs = []
	for i in range(0, len(f1)):
		sqr_diff = []
		for j in range(0, len(f2)):
			diff = np.sum((f1[i] - f2[j])**2)
			sqr_diff.append(diff)
		sqr_diff = np.array(sqr_diff)
		diff_sort = np.argsort(sqr_diff)
		sqr_diff_sorted = sqr_diff[diff_sort]
		ratio = sqr_diff_sorted[0]/(sqr_diff_sorted[1])
		if ratio < match_ratio:
			matched_pairs.append((corners1[i,1:3], corners2[diff_sort[0],1:3]))	
	
	return matched_pairs

def keypoint(points):
	kp1 = []
	for i in range(len(points)):
		kp1.append(cv2.KeyPoint(int(points[i][0]), int(points[i][1]), 3))
	return kp1

def matches(points):
	m = []
	for i in range(len(points)):
		m.append(cv2.DMatch(int(points[i][0]), int(points[i][1]), 2))
	return m

def draw_matches(images, matched_pairs):
	img1 = copy.deepcopy(images[0])
	img2 = copy.deepcopy(images[1])
	key_points_1 = [x[0] for x in matched_pairs]
	keypoints1 = keypoint(key_points_1)
	key_points_2 = [x[1] for x in matched_pairs]
	keypoints2 = keypoint(key_points_2)
	matched_pairs_idx = [(i,i) for i,j in enumerate(matched_pairs)]
	matches1to2 = matches(matched_pairs_idx)
	out = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, None, flags =2)
	# plt.imshow(out)
	# plt.show()



def homography(point1, point2):
	h_matrix  = cv2.getPerspectiveTransform(np.float32(point1), np.float32(point2))
	return h_matrix

def dot_product(h_mat, keypoint):
	keypoint = np.expand_dims(keypoint, 1)
	keypoint = np.vstack([keypoint, 1])
	product = np.dot(h_mat, keypoint)
	if product[2]!=0:
		product = product/product[2]
	else:
		product = product/0.000001
	# print(product)
	return product[0:2,:]

def ransac(matched_pairs, threshold):

	inliers = []   #to store ssd's and corresponding homography matrices
	COUNT = []
	for i in range(1000):    #Nmax iterations
		
		keypoints_1 = [x[0] for x in matched_pairs]
		keypoints_2 = [x[1] for x in matched_pairs]
		length = len(keypoints_1)
		
		randomlist = random.sample(range(0, length), 4)
		points_1 = [keypoints_1[idx] for idx in randomlist]
		points_2 = [keypoints_2[idx] for idx in randomlist]

		h_matrix = homography(points_1, points_2)
		# print(h_matrix)
		points = []
		count_inliers = 0
		for i in range(length):
			a = (np.array(keypoints_2[i]))
			# ssd = np.sum((np.expand_dims(np.array(keypoints_2[i]), 1) - dot_product(h_matrix, keypoints_1[i]))**2)
			ssd = np.linalg.norm(np.expand_dims(np.array(keypoints_2[i]), 1) - dot_product(h_matrix, keypoints_1[i]))
			# print("ssd",ssd)
			if ssd < threshold:
				count_inliers += 1
				points.append((keypoints_1[i], keypoints_2[i]))
		COUNT.append(-count_inliers)
		inliers.append((h_matrix, points))
	max_count_idx = np.argsort(COUNT)
	max_count_idx = max_count_idx[0]
	# print(max_count_idx)
	# final_h_matrix = inliers[max_count_idx][0]
	final_matched_pairs = inliers[max_count_idx][1]
	# print("Matched pairs", len(final_matched_pairs))

	pts_1 = [x[0] for x in final_matched_pairs]
	pts_2 = [x[1] for x in final_matched_pairs]
	h_final_matrix, status = cv2.findHomography(np.float32(pts_1),np.float32(pts_2))
	# print(h_final_matrix)
	return h_final_matrix, final_matched_pairs

def warpTwoImages(images, H):

	img1 = copy.deepcopy(images[1])
	img2 = copy.deepcopy(images[0])
	h1,w1 = img1.shape[:2]
	h2,w2 = img2.shape[:2]
	pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
	pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
	pts2_ = cv2.perspectiveTransform(pts2, H)
	pts = np.concatenate((pts1, pts2_), axis=0)
	[xmin, ymin] = np.int32(pts.min(axis=0).ravel())
	[xmax, ymax] = np.int32(pts.max(axis=0).ravel())
	t = [-xmin,-ymin]
	Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
	result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin), flags = cv2.INTER_LINEAR)
	# a,b = result.shape[:2]
	# img11 = np.pad(img1, ((t[1], t[0], 0), (0,0,0)), mode='constant', constant_values=0)
	# plt.imshow(img11)
	# plt.show()
	result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

	return result

def stitching(data_path):
	# folder_img = os.path.join("..", "Data", "Train", "Set2")
	folder_img = os.path.join("Data", data_path)
	name = os.path.split(folder_img)
	name = name[1]
	if os.path.exists(folder_img): 
		image_list = os.listdir(folder_img)
		image_list.sort()
	else:
		raise Exception ("Directory Image1 doesn't exist")
	images_path = []
	for i in range(len(image_list)):
		image_path = os.path.join(folder_img,image_list[i])
		images_path.append(image_path)

	# print(images_path)
	images1 = []
	gray_images1 = []
	# for i in range(1,len(image_list)+1):   #pass range as no. of images to be stitched
	for i, img in enumerate(images_path):
		# img = cv2.imread('../Data/Train/Set1/' + str(i) + '.jpg')
		img = cv2.imread(img)
		images1.append(img)
		img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray_images1.append(img_g)

	img_count = len(images1)
	corners = 1000
	match_ratio = 0.4
	threshold = 30
	c1 = 0
	c2 = 0
	c3 = 0
	for j in range(img_count-1):
		# print("Image number", j+1)
		img_desc = []
		best_corners = []
		c2 = j+1
		images = [images1[c1], images1[c2]]
		print("Matching image", c1+1, "and", c2+1)
		gray_images = [gray_images1[c1], gray_images1[c2]]
		for i in range(2):
			img = images[i]
			img_g = gray_images[i]
			coords, img_h = harriscorners(copy.deepcopy(img), img_g)
			Nbest_corners = ANMS(copy.deepcopy(img), img_h, corners, coords)
			best_corners.append(Nbest_corners)
			f_vectors = feature_descriptors(copy.deepcopy(img), img_g, Nbest_corners, 40)
			img_desc.append(f_vectors)
		
		matched_pairs = feature_matching(images, gray_images, img_desc, best_corners, match_ratio)
		print("Matched pairs", len(matched_pairs))
		# print(len(matched_pairs))
		# draw_matches(images, matched_pairs)
		if len(matched_pairs) > 20:
			c1 = c2
			final_h_mat, final_matched = ransac(matched_pairs, threshold)
			# draw_matches(images, final_matched)
			warped = warpTwoImages(images, final_h_mat)
			plt.imshow(warped)
			# cv2.imwrite("Panorama", name + '.png')
			plt.savefig('results/' + name + '/' + name + '.png')
			# plt.show()
		else:
			c3 += 1
			continue

		images1[c1] = warped
		gray_images1[c1] = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

def main():
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--DataPath', default='TestSet3', help='Contains images to stitch')
	Args = Parser.parse_args()
	data_path = Args.DataPath
	stitching(data_path)

if __name__ == "__main__":
	main()
