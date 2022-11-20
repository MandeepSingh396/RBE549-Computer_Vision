import cv2
import numpy as np
from EstimateFundamentalMatrix import *


def error_F(pts1,pts2,F):
    
    "Checking the epipolar constraint"
    x1 = np.array([pts1[0], pts1[1],1])
    x2 = np.array([pts2[0], pts2[1],1]).T

    error = np.dot(x2,np.dot(F,x1))

    return np.abs(error)

def getInliers(pts1,pts2,idx):

    "Point Correspondence are computed using SIFT feature descriptors, data becomes noisy, RANSAC is used with fundamental matrix with maximum no of Inliers"
    
    no_iterations = 2000
    error_threshold = 0.002
    inliers_threshold = 0
    inliers_indices = []
    f_inliers = None

    for i in range(0, no_iterations):
        # We need 8 points randomly for 8 points algorithm
        n_rows = pts1.shape[0]
        rand_indxs = np.random.choice(n_rows,8)
        x1 = pts1[rand_indxs,:]
        x2 = pts2[rand_indxs,:]
        F = EstimateFundamentalMatrix(x1,x2)
        indices = []

        if F is not None:
            for j in range(n_rows):
                error = error_F(pts1[j,:],pts2[j,:],F)  #x2.TFx1 = 0
                if error < error_threshold:
                    indices.append(idx[j])

        if len(indices) > inliers_threshold:
            inliers_threshold = len(indices)
            inliers_indices = indices
            f_inliers = F                       #We choose F with Maximum no of Inliers.

    return F, inliers_indices


