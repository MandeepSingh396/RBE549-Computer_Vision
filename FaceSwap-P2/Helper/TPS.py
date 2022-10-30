import cv2
import dlib
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
from Helper.utils import *
import argparse

def plot_corners(img, corners):
    img = copy.deepcopy(img)
    start_point = (corners[0].left(), corners[0].top())
    end_point = (corners[0].right(), corners[0].bottom())
    img = cv2.rectangle(img, start_point, end_point, color = (255, 0, 0), thickness = 2)
    x = corners[0].left()
    y = corners[0].top()
    w = corners[0].right() - x
    h = corners[0].bottom() - y
    # cv2.imshow("img", img)
    # cv2.waitKey()
    # return img
    return x,y,w,h

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# def landmark_detection(img):
#     img = copy.deepcopy(img)
#     detector = dlib.get_frontal_face_detector()
#     predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
#     # cap = cv2.VideoCapture(0)
#     # Convert the image color to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 1)
#     # plot_corners(img, rects)
#     face_landmarks = []
#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape_np = []
#         for i in range(0, 68):
#             # shape_np[i] = (shape.part(i).x, shape.part(i).y)
#             shape_np.append((shape.part(i).x, shape.part(i).y))
#         face_landmarks.append(shape_np)

#         # Display the landmarks
#         # for i, (x, y) in enumerate(face_landmarks):
#         #     cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
#         # cv2.imshow('Landmark Detection', img)
#         # cv2.waitKey()
#     return face_landmarks, rects

def bilinear_interpolate(img,cords):
    # print(img.shape)
    h,w = img.shape[0:2]
    int_cords = np.int32(cords)
    x0, y0 = int_cords[0], int_cords[1]
    dx,dy = cords - int_cords
    # print(x0,y0)
    # print(dx,dy)

    # 4 Neighbour pixels
    q11 = img[y0,x0]
    if x0+1 < w and x0+1 >=0:
        q21 = img[y0,x0+1]
    else:
        q21 = img[y0,x0]
    if y0+1 < h and y0+1 >=0:
        q12 = img[y0+1,x0]
    else:
        q12 = img[y0,x0]
    if x0+1 < w and x0+1 >=0 and y0+1 < h and y0+1 >=0:
        q22 = img[y0+1,x0+1]
    else:
        q22 = img[y0,x0]

    btm = q21.T * dx + q11.T * (1-dx)
    top = q22.T * dx + q12.T * (1-dx)
    inter_pixel = top * dy + btm * (1-dy)

    return inter_pixel.T

def U_function(r):
    if r==0:
        return 0
    else:
        return (r**2) * np.log(r)

def spline_eqn(x,y,landmarks,solution):
    affine_part = solution[-1] + y * solution[-2] + x * solution[-3]
    spline_part = 0
    for i in range(len(landmarks)):
        # print(solution[i])
        # print(U_function(np.sqrt((landmarks1[i][0] - x)**2 + (landmarks1[i][1] - y)**2)))
        spline_part += solution[i] * U_function(np.sqrt((landmarks[i][0] - x)**2 + (landmarks[i][1] - y)**2))

    f = affine_part + spline_part
    # print(spline_part)

    return f

def swap_images_tps(img1, img2, landmarks1, landmarks2, lmbda):
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)

    solution = tps_model(landmarks1, landmarks2, img1, img2, lmbda)
    blended_img = warping(landmarks1, landmarks2, img1, img2, solution)
    blended_img = cv2.medianBlur(blended_img, 7)
    return blended_img
    
def warping(landmarks1, landmarks2, img1, img2, solution):
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)
    w = img1.shape[0]
    h = img1.shape[1]

    xmin = np.min(landmarks2[:,0])
    xmax = np.max(landmarks2[:,0])
    ymax = np.max(landmarks2[:,1])
    ymin = np.min(landmarks2[:,1])

    xmin1 = np.min(landmarks1[:,0])
    xmax1 = np.max(landmarks1[:,0])
    ymax1 = np.max(landmarks1[:,1])
    ymin1 = np.min(landmarks1[:,1])

    x_grid = np.arange(xmin, xmax,1)
    y_grid = np.arange(ymin, ymax,1)
    warped_img = copy.deepcopy(img1)
    
    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            x_warped = spline_eqn(x_grid[i], y_grid[j], landmarks2, solution[:,0])
            y_warped = spline_eqn(x_grid[i], y_grid[j], landmarks2, solution[:,1])
            if x_warped > 0 and x_warped < h and y_warped > 0 and y_warped < w:
                warped_img[int(y_warped), int(x_warped)] = img2[y_grid[j],x_grid[i]]

    mask_warped_img = mask(warped_img, landmarks1)
    center_p = (int((xmin1+xmax1)/2), int((ymin1+ymax1)/2))
    blended_img = cv2.seamlessClone(warped_img, img1, mask_warped_img, center_p, cv2.NORMAL_CLONE)
    return blended_img

def tps_model(landmarks1, landmarks2, img1, img2, lmbda):
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)

    n = len(landmarks1)    #number of features

    #initialize all matrices
    K = np.zeros((n,n))
    P = np.ones((n,3))
    V = np.zeros((n+3, 2))

    #modify all matrices
    for i in range(n):
        for j in range(n):
            X_sub = landmarks2[i][0] - landmarks2[j][0]
            Y_sub = landmarks2[i][1] - landmarks2[j][1]
            r = np.sqrt(X_sub**2 + Y_sub**2)
            K[i][j] = U_function(r)

    P[:,:2] = landmarks2    #first two columns are x and y, last one is 1's
    V[:n] = landmarks1      #first n rows are x and y, last three are 0's
    mat_1 = np.concatenate((K, P),axis = 1)
    mat_2 = np.concatenate((np.transpose(P), np.zeros((3,3))), axis = 1)
    mat = np.concatenate((mat_1, mat_2), axis = 0)
    lmda_mat = lmbda * np.identity(n+3)
    inv_mat = np.linalg.inv(mat + lmda_mat)
    solution = np.dot(inv_mat, V)    #dimension is (n+3) x 2   - first n rows are x and y, last three are ax, ay and 1

    return solution






