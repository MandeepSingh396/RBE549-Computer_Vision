import cv2
import dlib
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
import os
from Helper.utils import *

def plot_corners(img, corners):
    img = copy.deepcopy(img)
    start_point = (corners[0].left(), corners[0].top())
    end_point = (corners[0].right(), corners[0].bottom())
    img = cv2.rectangle(img, start_point, end_point, color = (255, 0, 0), thickness = 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(2)
    # return img

def extract_index_nparray(nparray):
    for num in nparray[0]:
        index = num
        break
    return index

def measure_triangle(img, Points):
    for points in Points:
        img = copy.deepcopy(img)
        rect = (0, 0, img.shape[1], img.shape[0])
        sub_div = cv2.Subdiv2D(rect)

        for p in points:
            sub_div.insert(p)

        triangle_list = sub_div.getTriangleList()
        index_triangles = []

        for t in triangle_list :
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            points = np.array(points)

            index_pt1 = np.where((points == pt1).all(axis = 1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis = 1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis = 1))
            index_pt3 = extract_index_nparray(index_pt3)
            
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                index_triangles.append(triangle)

            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
                cv2.line(img, pt1, pt2, (255,0,0), 1)
                cv2.line(img, pt2, pt3, (0,255,0), 1)
                cv2.line(img, pt3, pt1, (0,0,255), 1)
            
    # cv2.imwrite("triangulation.png", img)
    
    return triangle_list, index_triangles

def triangle_2(index_triangles, landmarks2, img2):
    img2 = copy.deepcopy(img2)
    triangle_list = []
    for triangle_index in index_triangles:
        pt1 = landmarks2[triangle_index[0]]
        pt2 = landmarks2[triangle_index[1]]
        pt3 = landmarks2[triangle_index[2]]

        triangle_list.append([pt1, pt2, pt3])

        cv2.line(img2, pt1, pt2,(0,0,255),2)
        cv2.line(img2, pt3, pt2,(0,0,255),2)
        cv2.line(img2, pt1, pt3,(0,0,255),2)

    triangle_list = np.array(triangle_list)
    triangle_list = np.reshape(triangle_list, (-1, 6))
    # plt.imshow(img2)
    # plt.show()
    return triangle_list
        

def compute_barycentric_coords(triangle_coords, rect):

    ax = triangle_coords[0][0]
    bx = triangle_coords[1][0]
    cx = triangle_coords[2][0]

    ay = triangle_coords[0][1]
    by = triangle_coords[1][1]
    cy = triangle_coords[2][1]

    delta_B = np.array(([ax, bx, cx],
                        [ay, by, cy],
                        [1, 1, 1]))

    # (x,y,w,h) = rect
    # print(delta_B)
    delta_B_inv = np.linalg.pinv(delta_B)
    min_x = int(np.amin([ax, bx, cx]))
    max_x = int(np.amax([ax, bx, cx]))
    min_y = int(np.amin([ay, by, cy]))
    max_y = int(np.amax([ay, by, cy]))
    barycentric_coords_list = []
    pixel_position_list = []
    # for i in range(0, w):
    for i in range(min_x, max_x):
        # for j in range(0, h):
        for j in range(min_y, max_y):

            point = np.array(([i],[j],[1]))
            barycentric_coords = np.matmul(delta_B_inv, point)
            flag = True
            for k in range(3):
                if barycentric_coords[k] < 0 or barycentric_coords[k] > 1:
                    flag = False
            if flag == True:
                # barycentric_coords_list.append([i,j])
                barycentric_coords_list.append(barycentric_coords)
                pixel_position_list.append([i, j])
    # print(barycentric_coords_list)
    return np.array(barycentric_coords_list), np.array(pixel_position_list)

def source_pixel_position(triangle_coords, barycentric_coords_list):
    ax = triangle_coords[0][0]
    bx = triangle_coords[1][0]
    cx = triangle_coords[2][0]

    ay = triangle_coords[0][1]
    by = triangle_coords[1][1]
    cy = triangle_coords[2][1]

    delta_A = np.array(([ax, bx, cx],
                        [ay, by, cy],
                        [1, 1, 1]))
    
    pixel_position_list = []
    for i in range(len(barycentric_coords_list)):
        barycentric_coords = np.array(([barycentric_coords_list[i][0],
                                       barycentric_coords_list[i][1],
                                       barycentric_coords_list[i][2]]))
        # print(barycentric_coords)
        pixel_position = np.matmul(delta_A, barycentric_coords)
        pixel_position_x = pixel_position[0]/pixel_position[2]
        pixel_position_y = pixel_position[1]/pixel_position[2]
        pixel_position_list.append([pixel_position_x, pixel_position_y])

    return np.array(pixel_position_list)


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


def warping_triangles(landmarks1, landmarks2, indexes_triangles, img1, img2):
    img1 = copy.deepcopy(img1)
    img2 = copy.deepcopy(img2)
    img1_new = np.zeros_like(img1, np.uint8)
    for triangle_index in indexes_triangles:

        pt1_1 = landmarks1[triangle_index[0]]
        pt2_1 = landmarks1[triangle_index[1]]
        pt3_1 = landmarks1[triangle_index[2]]

        triangle1 = np.array([pt1_1, pt2_1, pt3_1], np.int32)
        rect1 = cv2.boundingRect(triangle1)
        (x1,y1,w1,h1) = rect1
        cropped_triangle1 = img1[y1:y1+h1, x1:x1+w1]
        points1 = np.array([[pt1_1[0] - x1, pt1_1[1] - y1],
                            [pt2_1[0] - x1, pt2_1[1] - y1],
                            [pt3_1[0] - x1, pt3_1[1] - y1]], np.int32)
        cropped_tri1_mask = np.zeros((h1,w1), np.uint8)
        cv2.fillConvexPoly(cropped_tri1_mask, points1, 255)
        cropped_triangle1 = cv2.bitwise_and(cropped_triangle1, cropped_triangle1, mask = cropped_tri1_mask)
        barycentric_list, pixel_pos_list = compute_barycentric_coords(points1, rect1)
        
        pt1_2 = landmarks2[triangle_index[0]]
        pt2_2 = landmarks2[triangle_index[1]]
        pt3_2 = landmarks2[triangle_index[2]]

        triangle2 = np.array([pt1_2, pt2_2, pt3_2], np.int32)
        rect2 = cv2.boundingRect(triangle2)
        (x,y,w,h) = rect2
        cropped_triangle2 = img2[y:y+h, x:x+w]
        points2 = np.array([[pt1_2[0] - x, pt1_2[1] - y],
                            [pt2_2[0] - x, pt2_2[1] - y],
                            [pt3_2[0] - x, pt3_2[1] - y]], np.int32)
        cropped_tri2_mask = np.zeros((h,w), np.uint8)
        cv2.fillConvexPoly(cropped_tri2_mask, points2, 255)
        cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask = cropped_tri2_mask)

        src_pixel_pos_list = source_pixel_position(points2, barycentric_list)
        warped_triangle = np.zeros_like(cropped_triangle1)
        for i in range(0, len(src_pixel_pos_list)):
            a = src_pixel_pos_list[i][0]
            b = src_pixel_pos_list[i][1]
            c = pixel_pos_list[i][0]
            d = pixel_pos_list[i][1]
            warped_triangle[d, c] = bilinear_interpolate(cropped_triangle2, (a,b))
        
      
        triangle_area = img1_new[y1:y1+h1, x1:x1+w1]
        triangle_area = cv2.add(triangle_area, warped_triangle)
        img1_new[y1:y1+h1, x1:x1+w1] = triangle_area
    return img1_new

def face_swapped(img1, warped_face, landmarks1):
    face_gray = cv2.cvtColor(warped_face, cv2.COLOR_BGR2GRAY)
    _,background = cv2.threshold(face_gray, 1, 255, cv2.THRESH_BINARY_INV)
    background1 = cv2.bitwise_and(img1, img1, mask= background)
    # warped_face = cv2.medianBlur(warped_face, 5)
    # result = cv2.add(background1, warped_face)
    # result = cv2.medianBlur(result, 7)
    landmarks1 = np.array(landmarks1, np.int32)
    mask_warped_img = mask(warped_face, landmarks1)
    xmin1 = np.min(landmarks1[:,0])
    xmax1 = np.max(landmarks1[:,0])
    ymax1 = np.max(landmarks1[:,1])
    ymin1 = np.min(landmarks1[:,1])
    center_p = (int((xmin1+xmax1)/2), int((ymin1+ymax1)/2))
    blended_img = cv2.seamlessClone(warped_face, img1, mask_warped_img, center_p, cv2.NORMAL_CLONE)
    blended_img = cv2.medianBlur(blended_img, 7)

    return blended_img






