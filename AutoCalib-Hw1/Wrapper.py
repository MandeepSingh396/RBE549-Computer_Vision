import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import os
import scipy.optimize as optimize
import math

def read_images():
    folder_img = os.path.join("Calibration_Imgs")
    if os.path.exists(folder_img): 
        image_list = os.listdir(folder_img)
        image_list.sort()
    else:
        raise Exception ("Directory doesn't exist")
    images_path = []
    for i in range(len(image_list)):
        image_path = os.path.join(folder_img,image_list[i])
        images_path.append(image_path)

    images = []
    for i, img in enumerate(images_path):
        
        img = cv2.imread(img)
        images.append(img)

    print("No. of images", len(images))
    return images

def plot_corners(img, corners, name):
    img = copy.deepcopy(img)
    for i in range(len(corners)):
        cv2.circle(img, (int(corners[i][0]),int(corners[i][1])), 7, (0,0,255), -1)
    cv2.imwrite("Results/" + name + ".png", img)

def find_img_corners(images, checker_size):
    images_corners = []
    count = 1
    for img in images:
        img = copy.deepcopy(img)
        img_corners = cv2.findChessboardCorners(img, checker_size, None)
        img_corners = np.array(img_corners[1], np.float32)
        img_corners = img_corners.squeeze(1)
        plot_corners(img, img_corners, "original_corners" + str(count))
        images_corners.append(img_corners)
        count += 1

    return np.array(images_corners)

def find_world_corners(checker_size, checker_box_size):
    corners = []
    for i in range(1,checker_size[1]+1):
        for j in range(1,checker_size[0]+1):
            corners.append((i*checker_box_size, j*checker_box_size))
    corners = np.array(corners, np.float32)
    return corners

def find_homography(img_corners, world_corners):
    h = []
    for i in range (len(img_corners)):
        x_i = img_corners[i][0]
        y_i = img_corners[i][1]
        X_i = world_corners[i][0]
        Y_i = world_corners[i][1]
        ax_T = np.array([-X_i, -Y_i, -1, 0, 0, 0, x_i*X_i, x_i*Y_i, x_i])
        h.append(ax_T)
        ay_T = np.array([0, 0, 0,-X_i, -Y_i, -1, y_i*X_i, y_i*Y_i, y_i])
        h.append(ay_T)
    
    h = np.array(h)
    U, S, V_T = np.linalg.svd(h, full_matrices = True)    #decomposed h into three matrices
    V = V_T.T                      
    H = V[:,-1]                     # H = last column of V(minimum singular value vector)
    H = H/H[8]                      # to retain homogenity
    H = np.reshape(H, (3,3))
    return H

def all_images_homography(images_corners, world_corners):
    H_all = []
    for img_corners in images_corners:
        H = find_homography(img_corners, world_corners)
        H_all.append(H)
    H_all = np.array(H_all)
    return H_all

def V_ij(H, i, j):
    H = H.T
    v = np.array([[H[i][0]*H[j][0]],
                  [H[i][0]*H[j][1] + H[i][1]*H[j][0]],
                  [H[i][1]*H[j][1]],
                  [H[i][2]*H[j][0] + H[i][0]*H[j][2]],
                  [H[i][2]*H[j][1] + H[i][1]*H[j][2]],
                  [H[i][2]*H[j][2]]])

    return v.T

def compute_B(H_all):
    v = []
    for i in range (len(H_all)):
        H = H_all[i]
        v.append(V_ij(H, 0,1))
        v.append((V_ij(H, 0, 0) - V_ij(H, 1, 1)))
    v = np.array(v)
    v = v.reshape(-1,6)
    U, S, V_T = np.linalg.svd(v)    #decomposed b into three matrices
    V_ = V_T.T                    
    b = V_[:,-1]

    B = np.array([[b[0],b[1],b[3]],
                   [b[1],b[2],b[4]],
                   [b[3],b[4],b[5]]])
    print("B matrix \n", B)
    return B

def compute_K(B):
    v0 = ((B[0][1] * B[0][2]) - (B[0][0] * B[1][2]))/((B[0][0] * B[1][1]) - (B[0][1])**2)
    lmbda = B[2][2] - ((B[0][2])**2 + v0*(B[0][1] * B[0][2]) - (B[0][0] * B[1][2]))/(B[0][0])
    alpha = np.sqrt(lmbda/B[0][0])
    beta = np.sqrt((lmbda*B[0][0])/((B[0][0] * B[1][1]) - (B[0][1])**2))
    gamma = -B[0][1] * (alpha**2) * beta/lmbda
    u0 = (gamma*v0/beta) - (B[0][2] * (alpha**2) / lmbda)

    K = np.array([[alpha,gamma,u0],
                  [0,beta,v0],
                  [0,0,1]])
    print("K matrix \n", K)

    return K

def get_A_matrix(param):
    alpha, gamma, beta, u0, v0, k1, k2 = param
    A = np.array([[alpha, gamma, u0],
                  [0, beta, v0],
                  [0, 0, 1]])
    k_distortion = np.array([[k1],[k2]])
    return A, k_distortion

def get_parameters(K, k_distortion):
    return np.array([K[0,0], K[0,1], K[1,1], K[0,2], K[1,2], k_distortion[0], k_distortion[1]])

def extrinsics(K, H):
    Rt_all = []
    for h in H:
        h1 = h[:,0]
        h2 = h[:,1]
        h3 = h[:,2]
        lmbda = 1/np.linalg.norm(np.matmul(np.linalg.pinv(K), h1),2)
        r1 = np.matmul(lmbda * np.linalg.pinv(K), h1)
        r2 = np.matmul(lmbda * np.linalg.pinv(K), h2)
        r3 = np.cross(r1, r2)
        t = np.matmul(lmbda * np.linalg.pinv(K), h3)
        Rt = np.vstack((r1, r2, r3, t)).T
        Rt_all.append(Rt)

    return Rt_all

#loss function for the optimizer
def function(x0, Rt_all, images_corners, world_corners):
    K, k_distortion = get_A_matrix(x0)
    error_all_images, _ = rms_error_reprojection(K, k_distortion, Rt_all, images_corners, world_corners)
    
    return np.array(error_all_images)

def rms_error_reprojection(K, K_distortion, Rt_all, images_corners, world_corners):
    alpha, gamma, beta, u0, v0, k1, k2 = get_parameters(K, K_distortion)
    error_all_images = []
    reprojected_corners_all  =[]
    for i in range(len(images_corners)):
        img_corners = images_corners[i]
        Rt = Rt_all[i]
        H = np.dot(K, Rt)
        error_img = 0
        reprojected_corners_img = []
        for j in range(len(img_corners)):
            world_point_2d = world_corners[j]
            world_corners_homo = np.array([[world_point_2d[0]],
                                        [world_point_2d[1]],
                                        [0], 
                                        [1]])

            corners = img_corners[j]
            corners = np.array([[corners[0]],
                                [corners[1]],
                                [1]], dtype = float)
            #pixel coords (u,v)
            proj_coords = np.matmul(H , world_corners_homo)
            u =  proj_coords[0] / proj_coords[2]
            v = proj_coords[1] / proj_coords[2]

            #image plane coords(x,y)
            normalized_coords = np.matmul(Rt , world_corners_homo)
            x =  normalized_coords[0] / normalized_coords[2]
            y = normalized_coords[1] / normalized_coords[2]
            r = np.sqrt(x**2 + y**2)

            u_hat = u+(u - u0)*(k1*r**2+k2*r**4)
            v_hat = v+(v - v0)*(k1*r**2+k2*r**4)
            corners_hat = np.array([[u_hat],
                                    [v_hat],
                                    [1]], dtype = float)
            reprojected_corners_img.append((corners_hat[0],corners_hat[1]))
            error = np.linalg.norm((corners - corners_hat), 2)
            error_img = error_img + error

        reprojected_corners_all.append(reprojected_corners_img)
        error_all_images.append(error_img / 54)
    return np.array(error_all_images), np.array(reprojected_corners_all)

def main():
    images = read_images()
    checker_size = (9,6)
    checker_box_size = 21.5 #mm
    images_corners = find_img_corners(images, checker_size)
    world_corners = find_world_corners(checker_size, checker_box_size)
    H_all = all_images_homography(images_corners, world_corners)
    B = compute_B(H_all)
    K = compute_K(B)
    Rt_all = extrinsics(K, H_all)
    k_distortion = np.array([[0],
                             [0]])
    param = get_parameters(K, k_distortion)
    print("Optimizing---------------------------")
    resultant_parameters = optimize.least_squares(function, x0=param, method="lm", args=[Rt_all, images_corners, world_corners])
    res = resultant_parameters.x
    K_new, K_distortion_new = get_A_matrix(res)
    print("Updated K matrix")
    print(K_new)
    print("Updated Distortion matrix")
    print(K_distortion_new)
    print("Calculating mean error and reprojections---------------------------")
    error_all_images, reprojected_points = rms_error_reprojection(K_new, K_distortion_new, Rt_all, images_corners, world_corners)
    mean_error = np.mean(error_all_images)
    K_distortion_new = np.array([K_distortion_new[0], K_distortion_new[1], 0, 0, 0], dtype = float)
    print("Reprojection error", mean_error)
    for i in range(len(images)):
        img = images[i]
        img = cv2.undistort(img, K_new, K_distortion_new)
        plot_corners(img, reprojected_points[i], "reprojected_corners" + str(i+1))

if __name__ == "__main__":
    main()