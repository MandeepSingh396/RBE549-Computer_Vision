import cv2
from pickletools import uint8
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
from moviepy.editor import ImageSequenceClip
import shutil
import copy
import dlib


def landmark_detection(img):
    img = copy.deepcopy(img)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # cap = cv2.VideoCapture(0)
    # Convert the image color to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    # plot_corners(img, rects)
    face_landmarks = []
    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = []
        for i in range(0, 68):
            # shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape_np.append((shape.part(i).x, shape.part(i).y))
        face_landmarks.append(shape_np)

        # Display the landmarks
    for j in range(len(face_landmarks)):
        for i, (x, y) in enumerate(face_landmarks[j]):
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    # cv2.imwrite('Landmarks.png', img)
    # plt.imshow(img)
    # plt.show()
    return face_landmarks, rects

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

def poission_blending(src, img, hull, mask):
    (x, y, w, h) = cv2.boundingRect(hull)
    center = (int((x+x+w)/2), int((y+y+h)/2))
    output = cv2.seamlessClone(src, img, mask, center, cv2.MIXED_CLONE)

    return output

def mask(orig_img, to_pts):
    # Find convex hull of these points to extract face region
    mask_img = np.full((orig_img.shape[0], orig_img.shape[1]), 0, dtype=np.uint8)
    hullPoints = cv2.convexHull(to_pts)
    cv2.fillConvexPoly(mask_img, hullPoints, (255,255,255))
    mask_img = np.dstack((mask_img, mask_img, mask_img))
    return mask_img

def video2images(video_path, img_dir):

    folder_img = os.path.join(img_dir)
    if os.path.exists(folder_img):
        shutil.rmtree(folder_img)
        os.mkdir(folder_img)
    else:
        os.mkdir(folder_img)

    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 10000
    while success:
        cv2.imwrite(img_dir + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1

def make_video(fps, path, video_file):
    print("Creating video {}, FPS={}".format(video_file, fps))
    clip = ImageSequenceClip(path, fps = fps)
    clip.write_videofile(video_file)

def load_images(folder_path):

    image_directory_path = os.path.join(folder_path)
    if os.path.exists(image_directory_path): 
        image_list = os.listdir(image_directory_path)
    else:
        raise Exception ("Directory doesn't exist")
    images_path = []
    # print(len(image_list))
    for i in range(len(image_list)):
        image_path = os.path.join(image_directory_path,image_list[i])
        images_path.append(image_path)
    images_path.sort()
    images = [cv2.imread(i, 1) for i in images_path]

    return images
