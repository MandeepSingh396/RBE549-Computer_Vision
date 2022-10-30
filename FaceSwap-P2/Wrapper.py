import cv2
import dlib
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy import interpolate
import os
import argparse
from Helper.utils import *
from Helper.delaunay import *
from Helper.TPS import *

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--Model', dest='model_name', default='delaunay', help='delaunay or tps')
    Parser.add_argument('--input_type', dest='input_type', default='image', help='image or video or images')
    Parser.add_argument('--input_path', dest='input_path', default='other/xy.jpg', help='input path to image or video')
    Args = Parser.parse_args()
    model = Args.model_name
    input = Args.input_type
    path = Args.input_path


#DELAUNAY TRIANGLES

    if model == "delaunay":
        print("Method -- ", model)

    #**************************************************************************************
    #code for two faces in one image

        if input == "image":
            print("Input type -- ", input)
            img1 = cv2.imread(path)
            landmarks, rects = landmark_detection(img1)

            triangle_list1, index_triangles = measure_triangle(img1, landmarks)
            # triangle_list2 = triangle_2(index_triangles, landmarks2, img2)
            warped_face1 = warping_triangles(landmarks[0], landmarks[1], index_triangles, img1, img1)
            result = face_swapped(img1, warped_face1, landmarks[0])

            
            warped_face2 = warping_triangles(landmarks[1], landmarks[0], index_triangles, result, img1)
            result_final = face_swapped(result, warped_face2, landmarks[1])
            cv2.imwrite("Results/res_del_2.png", result_final)
            cv2.imshow("result", result_final)
            cv2.waitKey()

        #************************************************************************************
        # for video input
        if input == "video":

            save_path = "output_del"
            folder_img = os.path.join(save_path)
            if os.path.exists(folder_img):
                shutil.rmtree(folder_img)
                os.mkdir(folder_img)
            else:
                os.mkdir(folder_img)

            print("Input type -- ", input)
            image_dir = "video_images"
            video2images(path, image_dir)
            images = load_images(image_dir)
            print("No. of frames in video", len(images))
            count = 10000
            for i in range(len(images)):
                print("count", i)
                img1 = images[i]
                landmarks, rects = landmark_detection(img1)
                triangle_list1, index_triangles = measure_triangle(img1, landmarks)
                warped_face1 = warping_triangles(landmarks[0], landmarks[1], index_triangles, img1, img1)
                result = face_swapped(img1, warped_face1, landmarks[0])

                warped_face2 = warping_triangles(landmarks[1], landmarks[0], index_triangles, result, img1)
                result_final = face_swapped(result, warped_face2, landmarks[1])
                filename = save_path + "/" + str(count) + ".png" 
                cv2.imwrite(filename, result_final)
                count += 1
            
            video_name = "Data/Data1OutputTri.mp4"
            make_video(30, save_path, video_name)
            shutil.rmtree(folder_img)
            shutil.rmtree(image_dir)

#**************************************************************************************
#**************************************************************************************
#**************************************************************************************
#**************************************************************************************
#**************************************************************************************

#TPS

    elif model == "tps":

    #**************************************************************************************
    #code for swapping two faces in one image 

        if input == "image":

            print("Input type -- ", input)
            img1 = cv2.imread(path)
            landmarks, rects = landmark_detection(img1)
            landmarks = np.array(landmarks)
            print("Swapping first face ************")
            result = swap_images_tps(img1, img1, landmarks[0], landmarks[1], 10)
            # cv2.imshow("result", result)
            # cv2.waitKey()
            print("Swapping second face ************")
            result_final = swap_images_tps(result, img1, landmarks[1], landmarks[0], 10)
            cv2.imwrite("Results/res_tps_2.png", result_final)
            cv2.imshow("result", result_final)
            cv2.waitKey()

    #************************************************************************************
    # for video input

        if input == "video":
            save_path = "output_tps"
            folder_img = os.path.join(save_path)
            if os.path.exists(folder_img):
                shutil.rmtree(folder_img)
                os.mkdir(folder_img)
            else:
                os.mkdir(folder_img)

            print("Input type -- ", input)
            image_dir = "video_images"
            video2images(path, image_dir)
            images = load_images(image_dir)
            print("No. of frames in video", len(images))
            count = 10000
            for i in range(len(images)):
                print("count",i, end = "\r")
                img1 = images[i]
                landmarks, rects = landmark_detection(img1)
                landmarks = np.array(landmarks)
                result = swap_images_tps(img1, img1, landmarks[0], landmarks[1], 10)
                result_final = swap_images_tps(result, img1, landmarks[1], landmarks[0], 10)
                filename = save_path + "/" + str(count) + ".png" 
                cv2.imwrite(filename, result_final)
                count += 1
            video_name = "Data/Data1OutputTPS.mp4"
            make_video(30, save_path, video_name)
            shutil.rmtree(folder_img)
            shutil.rmtree(image_dir)

    else:
        print("Invalid input")

if __name__ == "__main__":
    main()