import numpy as np
import cv2
from skimage.color import rgb2hsv,hsv2rgb
import os
import natsort

def stretching(img):
    height = len(img)
    width = len(img[0])
    for k in range(0, 3):
        Max_channel  = np.max(img[:,:,k])
        Min_channel  = np.min(img[:,:,k])
        # print('Max_channel',Max_channel)
        # print('Min_channel',Min_channel)
        for i in range(height):
            for j in range(width):
                img[i,j,k] = (img[i,j,k] - Min_channel) * (255 - 0) / (Max_channel - Min_channel)+ 0
    return img

def global_stretching(img_L,height, width):
    I_min = np.min(img_L)
    I_max = np.max(img_L)
    I_mean = np.mean(img_L)

    array_Global_histogram_stretching_L = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            p_out = (img_L[i][j] - I_min) * ((1) / (I_max - I_min))
            array_Global_histogram_stretching_L[i][j] = p_out

    return array_Global_histogram_stretching_L

def  HSVStretching(sceneRadiance):
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)
    img_s_stretching = global_stretching(s, height, width)
    img_v_stretching = global_stretching(v, height, width)

    labArray = np.zeros((height, width, 3), 'float64')
    labArray[:, :, 0] = h
    labArray[:, :, 1] = img_s_stretching
    labArray[:, :, 2] = img_v_stretching
    img_rgb = hsv2rgb(labArray) * 255

    return img_rgb

def sceneRadianceRGB(sceneRadiance):

    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)

    return sceneRadiance

def ICM(img):
    sceneRadiance = sceneRadianceRGB(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    return sceneRadiance


def process_images(input_folder, output_folder):
    files = os.listdir(input_folder)
    files = natsort.natsorted(files)
    for file in files:
        filepath = os.path.join(input_folder, file)
        if os.path.isfile(filepath):
            print('Processing file:', file)
            img = cv2.imread(filepath)
            img = stretching(img)
            sceneRadiance = sceneRadianceRGB(img)
            sceneRadiance = HSVStretching(sceneRadiance)
            sceneRadiance = sceneRadianceRGB(sceneRadiance)
            output_path = os.path.join(output_folder, file.split('.')[0] + '_ICM.jpg')
            cv2.imwrite(output_path, sceneRadiance)

if __name__ == '__main__':
    input_folder = "./input"
    output_folder = "./output"
    process_images(input_folder, output_folder)