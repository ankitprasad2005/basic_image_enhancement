import numpy as np
import cv2
from skimage.color import rgb2hsv,hsv2rgb
import numpy as np
import os
import matplotlib.pyplot as plt



def cal_equalisation(img, ratio):
    Array = img * ratio
    Array = np.clip(Array, 0, 255)
    return Array

def RGB_equalisation(img):
    img = np.float32(img)
    avg_RGB = [np.mean(img[:, :, i]) for i in range(3)]
    a_r = avg_RGB[0] / avg_RGB[2]
    a_g = avg_RGB[0] / avg_RGB[1]
    ratio = [0, a_g, a_r]
    for i in range(1, 3):
        img[:, :, i] = cal_equalisation(img[:, :, i], ratio[i])
    return img

def global_stretching(img_L):
    I_min, I_max = np.min(img_L), np.max(img_L)
    return (img_L - I_min) * (1 / (I_max - I_min))

def HSVStretching(sceneRadiance):
    sceneRadiance = np.uint8(sceneRadiance)
    img_hsv = rgb2hsv(sceneRadiance)
    img_hsv[:, :, 1] = global_stretching(img_hsv[:, :, 1])
    img_hsv[:, :, 2] = global_stretching(img_hsv[:, :, 2])
    return hsv2rgb(img_hsv) * 255

def sceneRadianceRGB(sceneRadiance):
    return np.clip(sceneRadiance, 0, 255).astype(np.uint8)

def UCM(img):
    sceneRadiance = RGB_equalisation(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    return sceneRadiance

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in os.listdir(input_folder):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_folder, file)
            img = cv2.imread(image_path)
            enhanced_image = UCM(img)
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, enhanced_image)
            print(f"Processed {file}")

if __name__ == '__main__':
    input_folder = './input'  # Replace with your input folder path
    output_folder = './output'  # Replace with your output folder path
    process_images(input_folder, output_folder)