import cv2
import os
import numpy as np


def CLAHE(sceneRadiance):
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        sceneRadiance[:, :, i] = clahe.apply(sceneRadiance[:, :, i])
    return sceneRadiance

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = os.listdir(input_folder)
    for file in files:
        filepath = os.path.join(input_folder, file)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                sceneRadiance = CLAHE(img)
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, sceneRadiance)

if __name__ == '__main__':
    input_folder = "./input"
    output_folder = "./output"
    process_images(input_folder, output_folder)