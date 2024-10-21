import cv2
import numpy as np
import os


def GC(sceneRadiance):
    sceneRadiance = sceneRadiance / 255.0
    for i in range(3):
        sceneRadiance[:, :, i] = np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 0.7)
    sceneRadiance = np.clip(sceneRadiance * 255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance

def enhance_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    files = os.listdir(input_folder)
    for file in files:
        filepath = os.path.join(input_folder, file)
        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            sceneRadiance = GC(img)
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, sceneRadiance)

if __name__ == '__main__':
    input_folder = "./input"
    output_folder = "./output"
    enhance_images(input_folder, output_folder)

