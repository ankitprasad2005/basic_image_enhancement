import cv2
import os
import numpy as np
import os
import numpy as np

def HE(image):
    for i in range(3):
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])
    return image
def process_images(input_folder, output_folder, method='he'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_{method.upper()}.jpg")
            
            image = cv2.imread(input_path)
            if image is not None:
                if method == 'he':
                    enhanced_image = HE(image)
                else:
                    raise ValueError("Unknown method: choose 'he'")
                
                cv2.imwrite(output_path, enhanced_image)
                print(f"Processed and saved: {output_path}")
                
if __name__ == '__main__':
    input_folder = "./input"
    output_folder = "./output"
    process_images(input_folder, output_folder, method='he')