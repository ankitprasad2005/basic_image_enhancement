import cv2
import os

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = cv2.mean(result[:, :, 1])[0]
    avg_b = cv2.mean(result[:, :, 2])[0]
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                balanced_img = white_balance(img)
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, balanced_img)

if __name__ == "__main__":
    input_folder = './input'
    output_folder = './output'
    process_images(input_folder, output_folder)