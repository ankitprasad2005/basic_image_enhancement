import os
import cv2
from whitebalance import white_balance
from clahe import CLAHE
from icm import ICM
from ucm import UCM

def process_image(image_path, output_path):
    image = cv2.imread(image_path)

    image = white_balance(image)
    image = CLAHE(image)
    image = ICM(image)
    image = UCM(image)

    cv2.imwrite(output_path, image)


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = white_balance(frame)
        frame = CLAHE(frame)
        frame = ICM(frame)
        frame = UCM(frame)
        out.write(frame)
    cap.release()
    out.release()


def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_image(input_path, output_path)
        elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_video(input_path, output_path)


if __name__ == "__main__":
    input_folder = 'input'
    output_folder = 'output'
    process_images(input_folder, output_folder)