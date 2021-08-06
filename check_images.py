from imutils import paths
import cv2
import os
import numpy as np

IMAGE_DIR = "coin-images-labels/valid"
for image_path in paths.list_images(IMAGE_DIR):
    img = cv2.imread(image_path)
    print(img.shape)
    if img is None:
        print(image_path)
