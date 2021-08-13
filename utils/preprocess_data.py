## A script created to preprocess the data for YOLOR model training
# The things done here are: split, resize, copy to train, valid, test set

import os
from imutils import paths
import shutil
import cv2
from sklearn.model_selection import train_test_split

BASE_DIR = "coin-images-labels"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")

VAL_SIZE = 0.15
TEST_SIZE = 0.05

# set this to 1 to test run for one image
TEST_RUN = 0
# set this to 1 to resize the images to 448 x 448 based on YOLO
PREPROCESS_YOLO = 1

# get the image paths and sort them to make sure they align with the label files,
# the label files have the exact same names as their image files
# except for different extensions
image_paths = sorted(list(paths.list_images(IMAGE_DIR)))
# get the label paths
label_paths = sorted(os.listdir(LABEL_DIR))
label_paths = [os.path.join(LABEL_DIR, i) for i in label_paths]
print(f"Total images = {len(image_paths)}")


# split the dataset into ratio of train:valid:test of 75%:15%:10%
# 207:42:28
X_train, X_val_test, y_train, y_val_test = train_test_split(
    image_paths, label_paths, test_size=(VAL_SIZE + TEST_SIZE), random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test,
    y_val_test,
    test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)),
    random_state=42,
)
print(f"Total training images = {len(y_train)}")
print(f"Total validation images = {len(y_val)}")
print(f"Total testing images = {len(y_test)}")


def preprocess_data(image_paths, label_paths, data_type, test_run=TEST_RUN):
    assert data_type in ("train", "valid", "test")
    image_dest = os.path.join(BASE_DIR, data_type, "images")
    label_dest = os.path.join(BASE_DIR, data_type, "labels")

    print(f"[INFO] Copying files from {BASE_DIR} to {image_dest}\n and {label_dest}")
    for destination in (image_dest, label_dest):
        # create the directories if not exists
        if not os.path.exists(destination):
            os.makedirs(destination)

    for image_path, label_path in zip(image_paths, label_paths):
        # copy the label file to the new directory
        # image_dest_path = shutil.copy2(image_path, image_dest)
        shutil.copy2(label_path, label_dest)

        # also resize the images to 448 x 448 at the same time
        # to prepare it for YOLO training
        img = cv2.imread(image_path)
        if PREPROCESS_YOLO:
            img = cv2.resize(img, (448, 448))
        image_dest_path = os.path.join(image_dest, os.path.basename(image_path))
        cv2.imwrite(image_dest_path, img)

        if test_run:
            print(f"copying {image_path} to {image_dest}")
            print(f"copying {label_path} to {label_dest}")
            break


# copy all the images and label files to the respective train, valid and test directories
if TEST_RUN:
    print("[INFO] Copying only for 1 image in each split ...")

preprocess_data(X_train, y_train, "train")
preprocess_data(X_val, y_val, "valid")
preprocess_data(X_test, y_test, "test")
print("[INFO] Files copied successfully.")
