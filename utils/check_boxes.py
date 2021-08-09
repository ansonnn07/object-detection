import cv2
import os
import imutils
from imutils import paths
import numpy as np
import yaml

# CONFIG
ONE_SAMPLE = 0

BASE_DIR = "coin-images-labels/valid"
with open(os.path.join(os.path.dirname(BASE_DIR), "data.yaml")) as f:
    dataMap = yaml.safe_load(f)
# print(dataMap)
CLASS_NAMES = dataMap["names"]

image_paths = sorted(list(paths.list_images(BASE_DIR)))
label_paths = sorted(os.listdir(os.path.join(BASE_DIR, "labels")))

if ONE_SAMPLE:
    random_idx = np.random.randint(len(label_paths))
    label_path = os.path.join(os.path.join(BASE_DIR, "labels", label_paths[random_idx]))
    # uncomment the line below and change the label filename to show specific sample
    # label_path = os.path.join(BASE_DIR, "labels", "00000011_tSNtJLr.txt")
    # append into a list to use in the `for` loop later
    label_paths = [label_path]

    image_path = image_paths[random_idx]
    # append into a list to use in the `for` loop later
    image_paths = [image_path]
else:
    # join the directory at the front to create the full path to the label file
    label_paths = (os.path.join(BASE_DIR, "labels", i) for i in label_paths)

for image_path, label_path in zip(image_paths, label_paths):
    img = cv2.imread(image_path)
    # resize the image to follow YOLO standard and display boxes properly
    # NOTE: YOLO normalized dimensions are based on **square** image shape,
    # therefore, the images must be reshaped to the any same width and height
    img = cv2.resize(img, (416, 416))
    img_width, img_height = img.shape[:2]
    print(f"Current image: {os.path.basename(image_path)}")
    print(f"Image shape: {img.shape}")

    # get the list of labels from the label text file
    with open(label_path) as f:
        labels = f.read().strip().split("\n")
    print(f"Total annotations for the image: {len(labels)}\n")

    # loop through the label file contents
    for label in labels:
        # split them into a list of values of:
        # [class_idx, bbox_x_center, bbox_y_center, bbox_width, bbox_height]
        label_properties = label.split()
        print(label_properties)
        # get the class_idx from first element
        class_idx = label_properties[0]

        # get the dimensions and rescale them back to original size
        box = np.array(label_properties[1:], dtype=np.float32) * np.array(
            [img_width, img_height, img_width, img_height]
        )
        (x_center, y_center, bbox_width, bbox_height) = box
        # get the top left coordinate (startX, startY)
        # and bottom right coordinate (endX, endY)
        # they must be integers to display them on image
        startX = int(x_center - (bbox_width / 2))
        startY = int(y_center - (bbox_height / 2))
        endX = int(startX + bbox_width)
        endY = int(startY + bbox_height)
        # print(f"startX = {startX}, startY = {startY}", end="; ")
        # print(f"endX = {endX}, endY = {endY}\n")

        # draw the bounding box
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 150, 0), 2)

        # get the class label name
        class_label = CLASS_NAMES[int(class_idx)]
        # get the size of the text of the label to draw the label nicely with background
        ((label_width, label_height), _) = cv2.getTextSize(
            class_label, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.75, thickness=2
        )

        # draw a squared background for the label
        cv2.rectangle(
            img,
            (int(startX), int(startY)),
            (
                int(startX + label_width + label_width * 0.05),
                int(startY + label_height + label_height * 0.5),
            ),
            color=(0, 150, 0),
            thickness=cv2.FILLED,
        )

        # draw the label text at the middle of the square background
        cv2.putText(
            img,
            class_label,
            org=(
                int(startX),
                int(startY + label_height + label_height * 0.3),
            ),  # bottom left
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.75,
            color=(255, 255, 255),
            thickness=2,
        )

    # show the annotated image
    cv2.imshow("image", img)
    # press any key to continue showing new images
    key = cv2.waitKey(0)
    if key == 27:
        # press "ESC" key to stop showing images
        break

# destroy the shown windows in the end
cv2.destroyAllWindows()
