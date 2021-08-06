import cv2
import os
import imutils
from imutils import paths
import numpy as np


CLASS_NAMES = ["50", "1r", "5", "10", "20", "1c"]
BASE_DIR = "coin-images-labels/valid"
SAMPLE = 0

if SAMPLE:
    label_paths = sorted(os.listdir(os.path.join(BASE_DIR, "labels")))
    random_idx = np.random.randint(len(label_paths))
    label_path = os.path.join(os.path.join(BASE_DIR, "labels", label_paths[random_idx]))
    # label_path = os.path.join(BASE_DIR, "labels", "one_cent_060.txt"
    label_paths = [label_path]

    image_dir = os.path.join(BASE_DIR, "images")
    filename = os.path.basename(label_path).split(".")[0]
    image_path = os.path.join(image_dir, filename + ".jpg")
    image_paths = [image_path]
else:
    image_paths = sorted(list(paths.list_images(BASE_DIR)))
    label_paths = sorted(os.listdir(os.path.join(BASE_DIR, "labels")))
    label_paths = (os.path.join(BASE_DIR, "labels", i) for i in label_paths)

for image_path, label_path in zip(image_paths, label_paths):
    img = cv2.imread(image_path)
    # resize the image to display it smaller
    img = imutils.resize(img, width=448)
    img_width, img_height = img.shape[:2]
    print(f"Current image: {image_path}")
    print(f"Image shape: {img.shape}")

    with open(label_path) as f:
        labels = f.read().strip().split("\n")
    print(f"Total annotations for sample image: {len(labels)}")
    print("class_index, bbox_x_center, bbox_y_center, bbox_width, bbox_height")
    print(labels)

    print(f"Total annotations for the image: {len(labels)}\n")

    for label in labels:
        label_properties = label.split()
        print(label_properties)
        class_idx = label_properties[0]

        box = np.array(label_properties[1:], dtype=np.float32) * np.array(
            [img_width, img_height, img_width, img_height]
        )
        (x_center, y_center, bbox_width, bbox_height) = box.astype("int")
        startX = int(x_center - (bbox_width / 2))
        startY = int(y_center - (bbox_height / 2))
        endX = int(startX + bbox_width)
        endY = int(startY + bbox_height)
        print(f"startX = {startX}, startY = {startY}")
        print(f"endX = {endX}, endY = {endY}\n")

        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 150, 0), 2)

        class_label = CLASS_NAMES[int(class_idx)]
        ((label_width, label_height), _) = cv2.getTextSize(
            class_label, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.75, thickness=2
        )

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

    cv2.imshow("image", img)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
