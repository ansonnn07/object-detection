"""
USAGE:
python inference.py -i <path to input image or directory> -e <EXPORTED_MODEL PATH> -l <label_map.pbtxt file path>
python inference.py -w -e <EXPORTED_MODEL PATH> -l <label_map.pbtxt file path>

Example:
# for single image
python inference.py -i images/test_images/maksssksksss27.png -e Tensorflow/workspace/models/my_centernet_hg104_512x512_coco17_tpu-8/export -l "Tensorflow/workspace/models/my_centernet_hg104_512x512_coco17_tpu-8/export/label_map.pbtxt"

# for multiple images in "images/test_images" directory
python inference.py -i images/test_images -e Tensorflow/workspace/models/my_centernet_hg104_512x512_coco17_tpu-8/export -l "Tensorflow/workspace/models/my_centernet_hg104_512x512_coco17_tpu-8/export/label_map.pbtxt"

# for webcam
python inference.py -w -e Tensorflow/workspace/models/my_centernet_hg104_512x512_coco17_tpu-8/export -l "Tensorflow/workspace/models/my_centernet_hg104_512x512_coco17_tpu-8/export/label_map.pbtxt"

NOTE: There are some images available in the `images/test_images` folder that you may use to test face mask detection
Some warnings also may show up but it does not matter.
"""

import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import cv2
import argparse
import imutils

# Initiate argument parser
parser = argparse.ArgumentParser(
    description="Sample TensorFlow XML-to-TFRecord converter"
)
parser.add_argument(
    "-e",
    "--export_dir",
    help="path to the directory of exported model",
    type=str,
    required=True,
)
parser.add_argument(
    "-l",
    "--labels_path",
    help="path to the label_map.pbtxt file",
    type=str,
    required=True,
)
parser.add_argument(
    "-i",
    "--input_dir",
    help="Input path to a single image, or a directory with multiple images. DOES NOT support video file",
    type=str,
    default="",
)
parser.add_argument(
    "-w", "--webcam", help="whether to run on webcam or not", action="store_true"
)
parser.add_argument(
    "-t",
    "--threshold",
    help="the min confidence threshold to consider a detection as positive,"
    " lower this to obtain more detections",
    type=float,
    default=0.5,
)
parser.add_argument(
    "-o", "--output_path", help="path to output images", type=str, default="output"
)

args = vars(parser.parse_args())


if args["webcam"]:
    assert not args["input_dir"], "Using webcam does not need input directory."
else:
    assert args["input_dir"], "Please specify input directory."

if os.path.isdir(args["input_dir"]):
    print(f"[INFO] Running inference on multiple images ...")
else:
    if os.path.splitext(args["input_dir"])[1] in (".mp4", ".mkv", ".avi"):
        raise ValueError(
            "Sorry, this script does not support inferencing on a saved video, "
            "but you may try using your webcam by passing -w"
        )
    if not args["webcam"]:
        print(f"[INFO] Running inference on a single image ...")


if not os.path.exists(args["output_path"]):
    os.makedirs(args["output_path"])


print(f"{args['labels_path'] = }")

# Loading the exported model from the saved_model directory
# you may change this to any other path based on where you exported and stored the model
PATH_TO_SAVED_MODEL = os.path.join(args["export_dir"], "saved_model")

print("Loading model...", end="")
start_time = time.perf_counter()
# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.perf_counter()
print(f"Done! Took {end_time - start_time} seconds")

# LOAD LABEL MAP DATA
category_index = label_map_util.create_category_index_from_labelmap(
    args["labels_path"], use_display_name=True
)

from imutils.paths import list_images
import numpy as np


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array of shape (height, width, channels), where channels=3 for RGB to feed into tensorflow graph.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img = cv2.imread(path)
    # convert from OpenCV's BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)


def detect(image_np, verbose=0):
    start_t = time.perf_counter()
    # Running the infernce on the image specified in the  image path
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.expand_dims`.
    input_tensor = input_tensor[tf.newaxis, ...]
    # input_tensor = tf.expand_dims(input_tensor, False)

    # running detection using the loaded model
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy() for key, value in detections.items()
    }
    detections["num_detections"] = num_detections

    # detection_classes should be ints.
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    # print(detections['detection_classes'])
    end_t = time.perf_counter()
    if verbose:
        print(f"[INFO] Done inference. [{end_t - start_t:.2f} secs]")

    return detections


def draw_results(detections, image_np, min_score_thresh=0.6):
    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"],
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
    )
    return image_np_with_detections


if not args["webcam"] and os.path.isfile(args["input_dir"]):
    ## Inference for a single image

    # specifically select one image
    img_path = args["input_dir"]

    image_np = load_image_into_numpy_array(img_path)
    print(f"[INFO] Detecting from the image {img_path} ...")
    detections = detect(image_np, verbose=1)
    image_np_with_detections = draw_results(detections, image_np, args["threshold"])
    resized_result = imutils.resize(image_np_with_detections, height=500)
    cv2.imshow("object detection", resized_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif os.path.isdir(args["input_dir"]):
    from imutils.paths import list_images

    img_paths = list_images(args["input_dir"])
    # run inference from the images in img_paths
    print(
        "\n[INFO] Running inference on multiple images ... You may press 'q' key if you wish to exit\n"
    )
    for p in img_paths:
        image_np = load_image_into_numpy_array(p)
        print(f"[INFO] Detecting from the image {p} ...")
        detections = detect(image_np, verbose=1)
        image_np_with_detections = draw_results(detections, image_np, args["threshold"])

        resized_result = imutils.resize(image_np_with_detections, height=500)
        cv2.imshow("object detection", resized_result)
        key = cv2.waitKey(0)

        if key & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

elif args["webcam"]:
    cap = cv2.VideoCapture(0)
    print(
        "\n[INFO] Running inference on webcam ... You may press 'q' key if you wish to exit\n"
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert to numpy array
        image_np = np.asarray(frame)

        # run inference
        detections = detect(image_np)
        # draw results
        image_np_with_detections = draw_results(detections, image_np)

        cv2.imshow("object detection", image_np_with_detections)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
