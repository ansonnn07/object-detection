import csv
from cv2 import cv2
import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import sys
from collections import namedtuple


def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + "/*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)
        for member in root.findall("object"):
            bndbox = member.find("bndbox")
            value = (
                filename,
                width,
                height,
                member.find("name").text,
                int(bndbox.find("xmin").text),
                int(bndbox.find("ymin").text),
                int(bndbox.find("xmax").text),
                int(bndbox.find("ymax").text),
            )
            xml_list.append(value)
    column_name = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
    ]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


IMAGE_DIR = r"T:\New Download Folder\face-mask-detection\images"
XML_DIR = r"C:\Users\user\Desktop\ANSON\Python Scripts\coin_detection\Tensorflow\workspace\images\train"

df = xml_to_csv(XML_DIR)
print(df)
sys.exit(0)

cnt = 0
error_cnt = 0
error = False
total_classes = []
error_images = []

print(f"[INFO] Checking {len(df)} annotations ...")

for row in df.values:

    if error == True:
        error_cnt += 1
        error = False

    (filename, width, height, class_label, xmin, ymin, xmax, ymax) = row
    if class_label not in total_classes:
        total_classes.append(class_label)
    # print(f"[INFO] Checking {filename}")

    cnt += 1

    image_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(image_path):
        # maybe missing file extension
        image_path += ".jpg"
    if not os.path.exists(image_path):
        # or maybe png format
        image_path = image_path.replace(".jpg", ".png")
    img = cv2.imread(image_path)

    if img is None:
        error = True
        print("Could not read image", image_path)
        continue

    org_height, org_width = img.shape[:2]

    if org_width != width:
        error = True
        print("Width mismatch for image: ", filename, width, "!=", org_width)

    if org_height != height:
        error = True
        print("Height mismatch for image: ", filename, height, "!=", org_height)

    if xmin > org_width:
        error = True
        print(f"XMIN > org_width, {xmin} > {org_width} for file", filename)

    if xmin < 0:
        error = True
        print(f"XMIN < 0, {xmin} < 0 for file", filename)

    if xmax > org_width:
        error = True
        print(f"XMAX > org_width, {xmax} > {org_width} for file", filename)

    if ymin > org_height:
        error = True
        print(f"YMIN > org_height, {ymin} > {org_height} for file", filename)

    if ymin < 0:
        error = True
        print(f"YMIN < 0, {ymin} < 0 for file", filename)

    if ymax > org_height:
        error = True
        print(f"YMAX > org_height, {ymax} > {org_height} for file", filename)

    if xmin >= xmax:
        error = True
        print(f"xmin >= xmax, {xmin} >= {xmax} for file", filename)

    if ymin >= ymax:
        error = True
        print(f"ymin >= ymax, {ymin} >= {ymax} for file", filename)

    if error == True:
        print("Error for file: %s" % filename)
        error_images.append(filename)
        print()

print("Checked %d annotations and found %d errors" % (cnt, error_cnt))
error_images_str = "\n".join(error_images)
print(f"Error with {len(error_images)} files: \n{error_images_str}")
