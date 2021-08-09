import os
from imutils import paths

IMAGE_DIR = "C:/Users/user/Desktop/ANSON/Python Scripts/coin_detection/images"
ONE_RINGGIT_DIR = os.path.join(IMAGE_DIR, "1_ringgit_images")
ONE_CENT_DIR = os.path.join(IMAGE_DIR, "1_sen_images")


def rename_images(img_dir, start_name=None, dry_run=0):
    """Function to rename images from names of digits into names with a start_name"""
    assert start_name is not None
    for img in paths.list_images(img_dir):
        filename = img.split(os.path.sep)[-1]
        # take the last 3 digits for the new filename
        filename, file_ext = filename.split(".")
        file_num = filename[-3:]
        new_filename = f"{start_name}_{file_num}.{file_ext}"
        new_path = os.path.join(img_dir, new_filename)
        print(f"renaming {img} to {new_path}")
        if not dry_run:
            os.rename(img, new_path)


dry_run = 0
rename_images(ONE_RINGGIT_DIR, "one_ringgit", dry_run)
rename_images(ONE_CENT_DIR, "one_cent", dry_run)
