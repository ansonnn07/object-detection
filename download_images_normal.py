import os
import sys
import requests
import time

import config

# this text file contains the urls for images downloaded from Google Search
# the file is obtained using the method taught in PyImageSearch
# https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
with open(config.URL_FILE) as f:
    urls = f.read().split("\n")

if not os.path.exists(config.IMAGE_DIR):
    # create the image folder if not exists
    os.mkdir(config.IMAGE_DIR)
else:
    if os.listdir(config.IMAGE_DIR):
        while True:
            user_input = input(
                f"Files are found in {config.IMAGE_DIR}, are you sure you want to overwrite them? (yes | no) "
            )
            if user_input in ("no", "n"):
                sys.exit(0)
            elif user_input in ("yes", "y"):
                break
            else:
                print("Please provide a valid input.")

total = 0
start = time.perf_counter()
for url in urls:
    try:
        # try to download the image
        r = requests.get(url, timeout=60)
        # save the image to disk
        p = os.path.sep.join(["test_images", "{}.jpg".format(str(total).zfill(8))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()
        # update the counter
        print("[INFO] downloaded: {}".format(p))
        total += 1
    # handle if any exceptions are thrown during the download process
    except:
        print("[INFO] error downloading {}...skipping".format(p))
end = time.perf_counter()
# around 68 secs
print(f"[INFO] Total time elapsed: {end - start}")
