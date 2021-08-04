import os
import requests
import time
import cv2
import numpy as np
import asyncio
from aiohttp import ClientSession
from imutils import paths

import config


async def fetch(url, session, total):
    """
    An asynchronous function to fetch and download images.
    Returns 0 for error, or returns 1 for successful download.
    """
    async with session.get(url) as response:
        if response.status != 200:
            # if the image link is not accessible, return 0 as error
            print(f"[ERROR] Error accessing {url}")
            return 0
    try:
        # read the image content
        r = await response.content.read()
    except:
        print(f"[ERROR] Error reading image from {url}")
        return 0
    # create the image filename based on the total count
    p = os.path.join(config.IMAGE_DIR, "{}.jpg".format(str(total).zfill(8)))
    # store the bytes as image
    with open(p, mode="wb") as f:
        f.write(r)
    print("[INFO] downloaded: {}".format(p))
    # return 1 as success
    return 1


async def fetch_with_sem(sem, url, session, total):
    async with sem:
        return await fetch(url, session, total)


async def async_download(urls, verbose=0):
    """An asynchronous function to download the images quickly"""
    # the tasks list to store the tasks for async scraping
    tasks = []
    # total count of images
    total = 0

    # limit the number of workers (semaphores) scraping at the same time
    # to reduce the load on their server
    sem = asyncio.Semaphore(10)
    async with ClientSession() as session:
        # loop through the urls extracted from urls.txt to store the async tasks
        for url in urls:
            if verbose:
                print(f"{url = }")
            tasks.append(asyncio.create_task(fetch_with_sem(sem, url, session, total)))
            # update the total image to use a the filename
            total += 1
        start_time = time.perf_counter()
        # gather all the tasks to run the async process
        # this will create a list of counts from 0 to n images
        download_success = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start_time
        # took only around 10 secs to finish downloading 99 images,
        # the non-async version took around 68 secs
        print(f"{total_time = :.4f} seconds")
        return download_success


if __name__ == "__main__":
    # the urls.txt file contains the urls for images downloaded from Google Search
    # the file is obtained using the method taught in PyImageSearch
    # https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
    with open("urls.txt") as f:
        urls = f.read().split("\n")

    # need to add this to avoid RuntimeError in Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # create the async coroutine to be run
    # also limiting the number of images to download
    download_coroutine = async_download(urls[: config.IMAGE_LIMIT], verbose=1)
    results = asyncio.run(download_coroutine)
    # because the results contain only 0 and 1 to indicate the success of download
    total_images = np.sum(results)
    print(f"Total images downloaded = {total_images}")

    # to delete unreadable images
    for image_path in paths.list_images(config.IMAGE_DIR):
        delete = False
        try:
            img = cv2.imread(image_path)
            if img is None:
                delete = True
        except:
            delete = True

        if delete:
            print(f"[INFO] deleting {image_path}")
            os.remove(image_path)
