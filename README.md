# Coin Detection
A project to train an object detection model to detect Malaysia coins and the value of the coins.

The project starts from downloading images from Google Search using a script prepared based on [PyImageSearch tutorial](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/). The URLs are scraped using a Javascript script `scrape_image.js` copied from the tutorial. This script will download all the images in the Google image search webpage, therefore, search for the images you want in Google Search, scroll down until the number of images you want (or until the end), then paste all the code here into the Console in your browser (Chrome is recommended). A new file `urls.txt` containing all the image URLs will be downloaded to your computer.

Then, download the images using either `download_images_async.py` (asynchronously download, very fast) or `download_images_normal.py` (normal sequential download). Remember to change the configurations in `config.py` as necessary to make sure the script works as you wanted.

Then, the images will be labelled using [Label Studio](https://labelstud.io/) (**Currently labelling**).