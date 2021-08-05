# Coin Detection
A project to train an object detection model to detect Malaysia coins and the value of the coins.

The project starts from downloading images from Google Search using a script prepared based on [PyImageSearch tutorial](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/). You may refer to the `download_images_async.py` for the asynchronous download script or the normal download script from `download_images_normal.py`.

Then, the images will be labelled using [Label Studio](https://labelstud.io/) (**Currently labelling**).