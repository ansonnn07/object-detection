# A script created to scrape images directly from Google Search link but do not work well

import os
import cv2
import io
from PIL import Image
import numpy as np
import requests
import urllib.parse
from bs4 import BeautifulSoup


query = "dog / cat"
encoded_query = urllib.parse.quote_plus(query, safe="")
GOOGLE_URL = "https://www.google.com/search?q={}&tbm=isch&ved=2ahUKEwiT_MvQx5byAhXloUsFHYnHAcoQ2-cCegQIABAA&oq={}&gs_lcp=CgNpbWcQDFAAWABg3gJoAHAAeACAAQCIAQCSAQCYAQCqAQtnd3Mtd2l6LWltZw&sclient=img&ei=fxsKYdPAEeXDrtoPiY-H0Aw&rlz=1C1CHBD_enMY896MY896#imgrc=204LlQFXnp1OEM"
# to encode the string back to URL supported query string
# https://www.urlencoder.io/python/
IMAGES_URL = GOOGLE_URL.format(encoded_query, encoded_query)

# The User-Agent request header contains a characteristic string
# that allows the network protocol peers to identify the application type,
# operating system, and software version of the requesting software user agent.
# needed for google search
user_agent = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "none",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}  # write: 'my user agent' in browser to get your browser user agent details

print(f"[INFO] Searching for '{query}'")
response = requests.get(IMAGES_URL, headers=user_agent)
html = response.text

soup = BeautifulSoup(html, "lxml")
img_tags = soup.findAll("img", attrs={"class": "rg_i"})
print(f"Found {len(img_tags)} images")
img_count = 0

for tag in img_tags:
    img_link = tag.get("src")
    try:
        response = requests.get(img_link)
    except:
        print(img_link)
        print("[ERROR] Cannot get request from the link")
        continue
    if response.status_code != 200:
        continue
    # print(img_link)
    image_bytes = io.BytesIO(response.content)
    # img = Image.open(image_bytes)
    # binary data stream to np.ndarray [np.uint8: 8-bit pixel]
    img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    # # Convert bgr to rbg
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    img_count += 1
    key = cv2.waitKey(0)

    if key == 27:
        # if "ESC" is pressed
        cv2.destroyAllWindows()
        break

    cv2.destroyAllWindows()

    # img = cv2.

print(f"Total scraped image = {img_count}")
