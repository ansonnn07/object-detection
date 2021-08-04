import os
import requests
import time


with open("urls.txt") as f:
    urls = f.read().split("\n")
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
