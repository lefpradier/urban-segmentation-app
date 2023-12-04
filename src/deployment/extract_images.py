import os
import cv2
from glob import glob

"""
Convert all images and masks to the input/output sizes in the model
This will allow an optimized storage in the deployed app
"""

for f in glob(
    "data/raw/P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/new_test/**/*.png",
    recursive=True,
):
    image = cv2.imread(f)
    image = cv2.resize(image, (256, 128))
    path = "deployment/frontend/static/images/"
    cv2.imwrite(path + os.path.basename(f), image)

for f in glob(
    "data/raw/P8_Cityscapes_gtFine_trainvaltest/gtFine/new_test/**/*labelIds.png",
    recursive=True,
):
    image = cv2.imread(f)
    image = cv2.resize(image, (256, 128))
    path = "deployment/frontend/static/masks/"
    cv2.imwrite(path + os.path.basename(f), image)
