#%%
import os
import numpy as np
import cv2
from pytesseract import pytesseract

from internals import LOG

#%% Constants
pytesseract.tesseract_cmd = R"C:\Program Files\Tesseract-OCR\tesseract.exe" # Path to tesseract installation 


#%%
IMG_PATH = "data/ztest_consult1.png"
IMG_PATH = "data/ztest_radiology1.png"
IMG_ROOT = os.path.splitext(IMG_PATH)[0]
image = cv2.imread(IMG_PATH)
LOG.info(type(image))

str_out = pytesseract.image_to_string(image)
LOG.info(str_out)
#%%
with open(F"{IMG_ROOT}.txt", "w+") as file:
    file.write(str_out)
# %%
