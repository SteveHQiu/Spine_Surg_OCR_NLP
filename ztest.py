#%% PDF conversion and retrieval 
import pypdfium2 as pdfium
from pypdfium2 import BitmapConv

from PIL.Image import Image

pdf = pdfium.PdfDocument("data/test_referral (4).pdf")
page = pdf.get_page(3)
img = page.render_to(BitmapConv.numpy_ndarray, scale=300/72) # convert directly to numpy
img: Image = page.render_to(BitmapConv.pil_image , scale=300/72) # scale = 300/72 ensures 300dpi: https://pypi.org/project/pypdfium2/
img.show()


#%% 
pages = []
for i in range(3, 6):
    page = pdf.get_page(i)
    img = page.render_to(BitmapConv.numpy_ndarray , scale=300/72) # convert directly to numpy
    pages.append(img[0]) # Image is stored as array where data is at index=0
    
img_conc = cv2.vconcat(pages)

if 0:
    img_conc = cv2.resize(img_conc, None, fx=0.2, fy=0.2) # Zoom out to see image

cv2.imshow('captcha_result', img_conc) # Image is stored as array where data is at index=0
cv2.waitKey()
cv2.destroyAllWindows()


#%% Extract text blocks
from pytesseract import pytesseract
import cv2
import numpy as np
pytesseract.tesseract_cmd = R"C:\Program Files\Tesseract-OCR\tesseract.exe" # Path to tesseract installation 

# img_cv2 = np.array(img) # Using PIL image converted from PDF
img_cv2 = img_conc.copy()


img2gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY_INV) # Inverse to get black background

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
dilated = cv2.dilate(mask, kernel, iterations=7) # Can tweak iterations to get optimal bounding box

if 0:
    cv2.imshow('captcha_result', dilated)
    cv2.waitKey()
    cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Retrieval modes https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71


start_y = 0
end_y = 0
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour) # get rectangle bounding contour
    
    if w < 35 and h < 35: # Don't plot small false positives that aren't text
        continue
    
    cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (255, 0, 255), 2) # draw rectangle around contour on original image

    if 1: # Send cropped image to OCR
        cropped = img_cv2[y:y+h, x:x+w]
        
        if 0: # Resize if needed
            new_dim = (cropped.shape[1]*2, cropped.shape[0]*2) # Width, height
            cropped = cv2.resize(cropped, new_dim, interpolation=cv2.INTER_AREA)
        
        str_out: str = pytesseract.image_to_string(cropped)
        if "summary paragraph" in str_out.lower():
            start_y = y
        if "thank you for involving" in str_out.lower():
            end_y = y
        print(str_out)



print(start_y, end_y)


cv2.imshow('captcha_result', img_cv2)
cv2.waitKey()
cv2.destroyAllWindows()

#%%
img_cv2 = img_conc.copy()
img_referral = img_cv2[start_y:end_y,]

img2gray = cv2.cvtColor(img_referral, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY_INV) # Inverse to get black background

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (6, 6))  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
dilated = cv2.dilate(mask, kernel, iterations=10) # Can tweak iterations to get optimal bounding box

cv2.imshow('captcha_result', dilated)
cv2.waitKey()
cv2.destroyAllWindows()


contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Retrieval modes https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour) # get rectangle bounding contour
    
    if w < 500 and h < 500: # Don't plot small false positives that aren't text
        continue
    
    cv2.rectangle(img_referral, (x, y), (x + w, y + h), (255, 0, 255), 2) # draw rectangle around contour on original image

    if 1: # Send cropped image to OCR
        cropped = img_referral[y:y+h, x:x+w]
        
        str_out: str = pytesseract.image_to_string(cropped)

        print(str_out)
        
cv2.imshow('captcha_result', img_referral)
cv2.waitKey()
cv2.destroyAllWindows()


#%% Label detection
from pytesseract import pytesseract
import cv2
import numpy as np
from matplotlib.colors import to_rgb

def visLabels(img: Image):
    colors = {1: [255*i for i in to_rgb("red")],
            2: [255*i for i in to_rgb("yellow")],
            3: [255*i for i in to_rgb("green")],
            4: [255*i for i in to_rgb("blue")],
            5: [255*i for i in to_rgb("purple")]}

    img_cv2 = np.array(img)

    data = pytesseract.image_to_data(img_cv2, output_type="dict")
    n_boxes = len(data["level"])
    for i in range(n_boxes):
        x, y, w, h = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
        box_level = data["level"][i]
        box_text = data["text"][i]
        confidence = data["conf"][i]
        cv2.rectangle(img_cv2, (x, y), (x + w, y + h), colors[box_level], 1)
        if box_level == 1: # Only label this level of text
            cv2.putText(img_cv2, F"{confidence}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, colors[box_level], 1)

    cv2.imshow("Image", img_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#%% Anonymization
import re

with open("data/ztest_consult1_fixed.txt", "r", encoding="utf8") as file:
    lines = file.readlines()
    text = "".join(lines)
    text = re.sub(R"\n([^\n])", lambda x: F" {x.group(1)}" or "", text) # Replace single lines
    text = re.sub(R"\n ", "\n", text) # Remove spaces for paragraphs
    print(text)

#%%
import spacy
from spacy import displacy

nlp_model_name = "en_core_sci_lg"
nlp_model_name = "en_core_sci_scibert"
nlp_model_name = "en_core_web_lg"
NLP = spacy.load(nlp_model_name)
doc = NLP(text)

repl_name = "Smith"
text_processed = text # Copy str
for ent in reversed(doc.ents): # Start at end so modifications don't shift frame
    if ent.label_ in ["PERSON"] and \
        not any(ex in ent.text.lower() for ex in ["guha", "achille", "hoffman"]): 
        text_processed = text_processed[:ent.start_char] + repl_name + text_processed[ent.end_char:]
print(text_processed)

#%%

if 1:
    for sent in doc.sents:
        displacy.render(sent, style="ent", options={"compact":True, "distance":100})


#%% OCR 
import os

import cv2
import numpy as np
from pytesseract import pytesseract
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from textblob import TextBlob


pytesseract.tesseract_cmd = R"C:\Program Files\Tesseract-OCR\tesseract.exe" # Path to tesseract installation 

img_path = "data/ztest_radiology_low1.png"
img_path = "data/ztest_consult1.png"
img_root = os.path.splitext(img_path)[0]

img = cv2.imread(img_path, 0)

if 0: # Thresholding techniques 
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.erode(img, np.ones((2,2)), iterations=1)
    img = cv2.dilate(img, np.ones((2,2)), iterations=1)


    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Alternate way to show image via matplotlib
    plt.figure(figsize = (10, 10))
    plt.imshow(img, aspect="auto")
    plt.show()
    
str_out: str = pytesseract.image_to_string(img)
print(str_out)


# Autocorrect
tb = TextBlob(str_out)
str_out_corr = tb.correct()
print(str_out_corr)
# Won't implment since it miscorrects certain medical terms (e.g., spasticity=elasticity, clonus=clouds)
#%%
