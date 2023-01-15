#%%
import os, re
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

# OCR
import cv2
from pytesseract import pytesseract
import pypdfium2 as pdfium
from pypdfium2 import BitmapConv
from PIL.Image import Image

# NLP
import spacy
from spacy import displacy

from internals import LOG

#%% Constants
pytesseract.tesseract_cmd = R"C:\Program Files\Tesseract-OCR\tesseract.exe" # Path to tesseract installation 
FIXES = {
    "|": "I",
    "wes": "was",
    "dise": "disc",
    "daiiy": "daily",
    "!eg": "leg",
    "Gespite": "Despite",
    "Mr,": "Mr.",
    "Ms,": "Ms.",
    "$1": "S1",
    "$2": "S2",
    "$3": "S3",
    "$4": "S4",
}
#%% NLP model
NLP = spacy.load("en_core_web_lg")
#%%

@dataclass
class TextBlock:
    # Class for contours of text 
    area: int
    x: int
    y: int
    w: int
    h: int

def processConsult(pdf_path, p_start, p_end):
    img_root = os.path.splitext(pdf_path)[0]
    
    img = getPdfPages(pdf_path, p_start, p_end)
    
    extractTextBlock(img, F"{img_root}.txt")
    anonText(F"{img_root}.txt")


def getPdfPages(pdf_path, p_start, p_end):
    # Returns concatenated numpy image of pages 
    pdf = pdfium.PdfDocument(pdf_path)
    pages = []
    for i in range(p_start, p_end):
        page = pdf.get_page(i)
        img = page.render_to(BitmapConv.numpy_ndarray , scale=300/72) # convert directly to numpy
        pages.append(img[0]) # Image is stored as array where data is at index=0
    
    img_conc = cv2.vconcat(pages)
    
    return img_conc

def getTextBlocks(img_cv2, kernel_size = (3, 3), iterations = 7):
    # Uses dilation to build contours that define text blocks
    
    img2gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY_INV) # Inverse to get black background

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)  # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more
    dilated = cv2.dilate(mask, kernel, iterations=iterations) # Can tweak iterations to get optimal bounding box
    
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Retrieval modes https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71

    return contours


def extractTextBlock(img_cv2, out_path = ""):
    # Takes numpy image and finds largest text block bound within landmarks
    text_start = "summary paragraph"
    text_end = "thank you for involving"
    
    img_copy = img_cv2.copy()
    
    contours = getTextBlocks(img_cv2, (3, 3), 7)

    start_y = 0
    end_y = 0
    LOG.info(F"Analyzing contours...")
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour) # get rectangle bounding contour
        
        if w < 35 and h < 35: # Don't plot small false positives that aren't text
            continue
        
        # Send cropped image to OCR
        cropped = img_cv2[y:y+h, x:x+w]

        str_out: str = pytesseract.image_to_string(cropped)
        if text_start in str_out.lower():
            start_y = y
        if text_end in str_out.lower():
            end_y = y        

    if end_y == 0:
        print(F"No block found")
        return "None found"
    
    img_referral = img_copy[start_y:end_y,]
    
    contours = getTextBlocks(img_referral, (6, 6), 15)

    LOG.info(F"Extracting text block")
    
    text_blocks: list[TextBlock] = []
    
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour) # get rectangle bounding contour
        
        if w < 500 and h < 500: # Don't plot small false positives that aren't text
            continue
        
        text_blocks.append(TextBlock(w*h, x, y, w, h))
        # Send cropped image to OCR
    
    block = max(text_blocks, key=lambda x: x.area)

    cropped = img_referral[block.y:block.y+block.h, block.x:block.x+block.w]
    
    if out_path:
        extractText(cropped, out_path)        
        return None
        
    else:
        str_out = extractText(cropped)
        return str_out


def extractText(img_input, out_path = ""):
    if isinstance(img_input, (str, os.PathLike)):
        img_root = os.path.splitext(img_input)[0]
        img = cv2.imread(img_input, 0)
    elif isinstance(img_input, ndarray):
        img_root = ""
        img = img_input
    else:
        print(F"Invalid input")
        return None
    
        
    str_out: str = pytesseract.image_to_string(img)
    
    str_out = re.sub(R"\n([^\n])", lambda x: F" {x.group(1)}" or "", str_out) # Replace single lines
    str_out = re.sub(R"\n ", "\n", str_out) # Remove spaces for paragraphs
    
    mistake_count = 0
    for mistake in FIXES: # Implement fixes
        if mistake in str_out:
            str_out = str_out.replace(mistake, FIXES[mistake])
            mistake_count += 1
            
    str_out = "".join([c if ord(c) < 128 else "" for c in str_out]) # Replace non-ASCII
    
    LOG.info(F"{mistake_count} mistake patterns corrected")
    
    if out_path:
        with open(out_path, "w+") as file:
            file.write(str_out)
            LOG.info(F"Wrote file to {out_path}")
        
    elif img_root:
        text_out_path = F"{img_root}.txt"
        with open(text_out_path, "w+") as file:
            file.write(str_out)
            LOG.info(F"Wrote file to {text_out_path}")
            
    else:
        return str_out

def anonText(text_path):
    repl_name = "Smith"
    text_root = os.path.splitext(text_path)[0]
    
    with open(text_path, "r", encoding="utf8") as file:
        lines = file.readlines()
        text = "".join(lines)

    doc = NLP(text)
    
    text_processed = text # Copy str
    for ent in reversed(doc.ents): # Start at end so modifications don't shift frame
        if ent.label_ in ["PERSON"] and \
            not any(ex in ent.text.lower() for ex in ["guha", "achille", "hoffman"]): 
            text_processed = text_processed[:ent.start_char] + repl_name + text_processed[ent.end_char:]
    
    text_out_path = F"{text_root}_anon.txt"
    with open(text_out_path, "w+") as file:
        file.write(text_processed)
        LOG.info(F"Wrote file to {text_out_path}")
    


#%%
if 0:
    img_path = "data/ztest_consult2.png"
    img_path = "data/ztest_consult1.png"

    img_root = os.path.splitext(img_path)[0]
    extractText(img_path)
    anonText(F"{img_root}.txt")

    img_path = "data/ztest_radiology_low2.png"
    img_path = "data/ztest_radiology_low1.png"
    extractText(img_path)

#%%
dir_path = "data/"
for file in os.listdir(dir_path):
    if file.endswith("pdf"):
        print(file)
        processConsult(F"{dir_path}{file}", 3, 6)
        # if not any([subs in file for subs in ["1", "2", "3",]]): # For skipping certain files


#%%
processConsult(F"data/test_referral (4).pdf", 3, 6)