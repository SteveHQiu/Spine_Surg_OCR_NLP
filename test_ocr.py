#%%
import os, re
from dataclasses import dataclass
#%%
# Internals 
from ocr import getPdfPages, extractText, extractTextBlock
from nlp_spacy import anonText
from pdf import processPDF
#%%

def processConsult(pdf_path, p_start, p_end):
    img_root = os.path.splitext(pdf_path)[0]
    
    img = getPdfPages(pdf_path, p_start, p_end)
    
    extractTextBlock(img, F"{img_root}.txt")
    anonText(F"{img_root}.txt")


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
processConsult(F"data/test_referral (1).pdf", 3, 6)