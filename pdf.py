#%%
import os, re

from bs4 import BeautifulSoup

from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from langchain.docstore.document import Document
import langchain


#%%



#%%
def iter_strings(elem):
    # iterate strings so that they can be replaced
    iter = elem.strings
    n = next(iter, None)
    while n is not None:
        current = n
        n = next(iter, None)
        yield current

def replace_strings(element, substring, newstring):
    # replace all found `substring`'s with newstring
    for string in iter_strings(element):
        new_str = string.replace(substring, newstring)
        string.replace_with(new_str)

def processPDF(filepath):

    loader = PDFMinerPDFasHTMLLoader(filepath)
    data = loader.load()[0]   # entire pdf is loaded as a single Document

    soup = BeautifulSoup(data.page_content,'html.parser')
    content = soup.find_all('div')

    for div in soup.find_all('div'):
        replace_strings(div, "!", "")

    cur_fs = None
    cur_text = ''
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        sp = c.find('span')
        if not sp:
            continue
        st = sp.get('style')
        if not st:
            continue
        fs = re.findall('font-size:(\d+)px',st)
        if not fs:
            continue
        fs = int(fs[0])
        if not cur_fs:
            cur_fs = fs
        if fs == cur_fs:
            cur_text += c.text
        else:
            snippets.append((cur_text,cur_fs))
            cur_fs = fs
            cur_text = c.text
    snippets.append((cur_text,cur_fs))
    # Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
    # headers/footers in a PDF appear on multiple pages so if we find duplicatess safe to assume that it is redundant info)


    cur_idx = -1
    semantic_snippets = []
    # Assumption: headings have higher font size than their respective content
    for s in snippets:
        # if current snippet's font size > previous section's heading => it is a new heading
        if not semantic_snippets or s[1] > semantic_snippets[cur_idx].metadata['heading_font']:
            metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
            metadata.update(data.metadata)
            semantic_snippets.append(Document(page_content='',metadata=metadata))
            cur_idx += 1
            continue
        
        # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
        # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
        if not semantic_snippets[cur_idx].metadata['content_font'] or s[1] <= semantic_snippets[cur_idx].metadata['content_font']:
            semantic_snippets[cur_idx].page_content += s[0]
            semantic_snippets[cur_idx].metadata['content_font'] = max(s[1], semantic_snippets[cur_idx].metadata['content_font'])
            continue
        
        # if current snippet's font size > previous section's content but less than previous section's heading than also make a new 
        # section (e.g. title of a pdf will have the highest font size but we don't want it to subsume all sections)
        metadata={'heading':s[0], 'content_font': 0, 'heading_font': s[1]}
        metadata.update(data.metadata)
        semantic_snippets.append(Document(page_content='',metadata=metadata))
        cur_idx += 1


    real_cont = []
    sum_par_obj = None
    end_par_obj = None

    for c in content:
        page_no = re.findall(R"Page \d", c.text)
        pt_name = re.findall(R"Patient Name", c.text)
        mrn = re.findall(R"MRN No.", c.text)
        printed = re.findall(R"Printed (On)|(by)", c.text)
        sum_par = re.findall(R"Summarizing Paragraph:", c.text)
        sum_par = re.findall(R"(Mr.)|(Mrs.)|(Ms.) [A-Z][a-z]+", c.text)
        end_par = re.findall(R"Thank you for involving me", c.text)
        
        if not any([page_no, pt_name, mrn, printed]):
            if sum_par and not sum_par_obj: # Only populate it if it doesn't exist
                sum_par_obj = c
                
            if end_par:
                end_par_obj = c
            # print("---------")
            # print(F"[{c.sourceline} - {c.sourcepos}]")
            # print(c.text)
            real_cont.append(c)

    filt_cont = []
    for c in real_cont:
        startline = sum_par_obj.sourceline 
        endline = end_par_obj.sourceline 
        line = c.sourceline
        
        if line >= startline and line <= endline:
            # print("---------")
            # print(F"[{c.sourceline} - {c.sourcepos}]")
            # print(c.text)
            filt_cont.append(c)
    
    text_out = "".join([t.text for t in filt_cont])
    
    return text_out

#%%
if __name__ == "__main__":
    from nlp_spacy import anonText
    text = processPDF("data2/75_referral.pdf")
    text_anon = anonText(text)
    print(text_anon)
