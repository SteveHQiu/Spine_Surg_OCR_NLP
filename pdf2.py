#%%
import fitz  # PyMuPDF

def extract_text_between_phrases(pdf_path, start_phrase, end_phrase):
    doc = fitz.open(pdf_path)
    text_between_phrases = []

    start_found = False
    end_found = False

    for page_num in range(doc.page_count):
        page = doc[page_num]
        words = page.get_text("words")

        text = ""
        y_pos = None

        for word in words:
            word_text = word[4]  # Word text
            word_y = word[3]     # Y position

            if start_phrase in word_text:
                start_found = True

            if start_found and end_phrase in word_text:
                end_found = True

            if start_found and not end_found:
                if y_pos is None:
                    y_pos = word_y
                elif word_y != y_pos:  # Different Y position, start new line
                    text_between_phrases.append((text.strip(), y_pos))
                    text = word_text
                    y_pos = word_y
                else:
                    text += " " + word_text

        if end_found:
            break

    doc.close()
    return text_between_phrases


#%%
pdf_path = "data2/75_referral.pdf"
start_phrase = "Summarizing Paragraph:"
end_phrase = "Thank you for involving me"

result = extract_text_between_phrases(pdf_path, start_phrase, end_phrase)
for text, y_pos in result:
    print(f"Text: {text}\nY Position: {y_pos}\n")


# %%
