import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        if text:
            pages.append({
                "page": page_num + 1,
                "text": text
            })
        else:
            # OCR fallback
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img)

            pages.append({
                "page": page_num + 1,
                "text": ocr_text
            })

    return pages
