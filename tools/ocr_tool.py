import pytesseract
import cv2
import os

# Point pytesseract to the scoop installation path
tesseract_path = os.path.expanduser(r'~\scoop\apps\tesseract\current\tesseract.exe')
tessdata_path = os.path.expanduser(r'~\scoop\apps\tesseract\current\tessdata')

pytesseract.pytesseract.tesseract_cmd = tesseract_path
os.environ['TESSDATA_PREFIX'] = tessdata_path

def extract_text(image_path):
    try:
        # Tesseract accepts image arrays or paths. If path is given, load it.
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
            
        # Run Tesseract to get all text at once
        full_text = pytesseract.image_to_string(image, lang='eng')
        
        # Pytesseract doesn't easily return line-by-line confidences without verbose data format,
        # so we'll assign a default high confidence if it succeeds or fallback to generating it later.
        confidence_score = 0.90
        
        return full_text.upper(), confidence_score

    except Exception as e:
        print(f"Tesseract OCR Error: {e}")
        return "", 0