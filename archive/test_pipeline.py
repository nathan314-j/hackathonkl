import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tools.preprocessing_tool import preprocess
from tools.ocr_tool import extract_text
from tools.parser_tool import parse_fields


def test_single_image(image_path):

    print(f"\nTesting image: {image_path}")

    # Step 1: Preprocess (Disabled to test raw Tesseract capabilities)
    # processed = preprocess(image_path)
    # import cv2
    # temp_path = "temp_test.jpg"
    # cv2.imwrite(temp_path, processed)

    # Step 2: OCR directly on raw image
    text, confidence = extract_text(image_path)

    print("\n--- RAW OCR TEXT ---")
    print(text)
    print("--------------------")

    # Step 3: Parse
    fields = parse_fields(text)

    print("\n--- EXTRACTED FIELDS ---")
    print("GSTIN:", fields["gstin"])
    print("Invoice Number:", fields["invoice_number"])
    print("Date:", fields["date"])
    print("Total:", fields["total_value"])
    print("Confidence:", round(confidence, 3))
    print("-------------------------")


if __name__ == "__main__":

    # Put exact image name here
    image_path = "sample_invoices/X51005268200.jpg"

    test_single_image(image_path)