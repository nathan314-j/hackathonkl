# GST Invoice OCR and Tax Calculator

This project automates the extraction of invoice data (GSTIN, Invoice Number, Date, Total Amounts, and Tax Splits) from receipt images using Python, OpenCV, and Tesseract OCR. It also generates the corresponding GSTR-1, GSTR-3B, and GSTR-9 CSV outputs.

## Setup
1. Install Tesseract OCR on your system.
2. Install Python requirements: `pandas`, `opencv-python`, `pytesseract`.
3. Place your invoice images (JPG/PNG) into the `sample_invoices/` directory.

## Usage
Run the main agent to process the invoices and generate the CSV files:
```bash
python agent.py --process sample_invoices/
```
The generated `gstr1.csv`, `gstr3b.csv`, and `gstr9.csv` files will be saved in the `output/` directory.
