import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix OpenMP error

import argparse
import pandas as pd
import config

from tools.preprocessing_tool import preprocess
from tools.ocr_tool import extract_text
from tools.parser_tool import parse_fields
from tools.gst_engine import calculate_tax_split
from tools.gstr1_generator import generate as generate_gstr1
from tools.gstr3b_generator import generate as generate_gstr3b
from tools.gstr9_generator import generate as generate_gstr9


class GSTAgent:

    def __init__(self):
        self.invoice_store = pd.DataFrame()

    def process_folder(self, folder_path):

        if not os.path.exists(folder_path):
            print("Folder does not exist.")
            return

        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                image_path = os.path.join(folder_path, file)
                print(f"\n[Agent] Processing: {file}")

                try:
                    # Step 1: Preprocess
                    processed_img = preprocess(image_path)
                    
                    # Step 2: OCR Extraction
                    text, confidence = extract_text(processed_img)

                    # Step 3: Parse fields
                    fields = parse_fields(text)

                    # Step 4: GST logic
                    
                    cgst, sgst, igst = calculate_tax_split(
                        fields["taxable_value"],
                        fields.get("gst_amount")
                        )

                    # Step 5: Store invoice
                    invoice_data = {
                        "filename": file,
                        **fields,
                        "cgst": cgst,
                        "sgst": sgst,
                        "igst": igst,
                        "confidence": confidence
                    }

                    self.invoice_store = pd.concat(
                        [self.invoice_store, pd.DataFrame([invoice_data])],
                        ignore_index=True
                    )

                    print("Date:", fields["date"])
                    print("Total:", fields["total_value"])
                    print("Confidence:", round(confidence, 3))

                except Exception as e:
                    print(f"Error processing {file}: {e}")

        print("\n[Agent] Processing complete.")

    def generate_returns(self):

        if self.invoice_store.empty:
            print("No invoices processed.")
            return

        os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

        generate_gstr1(self.invoice_store, f"{config.OUTPUT_FOLDER}/gstr1.csv")
        generate_gstr3b(self.invoice_store, f"{config.OUTPUT_FOLDER}/gstr3b.csv")
        generate_gstr9(self.invoice_store, f"{config.OUTPUT_FOLDER}/gstr9.csv")

        print("\n[Agent] All returns generated in /output folder.")


# =========================
# CLI ENTRY
# =========================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--process", help="Folder path of invoices")

    args = parser.parse_args()

    agent = GSTAgent()

    if args.process:
        agent.process_folder(args.process)
        agent.generate_returns()
    else:
        print("Please provide --process <folder_path>")