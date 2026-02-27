import os
import json

CGST_RATE = 0.09
SGST_RATE = 0.09

def calculate_tax_from_entities(entities_folder):

    total_turnover = 0
    total_cgst = 0
    total_sgst = 0

    for file in os.listdir(entities_folder):

        if file.endswith(".json") or file.endswith(".txt"):

            file_path = os.path.join(entities_folder, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                total_value = float(data.get("total", 0))

                cgst = round(total_value * CGST_RATE, 2)
                sgst = round(total_value * SGST_RATE, 2)

                total_turnover += total_value
                total_cgst += cgst
                total_sgst += sgst

                print(f"\nFile: {file}")
                print("Total:", total_value)
                print("CGST:", cgst)
                print("SGST:", sgst)

            except Exception as e:
                print(f"Error reading {file}: {e}")

    print("\n===== SUMMARY =====")
    print("Total Turnover:", round(total_turnover, 2))
    print("Total CGST:", round(total_cgst, 2))
    print("Total SGST:", round(total_sgst, 2))
    print("Total Tax Paid:", round(total_cgst + total_sgst, 2))


if __name__ == "__main__":
    calculate_tax_from_entities("SROIE2019/train/entities")