import pandas as pd

CGST_RATE = 0.09
SGST_RATE = 0.09

def validate_tax(csv_path):

    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV is empty.")
        return

    print("\n--- TAX VALIDATION REPORT ---")

    total_turnover = 0
    total_calculated_tax = 0
    total_recorded_tax = 0

    for index, row in df.iterrows():

        taxable = float(row["Taxable Value"])

        calculated_cgst = round(taxable * CGST_RATE, 2)
        calculated_sgst = round(taxable * SGST_RATE, 2)

        recorded_cgst = float(row["CGST Amount"])
        recorded_sgst = float(row["SGST Amount"])

        total_turnover += taxable
        total_calculated_tax += (calculated_cgst + calculated_sgst)
        total_recorded_tax += (recorded_cgst + recorded_sgst)

        print(f"\nInvoice {row['Invoice Number']}")
        print("Taxable:", taxable)
        print("Calculated CGST:", calculated_cgst, "| Recorded:", recorded_cgst)
        print("Calculated SGST:", calculated_sgst, "| Recorded:", recorded_sgst)

    print("\n--- SUMMARY ---")
    print("Total Turnover:", round(total_turnover, 2))
    print("Total Calculated Tax:", round(total_calculated_tax, 2))
    print("Total Recorded Tax:", round(total_recorded_tax, 2))
    print("----------------------------")


if __name__ == "__main__":
    validate_tax("output/gstr1.csv")