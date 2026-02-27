def generate(invoice_df, output_path):

    if invoice_df.empty:
        print("No invoices found.")
        return

    gstr1 = invoice_df[[
        "filename", "gstin", "invoice_number", "date",
        "taxable_value", "cgst", "sgst", "igst"
    ]].copy()

    # Round numbers
    numeric_cols = ["taxable_value", "cgst", "sgst", "igst"]
    gstr1[numeric_cols] = gstr1[numeric_cols].round(2)

    gstr1.columns = [
        "Filename",
        "Recipient GSTIN",
        "Invoice Number",
        "Invoice Date",
        "Taxable Value",
        "CGST Amount",
        "SGST Amount",
        "IGST Amount"
    ]

    gstr1.to_csv(output_path, index=False)

    print(f"GSTR-1 generated → {output_path}")