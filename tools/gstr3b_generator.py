import pandas as pd

def generate(invoice_df, output_path):

    summary = pd.DataFrame([{
        "3.1(a) Taxable Value": invoice_df["taxable_value"].sum(),
        "CGST": invoice_df["cgst"].sum(),
        "SGST": invoice_df["sgst"].sum(),
        "IGST": invoice_df["igst"].sum()
    }])

    summary.to_csv(output_path, index=False)