import pandas as pd

def generate(invoice_df, output_path):

    annual_turnover = round(invoice_df["taxable_value"].sum(), 2)

    total_tax_paid = round(
        invoice_df["cgst"].sum() +
        invoice_df["sgst"].sum() +
        invoice_df["igst"].sum(),
        2
    )

    annual = pd.DataFrame([{
        "Annual Turnover": annual_turnover,
        "Total Tax Paid": total_tax_paid
    }])

    annual.to_csv(output_path, index=False)