def calculate_tax_split(taxable_value, gst_amount=None):

    if not taxable_value:
        return 0.0, 0.0, 0.0

    if gst_amount:
        cgst = gst_amount / 2
        sgst = gst_amount / 2
        igst = 0.0
    else:
        cgst = taxable_value * 0.09
        sgst = taxable_value * 0.09
        igst = 0.0

    return round(cgst, 2), round(sgst, 2), round(igst, 2)