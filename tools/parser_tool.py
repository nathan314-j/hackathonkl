import re

GSTIN_PATTERN = r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b"
DATE_PATTERN = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
# Match numbers formatted like 1,000.00 or 1000.00 or 1000 or 1 000.00
AMOUNT_PATTERN = r"\b\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})\b"

def extract_number_from_line(line):
    # Find all matching amounts in the line
    matches = re.findall(AMOUNT_PATTERN, line)
    if not matches:
        return None
        
    # Pick the last match on the line (typically the actual value, not a quantity or rate)
    val_str = matches[-1]
    
    # Clean string: remove commas and spaces
    clean_val = val_str.replace(",", "").replace(" ", "")
    
    try:
        return float(clean_val)
    except ValueError:
        return None


def parse_fields(text):

    lines = text.split("\n")

    gstin = None
    date = None
    invoice_number = None
    subtotal = None
    gst_amount = None
    total = None

    for i, line in enumerate(lines):
        line_upper = line.upper()

        # GSTIN
        if not gstin:
            gstin_match = re.search(GSTIN_PATTERN, line_upper)
            if gstin_match:
                gstin = gstin_match.group()

        # Date
        if not date:
            date_match = re.search(DATE_PATTERN, line_upper)
            if date_match:
                date = date_match.group()

        # Invoice Number
        if not invoice_number:
            # First try: strict matching for common invoice prefixes
            inv_match = re.search(r"(?:INVOICE|INV|RECEIPT)[\s\.:]*(?:NO|#)?[\s\.:]*([A-Z0-9/-]+)", line_upper)
            if inv_match and len(inv_match.group(1)) > 3 and inv_match.group(1) not in ["OICE", "VOICE", "NO"]:
                invoice_number = inv_match.group(1)
            # Fallback: OCR sometimes breaks "INVOICE" into "OICE"
            elif "OICE" in line_upper or "INV" in line_upper:
                # Capture the first alphanumerical block after the split that looks like an ID
                fallback_match = re.search(r"(?:OICE|INV)[^\d]*([A-Z0-9/-]{4,})", line_upper)
                if fallback_match and fallback_match.group(1) not in ["OICE", "VOICE"]:
                    invoice_number = fallback_match.group(1)

        # Taxable Value / Subtotal
        if any(keyword in line_upper for keyword in ["SUBTOTAL", "TAXABLE", "SUB TOTAL"]):
            val = extract_number_from_line(line_upper)
            if val:
                subtotal = val

        # GST amount (CGST / SGST / IGST line)
        if "GST" in line_upper and "IN" not in line_upper: # avoid GSTIN line
            val = extract_number_from_line(line_upper)
            if val:
                # Accumulate tax amounts if there are multiple lines (CGST + SGST etc)
                if gst_amount is None:
                    gst_amount = val
                else:
                    gst_amount += val

        # Total
        if "TOTAL" in line_upper and "SUB" not in line_upper and "TAXABLE" not in line_upper:
            val = extract_number_from_line(line_upper)
            if val and (total is None or val > total): # Typically total is the largest value parsed as 'TOTAL'
                total = val

    # If subtotal missing but GST exists
    if total and gst_amount and not subtotal:
        subtotal = round(total - gst_amount, 2)

    return {
        "gstin": gstin,
        "invoice_number": invoice_number,
        "date": date,
        "taxable_value": subtotal if subtotal else total,
        "gst_amount": gst_amount,
        "total_value": total
    }