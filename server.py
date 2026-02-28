from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from tools.preprocessing_tool import preprocess
from tools.ocr_tool import extract_text
from tools.parser_tool import parse_fields
from tools.gst_engine import calculate_tax_split

app = FastAPI(title="Invoice OCR API")

# Enable CORS so the frontend website can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for hackathon purposes)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload-invoice")
async def upload_invoice(file: UploadFile = File(...)):
    try:
        # Read the uploaded image into memory
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image format."}

        # Run the OCR Pipeline
        processed_img = preprocess(image)
        text, confidence = extract_text(processed_img)
        fields = parse_fields(text)
        
        # Calculate Taxes
        cgst, sgst, igst = calculate_tax_split(fields["total_value"])

        # Build response
        response_data = {
            "success": True,
            "filename": file.filename,
            "extracted_data": {
                "gstin": fields["gstin"],
                "invoice_number": fields["invoice_number"],
                "date": fields["date"],
                "taxable_value": fields["total_value"],
                "cgst": cgst,
                "sgst": sgst,
                "igst": igst,
                "confidence": round(confidence, 3)
            }
        }

        return response_data

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import os
    # Set tesseract prefix just in case as we did in the terminal
    os.environ['TESSDATA_PREFIX'] = os.path.expanduser(r'~\scoop\apps\tesseract\current\tessdata')
    print("Starting FastAPI Server on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
