from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import os
import io
import base64
import json
import logging
import datetime

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from openai import OpenAI

# =========================================
# Setup
# =========================================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY not set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# Helpers
# =========================================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    return img

def ocr_text(img: Image.Image, psm: int = 6) -> str:
    try:
        proc = preprocess_image(img)
        config = f"--psm {psm} --oem 3"
        return pytesseract.image_to_string(proc, lang="eng", config=config)
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""

def extract_text_from_pdf_embedded(pdf_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(parts)
    except Exception as e:
        logger.debug(f"Embedded text failed: {e}")
        return ""

def extract_from_estimate(pdf_bytes: bytes, dpi: int = 300) -> Dict[str, Any]:
    extracted = {
        "claim_number": None,
        "vin": None,
        "year": None,
        "make": None,
        "model": None,
        "labor_rate": None,
        "tax_rate": None,
        "mileage": None,
        "estimate_items": [],
        "estimate_text": ""
    }
    try:
        embedded = extract_text_from_pdf_embedded(pdf_bytes)
        if embedded.strip():
            text = embedded
        else:
            pages = convert_from_bytes(pdf_bytes, dpi=dpi)
            text = '\n'.join(ocr_text(p, psm=4) for p in pages)
        extracted["estimate_text"] = text
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            max_tokens=800,
            messages=[
                {"role": "system", "content": 'Extract from this estimate text as JSON: {"claim_number": str or null, "vin": 17-char str or null, "year": int or null, "make": str or null, "model": str or null, "labor_rate": "$XX.XX /hr" or null, "tax_rate": "X.XXXX%" or null, "mileage": int or null, "estimate_items": list of {"line": str, "oper": str, "description": str, "part_number": str or null, "qty": int or null, "price": float or null, "labor": float or null, "paint": float or null, "type": "OEM" or "A/M" or "USED" or "RECOND" or "OTHER"}}. Be accurate, null if not found. For vehicle, look for the line describing the vehicle to extract year, make, model (include trim and other details in model). Parse the table for items, including indicators like ** for A/M or USED.'},
                {"role": "user", "content": text}
            ]
        )
        try:
            data = json.loads(response.choices[0].message.content)
            extracted.update(data)
        except Exception as e:
            logger.warning(f"JSON parse error: {e}")
    except Exception as e:
        logger.error(f"Estimate extraction error: {e}")
    return extracted

def extract_vin_from_photo(photo_bytes: bytes) -> Optional[str]:
    try:
        buf = io.BytesIO()
        Image.open(io.BytesIO(photo_bytes)).convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Extract the 17-character VIN from this image if it is a VIN plate or label photo. Output only the VIN or null if not a VIN plate or unclear."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract VIN."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
        )
        vin = response.choices[0].message.content.strip()
        if len(vin) == 17 and vin.isalnum():
            return vin.upper()
        return None
    except Exception as e:
        logger.warning(f"VIN photo extraction error: {e}")
        return None

def compare_estimate_to_photos(estimate_items: List[Dict], photos: List[bytes]) -> Dict:
    images_for_vision = []
    for b in photos:
        buf = io.BytesIO()
        try:
            Image.open(io.BytesIO(b)).convert("RGB").save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            images_for_vision.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        except:
            pass
    if not images_for_vision:
        return {"overall": "No photos provided.", "per_item": [], "missing_in_estimate": [], "not_in_photos": []}
    prompt = f"Review the estimate items against the damage photos for consistency. For each item, determine if the described damage is visible in the photos. Provide confidence level. List any additional damages in photos not covered in the estimate, and any items in estimate not visible in photos.\nEstimate Items: {json.dumps(estimate_items)}"
    messages = [
        {"role": "system", "content": "Output JSON: {'per_item': list of {'description': str, 'visible': bool, 'confidence': float 0-1, 'note': str}, 'not_in_photos': list str, 'missing_in_estimate': list str, 'overall': str summary of the review}"},
        {"role": "user", "content": [{"type": "text", "text": prompt}] + images_for_vision}
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Comparison error: {e}")
        return {"overall": "Review failed."}

def compare_to_guidelines(estimate_text: str, estimate_items: List[Dict], guidelines_text: str) -> str:
    if not guidelines_text:
        return "No client guidelines provided."
    prompt = f"Review the estimate against these client guidelines. Check for compliance in parts, labor, taxes, documentation, etc. Provide a detailed markdown summary, highlighting any non-compliance.\nGuidelines: {guidelines_text}\n\nEstimate Text: {estimate_text}\nItems: {json.dumps(estimate_items)}"
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Output a detailed review in markdown format."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# =========================================
# Route
# =========================================
@app.post("/vision-review")
async def vision_review(
    estimate: UploadFile = File(...),
    photos: List[UploadFile] = File(None),
    guidelines: UploadFile = File(None),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...)
):
    estimate_bytes = await estimate.read()
    estimate_data = extract_from_estimate(estimate_bytes)

    photo_bytes_list = []
    vin_photo = None
    if photos:
        for p in photos:
            pb = await p.read()
            photo_bytes_list.append(pb)
            if not vin_photo:
                possible_vin = extract_vin_from_photo(pb)
                if possible_vin:
                    vin_photo = possible_vin

    guidelines_text = ""
    if guidelines:
        g_bytes = await guidelines.read()
        if guidelines.filename.endswith(".docx"):
            guidelines_text = "\n".join(p.text for p in Document(io.BytesIO(g_bytes)).paragraphs if p.text.strip())
        elif guidelines.filename.endswith(".pdf"):
            guidelines_text = extract_text_from_pdf_embedded(g_bytes)
        elif guidelines.filename.endswith(".txt"):
            guidelines_text = g_bytes.decode("utf-8", errors="ignore")

    vin_est = estimate_data.get("vin")
    vin_verification = "VIN unavailable"
    if vin_est and vin_photo:
        vin_verification = "Match" if vin_est == vin_photo else f"No Match (photo: {vin_photo})"
    elif vin_est and not vin_photo:
        vin_verification = "VIN photo not found"
    elif not vin_est and vin_photo:
        vin_verification = "VIN not found in estimate"

    labor_present = bool(estimate_data.get("labor_rate"))
    tax_present = bool(estimate_data.get("tax_rate"))

    consistency = compare_estimate_to_photos(estimate_data.get("estimate_items", []), photo_bytes_list)

    guidelines_review = compare_to_guidelines(estimate_data["estimate_text"], estimate_data.get("estimate_items", []), guidelines_text)

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="AI Review Report", ln=1, align="C")
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, f"File Number: {file_number}")
    pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 10, f"Appraiser ID: {appraiser_id}")
    pdf.multi_cell(0, 10, f"Claim #: {estimate_data.get('claim_number', 'N/A')}")
    pdf.multi_cell(0, 10, f"VIN: {vin_est or 'N/A'}")
    pdf.multi_cell(0, 10, f"VIN Verification: {vin_verification}")
    pdf.multi_cell(0, 10, f"Vehicle: {estimate_data.get('year', 'N/A')} {estimate_data.get('make', 'N/A')} {estimate_data.get('model', 'N/A')}, Mileage: {estimate_data.get('mileage', 'N/A')}")
    pdf.multi_cell(0, 10, f"Labor Rates: {'Present' if labor_present else 'Missing'} ({estimate_data.get('labor_rate', 'N/A')})")
    pdf.multi_cell(0, 10, f"Tax Rate: {'Present' if tax_present else 'Missing'} ({estimate_data.get('tax_rate', 'N/A')})")
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Estimate vs Photos Review:")
    pdf.multi_cell(0, 10, json.dumps(consistency, indent=2))
    pdf.ln(10)
    pdf.multi_cell(0, 10, "Estimate vs Guidelines Review:")
    pdf.multi_cell(0, 10, guidelines_review)

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    pdf.output(pdf_path)

    # Email
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI Review: {estimate_data.get('claim_number', 'N/A')}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"Report for file {file_number}.")
        with open(pdf_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename=f"{file_number}.pdf")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error: {e}")

    return {"message": "Review complete", "pdf_path": f"/download-pdf?file_number={file_number}"}

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})























