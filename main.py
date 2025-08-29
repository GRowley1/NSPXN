from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import re
import base64
import io
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("\u274c OPENAI_API_KEY environment variable is NOT set.")

app = FastAPI()

# Global request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        logger.debug(f"Received {request.method} {request.url.path} with headers={dict(request.headers)}")
        body = await request.body()
        logger.debug(f"Raw request body (hex): {body.hex()}")
        form = await request.form() if request.method in ["POST", "PUT"] else None
        if form:
            logger.debug(f"Raw form data: {dict(form)}")
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Middleware error: {str(e)} with body={body.decode('utf-8', errors='ignore') if 'body' in locals() else 'No body'}")
        return JSONResponse(status_code=400, content={"error": f"Invalid request: {str(e)}"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com",
        "https://www.nspxn.com",
        "http://nspxn.com",
        "http://www.nspxn.com",
        "https://nspxn.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img)
    img = ImageOps.invert(img)
    return img

def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        images = convert_from_bytes(file.read(), dpi=200)
        text_output = ""
        for i, img in enumerate(images, 1):
            processed = preprocess_image(img)
            try:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 3')
            except Exception as e:
                logger.warning(f"PSM 3 failed for page {i}: {str(e)}, retrying with PSM 6")
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6')
            if len(ocr_text.strip()) < 50 or re.search(r"[\:/\d\s]{50,}", ocr_text):
                logger.warning(f"Page {i} OCR output skipped (garbled): {ocr_text[:100]}...")
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return f"\n\u274c OCR error: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"Extracted DOCX text: {text[:500]}...")
    return text

def extract_text_from_images(image_files: List[UploadFile]) -> str:
    text_output = ""
    for i, img in enumerate(image_files, 1):
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr_text = pytesseract.image_to_string(processed, lang="eng")
            text_output += f"\n[Image {i}]\n{ocr_text}"
            logger.debug(f"Extracted text from image {i}: {ocr_text[:500]}...")
        except Exception as e:
            logger.error(f"Image {i} OCR error: {str(e)}")
    return text_output

@app.post("/vision-review")
async def vision_review(file_number: str = Form(...), ia_company: str = Form(...), appraiser_id: str = Form(...), estimate: UploadFile = File(...), image_files: List[UploadFile] = File(...)):
    logger.debug(f"Starting vision_review with headers={dict(request.headers)}")
    # Validate form fields with detailed logging
    logger.debug(f"Received form data: file_number='{file_number}', ia_company='{ia_company}', appraiser_id='{appraiser_id}'")
    if not all([field.strip() for field in [file_number, ia_company, appraiser_id]]):
        missing = [f for f in ['file_number', 'ia_company', 'appraiser_id'] if not f.strip()]
        logger.error(f"Validation failed: Empty fields - {', '.join(missing)}")
        return JSONResponse(status_code=422, content={"error": f"Missing or empty required fields: {', '.join(missing)}"})
    
    # Validate estimate file type
    logger.debug(f"Estimate file: filename='{estimate.filename}', content_type='{estimate.content_type}'")
    if not estimate.filename.lower().endswith(('.pdf', '.docx')):
        logger.error(f"Validation failed: Invalid estimate file type - {estimate.filename}")
        return JSONResponse(status_code=422, content={"error": f"Estimate must be a PDF or DOCX file, got {estimate.filename}"})
    
    # Process estimate
    estimate_text = extract_text_from_pdf(estimate) if estimate.filename.lower().endswith('.pdf') else extract_text_from_docx(estimate)
    if "OCR error" in estimate_text:
        logger.error(f"OCR error on estimate {estimate.filename}: {estimate_text}")
        return JSONResponse(status_code=422, content={"error": "Failed to extract text from estimate due to OCR error"})
    
    # Process images
    image_text = extract_text_from_images(image_files)
    if not image_text.strip():
        logger.warning("No valid text extracted from images")

    # Load client guidelines
    client_rules = extract_text_from_docx(open(os.path.join("client_rules", "SCA.docx"), 'rb')) if os.path.exists(os.path.join("client_rules", "SCA.docx")) else "No client guidelines available"
    logger.debug(f"Client rules: {client_rules[:500]}...")

    # Simple comparison
    comparison = "Comparison Report:\n"
    comparison += f"Estimate Text: {estimate_text[:500]}...\n"
    comparison += f"Image Text: {image_text[:500]}...\n"
    comparison += f"Client Guidelines: {client_rules[:500]}...\n"
    matches = set(re.findall(r'\w+', estimate_text.lower())) & set(re.findall(r'\w+', image_text.lower())) & set(re.findall(r'\w+', client_rules.lower()))
    if matches:
        comparison += f"Matching terms across estimate, images, and guidelines: {', '.join(matches)}\n"
    else:
        comparison += "No matching terms found.\n"

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=11)
    try:
        pdf.cell(200, 10, txt="NSPXN.com Comparison Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "Comparison Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, comparison)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)
    except Exception as pdf_e:
        logger.error(f"PDF generation error: {str(pdf_e)}")
        return JSONResponse(status_code=500, content={"error": f"PDF generation failed: {str(pdf_e)}", "comparison": comparison})

    # Send email
    msg = EmailMessage()
    msg["Subject"] = f"Comparison Report: {file_number}"
    msg["From"] = "noreply@nspxn.com"
    msg["To"] = "info@nspxn.com"
    email_body = f"""NSPXN.com Comparison Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Comparison Summary:
{comparison}
"""
    msg.set_content(email_body.encode("utf-8", errors="ignore").decode("utf-8"))
    with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
        try:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
        except Exception as email_e:
            logger.error(f"Email sending error: {str(email_e)}")

    return {
        "comparison": comparison,
        "file_number": file_number,
        "ia_company": ia_company,
        "appraiser_id": appraiser_id
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.debug(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)












