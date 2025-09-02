from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
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
from openai import OpenAI
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("\u274c OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

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
    img = img.convert("L")  # Convert to grayscale
    img = ImageEnhance.Contrast(img).enhance(2.0)  # Enhance contrast
    img = img.filter(ImageFilter.MedianFilter(size=3))  # Noise reduction
    img = ImageOps.autocontrast(img)  # Adaptive thresholding
    img = ImageOps.invert(img)  # Invert for better OCR
    return img

def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        images = convert_from_bytes(file.read(), dpi=200)  # Stable DPI
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

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...)
):
    logger.debug(f"Starting vision_review with raw form data: file_number={file_number}, ia_company={ia_company}, appraiser_id={appraiser_id}, client_rules={client_rules[:100]}...")
    # Validate form fields with detailed logging
    fields = [file_number, ia_company, appraiser_id, client_rules]
    field_names = ['file_number', 'ia_company', 'appraiser_id', 'client_rules']
    if not all(f.strip() for f in fields):
        empty_fields = [name for name, f in zip(field_names, fields) if not f.strip()]
        logger.error(f"Validation failed: Empty fields - {', '.join(empty_fields)}")
        return JSONResponse(status_code=422, content={"error": f"Missing or empty required fields: {', '.join(empty_fields)}"})

    # Process files
    estimate_text = ""
    image_files = []
    for file in files:
        content = await file.read()
        name = file.filename.lower()
        if name.endswith((".pdf", ".docx")) and not estimate_text:
            estimate_text = extract_text_from_pdf(io.BytesIO(content)) if name.endswith(".pdf") else extract_text_from_docx(io.BytesIO(content))
            if "OCR error" in estimate_text:
                logger.error(f"OCR error on estimate {file.filename}: {estimate_text}")
                return JSONResponse(status_code=422, content={"error": "Failed to extract text from estimate due to OCR error"})
        elif name.endswith((".jpg", ".jpeg", ".png")):
            image_files.append(file)

    if not estimate_text:
        logger.error("No valid estimate (PDF/DOCX) provided")
        return JSONResponse(status_code=422, content={"error": "No valid estimate (PDF or DOCX) provided"})

    image_text = extract_text_from_images(image_files)
    if not image_text.strip():
        logger.warning("No valid text extracted from images")

    # Prepare data for OpenAI
    vision_message = {"role": "user", "content": [
        {"type": "text", "text": f"Estimate Text:\n{estimate_text}\n\nImage Text:\n{image_text}\n\nClient Rules:\n{client_rules}"}
    ]}
    prompt = f"""
    You are an AI auto damage auditor tasked with comparing an estimate, photos, and client guidelines. Provide a full review.

    INSTRUCTIONS:
    - Analyze the provided 'Estimate Text' and 'Image Text' to identify key details (e.g., labor rates, tax rates, vehicle info, damage descriptions).
    - Compare these details against the 'Client Rules' to assess compliance and highlight matches or discrepancies.
    - Include a section titled 'Full Review' with:
      - A summary of extracted details from the estimate and photos.
      - A comparison against client rules, noting any deviations or missing elements.
      - Specific examples from the text where possible.
    - Do not apply deductions or scores unless explicitly required by the client rules.
    - Use only the provided text and images; do not assume additional data.

    At the top of your response, include:
    File Number: (from input)
    IA Company: (from input)
    Appraiser ID: (from input)

    Then provide the 'Full Review' section.
    """

    try:
        logger.debug(f"OpenAI request: prompt={prompt[:500]}..., vision_message={json.dumps(vision_message)}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=3500
        )
        logger.debug(f"OpenAI raw response: {json.dumps(response.dict(), default=str)[:1000]}...")
        gpt_output = response.choices[0].message.content if response.choices and response.choices[0].message and response.choices[0].message.content else "⚠️ No GPT output."
        logger.debug(f"OpenAI extracted content: {gpt_output[:1000]}...")
        if gpt_output.startswith("⚠️"):
            logger.error(f"OpenAI API returned no content: raw_response={json.dumps(response.dict(), default=str)}")
            return JSONResponse(status_code=500, content={"error": "OpenAI API returned no content", "review": gpt_output})

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com Comparison Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "Full Review:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)

        # Send email
        msg = EmailMessage()
        msg["Subject"] = f"Comparison Review Report: {file_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com Comparison Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Full Review:
{gpt_output}
"""
        msg.set_content(email_body.encode("utf-8", errors="ignore").decode("utf-8"))
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)

        return {
            "review": gpt_output,
            "file_number": file_number,
            "ia_company": ia_company,
            "appraiser_id": appraiser_id
        }

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e), "review": "⚠️ Review failed."})

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"
    file_name = f"{client_name}.docx"
    file_path = os.path.join(rules_dir, file_name)
    if os.path.exists(file_path):
        try:
            doc = Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            logger.debug(f"Client rules for {client_name}: {text[:500]}...")
            return {"text": text}
        except Exception as e:
            logger.error(f"Client rules error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        logger.error(f"Rules not found for client: {client_name}")
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})


















