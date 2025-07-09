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
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from openai import OpenAI
import logging

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
        images = convert_from_bytes(file.read(), dpi=200)  # Increased DPI for better clarity
        text_output = ""
        for i, img in enumerate(images, 1):
            processed = preprocess_image(img)
            ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6')
            text_output += f"\n[Page {i}]\n{ocr_text}"
            logger.debug(f"OCR output for page {i}: {ocr_text[:500]}...")  # Log first 500 chars
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return f"\n\u274c OCR error during combined extraction: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"Extracted DOCX text: {text[:500]}...")
    return text

def extract_field(label, text) -> str:
    # More flexible regex to handle OCR artifacts (e.g., spaces, special chars)
    pattern = re.compile(rf"{label}\s*[:\-#=]?\s*([^\n\r;]+)", re.IGNORECASE)
    match = pattern.search(text)
    result = match.group(1).strip() if match else "N/A"
    logger.debug(f"Extracted field '{label}': {result}")
    return result

def advisor_report_present(texts: List[str], image_files: List[UploadFile]) -> bool:
    for t in texts:
        if any(term in t.lower() for term in ["ccc advisor report", "advisor report"]):
            logger.debug("Advisor report found in text")
            return True
    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            if "advisor report" in ocr.lower():
                logger.debug("Advisor report found in image OCR")
                return True
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            continue
    return False

def check_required_photos(image_files: List[UploadFile]) -> List[str]:
    required_photos = ["four corners", "odometer", "vin", "license plate"]
    found_photos = []
    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang="eng").lower()
            for photo_type in required_photos:
                if photo_type in ocr or any(keyword in ocr for keyword in photo_type.split()):
                    found_photos.append(photo_type)
                    logger.debug(f"Found photo type: {photo_type}")
        except Exception as e:
            logger.error(f"Photo validation error: {str(e)}")
            continue
    missing = [p for p in required_photos if p not in found_photos]
    return missing

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    score_adj = 0
    # Check labor rate (less punitive: -50 instead of -100 for missing/zero rate)
    if re.search(r"labor hours[:\s]*\d+", text, re.IGNORECASE):
        if not re.search(r"labor rate[:\s]*\$?\d+\.?\d*", text, re.IGNORECASE):
            score_adj -= 50
            logger.debug("Labor rate missing or zero")
    # Check tax compliance
    if re.search(r"utilize applicable tax rate", client_rules, re.IGNORECASE):
        if not re.search(r"tax[:\s]*\$?\d+|\d+%", text, re.IGNORECASE):
            score_adj -= 25
            logger.debug("Tax rate missing")
    return score_adj

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
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error": "Appraiser ID is required."})

    images = []
    texts = []
    image_files = []

    for file in files:
        content = await file.read()
        name = file.filename.lower()
        if name.endswith((".jpg", ".jpeg", ".png")):
            image_files.append(file)
            b64 = base64.b64encode(content).decode("utf-8")
            images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(content)))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    combined_text = '\n'.join(texts).lower()
    logger.debug(f"Combined text: {combined_text[:1000]}...")
    logger.debug(f"Client rules: {client_rules[:500]}...")
    advisor_confirmed = advisor_report_present(texts, image_files)
    advisor_hint = "\n\nCONFIRMED: CCC Advisor Report is included based on OCR or filename." if advisor_confirmed else ""
    missing_photos = check_required_photos(image_files)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}" 

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": '\n\n'.join(texts) + advisor_hint + photo_hint})
    if images:
        vision_message["content"].extend(images)

    prompt = f"""
    You are an AI auto damage auditor. You have access to both text and images (or scans).

    IMPORTANT RULES:
    - If labor hours are present but no labor rate is specified or is $0, reduce Compliance Score by 50%.
    - If tax is required by client rules but no tax rate is found, reduce Compliance Score by 25%.
    - Never assume compliance if required elements (like labor rate, taxes, or photos) are missing.
    - Treat mentions or OCR detection of "Clean Retail Value", "NADA Value", "Fair Market Range", "Estimated Trade-In Value", or synonyms like "retail value" or "market value" as CONFIRMATION that the value was included.
    - Treat mentions or OCR detection of "CCC Advisor Report" or "Advisor Report" as CONFIRMATION that the Advisor Report was included.
    - Do NOT rely on assumptions. Only acknowledge presence of documents or data when clearly present in text or visible in photos.
    - Only evaluate Total Loss protocols if the estimate or documentation explicitly indicates the vehicle was a total loss (e.g., mentions "total loss" or "salvage").
    - Do not assume a total loss condition based on estimate formatting or value alone.
    - If no mention of Total Loss or salvage is found, do not apply deductions for missing Total Loss evaluation details.
    - For parts usage, flag non-compliance if alternative parts are used for vehicles of the current model year (2025) or previous year (2024), as per client rules.
    - Deduct 25% from Compliance Score if required photos (four corners, odometer, VIN, license plate) are missing, per photo type.

    PHOTO EVIDENCE RULES:
    - Required photos: four corners, odometer, VIN, license plate.
    - If photo types are missing (indicated in input as "MISSING PHOTOS"), deduct 25% per missing type from Compliance Score.

    At the top of your response, ALWAYS include:
    Claim #: (from estimate)
    VIN: (from estimate or photos)
    Vehicle: (make, model, mileage from estimate)
    Compliance Score: (0–100%)

    Then summarize findings and rule violations based STRICTLY on the following rules:
    {client_rules}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=3500
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
        logger.debug(f"GPT output: {gpt_output[:1000]}...")
        claim_number = extract_field("Claim", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        try:
            score = int(score.strip("%"))
        except:
            score = 100

        score_adj = check_labor_and_tax_score(combined_text, client_rules)
        # Adjust score for missing photos (25% per missing type)
        score_adj -= 25 * len(missing_photos)
        score = max(0, score + score_adj)

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.multi_cell(0, 10, f"AI-4-IA Final Compliance Score: {score}%")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI-4-IA Review Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)

        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Adjusted Compliance Score: {score}%

AI Review Summary:
{gpt_output}
"""
        msg.set_content(email_body.encode("utf-8", errors="ignore").decode("utf-8"))
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)

        return {
            "gpt_output": gpt_output,
            "file_number": file_number,
            "claim_number": claim_number,
            "vehicle": vehicle,
            "score": f"{score}%"
        }

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})

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

