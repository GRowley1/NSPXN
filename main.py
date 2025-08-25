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
                ocr_text = pytesseract.image_to_string(processed, lang='eng', config='--psm 3')
            except Exception as e:
                logger.warning(f"PSM 3 failed for page {i}: {str(e)}, retrying with PSM 6")
                ocr_text = pytesseract.image_to_string(processed, lang='eng', config='--psm 6')
            if len(ocr_text.strip()) < 50 or re.search(r"[\:/\d\s]{50,}", ocr_text):
                logger.warning(f"Page {i} OCR output skipped (garbled): {ocr_text[:100]}...")
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
            if i == 5:
                logger.debug(f"Page 5 OCR (labor/tax): {ocr_text[:500]}...")
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
        return text_output
    except Exception as e:
        logger.error(f"OCR error (possible network failure): {str(e)}")
        return f"\n\u274c OCR error during combined extraction: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"Extracted DOCX text: {text[:500]}...")
    return text

def extract_field(label, text) -> str:
    pattern = re.compile(rf"{label}\s*[:\-#=]?\s*(R226\d+.*|[A-HJ-NPR-Z0-9]{17}|[^\n\r;]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        from collections import Counter
        return Counter(matches).most_common(1)[0][0].strip()
    return "N/A"

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
            ocr = pytesseract.image_to_string(processed, lang='eng')
            if "advisor report" in ocr.lower():
                logger.debug("Advisor report found in image OCR")
                return True
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            continue
    return False

def check_required_photos(image_files: List[UploadFile], ocr_text: str) -> List[str]:
    required_photos = ["four corners", "odometer", "vin", "license plate", "registration"]
    found_photos = []
    ocr_lower = ocr_text.lower()
    
    if any(term in ocr_lower for term in ["license plate", "plate photo", "registration plate"]):
        found_photos.append("license plate")
        logger.debug("Found license plate photo via OCR keywords")
    if any(term in ocr_lower for term in ["odometer", "mileage photo", "dashboard mileage"]):
        found_photos.append("odometer")
        logger.debug("Found odometer photo via OCR keywords")
    if any(term in ocr_lower for term in ["vin", "vehicle identification number", "vin photo"]):
        found_photos.append("vin")
        logger.debug("Found VIN photo via OCR keywords")
    if any(term in ocr_lower for term in ["registration", "reg photo", "vehicle registration"]):
        found_photos.append("registration")
        logger.debug("Found registration photo via OCR keywords")
    
    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang='eng')
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr, re.IGNORECASE):
                found_photos.append("vin")
                logger.debug("Found VIN photo via image OCR")
            if re.search(r"\d{1,3}(,\d{3})*\s*(miles|km)", ocr, re.IGNORECASE):
                found_photos.append("odometer")
                logger.debug("Found odometer photo via image OCR")
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", ocr, re.IGNORECASE):
                found_photos.append("license plate")
                logger.debug("Found license plate photo via image OCR")
            if re.search(r"registration\s*(document|card)", ocr, re.IGNORECASE):
                found_photos.append("registration")
                logger.debug("Found registration photo via image OCR")
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
    
    found_photos = list(set(found_photos))
    missing = [p for p in required_photos if p not in found_photos]
    logger.debug(f"Found photos: {found_photos}, Missing photos: {missing}")
    return missing

@app.post("/vision-review")
async def vision_review(
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...),
    estimate: UploadFile = File(...),
    image_files: List[UploadFile] = File(...)
):
    if not all([file_number.strip(), ia_company.strip(), appraiser_id.strip()]):
        return JSONResponse(status_code=422, content={"error": "Missing or empty required form fields (file_number, ia_company, appraiser_id)"})
    if not estimate.filename.endswith(('.pdf', '.docx')):
        return JSONResponse(status_code=422, content={"error": "Estimate must be a PDF or DOCX file"})
    
    combined_text = extract_text_from_pdf(estimate) if estimate.filename.endswith('.pdf') else extract_text_from_docx(estimate)
    if "OCR error" in combined_text:
        return JSONResponse(status_code=422, content={"error": "Failed to extract text from estimate due to OCR error"})
    
    missing_photos = check_required_photos(image_files, combined_text)
    vision_message = {"role": "user", "content": [
        {"type": "text", "text": combined_text},
        *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(img.file.read()).decode('utf-8')}"}} for img in image_files]
    ]}
    client_rules = extract_text_from_docx(open(os.path.join("client_rules", "SCA.docx"), 'rb')) if os.path.exists(os.path.join("client_rules", "SCA.docx")) else ""

    prompt = f"""
    You are an AI auto damage auditor. You have access to both text and images (or scans).

    IMPORTANT RULES:
    - If labor rates are missing for ALL sections (body, paint, mechanical, structural), reduce Compliance Score by 50%. If any labor rate is present, no deduction applies.
    - If tax is required but missing, deduct 25%.
    - Never assume compliance if required elements are missing.
    - Treat mentions of 'J.D. Power' or similar as retail value confirmation.
    - Treat 'Advisor Report' mentions as included.
    - Deduct 25% per missing photo (four corners, odometer, VIN, license plate) unless virtual (keywords: 'virtual', 'photo inspection').
    - Four corners require all unique views; multiple of same view count as one.
    - Deductions only for explicit violations.

    PHOTO EVIDENCE RULES:
    - Required photos: four corners, odometer, VIN, license plate.
    - Classify each image's view and list findings.
    - Deduct 25% if any required photo is missing, unless virtual.

    DAMAGE REVIEW AND COMPARISON:
    - Detect damage (dents, scratches) in photos with locations.
    - Extract damage from estimate text.
    - Compare: list matches, photo-only, estimate-only damage.

    At the top, include:
    Claim #: (from estimate)
    VIN: (from estimate or photos)
    Vehicle: (make, model, mileage)
    Compliance Score: (0–100%)

    Summarize findings based on rules, listing:
    - Virtual assignment status.
    - Photo presence/missing.
    - Damage review comparison.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=3000  # Reduced to avoid token limit issues
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
        score_adj -= 25 * len(missing_photos)
        logger.debug(f"Score calculation: AI score={score}, labor_tax_adj={check_labor_and_tax_score(combined_text, client_rules)}, photo_adj={-25 * len(missing_photos)}, final_score={max(0, score + score_adj)}")
        score = max(0, score + score_adj)
        if score < 100 and score_adj == 0:
            logger.warning(f"AI score ({score}) inconsistent with no deductions. Overriding to 100.")
            score = 100
          
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI-4-IA Review Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)
        pdf.ln(5)
        pdf.multi_cell(0, 10, "Damage Photo Review and Comparison:", align='L')
        damage_section = gpt_output.split("Damage Review and Comparison:")[-1] if "Damage Review and Comparison:" in gpt_output else "No damage comparison data available."
        pdf.multi_cell(0, 10, damage_section)

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

















