from fastapi import FastAPI, File, UploadFile, Form, Request
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
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("\u274c OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = FastAPI()

# Global request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Received {request.method} {request.url.path} with requestID={request.headers.get('X-Request-ID', 'unknown')}")
    form = await request.form() if request.method in ["POST", "PUT"] else None
    if form:
        logger.debug(f"Raw form data: {dict(form)}")
    response = await call_next(request)
    logger.debug(f"Response status: {response.status_code}")
    return response

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
            ocr = pytesseract.image_to_string(processed, lang="eng")
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
            ocr = pytesseract.image_to_string(processed, lang="eng")
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

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    deduction = 0
    # Check for labor rates across all sections
    if not re.search(r'labor\s*rate\s*(body|paint|mechanical|structural)', text, re.IGNORECASE):
        deduction -= 50  # All labor rates missing
    # Check for tax
    if not re.search(r'(tax|sales tax)\s*[\d.]+%', text, re.IGNORECASE):
        deduction -= 25  # Tax missing or no percentage
    return deduction

@app.post("/vision-review")
async def vision_review(request: Request, file_number: str = Form(...), ia_company: str = Form(...), appraiser_id: str = Form(...), estimate: UploadFile = File(...), image_files: List[UploadFile] = File(...)):
    # Validate form fields
    logger.debug(f"Processed form data: file_number='{file_number}', ia_company='{ia_company}', appraiser_id='{appraiser_id}'")
    if not all([field.strip() for field in [file_number, ia_company, appraiser_id]]):
        logger.error(f"Validation failed: Empty fields - file_number='{file_number}', ia_company='{ia_company}', appraiser_id='{appraiser_id}'")
        return JSONResponse(status_code=422, content={"error": "Missing or empty required form fields (file_number, ia_company, appraiser_id)"})
    
    # Validate estimate file type
    logger.debug(f"Estimate file: filename='{estimate.filename}', content_type='{estimate.content_type}'")
    if not estimate.filename.lower().endswith(('.pdf', '.docx')):
        logger.error(f"Validation failed: Invalid estimate file type - {estimate.filename}")
        return JSONResponse(status_code=422, content={"error": f"Estimate must be a PDF or DOCX file, got {estimate.filename}"})
    
    # Log initial file details and size limits
    max_file_size = 5 * 1024 * 1024  # 5MB limit
    estimate.file.seek(0, os.SEEK_END)
    size = estimate.file.tell()
    estimate.file.seek(0)
    logger.debug(f"Estimate file size: {size} bytes")
    if size > max_file_size:
        logger.error(f"Estimate {estimate.filename} exceeds 5MB limit ({size} bytes)")
        return JSONResponse(status_code=422, content={"error": "Estimate file exceeds 5MB limit"})
    
    for i, img in enumerate(image_files):
        img.file.seek(0, os.SEEK_END)
        size = img.file.tell()
        img.file.seek(0)
        logger.debug(f"Image {i+1}: filename='{img.filename}', size={size} bytes")
        if size > max_file_size:
            logger.error(f"Image {img.filename} exceeds 5MB limit ({size} bytes)")
            return JSONResponse(status_code=422, content={"error": f"Image file {img.filename} exceeds 5MB limit"})
    
    # Process estimate
    combined_text = extract_text_from_pdf(estimate) if estimate.filename.lower().endswith('.pdf') else extract_text_from_docx(estimate)
    if "OCR error" in combined_text:
        logger.error(f"OCR error on estimate {estimate.filename}: {combined_text}")
        return JSONResponse(status_code=422, content={"error": "Failed to extract text from estimate due to OCR error"})
    
    # Check required photos
    missing_photos = check_required_photos(image_files, combined_text)
    # Reset file pointers for images
    for img in image_files:
        img.file.seek(0)
    vision_message = {"role": "user", "content": [
        {"type": "text", "text": combined_text},
        *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(await img.read()).decode('utf-8')}"}} for img in image_files]
    ]}
    client_rules = extract_text_from_docx(open(os.path.join("client_rules", "SCA.docx"), 'rb')) if os.path.exists(os.path.join("client_rules", "SCA.docx")) else ""

    prompt = f"""
    You are an AI auto damage auditor. You have access to both text and images (or scans).

    IMPORTANT RULES:
    - If labor rates are missing for ALL sections (body, paint, mechanical, structural), reduce Compliance Score by 50%. If any labor rate is present (e.g., body or paint), no deduction applies.
    - If tax is required by client rules but no tax rate or amount (e.g., percentage or dollar value) is found, reduce Compliance Score by 25%.
    - Never assume compliance if required elements (like labor rates, taxes, or photos) are missing.
    - Treat mentions or OCR detection of "Clean Retail Value", "NADA Value", "Fair Market Range", "Estimated Trade-In Value", "market value", "J.D. Power", "JD Power", or "Average Price Paid" as CONFIRMATION that the retail/market value requirement is met.
    - Treat mentions or OCR detection of "CCC Advisor Report" or "Advisor Report" as CONFIRMATION that the Advisor Report was included.
    - Do NOT rely on assumptions. Only acknowledge presence of documents or data when clearly present in text or visible in photos.
    - Only evaluate Total Loss protocols if the estimate or documentation explicitly indicates the vehicle was a total loss (e.g., mentions "total loss" or "salvage"). If declared a total loss, no forms or bids are required.
    - Do not assume a total loss condition based on estimate formatting or value alone.
    - If no mention of Total Loss or salvage is found, do not apply deductions for missing Total Loss evaluation details.
    - For parts usage, flag non-compliance if alternative parts (e.g., LKQ, aftermarket) are used for vehicles of the current model year (2025) or previous year (2024), as per client rules. Deduct 25% for this violation. For older models (e.g., 2012), LKQ/aftermarket parts are compliant.
    - Deduct 25% from Compliance Score for each missing required photo type (four corners, odometer, VIN, license plate, registration).
    - For four corners photos, the requirement is met if all four unique views are present across the images: front-left (front and driver side), front-right (front and passenger side), rear-left (rear and driver side), rear-right (rear and passenger side). Three-quarter views or partial zooms count as long as the corner is clearly visible for damage assessment. Multiple images of the same view count as one. Deduct 25% if any corner is missing.
    - If this is a VIRTUAL ASSIGNMENT (determine from text: look for keywords like 'virtual inspection', 'photo estimate', 'Streamline', 'customer photos', 'remote appraisal', or absence of physical inspection date/notes), do not apply deduction for missing registration photo. Otherwise, deduct 25% if registration photo is missing.
    - Do NOT apply deductions for unmentioned elements or assumed violations. Deductions must be explicitly listed in the findings and supported by evidence in the input or client rules.
    - The Compliance Score starts at 100% and is only reduced by explicit deductions for labor rates (50% if all missing), tax (25% if missing), photos (25% per missing type), or parts (25% for violations).
    - Respect the MISSING PHOTOS hint provided in the input to determine photo compliance, but override with your visual analysis of the images if the hint conflicts (e.g., if images clearly show a required photo but OCR missed it).

    PHOTO EVIDENCE RULES:
    - Required photos: four corners, odometer, VIN, license plate, registration (photo of the vehicle registration document/card, separate from license plate).
    - Examine each provided image and classify its primary view (e.g., 'Image 1: rear-left corner', 'Image 2: close-up rear-left corner'). List these classifications in your findings.
    - Four corners is one type: satisfied only if all four unique corners are covered (deduct 25% if any are missing, and specify which one(s)).
    - Odometer: deduct 25% if no image shows the dashboard mileage reading.
    - VIN: deduct 25% if no image shows the VIN plate/sticker.
    - License plate: satisfied if visible in any image (e.g., rear views); deduct 25% if missing.
    - Registration: deduct 25% if no image shows the registration document/card (unless virtual assignment).
    - Respect the MISSING PHOTOS hint provided in the input, but use your visual analysis to confirm or override.

    DAMAGE REVIEW AND COMPARISON:
    - Include a section titled 'Damage Review and Comparison:'.
    - Detect damage (e.g., dents, scratches) in photos with locations (e.g., 'front-left door', 'rear bumper').
    - Extract damage descriptions from estimate text.
    - Compare: list matches (damage in both photos and estimate), photo-only damage, and estimate-only damage.

    At the top of your response, ALWAYS include:
    Claim #: (from estimate)
    VIN: (from estimate or photos)
    Vehicle: (make, model, mileage from estimate)
    Compliance Score: (0–100%)

    Then summarize findings and rule violations based STRICTLY on the following rules:
    {client_rules}

    In your findings, explicitly list:
    - Whether this is a virtual assignment (with evidence from text).
    - Which photo types are present/missing, with evidence from the images (e.g., 'Four corners: All present - rear-left in Images 1 and 2, rear-right in Image 3, front-right in Image 4, front-left in Image 5'; 'Registration: Missing - no image of registration document').
    - 'Damage Review and Comparison:' section with comparison results.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=4000
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
        photo_deduction = 25 * len(missing_photos)
        score_adj -= photo_deduction
        logger.debug(f"Score calculation: AI score={score}, labor_tax_adj={check_labor_and_tax_score(combined_text, client_rules)}, photo_adj={-photo_deduction}, final_score={max(0, score + score_adj)}")
        score = max(0, score + score_adj)
        if score < 100 and score_adj == 0:
            logger.warning(f"AI score ({score}) inconsistent with no deductions (labor_tax_adj=0, photo_adj=0). Overriding to 100.")
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
        damage_section = gpt_output.split("Damage Review and Comparison:")[-1].strip() if "Damage Review and Comparison:" in gpt_output else "No damage comparison data available."
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
            try:
                smtp.login("info@nspxn.com", "grr2025GRR")
                smtp.send_message(msg)
            except Exception as email_e:
                logger.error(f"Email sending error: {str(email_e)}")

        return {
            "gpt_output": gpt_output,
            "file_number": file_number,
            "claim_number": claim_number,
            "vehicle": vehicle,
            "score": f"{score}%"
        }

    except Exception as e:
        logger.error(f"API error during processing: {str(e)}")
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
    logger.debug(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)





