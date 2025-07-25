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
from PIL import Image, ImageEnhance, ImageFilter
from openai import OpenAI
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
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

# Fraud detection module
def run_fraud_checks(texts: List[str], image_files: List[UploadFile]) -> dict:
    indicators = {
        "manipulated_photos": False,
        "repeated_damage_claims": False,
        "timestamp_conflict": False,
        "salvage_resubmission": False,
        "suspicious_terms": [],
    }

    combined_text = '\n'.join(texts).lower()

    suspicious_keywords = ["prior damage", "again", "second time", "wasn't paid", "already repaired", "previous total loss"]
    found_keywords = [term for term in suspicious_keywords if term in combined_text]
    if found_keywords:
        indicators["repeated_damage_claims"] = True
        indicators["suspicious_terms"] = found_keywords

    if "salvage" in combined_text or "total loss" in combined_text:
        indicators["salvage_resubmission"] = True

    try:
        for file in image_files:
            file.file.seek(0)
            image = Image.open(io.BytesIO(file.file.read()))
            exif_data = image.getexif()
            if exif_data:
                date = exif_data.get(36867)
                if date and not re.search(r"202[3-5]", date):
                    indicators["timestamp_conflict"] = True
                    break
    except Exception as e:
        logger.warning(f"EXIF check failed: {str(e)}")

    manipulated = any("photoshop" in t.lower() or "edited" in t.lower() for t in texts)
    if manipulated:
        indicators["manipulated_photos"] = True

    score = 0
    if indicators["manipulated_photos"]:
        score += 40
    if indicators["repeated_damage_claims"]:
        score += 30
    if indicators["timestamp_conflict"]:
        score += 20
    if indicators["salvage_resubmission"]:
        score += 25

    score = min(100, score)
    return {"risk_score": score, "flags": indicators}

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.2)  # Further reduced contrast
    img = img.filter(ImageFilter.SHARPEN)  # Added sharpening
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        pdf_data = file.read()
        images = convert_from_bytes(pdf_data, dpi=200)  # Increased to 200 for clarity
        if not images:
            logger.error("No images generated from PDF")
            return "\n❌ No images generated from PDF"
        text_output = ""
        for i, img in enumerate(images):
            start_time = time.time()
            processed = preprocess_image(img)
            try:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 3', timeout=30)
            except Exception as e:
                logger.warning(f"PSM 3 failed on page {i+1}: {str(e)}")
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6', timeout=30)
            if time.time() - start_time > 30:
                logger.warning(f"OCR timeout on page {i+1}")
                break
            if ocr_text.strip() and not re.search(r"[\:/\d\s]{50,}", ocr_text):
                text_output += f"\n[Page {i+1}]\n{ocr_text.strip()}"
                logger.debug(f"Page {i+1} OCR text: {ocr_text[:200]}...")
            if i > 0 and i % 10 == 0:
                logger.debug(f"Processed {i+1} pages successfully")
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
            return "\n❌ No valid text extracted from PDF"
        return text_output
    except MemoryError as me:
        logger.error(f"Memory error during PDF processing after {i+1 if 'i' in locals() else 0} pages: {str(me)}", exc_info=True)
        return text_output if text_output else f"\n❌ Memory error after {i+1 if 'i' in locals() else 0} pages: {str(me)}"
    except Exception as e:
        logger.error(f"OCR error on page {i+1 if 'i' in locals() else 0}: {str(e)}", exc_info=True)
        return text_output if text_output else f"\n❌ OCR error on page {i+1 if 'i' in locals() else 0}: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return '\n'.join(p.text.strip() for p in doc.paragraphs if p.text.strip())

def extract_field(label: str, text: str) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*[:\-#=]?\s*(R226\d+.*|[A-HJ-NPR-Z0-9]{{10,17}}|[^\n\r;]+)", re.IGNORECASE)  # Expanded VIN range
    matches = pattern.findall(text)
    if matches:
        from collections import Counter
        return Counter(matches).most_common(1)[0][0].strip()
    # Fallback for VIN with partial matches
    if label.lower() == "vin":
        vin_match = re.search(r"\b[A-HJ-NPR-Z0-9]{10,17}\b", text, re.IGNORECASE)
        return vin_match.group(0) if vin_match else "N/A"
    return "N/A"

def advisor_report_present(texts: List[str], image_files: List[UploadFile]) -> bool:
    for t in texts:
        if any(term in t.lower() for term in ["ccc advisor report", "advisor report"]):
            return True
    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            if "advisor report" in ocr.lower():
                return True
        except Exception as e:
            logger.error(f"Image OCR error: {str(e)}")
    return False

def check_required_photos(image_files: List[UploadFile], ocr_text: str) -> List[str]:
    required_photos = ["four corners", "odometer", "vin", "license plate"]
    found_photos = []
    ocr_lower = ocr_text.lower()
    corner_keywords = [
        "four corners", "four corner photo", "vehicle corners",
        "front left", "front right", "rear left", "rear right",
        "left front", "right front", "left rear", "right rear",
        "corner view", "all corners", "vehicle angles"
    ]
    license_plate_keywords = [
        "license plate", "plate photo", "registration plate",
        "license", "reg plate", "license #", r"\b[A-Z0-9]{5,8}\b"
    ]

    # Check OCR text
    if any(re.search(rf"{re.escape(term)}", ocr_lower) for term in license_plate_keywords) or \
       re.search(r"\b[A-Z0-9]{5,8}\b", ocr_lower):
        found_photos.append("license plate")
        logger.debug("Detected license plate in OCR text")
    if any(re.search(rf"{re.escape(term)}", ocr_lower) for term in ["odometer", "mileage photo", "dashboard mileage", "mileage reading", r"\d{1,6}(?:,\d{3})*(?:\s*miles|\s*km)?"]):  # Expanded mileage range
        found_photos.append("odometer")
        logger.debug("Detected odometer in OCR text")
    if any(term in ocr_lower for term in ["vin", "vehicle identification number", "vin photo"]):
        found_photos.append("vin")
        logger.debug("Detected VIN in OCR text")
    if any(re.search(rf"{re.escape(term)}", ocr_lower) for term in corner_keywords):
        found_photos.append("four corners")
        logger.debug("Detected four corners in OCR text")

    # Enhanced image analysis with detailed logging
    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            logger.debug(f"Raw image OCR text: {ocr[:500]}...")  # Increased log length
            if re.search(r"\b[A-HJ-NPR-Z0-9]{10,17}\b", ocr):  # Relaxed VIN
                found_photos.append("vin")
                logger.debug("Detected VIN in image")
            if re.search(r"\d{1,6}(?:,\d{3})*(?:\s*miles|\s*km)?", ocr, re.IGNORECASE):  # Expanded mileage
                found_photos.append("odometer")
                logger.debug("Detected odometer in image")
            if any(re.search(rf"{re.escape(term)}", ocr.lower()) for term in license_plate_keywords) or \
               re.search(r"\b[A-Z0-9]{5,8}\b", ocr):
                found_photos.append("license plate")
                logger.debug("Detected license plate in image")
            if any(re.search(rf"{re.escape(term)}", ocr.lower()) for term in corner_keywords):
                found_photos.append("four corners")
                logger.debug("Detected four corners in image")
        except Exception as e:
            logger.error(f"OCR image read error: {str(e)}")

    found_photos = list(set(found_photos))
    missing = [p for p in required_photos if p not in found_photos]
    logger.debug(f"Found photos: {found_photos}, Missing photos: {missing}")
    return missing

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    score_adj = 0
    required_sections = ["body labor", "paint labor", "mechanical labor", "structural labor"]
    found_sections = []
    for section in required_sections:
        if re.search(rf"{section}[:\s]*(?:\$?\d+\.?\d*\s*(?:/hr|hour)?)", text, re.IGNORECASE):
            found_sections.append(section)
    if not found_sections:
        score_adj -= 50
    if re.search(r"utilize applicable tax rate", client_rules, re.IGNORECASE):
        if not re.search(r"tax[:\s]*(?:\$?\d+\.?\d*|\d+\.?\d*%?)", text, re.IGNORECASE):
            score_adj -= 25
    return score_adj

def is_total_loss(text: str, gpt_output: str) -> bool:
    combined_text = text.lower()
    return any(term in combined_text or term in gpt_output.lower() for term in ["total loss", "totaled", "write-off"])

def get_fraud_risk_explanation(fraud_flags: List[str], score: int) -> str:
    explanations = {
        "🚩 Repeated Claim Number": "Multiple instances of the claim number were detected, which may indicate data duplication or potential fraud.",
        "🚩 Multiple Total Loss Mentions": "The term 'total loss' appears excessively, suggesting possible manipulation or inconsistency.",
        "🚩 Edited or manipulated content terms found": "Terms like 'copy', 'edited', 'photoshop', or 'manipulated' were found, indicating potential image or document alteration.",
        "🚩 Future Date of Loss": "The date of loss is set in the future relative to the current date, which may suggest a data entry error or intentional misrepresentation.",
        "🚩 Claim Number Mismatch": "Different claim numbers were identified across the documents, which could indicate a mismatch or fraudulent activity.",
        "🚩 GPT flagged suspicious content": "The AI model flagged the content as suspicious, potentially due to unusual patterns or language.",
        "🚩 Possible duplicate photos or content": "The AI detected possible duplicate photos or content, which may suggest tampering or resubmission."
    }
    if not fraud_flags or fraud_flags == ["No fraud indicators detected."]:
        if score > 0:
            return "A baseline risk is present due to the identification of a total loss with a specified date of loss, which may warrant further review."
        return "No fraud indicators detected."
    return "\n".join([explanations.get(flag, "Unknown fraud indicator.") for flag in fraud_flags])

def fraud_risk_score(combined_text: str, gpt_output: str) -> dict:
    fraud_flags = []
    current_date = datetime.now().date()

    # Extract estimate date if available
    estimate_date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})\s*(?:AM|PM)?", combined_text, re.IGNORECASE)
    estimate_date = datetime.strptime(estimate_date_match.group(1), "%m/%d/%Y").date() if estimate_date_match else current_date

    # Extract claim numbers within function
    claim_pattern = re.compile(r"claim\s*#?\s*[:\-]?\s*(C[A-Z]\d+[A-Z]?\d*|225\d{6}V\d|\d{5,6})", re.IGNORECASE)
    claim_numbers = claim_pattern.findall(combined_text)

    if len(claim_numbers) > 3:  # Check for repetition
        fraud_flags.append("🚩 Repeated Claim Number")
    if claim_numbers and len(set(claim_numbers)) > 1:  # Safe mismatch check
        fraud_flags.append("🚩 Claim Number Mismatch")
    if "total loss" in combined_text and combined_text.count("total loss") > 2:
        fraud_flags.append("🚩 Multiple Total Loss Mentions")
    if any(term in combined_text for term in ["copy", "edited", "photoshop", "manipulated"]):
        fraud_flags.append("🚩 Edited or manipulated content terms found")

    date_match = re.search(r"date of loss\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4})", combined_text, re.IGNORECASE)
    if date_match:
        loss_date = datetime.strptime(date_match.group(1), "%m/%d/%Y").date()
        if loss_date > current_date or (estimate_date_match and loss_date > estimate_date):
            fraud_flags.append("🚩 Future Date of Loss")

    fraud_risk_score = 0
    if "fraud" in gpt_output.lower() or "suspicious" in gpt_output.lower():
        fraud_risk_score += 30
        fraud_flags.append("🚩 GPT flagged suspicious content")
    if "duplicate" in gpt_output.lower():
        fraud_risk_score += 20
        fraud_flags.append("🚩 Possible duplicate photos or content")
    if is_total_loss(combined_text, gpt_output) and "date of loss" in combined_text.lower():
        fraud_risk_score += 15
    fraud_risk_score += 10 * len(fraud_flags)
    fraud_risk_score = min(fraud_risk_score, 100)

    logger.debug(f"Fraud risk score: {fraud_risk_score}, flags: {fraud_flags}")
    return {
        "score": fraud_risk_score,
        "flags": fraud_flags or ["No fraud indicators detected."]
    }

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
            text = extract_text_from_pdf(io.BytesIO(content))
            if "❌" in text:
                logger.error(f"PDF processing failed: {text}")
                return JSONResponse(status_code=500, content={"error": f"PDF processing failed: {text}"})
            texts.append(text)
            logger.debug(f"Extracted text from PDF: {text[:200]}...")  # Debug log
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {file.filename}")

    combined_text = '\n'.join(texts).lower()
    advisor_confirmed = advisor_report_present(texts, image_files)
    advisor_hint = "\n\nCONFIRMED: CCC Advisor Report is included." if advisor_confirmed else ""
    missing_photos = check_required_photos(image_files, combined_text)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({
            "type": "text",
            "text": '\n'.join(texts) + advisor_hint + photo_hint
        })
        logger.debug(f"Vision message text: {vision_message['content'][0]['text'][:200]}...")  # Debug log
    if images:
        vision_message["content"].extend(images)
        logger.debug(f"Vision message includes {len(images)} images")  # Debug log

    if not vision_message["content"]:
        logger.error("No content available for GPT processing")
        return JSONResponse(status_code=400, content={"error": "No valid data extracted for processing"})

    prompt = f"""
    You are an AI auto damage auditor. Always output:

    Claim #: (from estimate)
    VIN: (from estimate or photos)
    Vehicle: (make, model, mileage)
    Compliance Score: (0–100%)
    Total Loss Status: (Yes/No)

    Then summarize findings based on these rules:
    - For repair estimates: Deduct 50% if ALL labor rates missing, 25% if tax missing and client requires it, 25% per missing photo type (four corners, odometer, VIN, license plate)
    - For total loss: Start at 100%, deduct 25% per missing field (insured, policy #, claim #, date of loss)
    - Treat mentions of “advisor report” as confirmation of inclusion for repair estimates
    - Don’t assume total loss unless explicitly mentioned
    - Use MISSING PHOTOS hint and advisor confirmation for repair estimates
    - Use total loss status to switch evaluation mode
    - Do not include disclaimers about processing personal data (e.g., names); focus solely on vehicle assessment data.

    {client_rules}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=3500
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
        logger.debug(f"GPT output: {gpt_output[:200]}...")  # Debug log
        claim_number = extract_field("Claim", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        mileage_match = re.search(r"mileage:\s*(\d{1,6}(?:,\d{3})*(?:\s*miles|\s*km)?)", vehicle.lower())
        if mileage_match:
            vehicle = re.sub(r"mileage:\s*\d{1,6}(?:,\d{3})*(?:\s*miles|\s*km)?", "", vehicle).strip()
        score = extract_field("Compliance Score", gpt_output)
        total_loss_status = extract_field("Total Loss Status", gpt_output).lower() == "yes"

        try:
            score = int(score.strip("%"))
        except ValueError:
            score = 100

        score_adj = 0
        if not total_loss_status:
            score_adj = check_labor_and_tax_score(combined_text, client_rules)
            score_adj -= 25 * len(missing_photos)
        else:
            required_fields = ["insured", "policy #", "claim #", "date of loss"]
            found_fields = [f for f in required_fields if f in combined_text]
            score_adj -= 25 * (len(required_fields) - len(found_fields))

        final_score = max(0, min(100, score + score_adj))

        # Fraud detection
        fraud_result = fraud_risk_score(combined_text, gpt_output)
        fraud_explanation = get_fraud_risk_explanation(fraud_result["flags"], fraud_result["score"])

        # Generate PDF with only file number, appending suffix if exists
        base_pdf_path = f"{file_number}.pdf"
        pdf_path = base_pdf_path
        counter = 1
        while os.path.exists(pdf_path):
            pdf_path = f"{file_number}_{counter}.pdf"
            counter += 1
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.set_font("DejaVu", size=10)
        pdf.ln(5)
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 10, f"Fraud Risk Score: {fraud_result['score']}%")
        if fraud_result['flags']:
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, "Fraud Indicators:")
            for flag in fraud_result['flags']:
                pdf.multi_cell(0, 10, f"- {flag}")
            pdf.ln(5)
            pdf.multi_cell(0, 10, "Fraud Risk Explanation:")
            pdf.multi_cell(0, 10, fraud_explanation)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"Total Loss Status: {'Yes' if total_loss_status else 'No'}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI-4-IA Review Summary:")
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)

        pdf.output(pdf_path)

        # Send Email
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Adjusted Compliance Score: {final_score}%
Fraud Risk Score: {fraud_result['score']}%
Total Loss Status: {'Yes' if total_loss_status else 'No'}

AI Review Summary:
{gpt_output}

Fraud Indicators:
{chr(10).join(fraud_result['flags']) if fraud_result['flags'] else "None Detected"}

Fraud Risk Explanation:
{fraud_explanation}
"""
        msg.set_content(email_body)
        try:
            with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
                smtp.login("info@nspxn.com", "grr2025GRR")
                smtp.send_message(msg)
        except Exception as e:
            logger.error(f"Email sending failed: {str(e)}")

        return {
            "gpt_output": gpt_output,
            "file_number": file_number,
            "claim_number": claim_number,
            "vehicle": vehicle,
            "score": f"{final_score}%",
            "fraud_risk_score": f"{fraud_result['score']}%",
            "fraud_indicators": fraud_result['flags'],
            "fraud_explanation": fraud_explanation,
            "total_loss": total_loss_status
        }

    except MemoryError as me:
        logger.error(f"Memory error during processing: {str(me)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Memory error: {str(me)}"})
    except Exception as e:
        logger.error(f"Review processing failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_files = [f for f in os.listdir('.') if f.startswith(f"{file_number}") and f.endswith(".pdf")]
    if not pdf_files:
        return JSONResponse(status_code=404, content={"detail": "PDF not found."})
    # Sort by creation time to get the latest
    latest_pdf = max(pdf_files, key=lambda x: os.path.getctime(x))
    pdf_path = latest_pdf
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
    return JSONResponse(status_code=404, content={"detail": "PDF not found."})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"
    file_name = f"{client_name}.docx"
    file_path = os.path.join(rules_dir, file_name)

    if os.path.exists(file_path):
        try:
            doc = Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            logger.debug(f"Retrieved client rules for {client_name}: {text[:500]}...")
            return {"text": text}
        except Exception as e:
            logger.error(f"Error reading client rules for {client_name}: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Failed to read client rules: {str(e)}"})
    else:
        logger.warning(f"Client rules not found for: {client_name}")
        return JSONResponse(status_code=404, content={"error": f"Rules not found for client: {client_name}"})
