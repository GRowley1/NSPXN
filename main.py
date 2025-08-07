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
from ultralytics import YOLO
import torch
from fraud_check import calculate_fraud_risk

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
        images = convert_from_bytes(file.read(), dpi=150)  # Stable DPI
        text_output = ""
        for i, img in enumerate(images, 1):
            processed = preprocess_image(img)
            try:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 3', timeout=30)
            except Exception as e:
                logger.warning(f"PSM 3 failed on page {i}: {str(e)}")
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6', timeout=30)
            if ocr_text.strip() and not re.search(r"[\:/\d\s]{50,}", ocr_text):
                text_output += f"\n[Page {i}]\n{ocr_text.strip()}"
                logger.debug(f"Page {i} OCR text: {ocr_text[:200]}...")
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
            return "\n\u274c No valid text extracted from PDF"
        return text_output
    except Exception as e:
        logger.error(f"PDF processing error: {str(e)}")
        return "\n\u274c PDF processing error: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return '\n'.join(p.text.strip() for p in doc.paragraphs if p.text.strip())

def extract_field(label: str, text: str) -> str:
    pattern = re.compile(rf"{re.escape(label)}\s*[:\-#=]?\s*(R226\d+.*|[A-HJ-NPR-Z0-9]{{17}}|[^\n\r;]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        from collections import Counter
        return Counter(matches).most_common(1)[0][0].strip()
    if label.lower() == "vin":
        vin_match = re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", text, re.IGNORECASE)
        return vin_match.group(0) if vin_match else "N/A"
    return "N/A"

def detect_corners_with_yolo(image: Image.Image) -> int:
    # Load YOLO model from local file
    try:
        model_path = os.path.join(os.getcwd(), "corner-detector.pt")
        if not os.path.exists(model_path):
            logger.error(f"YOLO model file not found at {model_path}")
            return 0
        model = YOLO(model_path)
        # Convert PIL image to RGB and ensure numpy compatibility
        image_rgb = image.convert("RGB")
        # Perform inference in headless mode
        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Force headless mode
        results = model(image_rgb)
        # Count detected corners (assuming class 0 is corner)
        corner_count = len([box for box in results[0].boxes if box.cls[0] == 0]) if results[0].boxes else 0
        logger.debug(f"YOLO detected {corner_count} corner views")
        return corner_count
    except Exception as e:
        logger.error(f"YOLO detection error: {str(e)}")
        return 0  # Default to 0 if detection fails

def check_required_photos(image_files: List[UploadFile]) -> tuple[List[str], float]:
    required_photos = ["four corners", "odometer", "vin", "license plate"]
    found_photos = []
    corner_deduction = 0

    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr_text = pytesseract.image_to_string(processed, lang="eng")
            logger.debug(f"Image OCR text: {ocr_text[:200]}...")

            # VIN detection
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr_text):
                found_photos.append("vin")
                logger.debug("Detected VIN in image")

            # Odometer detection
            if re.search(r"\d{1,6}(?:,\d{3})*(?:\s*miles|\s*km)?", ocr_text, re.IGNORECASE):
                found_photos.append("odometer")
                logger.debug("Detected odometer in image")

            # License plate detection
            if re.search(r"\b[A-Z0-9]{5,8}\b", ocr_text):
                found_photos.append("license plate")
                logger.debug("Detected license plate in image")

            # Four corners detection using YOLO only
            corner_count = detect_corners_with_yolo(processed)
            if corner_count >= 2:
                found_photos.append("four corners")
                logger.debug("Detected 2+ corner views with YOLO")
            elif corner_count == 1:
                corner_deduction = 12.5  # Partial deduction
                logger.debug("Detected 1 corner view with YOLO")
            else:
                corner_deduction = 25  # Default deduction for 0 corners

        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            corner_deduction = 25  # Default to 25% if processing fails

    found_photos = list(set(found_photos))
    missing = [p for p in required_photos if p not in found_photos]
    if "four corners" not in found_photos:
        missing.append("four corners")  # Ensure four corners is marked missing if not compliant
    # Apply 50% deduction if all required photos are missing
    if all(p in missing for p in required_photos):
        corner_deduction = 50
    logger.debug(f"Found photos: {found_photos}, Missing photos: {missing}, Corner deduction: {corner_deduction}%")
    return missing, corner_deduction

def check_labor_and_tax_score(text: str, client_rules: str, skip_labor_tax_checks: bool) -> int:
    if skip_labor_tax_checks:
        logger.debug("Skipping labor and tax checks due to no damage found")
        return 0
    score_adj = 0
    required_sections = ["body labor", "paint labor", "mechanical labor", "structural labor"]
    found_sections = []
    for section in required_sections:
        if re.search(rf"{section}[:\s]*(?:\$?\d+\.?\d*\s*(?:/hr|hour)?)", text, re.IGNORECASE):
            found_sections.append(section)
    if not found_sections:
        score_adj -= 50
    if re.search(r"utilize applicable tax rate", client_rules, re.IGNORECASE):
        tax_pattern = r"sales\s+tax\s+tier\s+\d+\s+\$\s*\d+\.?\d*\s+@\s*\d+\.?\d*%\s+\$\s*\d+\.?\d*"
        if re.search(tax_pattern, text.lower()):
            logger.debug("Tax information detected, no deduction applied")
        else:
            score_adj -= 25
    return score_adj

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

@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...),
    claim_number: str = Form("010683-162546-AD-01")  # Default to provided claim number
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
            if "\u274c" in text:
                logger.error(f"PDF processing failed: {text}")
                return JSONResponse(status_code=500, content={"error": f"PDF processing failed: {text}"})
            texts.append(text)
            logger.debug(f"Extracted text from PDF: {text[:200]}...")
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(content)))
        elif name.endswith(".txt"):
            texts.append(content.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"\u26a0\ufe0f Skipped unsupported file: {file.filename}")

    combined_text = '\n'.join(texts).lower()
    advisor_confirmed = advisor_report_present(texts, image_files)
    advisor_hint = "\n\nCONFIRMED: CCC Advisor Report is included." if advisor_confirmed else ""
    missing_photos, corner_deduction = check_required_photos(image_files)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"

    # Check for no damage phrases
    skip_labor_tax_checks = any(phrase in combined_text for phrase in ["no damage found", "no damage identified"])

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({
            "type": "text",
            "text": '\n'.join(texts) + advisor_hint + photo_hint
        })
        logger.debug(f"Vision message text: {vision_message['content'][0]['text'][:200]}...")
    if images:
        vision_message["content"].extend(images)
        logger.debug(f"Vision message includes {len(images)} images")

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
    - For repair estimates: Deduct {corner_deduction}% for four corners photo compliance based on YOLO detection, 25% if tax missing and client requires it, 50% if ALL labor rates missing (only if damage is indicated)
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
        gpt_output = response.choices[0].message.content or "\u274c GPT returned no output."
        logger.debug(f"GPT output: {gpt_output[:200]}...")
        claim_number_from_gpt = extract_field("Claim", gpt_output)
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
            score_adj = check_labor_and_tax_score(combined_text, client_rules, skip_labor_tax_checks)
            score_adj -= corner_deduction  # Apply YOLO-based corner deduction
        else:
            required_fields = ["insured", "policy #", "claim #", "date of loss"]
            found_fields = [f for f in required_fields if f in combined_text]
            score_adj -= 25 * (len(required_fields) - len(found_fields))

        final_score = max(0, min(100, score + score_adj))

        # Calculate fraud risk with the provided claim number
        fraud_result = calculate_fraud_risk(combined_text, image_files, claim_number)
        fraud_explanation = fraud_result.get("explanation", "No fraud indicators detected.")

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
        pdf.ln(5)
        pdf.set_text_color(200, 0, 0)
        pdf.multi_cell(0, 10, f"Fraud Risk Score: {fraud_result['score']}%")
        if fraud_result["flags"]:
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 10, "Fraud Indicators:")
            for flag in fraud_result["flags"]:
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
            "claim_number": claim_number_from_gpt or claim_number,
            "vehicle": vehicle,
            "score": f"{final_score}%",
            "fraud_score": f"{fraud_result['score']}%"
        }

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "\u274c AI review failed."})

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








