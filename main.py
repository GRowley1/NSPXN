from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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
import asyncio

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes

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
    try:
        img = img.convert("L")  # Convert to grayscale
        img = ImageEnhance.Contrast(img).enhance(2.0)  # Enhance contrast
        img = img.filter(ImageFilter.MedianFilter(size=3))  # Noise reduction
        img = ImageOps.autocontrast(img)  # Adaptive thresholding
        img = ImageOps.invert(img)  # Invert for better OCR
        return img
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise

async def extract_text_from_pdf(file: io.BytesIO) -> str:
    try:
        file.seek(0)
        file_size = len(file.read())
        file.seek(0)
        if file_size > MAX_FILE_SIZE:
            logger.error(f"PDF file size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE} bytes)")
            raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit")

        images = await asyncio.to_thread(convert_from_bytes, file.read(), dpi=200)
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
        logger.error(f"PDF OCR error: {str(e)}")
        return f"\n\u274c OCR error during PDF extraction: {str(e)}"

def extract_text_from_docx(file: io.BytesIO) -> str:
    try:
        doc = Document(file)
        text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
        logger.debug(f"Extracted DOCX text: {text[:500]}...")
        return text
    except Exception as e:
        logger.error(f"DOCX extraction error: {str(e)}")
        return f"\n\u274c DOCX extraction error: {str(e)}"

def extract_field(label: str, text: str) -> str:
    pattern = re.compile(rf"{label}\s*[:\-#=]?\s*(R226\d+.*|[A-HJ-NPR-Z0-9]{{17}}|[^\n\r;]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        from collections import Counter
        return Counter(matches).most_common(1)[0][0].strip()
    return "N/A"

def advisor_report_present(texts: List[str], image_files: List[UploadFile]) -> bool:
    advisor_terms = [
        "ccc advisor report", "advisor report", "estimate of record", "supplement of record",
        "sca claim services", "motor crash estimating guide"
    ]
    for t in texts:
        if any(term in t.lower() for term in advisor_terms):
            logger.debug(f"Advisor report found in text with term: {term}")
            return True
    for img in image_files:
        try:
            img.file.seek(0)
            content = img.file.read()
            if len(content) > MAX_FILE_SIZE:
                logger.error(f"Image file {img.filename} size ({len(content)} bytes) exceeds limit")
                continue
            image = Image.open(io.BytesIO(content))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            if any(term in ocr.lower() for term in advisor_terms):
                logger.debug(f"Advisor report found in image OCR with term: {term}")
                return True
        except Exception as e:
            logger.error(f"Image processing error for {img.filename}: {str(e)}")
            continue
    return False

async def check_required_photos(image_files: List[UploadFile], ocr_text: str) -> List[str]:
    required_photos = ["four corners", "odometer", "vin", "license plate"]
    found_photos = []
    ocr_lower = ocr_text.lower()
    corner_keywords = [
        "four corners", "four corner photo", "vehicle corners",
        "front left", "front right", "rear left", "rear right",
        "left front", "right front", "left rear", "right rear"
    ]
    corner_matches = []

    if any(term in ocr_lower for term in ["license plate", "plate photo", "registration plate"]):
        found_photos.append("license plate")
        logger.debug("Found license plate photo via OCR keywords")
    if any(term in ocr_lower for term in ["odometer", "mileage photo", "dashboard mileage"]):
        found_photos.append("odometer")
        logger.debug("Found odometer photo via OCR keywords")
    if any(term in ocr_lower for term in ["vin", "vehicle identification number", "vin photo"]):
        found_photos.append("vin")
        logger.debug("Found VIN photo via OCR keywords")
    for term in corner_keywords:
        if term in ocr_lower:
            corner_matches.append(term)
    if len(corner_matches) >= 2:
        found_photos.append("four corners")
        logger.debug(f"Found four corners photo via OCR keywords: {corner_matches}")

    for img in image_files:
        try:
            img.file.seek(0)
            content = img.file.read()
            if len(content) > MAX_FILE_SIZE:
                logger.error(f"Image file {img.filename} size ({len(content)} bytes) exceeds limit")
                continue
            image = Image.open(io.BytesIO(content))
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
            corner_matches_img = [term for term in corner_keywords if term in ocr.lower()]
            if len(corner_matches_img) >= 2:
                found_photos.append("four corners")
                logger.debug(f"Found four corners photo via image OCR: {corner_matches_img}")
            if corner_matches_img:
                corner_matches.extend(corner_matches_img)
        except Exception as e:
            logger.error(f"Image processing error for {img.filename}: {str(e)}")

    found_photos = list(set(found_photos))
    missing = [p for p in required_photos if p not in found_photos]
    logger.debug(f"Found photos: {found_photos}, Missing photos: {missing}, Corner matches: {corner_matches}")
    return missing

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    score_adj = 0
    required_sections = ["body labor", "paint labor", "mechanical labor", "structural labor"]
    found_sections = []
    for section in required_sections:
        if re.search(rf"{section}\s*[:\s]*\$?\d+\.?\d*\s*(?:/hr|hour)?", text, re.IGNORECASE):
            found_sections.append(section)
            logger.debug(f"Found labor rate for {section}")
    if not found_sections:
        score_adj -= 50
        logger.debug("All labor rates missing")
    else:
        logger.debug(f"Found labor rates in sections: {found_sections}")
    if re.search(r"utilize applicable tax rate", client_rules, re.IGNORECASE):
        if not re.search(r"tax[:\s]*(?:\$?\d+\.?\d*|\d+\.?\d*%?)", text, re.IGNORECASE):
            score_adj -= 25
            logger.debug("Tax rate missing")
        else:
            logger.debug("Tax rate found")
    return score_adj

def check_parts_compliance(text: str, vehicle_year: int) -> int:
    score_adj = 0
    if vehicle_year in [2024, 2025]:
        if re.search(r"\b(?:LKQ|aftermarket|A/M)\b", text, re.IGNORECASE):
            score_adj -= 25
            logger.debug(f"Non-compliant parts (LKQ/aftermarket) detected for vehicle year {vehicle_year}")
    else:
        logger.debug(f"Vehicle year {vehicle_year} allows LKQ/aftermarket parts")
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
        raise HTTPException(status_code=400, detail="Appraiser ID is required.")

    images = []
    texts = []
    image_files = []

    try:
        for file in files:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                logger.error(f"File {file.filename} size ({len(content)} bytes) exceeds limit")
                raise HTTPException(status_code=400, detail=f"File {file.filename} exceeds {MAX_FILE_SIZE // 1024 // 1024}MB limit")
            name = file.filename.lower()
            if name.endswith((".jpg", ".jpeg", ".png")):
                image_files.append(file)
                b64 = base64.b64encode(content).decode("utf-8")
                images.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            elif name.endswith(".pdf"):
                texts.append(await extract_text_from_pdf(io.BytesIO(content)))
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
        advisor_hint = "\n\nCONFIRMED: CCC Advisor Report or equivalent (Estimate/Supplement of Record) is included based on OCR or filename." if advisor_confirmed else "\n\nWARNING: No CCC Advisor Report or equivalent detected."
        missing_photos = await check_required_photos(image_files, combined_text)
        photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"

        vehicle_year_match = re.search(r"\b(20\d{2})\b", combined_text)
        vehicle_year = int(vehicle_year_match.group(1)) if vehicle_year_match else 0
        logger.debug(f"Extracted vehicle year: {vehicle_year}")

        vision_message = {"role": "user", "content": []}
        if texts:
            vision_message["content"].append({"type": "text", "text": '\n\n'.join(texts) + advisor_hint + photo_hint})
        if images:
            vision_message["content"].extend(images)

        prompt = f"""
        You are an AI auto damage auditor. You have access to both text and images (or scans).

        IMPORTANT RULES:
        - If labor rates are missing for ALL sections (body, paint, mechanical, structural), reduce Compliance Score by 50%. If any labor rate is present (e.g., body or paint), no deduction applies.
        - If tax is required by client rules but no tax rate or amount (e.g., percentage or dollar value) is found, reduce Compliance Score by 25%.
        - Never assume compliance if required elements (like labor rates, taxes, or photos) are missing.
        - Treat mentions or OCR detection of "Clean Retail Value", "NADA Value", "Fair Market Range", "Estimated Trade-In Value", "market value", "J.D. Power", "JD Power", or "Average Price Paid" as CONFIRMATION that the retail/market value requirement is met.
        - Treat mentions or OCR detection of "CCC Advisor Report", "Advisor Report", "Estimate of Record", "Supplement of Record", "SCA Claim Services", or "MOTOR CRASH ESTIMATING GUIDE" as CONFIRMATION that the Advisor Report was included.
        - Do NOT rely on assumptions. Only acknowledge presence of documents or data when clearly present in text or visible in photos.
        - Only evaluate Total Loss protocols if the estimate or documentation explicitly indicates the vehicle was a total loss (e.g., mentions "total loss" or "salvage"). If declared a total loss, no forms or bids are required.
        - Do not assume a total loss condition based on estimate formatting or value alone.
        - If no mention of Total Loss or salvage is found, do not apply deductions for missing Total Loss evaluation details.
        - For parts usage, flag non-compliance if alternative parts (e.g., LKQ, aftermarket) are used for vehicles of the current model year (2025) or previous year (2024), as per client rules. Deduct 25% for this violation. For older models (e.g., 2019), LKQ/aftermarket parts are compliant.
        - Deduct 25% from Compliance Score for each missing required photo type (four corners, odometer, VIN, license plate).
        - For four corners photos, the requirement is met if at least two views (e.g., front left, front right, rear left, rear right, or synonyms like left front, right front, left rear, right rear) are present in text or images, as indicated in the MISSING PHOTOS hint.
        - Do NOT apply deductions for unmentioned elements or assumed violations. Deductions must be explicitly listed in the findings and supported by evidence in the input or client rules.
        - The Compliance Score starts at 100% and is only reduced by explicit deductions for labor rates (50% if all missing), tax (25% if missing), photos (25% per missing type), or parts (25% for violations).
        - Respect the MISSING PHOTOS hint provided in the input to determine photo compliance.

        PHOTO EVIDENCE RULES:
        - Required photos: four corners, odometer, VIN, license plate.
        - Four corners is satisfied if at least two views (e.g., front left, front right, rear left, right rear, or synonyms like left front, right front, left rear, right rear) are detected in text or images, as indicated in the MISSING PHOTOS hint.
        - If photo types are missing (indicated in input as "MISSING PHOTOS"), deduct 25% per missing type from Compliance Score.
        - Respect the MISSING PHOTOS hint provided in the input to determine photo compliance.

        At the top of your response, ALWAYS include:
        Claim #: (from estimate)
        VIN: (from estimate or photos)
        Vehicle: (make, model, mileage from estimate)
        Compliance Score: (0–100%)

        Then summarize findings and rule violations based STRICTLY on the following rules:
        {client_rules}
        """

        try:
            async with asyncio.timeout(60):  # 60-second timeout for API call
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role": "system", "content": prompt}, vision_message],
                    max_tokens=3500
                )
            gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
            logger.debug(f"GPT output: {gpt_output[:1000]}...")
            claim_number = extract_field("Claim", gpt_output) or extract_field("Claim Number", combined_text) or "N/A"
            vehicle = extract_field("Vehicle", gpt_output) or extract_field("2019 HOND Civic", combined_text) or "N/A"
            score = extract_field("Compliance Score", gpt_output)

            try:
                score = int(score.strip("%"))
            except:
                score = 100

            score_adj = check_labor_and_tax_score(combined_text, client_rules)
            score_adj += check_parts_compliance(combined_text, vehicle_year)
            score_adj -= 25 * len(missing_photos)
            logger.debug(f"Score calculation: AI score={score}, labor_tax_adj={check_labor_and_tax_score(combined_text, client_rules)}, parts_adj={check_parts_compliance(combined_text, vehicle_year)}, photo_adj={-25 * len(missing_photos)}, final_score={max(0, score + score_adj)}")
            score = max(0, score + score_adj)
            if score < 100 and score_adj == 0:
                logger.warning(f"AI score ({score}) inconsistent with no deductions (labor_tax_adj=0, parts_adj=0, photo_adj=0). Overriding to 100.")
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
            pdf.multi_cell(0, 10, f"Final Compliance Score: {score}%")
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

        except asyncio.TimeoutError:
            logger.error("OpenAI API call timed out after 60 seconds")
            raise HTTPException(status_code=504, detail="OpenAI API call timed out")
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"AI review failed: {str(e)}")

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_path)
    raise HTTPException(status_code=404, detail="PDF not found")

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
            raise HTTPException(status_code=500, detail=f"Client rules error: {str(e)}")
    else:
        logger.error(f"Rules not found for client: {client_name}")
        raise HTTPException(status_code=404, detail="Rules not found for this client."})
