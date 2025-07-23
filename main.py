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
from collections import Counter

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

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")  # Grayscale
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
            except Exception:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6')
            if len(ocr_text.strip()) < 50:
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        return text_output or "❌ No valid text extracted."
    except Exception as e:
        return f"❌ OCR Error: {e}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())

def extract_field(label, text) -> str:
    pattern = re.compile(rf"{label}\s*[:\-#=]?\s*(R226\d+.*|[A-HJ-NPR-Z0-9]{{17}}|[^\n\r;]+)", re.IGNORECASE)
    matches = pattern.findall(text)
    return Counter(matches).most_common(1)[0][0].strip() if matches else "N/A"

def advisor_report_present(texts: List[str], image_files: List[UploadFile]) -> bool:
    for t in texts:
        if "advisor report" in t.lower():
            return True
    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            if "advisor report" in ocr.lower():
                return True
        except:
            continue
    return False
def check_required_photos(image_files: List[UploadFile], ocr_text: str) -> List[str]:
    required_photos = ["four corners", "odometer", "vin", "license plate"]
    found_photos = []
    ocr_lower = ocr_text.lower()
    corner_keywords = ["front left", "front right", "rear left", "rear right", 
                       "left front", "right front", "left rear", "right rear"]
    if any(k in ocr_lower for k in ["license plate", "registration plate"]):
        found_photos.append("license plate")
    if any(k in ocr_lower for k in ["odometer", "mileage"]):
        found_photos.append("odometer")
    if "vin" in ocr_lower:
        found_photos.append("vin")
    if sum(1 for term in corner_keywords if term in ocr_lower) >= 2:
        found_photos.append("four corners")

    for img in image_files:
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            ocr = pytesseract.image_to_string(preprocess_image(image))
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr):
                found_photos.append("vin")
            if re.search(r"\d{1,3}(,\d{3})*\s*(miles|km)", ocr, re.IGNORECASE):
                found_photos.append("odometer")
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", ocr):
                found_photos.append("license plate")
            if sum(1 for term in corner_keywords if term in ocr.lower()) >= 2:
                found_photos.append("four corners")
        except:
            continue
    return [p for p in required_photos if p not in set(found_photos)]

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    score = 0
    required_labor = ["body labor", "paint labor", "mechanical labor", "structural labor"]
    if not any(re.search(rf"{lab}[:\s]*\$?\d+", text, re.IGNORECASE) for lab in required_labor):
        score -= 50
    if "utilize applicable tax rate" in client_rules.lower():
        if not re.search(r"tax[:\s]*\$?\d+|\d+\%", text, re.IGNORECASE):
            score -= 25
    return score

def detect_fraud_signals(text: str, score: int, missing_photos: List[str], has_advisor: bool) -> List[str]:
    red_flags = []
    if score <= 50:
        red_flags.append("🚩 Low compliance score (< 50%)")
    if len(missing_photos) >= 2:
        red_flags.append(f"🚩 Missing multiple required photos: {', '.join(missing_photos)}")
    if not has_advisor and "advisor" not in text.lower():
        red_flags.append("🚩 Advisor report not found")
    if "same damage repeated" in text.lower():
        red_flags.append("🚩 Duplicate or recycled damage description")
    if re.search(r"(edited|photoshopped|tampered)", text.lower()):
        red_flags.append("🚩 Possible image manipulation reference")
    if re.search(r"repaired.*before photos", text.lower()):
        red_flags.append("🚩 Repaired prior to inspection")
    return red_flags

def compute_fraud_risk_score(red_flags: List[str]) -> str:
    level = len(red_flags)
    if level == 0:
        return "Low"
    elif level <= 2:
        return "Moderate"
    elif level <= 4:
        return "High"
    else:
        return "Critical"
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

    images, texts, image_files = [], [], []

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
    advisor_confirmed = advisor_report_present(texts, image_files)
    advisor_hint = "\n\nCONFIRMED: CCC Advisor Report is included." if advisor_confirmed else ""
    missing_photos = check_required_photos(image_files, combined_text)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"

    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": '\n\n'.join(texts) + advisor_hint + photo_hint})
    if images:
        vision_message["content"].extend(images)

    prompt = f"""You are an AI auto damage auditor. Review based on the following rules:
{client_rules}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=3500
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
        claim_number = extract_field("Claim", gpt_output)
        vehicle = extract_field("Vehicle", gpt_output)
        score = extract_field("Compliance Score", gpt_output)

        try:
            score = int(score.strip("%"))
        except:
            score = 100

        score_adj = check_labor_and_tax_score(combined_text, client_rules)
        score_adj -= 25 * len(missing_photos)
        score = max(0, score + score_adj)
        if score < 100 and score_adj == 0:
            score = 100

        fraud_flags = detect_fraud_signals(gpt_output, score, missing_photos, advisor_confirmed)
        fraud_level = compute_fraud_risk_score(fraud_flags)

        # continue in next part...
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
        pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.multi_cell(0, 10, f"Compliance Score: {score}%")
        pdf.multi_cell(0, 10, f"Fraud Risk Score: {fraud_level}/10")
        pdf.multi_cell(0, 10, "Fraud Red Flags: " + (", ".join(fraud_flags) if fraud_flags else "None"))
        pdf.ln(5)
        pdf.multi_cell(0, 10, "AI-4-IA Review Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, gpt_output)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)

        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number} (Risk: {fraud_level}/10)"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Adjusted Compliance Score: {score}%
Fraud Risk Score: {fraud_level}/10
Fraud Red Flags: {', '.join(fraud_flags) if fraud_flags else "None"}

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
            "score": f"{score}%",
            "fraud_risk_score": fraud_level,
            "fraud_flags": fraud_flags
        }

    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})


