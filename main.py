from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional
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
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
from openai import OpenAI
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# OpenAI client
# ----------------------------
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ----------------------------
# FastAPI app + CORS
# ----------------------------
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

# ----------------------------
# OCR helpers
# ----------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")                       # grayscale
    img = ImageEnhance.Contrast(img).enhance(2) # boost contrast
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def extract_text_from_pdf(file_like: io.BytesIO) -> str:
    try:
        file_like.seek(0)
        images = convert_from_bytes(file_like.read(), dpi=200)
        text_output = ""
        for i, img in enumerate(images, 1):
            processed = preprocess_image(img)
            try:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6')
            except Exception as e:
                logger.warning(f"PSM 6 failed for page {i}: {str(e)}, retrying with PSM 3")
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 3')
            if len(ocr_text.strip()) < 30:
                logger.warning(f"OCR page {i} very short, skipping likely noise.")
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return f"\n❌ OCR error during extraction: {str(e)}"

def extract_text_from_docx(file_like: io.BytesIO) -> str:
    doc = Document(file_like)
    txt = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"DOCX extract sample: {txt[:400]}...")
    return txt

# ----------------------------
# Field extraction (robust)
# ----------------------------
def extract_claim_from_text(text: str) -> Optional[str]:
    """
    Prefer explicit 'Claim #', 'Claim No', etc. Avoid 'Workfile ID' / 'Job Number'.
    """
    patterns = [
        r"(?:^|\s)(?:Claim\s*(?:#|No\.?|Number)[:\s]*)\s*([A-Za-z0-9\-]+)",
        r"(?:^|\s)Claim\s*[:#]\s*([A-Za-z0-9\-]+)"
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    m = re.search(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def extract_vehicle_from_text(text: str) -> Optional[str]:
    """
    Grab year + make + model and mileage if present.
    Examples in OCR:
      '2025 NISS Sentra ...'
      'Odometer: 9,792'
    """
    # Year + make/model (loose)
    m1 = re.search(r"\b(20\d{2})\s+([A-Za-z]{3,})\s+([A-Za-z0-9\-]{2,})", text)
    # mileage
    m2 = re.search(r"Odometer\s*:\s*([\d,]+)", text, flags=re.IGNORECASE)
    if m1:
        year, make, model = m1.group(1), m1.group(2), m1.group(3)
        miles = m2.group(1) if m2 else "Mileage unknown"
        return f"{year} {make} {model}, {miles} miles"
    return None

# ----------------------------
# Photo checks
# ----------------------------
def _image_is_exterior_wide(img: Image.Image) -> bool:
    """
    Heuristic: exterior wide shots typically have high entropy and very low OCR text.
    """
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    entropy = ImageStat.Stat(processed).var[0]
    return len(text.strip()) < 10 and entropy > 150  # simple + effective for corner shots

def check_required_photos(
    image_blobs: List[Tuple[str, bytes]],
    ocr_text: str
) -> List[str]:
    """
    Required: four corners, odometer, VIN, license plate.
    We now detect 'four corners' visually: if >=2 exterior angles (heuristic) are present
    we treat four corners as satisfied (your rule allows two views).
    """
    required = ["four corners", "odometer", "vin", "license plate"]
    present = set()

    # OCR hints still count
    txt = ocr_text.lower()
    if any(k in txt for k in ["odometer", "mileage photo", "dashboard mileage"]):
        present.add("odometer")
    if any(k in txt for k in ["vin", "vehicle identification number", "vin photo"]):
        present.add("vin")
    if any(k in txt for k in ["license plate", "registration plate"]):
        present.add("license plate")

    exterior_count = 0
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            processed = preprocess_image(img)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            # direct detections
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr, re.IGNORECASE):
                present.add("vin")
            if re.search(r"\d{1,3}(,\d{3})*\s*(miles|km)", ocr, re.IGNORECASE):
                present.add("odometer")
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", ocr, re.IGNORECASE):
                present.add("license plate")
            # exterior heuristic
            if _image_is_exterior_wide(img):
                exterior_count += 1
        except Exception as e:
            logger.warning(f"Image check error for {name}: {e}")

    if exterior_count >= 2:   # “four corners” satisfied per your rule
        present.add("four corners")

    missing = [p for p in required if p not in present]
    logger.debug(f"Photo check → present={sorted(list(present))}, missing={missing}, exterior_count={exterior_count}")
    return missing

# ----------------------------
# Labor/tax compliance
# ----------------------------
def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    """
    -50% if *all* labor rates (body, paint, mechanical, structural) are missing.
    -25% if tax required by rules but no tax/percent found in the estimate text.
    """
    adj = 0
    # find any rate near the label within 120 chars
    def has_rate(label: str) -> bool:
        pat = rf"{label}[^\n]{{0,120}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        return re.search(pat, text, flags=re.IGNORECASE) is not None

    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    found_any = any(has_rate(lbl) for lbl in labels)

    if not found_any:
        adj -= 50
        logger.debug("Labor rates missing for all sections → -50%")
    else:
        logger.debug("At least one labor rate detected → no labor deduction")

    # Tax only if rules require
    if re.search(r"tax\s*(required|must|utilize|apply)", client_rules, re.IGNORECASE):
        if not re.search(r"(sales\s*tax|tax)[^\n]{0,80}?(\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(\.\d{2})?)", text, re.IGNORECASE):
            adj -= 25
            logger.debug("Tax required but not found → -25%")
        else:
            logger.debug("Tax found → no tax deduction")

    return adj

# ----------------------------
# Root
# ----------------------------
@app.get("/")
async def root():
    return {"status": "ok"}

# ----------------------------
# Vision Review
# ----------------------------
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

    texts: List[str] = []
    image_blobs: List[Tuple[str, bytes]] = []
    images_for_vision = []

    # Read all uploads once
    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_blobs.append((name, raw))
            b64 = base64.b64encode(raw).decode("utf-8")
            images_for_vision.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(raw)))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {f.filename}")

    combined_text = "\n".join(texts)
    logger.debug(f"Combined OCR/text sample: {combined_text[:1200]}...")

    # Photo checks (now robust for 4 corners)
    missing_photos = check_required_photos(image_blobs, combined_text)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"

    # Advisor report quick check
    advisor_hint = ""
    if re.search(r"advisor report", combined_text, re.IGNORECASE):
        advisor_hint = "\n\nCONFIRMED: CCC Advisor Report is included based on OCR."

    # Compose vision message
    vision_message = {"role": "user", "content": []}
    if texts:
        vision_message["content"].append({"type": "text", "text": combined_text + advisor_hint + photo_hint})
    if images_for_vision:
        vision_message["content"].extend(images_for_vision)

    # System prompt
    system_prompt = f"""
You are an AI auto damage auditor. Evaluate STRICTLY by these rules:

- Start at 100% and deduct only for: labor (-50% if ALL sections missing), tax (-25% if rules require but not present), photos (-25% per missing type), parts (-25% if 2024–2025 vehicle uses LKQ/AM in violation).
- Required photos: four corners, odometer, VIN, license plate.
- "Four corners" is satisfied only if all four exterior corner views are present. Use the MISSING PHOTOS hint computed for you.
- Do NOT assume total loss unless explicitly stated.
- If any labor rate is present (body OR paint OR mechanical OR structural), do NOT apply the -50% deduction.
- Only mark items present when clearly found in the estimate text or photos.

Now summarize the findings and list each deduction you actually applied.
Rules to follow from client:
{client_rules}
"""

    # Call OpenAI (vision)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, vision_message],
            max_tokens=3000,
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
        logger.debug(f"GPT output sample: {gpt_output[:1000]}...")
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "gpt_output": "⚠️ AI review failed."})

    # Parse key fields from OCR text first (authoritative), fall back to GPT text
    claim_number = extract_claim_from_text(combined_text) or extract_claim_from_text(gpt_output) or "N/A"
    vin = extract_vin_from_text(combined_text) or extract_vin_from_text(gpt_output) or "N/A"
    vehicle_desc = extract_vehicle_from_text(combined_text) or extract_vehicle_from_text(gpt_output) or "N/A"

    # Score adjustments (labor/tax + missing photos)
    try:
        ai_score_text = re.search(r"Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?", gpt_output, re.IGNORECASE)
        ai_score = int(ai_score_text.group(1)) if ai_score_text else 100
    except Exception:
        ai_score = 100

    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    final_score = max(0, ai_score + labor_tax_adj + photo_adj)
    if final_score < 100 and (labor_tax_adj == 0 and photo_adj == 0):
        # sanity override: if AI deducted but our checks show no reasons, restore to 100
        final_score = 100

    # ----------------------------
    # PDF generation (font fallback)
    # ----------------------------
    pdf = FPDF()
    pdf.add_page()
    # try DejaVu; fallback to built-in
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align='C')
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"File Number: {file_number}")
    pdf.multi_cell(0, 8, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 8, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(5)
    pdf.set_font_size(10)
    header = (
        f"Claim #: {claim_number}\n"
        f"VIN: {vin}\n"
        f"Vehicle: {vehicle_desc}\n"
        f"Adjusted Compliance Score: {final_score}%\n"
    )
    pdf.multi_cell(0, 8, header)
    pdf.ln(2)
    pdf.multi_cell(0, 8, "AI-4-IA Review Summary:")
    pdf.ln(1)
    pdf.multi_cell(0, 6, gpt_output)

    pdf_path = f"{file_number}.pdf"
    try:
        pdf.output(pdf_path)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # ----------------------------
    # Email
    # ----------------------------
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_number}
VIN: {vin}
Vehicle: {vehicle_desc}

Adjusted Compliance Score: {final_score}%

AI Review Summary:
{gpt_output}
"""
        msg.set_content(email_body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing without failure): {e}")

    return {
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin,
        "score": f"{final_score}%"
    }

# ----------------------------
# Download PDF
# ----------------------------
@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = f"{file_number}.pdf"
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=pdf_path)
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

# ----------------------------
# Client rules endpoint
# ----------------------------
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

