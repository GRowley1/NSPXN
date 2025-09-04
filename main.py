from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os
import re
import io
import base64
import json
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
        "https://nspxn.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# OCR helpers
# ----------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2)
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
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config="--psm 6")
            except Exception:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config="--psm 3")
            if len(ocr_text.strip()) < 30:
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def extract_text_from_docx(file_like: io.BytesIO) -> str:
    doc = Document(file_like)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

# ----------------------------
# Field extraction
# ----------------------------
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

def normalize_vin(s: str) -> Optional[str]:
    s = s.strip().upper()
    # remove common OCR noise
    s = s.replace(" ", "").replace("O", "0").replace("I", "1").replace("Q", "0")
    if len(s) == 17 and all(ch in VIN_ALLOWED for ch in s):
        return s
    return None

def extract_claim_from_text(text: str) -> Optional[str]:
    pats = [
        r"(?:^|\s)(?:Claim\s*(?:#|No\.?|Number)[:\s]*)\s*([A-Za-z0-9\-]+)",
        r"(?:^|\s)Claim\s*[:#]\s*([A-Za-z0-9\-]+)"
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    candidates = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, re.IGNORECASE)
    for c in candidates:
        vin = normalize_vin(c)
        if vin:
            return vin
    return None

def extract_vehicle_from_text(text: str) -> Optional[str]:
    m1 = re.search(r"\b(20\d{2})\s+([A-Za-z]{3,})\s+([A-Za-z0-9\-]{2,})", text)
    m2 = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    if m1:
        year, make, model = m1.group(1), m1.group(2), m1.group(3)
        miles = m2.group(1) if m2 else "Mileage unknown"
        return f"{year} {make} {model}, {miles} miles"
    return None

# ----------------------------
# Photo checks
# ----------------------------
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    entropy = ImageStat.Stat(processed).var[0]
    return len(text.strip()) < 10 and entropy > 150

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            processed = preprocess_image(img)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            for c in re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", ocr.upper()):
                vin = normalize_vin(c)
                if vin:
                    return vin
        except Exception as e:
            logger.warning(f"VIN photo OCR error ({name}): {e}")
    return None

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            processed = preprocess_image(img)
            ocr = pytesseract.image_to_string(processed, lang="eng")
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", ocr, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception as e:
            logger.warning(f"Odo photo OCR error ({name}): {e}")
    return None

def check_required_photos(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> List[str]:
    required = ["four corners", "odometer", "vin", "license plate"]
    present = set()
    txt = ocr_text.lower()

    if any(k in txt for k in ["odometer", "mileage photo", "dashboard mileage"]):
        present.add("odometer")
    if any(k in txt for k in ["vin", "vehicle identification number", "vin photo"]):
        present.add("vin")
    if any(k in txt for k in ["license plate", "registration plate"]):
        present.add("license plate")

    ext = 0
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = preprocess_image(img)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr, re.IGNORECASE):
                present.add("vin")
            if re.search(r"\d{1,3}(,\d{3})*\s*(miles|km)", ocr, re.IGNORECASE):
                present.add("odometer")
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", ocr, re.IGNORECASE):
                present.add("license plate")
            if _image_is_exterior_wide(img):
                ext += 1
        except Exception as e:
            logger.warning(f"Image parse error {name}: {e}")

    if ext >= 2:
        present.add("four corners")

    return [p for p in required if p not in present]

# ----------------------------
# Labor/tax compliance
# ----------------------------
def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0

    def has_rate(label: str) -> bool:
        pat = rf"{label}[^\n]{{0,120}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        return re.search(pat, text, re.IGNORECASE) is not None

    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    if not any(has_rate(lbl) for lbl in labels):
        adj -= 50

    if re.search(r"tax\s*(required|must|utilize|apply)", client_rules, re.IGNORECASE):
        if not re.search(r"(sales\s*tax|tax)[^\n]{0,80}?(\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(\.\d{2})?)",
                         text, re.IGNORECASE):
            adj -= 25
    return adj

# ----------------------------
# Estimate parsing (line items)
# ----------------------------
PANELS = [
    "bumper", "fender", "door", "hood", "grille", "headlamp", "headlight",
    "taillamp", "tail lamp", "quarter panel", "rocker", "roof", "trunk",
    "decklid", "mirror", "apron", "radiator support", "wheel", "tire",
    "pillar", "garnish", "molding", "fog lamp", "reinforcement", "cover"
]
OPS = ["replace", "repair", "refinish", "r&i", "r & i", "align", "blend", "calibrate"]

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for line in text.splitlines():
        l = line.strip().lower()
        if not l or len(l) < 6:
            continue
        if any(op in l for op in OPS) and any(p in l for p in PANELS):
            # side detection
            side = "unspecified"
            if "left" in l or "lh" in l:
                side = "left"
            if "right" in l or "rh" in l:
                side = "right"
            op = next((op for op in OPS if op in l), "unspecified")
            # panel/part (best-effort)
            panel = next((p for p in PANELS if p in l), "component")
            items.append({"op": op, "part": panel, "side": side, "raw": line.strip()})
    # de-dup conservative
    uniq = []
    seen = set()
    for it in items:
        key = (it["op"], it["part"], it["side"])
        if key not in seen:
            uniq.append(it)
            seen.add(key)
    return uniq

# ----------------------------
# GPT: compare estimate ↔ photos
# ----------------------------
def compare_estimate_with_photos(items: List[Dict[str, str]],
                                 images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns dict:
      {
        "per_item": [{"op": "...", "part":"...", "side":"...", "photo_evidence": true/false, "confidence": 0-1, "note": ""}, ...],
        "not_in_photos": [...raw lines...],
        "extra_damage_in_photos": ["desc", ...],
        "overall": "short summary"
      }
    """
    schema = {
        "type": "object",
        "properties": {
            "per_item": {"type":"array","items":{
                "type":"object",
                "properties":{
                    "op":{"type":"string"},
                    "part":{"type":"string"},
                    "side":{"type":"string"},
                    "photo_evidence":{"type":"boolean"},
                    "confidence":{"type":"number"},
                    "note":{"type":"string"}
                },
                "required":["op","part","side","photo_evidence","confidence","note"]
            }},
            "not_in_photos":{"type":"array","items":{"type":"string"}},
            "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
            "overall":{"type":"string"}
        },
        "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]
    }

    sys = (
        "You are an auto-damage visual auditor. "
        "Given estimate line items and vehicle photos, decide for EACH item whether there is visible photo evidence. "
        "Use common sense: some operations (calibration, R&I hidden parts) may not be visible → mark as no-evidence with a short note. "
        "Also list obvious damages seen in photos that are NOT listed in the estimate."
        "\nReturn STRICT JSON ONLY matching this schema:\n"
        + json.dumps(schema)
        + "\nValues rules: confidence 0.0–1.0; notes 3–10 words."
    )

    user_content = [{"type": "text", "text": "Estimate items:\n" + json.dumps(items, ensure_ascii=False)}]
    user_content.extend(images_for_vision)

    try:
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1500,
            temperature=0
        )
        txt = rsp.choices[0].message.content or "{}"
        # Trim code fences if any
        txt = txt.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(txt)
        # basic shape check
        if not isinstance(data, dict) or "per_item" not in data:
            raise ValueError("JSON shape mismatch")
        return data
    except Exception as e:
        logger.error(f"Vision compare JSON error: {e}")
        # graceful fallback
        return {
            "per_item": [],
            "not_in_photos": [],
            "extra_damage_in_photos": [],
            "overall": "Comparison unavailable (AI parsing error)."
        }

# ----------------------------
# PDF helpers
# ----------------------------
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12)
    pdf.cell(0, 8, txt=title, ln=True)
    pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"{key}: {val}")

# ----------------------------
# API
# ----------------------------
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

    texts: List[str] = []
    image_blobs: List[Tuple[str, bytes]] = []
    images_for_vision = []

    # Read uploads once
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

    combined_text = "\n".join(texts)

    # Photo presence checks
    missing_photos = check_required_photos(image_blobs, combined_text)

    # VIN & odometer consolidation
    vin_est = extract_vin_from_text(combined_text)
    vin_photos = extract_vin_from_photos(image_blobs)
    vin_final = vin_est or vin_photos or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    # Parse estimate line items
    est_items = extract_estimate_items(combined_text)

    # Compare estimate ↔ photos (per item + extras)
    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    # Vision narrative (same as before, but we add hints)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"
    system_prompt = f"""
You are an AI auto damage auditor. Evaluate STRICTLY by these rules:

- Start at 100% and deduct only for: labor (-50% if ALL sections missing), tax (-25% if rules require but not present), photos (-25% per missing type), parts (-25% if 2024–2025 vehicle uses LKQ/AM in violation).
- Required photos: four corners, odometer, VIN, license plate.
- "Four corners" is satisfied if at least two exterior corner views are present (already computed for you).
- Do NOT assume total loss unless explicitly stated.
- If any labor rate is present (body OR paint OR mechanical OR structural), do NOT apply the -50% deduction.

Rules to follow from client:
{client_rules}
"""
    vision_message = {"role": "user", "content": []}
    if combined_text:
        vision_message["content"].append({"type": "text", "text": combined_text + photo_hint})
    if images_for_vision:
        vision_message["content"].extend(images_for_vision)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, vision_message],
            max_tokens=500
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        gpt_output = "⚠️ AI review failed."
    
    # Parse core fields
    claim_number = extract_claim_from_text(combined_text) or "N/A"
    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    vin = vin_final

    # Score adjustments
    ai_score_match = re.search(r"Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?", gpt_output, re.IGNORECASE)
    ai_score = int(ai_score_match.group(1)) if ai_score_match else 100
    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    final_score = max(0, ai_score + labor_tax_adj + photo_adj)
    if final_score < 100 and (labor_tax_adj == 0 and photo_adj == 0):
        final_score = 100

    # ----------------------------
    # PDF
    # ----------------------------
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(0, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.ln(2)
    pdf_kv(pdf, "File Number", file_number)
    pdf_kv(pdf, "IA Company", ia_company)
    pdf_kv(pdf, "Appraiser ID #", appraiser_id)
    pdf.ln(2)
    pdf_kv(pdf, "Claim #", claim_number)
    pdf_kv(pdf, "VIN", vin)
    pdf_kv(pdf, "Vehicle", vehicle_desc)
    if odo_photos:
        pdf_kv(pdf, "Odometer (from photos)", odo_photos)
    pdf_kv(pdf, "Adjusted Compliance Score", f"{final_score}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, gpt_output)

    # NEW: Consistency section
    pdf.ln(4)
    pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")

    # Per-item table (compact)
    if consistency["per_item"]:
        pdf.set_font_size(10)
        for it in consistency["per_item"][:40]:  # keep PDF compact
            ev = "YES" if it.get("photo_evidence") else "NO"
            conf = f"{round(float(it.get('confidence', 0))*100)}%"
            line = f"- {it['side'].title()} {it['part']} · {it['op']} → Photo: {ev} ({conf}); {it.get('note','')}"
            pdf.multi_cell(0, 6, line)
    else:
        pdf.multi_cell(0, 6, "Per-item comparison unavailable.")

    # Not in photos
    if consistency["not_in_photos"]:
        pdf.ln(2)
        pdf_add_section_title(pdf, "Items Estimated but Not Evident in Photos")
        for raw in consistency["not_in_photos"][:20]:
            pdf.multi_cell(0, 6, f"- {raw}")

    # Extra damage
    if consistency["extra_damage_in_photos"]:
        pdf.ln(2)
        pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:20]:
            pdf.multi_cell(0, 6, f"- {d}")

    # Overall
    pdf.ln(2)
    pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    pdf_path = f"{file_number}.pdf"
    try:
        pdf.output(pdf_path)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # Email (best-effort; ignore failures)
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_number}
VIN: {vin}
Vehicle: {vehicle_desc}
Adjusted Compliance Score: {final_score}%

Consistency Overall: {consistency.get('overall','')}

(See attached PDF for full details.)
"""
        msg.set_content(body)
        # NOTE: Fill in your SMTP credentials or keep disabled in production build script
        # with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
        #     smtp.login("info@nspxn.com", "YOUR_PASSWORD")
        #     smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error: {e}")

    return {
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin,
        "score": f"{final_score}%",
        "consistency_review": consistency
    }

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
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return {"text": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})


