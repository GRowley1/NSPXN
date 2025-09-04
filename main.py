from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os
import re
import io
import base64
import json
import logging

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
from openai import OpenAI

# =========================================
# PDF storage: save to /tmp but filename stays {file_number}.pdf (no _tmp_ prefix)
# =========================================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

# =========================================
# Logging
# =========================================
logging.basicConfig(
    level=logging.DEBUG,
    filename="app.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================
# OpenAI client (fixed to gpt-4o)
# =========================================
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

# =========================================
# FastAPI app + CORS
# =========================================
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

# =========================================
# OCR helpers
# =========================================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
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
                logger.warning(f"OCR page {i} too short; skipping noise.")
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return ""

def extract_text_from_docx(file_like: io.BytesIO) -> str:
    try:
        doc = Document(file_like)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX read error: {e}")
        return ""

# =========================================
# VIN utilities (normalization + checksum)
# =========================================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

def normalize_vin(s: str) -> Optional[str]:
    s = s.strip().upper()
    s = s.replace(" ", "")
    # Common OCR fixes
    s = s.replace("O", "0").replace("I", "1").replace("Q", "0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

# ISO 3779 VIN checksum
_translit = {**{str(i): i for i in range(10)},
             **dict(A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8,
                    J=1, K=2, L=3, M=4, N=5, P=7, R=9,
                    S=2, T=3, U=4, V=5, W=6, X=7, Y=8, Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def vin_checksum_ok(v: str) -> bool:
    if len(v) != 17: return False
    try:
        total = 0
        for i, ch in enumerate(v):
            val = _translit[ch]
            total += val * _weights[i]
        check = total % 11
        return v[8] == ("X" if check == 10 else str(check))
    except Exception:
        return False

def best_vin_candidate(cands: List[str]) -> Optional[str]:
    for c in cands:
        vin = normalize_vin(c)
        if vin and vin_checksum_ok(vin):
            return vin
    # fallback: accept normalized even if checksum fails (some imports/older docs)
    for c in cands:
        vin = normalize_vin(c)
        if vin:
            return vin
    return None

# =========================================
# Field extraction (Claim #, VIN, Vehicle)
# =========================================
def extract_claim_from_text(text: str) -> Optional[str]:
    patterns = [
        r"(?:^|\s)(?:Claim\s*(?:#|No\.?|Number)[:\s]*)\s*([A-Za-z0-9\-]+)",
        r"(?:^|\s)Claim\s*[:#]\s*([A-Za-z0-9\-]+)"
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    # prioritize labeled VIN lines
    label_block = re.findall(r"(?:^|\n).{0,40}VIN[:\s\-]*([A-HJ-NPR-Z0-9]{10,20}).*", text, re.IGNORECASE)
    if label_block:
        vin = best_vin_candidate(label_block)
        if vin:
            return vin
    # fallback: any 17-char VIN-like
    candidates = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, re.IGNORECASE)
    return best_vin_candidate(candidates)

def extract_vehicle_from_text(text: str) -> Optional[str]:
    m1 = re.search(r"\b(20\d{2})\s+([A-Za-z]{3,})\s+([A-Za-z0-9\-]{2,})", text)
    m2 = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    if m1:
        year, make, model = m1.groups()
        miles = m2.group(1) if m2 else "Mileage unknown"
        return f"{year} {make} {model}, {miles} miles"
    return None

# =========================================
# Photo parsing & requirements
# =========================================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    # Try OCR at 4 rotations for robustness
    rots = [0, 90, 180, 270]
    found: List[str] = []
    for name, blob in image_blobs:
        try:
            base = Image.open(io.BytesIO(blob))
            for r in rots:
                img = base.rotate(r, expand=True)
                proc = preprocess_image(img)
                # more permissive PSMs
                for psm in ("--psm 7", "--psm 6", "--psm 11"):
                    ocr = pytesseract.image_to_string(proc, lang="eng", config=psm)
                    cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", ocr.upper())
                    if cands:
                        found.extend(cands)
        except Exception as e:
            logger.warning(f"VIN photo OCR error ({name}): {e}")
    return best_vin_candidate(found)

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = preprocess_image(img)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", ocr, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception as e:
            logger.warning(f"Odometer photo OCR error ({name}): {e}")
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

# =========================================
# Labor/tax compliance checks
# =========================================
def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    def has_rate(label: str) -> bool:
        pat = rf"{label}[^\n]{{0,120}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        return re.search(pat, text, re.IGNORECASE) is not None
    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    if not any(has_rate(lbl) for lbl in labels):
        adj -= 50
    if re.search(r"tax\s*(required|must|utilize|apply)", client_rules, re.IGNORECASE):
        if not re.search(r"(sales\s*tax|tax)[^\n]{0,80}?(\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(\.\d{2})?)", text, re.IGNORECASE):
            adj -= 25
    return adj

# =========================================
# Estimate parsing (line items for comparison)
# =========================================
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
            side = "unspecified"
            if "left" in l or re.search(r"\blh\b", l): side = "left"
            if "right" in l or re.search(r"\brh\b", l): side = "right"
            op = next((op for op in OPS if op in l), "unspecified")
            panel = next((p for p in PANELS if p in l), "component")
            items.append({"op": op, "part": panel, "side": side, "raw": line.strip()})
    uniq, seen = [], set()
    for it in items:
        key = (it["op"], it["part"], it["side"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

# =========================================
# GPT compare: estimate ↔ photos (JSON)
# =========================================
def compare_estimate_with_photos(items: List[Dict[str, str]],
                                 images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns dict:
      per_item: [{op,part,side,photo_evidence,confidence,note}]
      not_in_photos: [raw...]
      extra_damage_in_photos: ["desc"...]
      overall: "short"
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

    system = (
        "You are an auto-damage visual auditor. "
        "Given estimate line items and vehicle photos, decide for EACH item whether visible photo evidence exists. "
        "Hidden ops (calibration, internal R&I) may not be visible → mark as no-evidence with a short 3–10 word note. "
        "Also list obvious damages seen in photos that are NOT listed in the estimate. "
        "Return STRICT JSON ONLY per this schema:\n" + json.dumps(schema)
    )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": "Estimate items:\n" + json.dumps(items, ensure_ascii=False)}
    ]
    user_parts.extend(images_for_vision)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_parts}
            ],
            max_tokens=1200,
            temperature=0
        )
        txt = (rsp.choices[0].message.content or "").strip()
        txt = txt.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(txt)
        if not isinstance(data, dict) or "per_item" not in data:
            raise ValueError("JSON shape mismatch")
        return data
    except Exception as e:
        logger.error(f"Vision compare JSON error: {type(e).__name__}: {e}")
        return {
            "per_item": [],
            "not_in_photos": [],
            "extra_damage_in_photos": [],
            "overall": f"Comparison unavailable ({type(e).__name__})."
        }

# =========================================
# PDF helpers
# =========================================
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12)
    pdf.cell(0, 8, txt=title, ln=True)
    pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"{key}: {val}")

# =========================================
# Routes
# =========================================
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

    # ----- read uploads once
    texts: List[str] = []
    image_blobs: List[Tuple[str, bytes]] = []
    images_for_vision: List[Dict[str, Any]] = []

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

    # ----- photo checks + VIN/odo
    missing_photos = check_required_photos(image_blobs, combined_text)

    vin_est = extract_vin_from_text(combined_text)
    vin_photos = extract_vin_from_photos(image_blobs)
    vin_final = vin_est or vin_photos or "N/A"

    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    claim_number = extract_claim_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    # ----- parse estimate items & compare to photos
    est_items = extract_estimate_items(combined_text)
    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    # ----- vision narrative (compliance summary)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"
    system_prompt = f"""
You are an AI auto damage auditor. Evaluate STRICTLY by these rules:

- Start at 100% and deduct only for: labor (-50% if ALL sections missing), tax (-25% if rules require but not present), photos (-25% per missing type), parts (-25% if a 2024–2025 vehicle uses LKQ/AM in violation).
- Required photos: four corners, odometer, VIN, license plate.
- "Four corners" is satisfied if at least two exterior corner views are present (already computed for you).
- Do NOT assume total loss unless explicitly stated.
- If any labor rate is present (body OR paint OR mechanical OR structural), do NOT apply the -50% deduction.

Rules to follow from client:
{client_rules}
""".strip()

    user_parts: List[Dict[str, Any]] = []
    if combined_text:
        user_parts.append({"type": "text", "text": combined_text + photo_hint})
    if images_for_vision:
        user_parts.extend(images_for_vision)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_parts}
            ],
            max_tokens=900
        )
        gpt_output = response.choices[0].message.content or "⚠️ GPT returned no output."
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.error(f"OpenAI error: {err}")
        gpt_output = f"⚠️ AI review failed: {err}"

    # ----- score adjustments
    ai_score_match = re.search(r"Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?", gpt_output, re.IGNORECASE)
    ai_score = int(ai_score_match.group(1)) if ai_score_match else 100
    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    final_score = max(0, ai_score + labor_tax_adj + photo_adj)
    if final_score < 100 and (labor_tax_adj == 0 and photo_adj == 0):
        final_score = 100
    # Use "Total Evaluation" percentage if the AI provided one; else fall back to final_score
    total_eval_match = re.search(r"Total\s*Evaluation\s*[:\-]?\s*(\d{1,3})\s*%?", gpt_output, re.IGNORECASE)
    total_eval_pct = int(total_eval_match.group(1)) if total_eval_match else final_score

    
    # =========================================
    # PDF build (uses your same style)
    # =========================================
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font_size(10)
    pdf.multi_cell(0, 8, f"File Number: {file_number}")
    pdf.multi_cell(0, 8, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 8, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Claim #: {claim_number}")
    pdf.multi_cell(0, 6, f"VIN: {vin_final}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    if odo_photos:
        pdf.multi_cell(0, 6, f"Odometer (from photos): {odo_photos}")
    pdf.multi_cell(0, 6, f"Compliance Score: {total_eval_pct}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, gpt_output)

    # ======== New: full Estimate ↔ Photos Consistency Review ========
    pdf.ln(4)
    pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")

    if consistency.get("per_item"):
        for it in consistency["per_item"][:40]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            try:
                conf = float(it.get("confidence", 0))
            except Exception:
                conf = 0.0
            conf_txt = f"{round(conf*100)}%"
            line = f"- {it.get('side','unspecified').title()} {it.get('part','component')} · {it.get('op','op')} → Photo: {ev} ({conf_txt}); {it.get('note','')}"
            pdf.multi_cell(0, 6, line)
    else:
        pdf.multi_cell(0, 6, "Per-item comparison unavailable.")

    if consistency.get("not_in_photos"):
        pdf.ln(2)
        pdf_add_section_title(pdf, "Items Estimated but Not Evident in Photos")
        for raw in consistency["not_in_photos"][:20]:
            pdf.multi_cell(0, 6, f"- {raw}")

    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2)
        pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:20]:
            pdf.multi_cell(0, 6, f"- {d}")

    pdf.ln(2)
    pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    # ---------- Save PDF to /tmp with name {file_number}.pdf (no prefix) ----------
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # OPTIONAL email (unchanged; comment out if undesired)
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
VIN: {vin_final}
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
        logger.error(f"Email error (continuing): {e}")

    return {
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin_final,
        "score": f"{final_score}%",
        "consistency_review": consistency
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
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
            logger.debug(f"Client rules for {client_name}: {text[:500]}...")
            return {"text": text}
        except Exception as e:
            logger.error(f"Client rules error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        logger.error(f"Rules not found for client: {client_name}")
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})






