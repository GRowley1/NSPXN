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
import math
import datetime
import hashlib

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
from openai import OpenAI

# =========================================
# PDF storage
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
# OpenAI client (gpt-4o)
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
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    return img

def ocr_text_fast(img: Image.Image, psm: int = 6) -> str:
    try:
        proc = preprocess_image(img)
        config = f"--psm {psm} --oem 3"
        return pytesseract.image_to_string(proc, lang="eng", config=config)
    except Exception as e:
        logger.warning(f"OCR fast error: {e}")
        return ""

def extract_text_from_pdf(file_like: io.BytesIO, max_ocr_pages: int = 8, dpi: int = 200) -> str:
    try:
        file_like.seek(0)
        pages = convert_from_bytes(file_like.read(), dpi=dpi)
        text_output = ""
        for i, img in enumerate(pages, 1):
            if i > max_ocr_pages:
                break
            page_text = ocr_text_fast(img, psm=7)
            if len(page_text.strip()) < 20:
                page_text = ocr_text_fast(img, psm=3)
            if not page_text.strip():
                continue
            text_output += f"\n[Page {i}]\n{page_text}"
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

def extract_text_from_pdf_embedded(pdf_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for p in reader.pages:
            t = p.extract_text() or ""
            parts.append(t)
        return "\n".join(parts)
    except Exception as e:
        logger.debug(f"Embedded text extraction failed: {e}")
        return ""

# =========================================
# Photo harvesting (skip for estimates)
# =========================================
def _page_var(img: Image.Image) -> float:
    g = img.convert("L")
    return ImageStat.Stat(g).var[0]

def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int = 20, dpi: int = 135) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
        for i, page in enumerate(pages, 1):
            var = _page_var(page)
            if var > 110:
                buf = io.BytesIO()
                page.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
                out.append((f"pdf-photo-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        logger.warning(f"harvest_photos_from_pdf error: {e}")
    return out

# =========================================
# VIN utilities
# =========================================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

def normalize_vin(s: str) -> Optional[str]:
    s = s.strip().upper()
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", s)
    s = s.replace("O", "0").replace("I", "1").replace("Q", "0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

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
            total += _translit[ch] * _weights[i]
        check = total % 11
        return v[8] == ("X" if check == 10 else str(check))
    except Exception:
        return False

def best_vin_candidate(cands: List[str]) -> Optional[str]:
    for c in cands:
        vin = normalize_vin(c)
        if vin and vin_checksum_ok(vin):
            return vin
    return None

VIN_LABEL = re.compile(r'(?i)\bV[\W_]*I[\W_]*N\b')
VIN_PHRASE = re.compile(r'(?i)\bVehicle\s*Identification\s*Number\b')
VIN_SEP_SEQ = re.compile(r'(?i)((?:[A-HJ-NPR-Z0-9][\s\.\-–—:_]){16}[A-HJ-NPR-Z0-9])')

def _extract_vin_near_positions(text: str, positions: List[int], radius: int = 240) -> Optional[str]:
    for pos in positions:
        window = text[pos: pos + radius]
        for m in VIN_SEP_SEQ.finditer(window):
            vin = normalize_vin(m.group(1))
            if vin and vin_checksum_ok(vin):
                return vin
        cands = re.findall(r'([A-HJ-NPR-Z0-9]{17})', window)
        vin = best_vin_candidate(cands)
        if vin: return vin
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    label_positions = [m.end() for m in VIN_LABEL.finditer(text)]
    label_positions += [m.end() for m in VIN_PHRASE.finditer(text)]
    vin = _extract_vin_near_positions(text, label_positions, radius=240)
    if vin:
        return vin
    for m in VIN_SEP_SEQ.finditer(text):
        vin = normalize_vin(m.group(1))
        if vin and vin_checksum_ok(vin):
            return vin
    cands = re.findall(r'\b([A-HJ-NPR-Z0-9]{17})\b', text)
    return best_vin_candidate(cands)

def extract_vin_from_image(img_bytes: bytes) -> Optional[str]:
    try:
        buf = io.BytesIO()
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Extract the 17-character VIN from the image if it's a VIN plate or label. Output only the VIN or nothing if not found or unclear."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is the VIN in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
        )
        vin = response.choices[0].message.content.strip()
        norm_vin = normalize_vin(vin)
        if norm_vin and vin_checksum_ok(norm_vin):
            return norm_vin
        return None
    except Exception as e:
        logger.warning(f"GPT VIN extraction error: {e}")
        return None

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for _, bytes in image_blobs:
        vin = extract_vin_from_image(bytes)
        if vin:
            return vin
    for _, bytes in image_blobs:
        img = Image.open(io.BytesIO(bytes))
        txt = ocr_text_fast(img, psm=7)
        vin = extract_vin_from_text(txt)
        if vin:
            return vin
    return None

def extract_vin_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 300) -> Optional[str]:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max(1, pages_to_scan)]
        for img in pages:
            vin = extract_vin_from_image(img.tobytes())
            if vin:
                return vin
            txt = ocr_text_fast(img, psm=7)
            v = extract_vin_from_text(txt)
            if v: return v
    except Exception as e:
        logger.warning(f"VIN first-pages OCR error: {e}")
    return None

# =========================================
# Claim extraction
# =========================================
CLAIM_AFTER_LABEL = re.compile(
    r'(?is)\bclaim\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'
)
ALT_CLAIM_LABELS = [
    re.compile(r'(?is)\bloss\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
    re.compile(r'(?is)\bfile\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
    re.compile(r'(?is)\bref(?:erence)?\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
    re.compile(r'(?is)\bassignment\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
]

_CLAIM_BLACKLIST = {"SERVICE", "SERVICES", "PHONE", "EMAIL", "FAX", "TOTAL", "POLICY"}

def _clean_claim(c: str) -> str:
    c = c.strip().strip(':').strip().strip('.').strip('-')
    c = c.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    c = c.replace("_", "")
    c = re.sub(r'\s+', '', c)
    c = re.sub(r'(?:V\d+)$', '', c, flags=re.IGNORECASE)
    return c

def _valid_claim_candidate(c: str) -> bool:
    if not c or len(c) < 3:
        return False
    if not re.search(r'\d', c):
        return False
    if c.upper() in _CLAIM_BLACKLIST:
        return False
    return True

def extract_claim_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for m in CLAIM_AFTER_LABEL.finditer(text):
        cand = _clean_claim(m.group(1))
        if _valid_claim_candidate(cand):
            return cand
    for pat in ALT_CLAIM_LABELS:
        for m in pat.finditer(text):
            cand = _clean_claim(m.group(1))
            if _valid_claim_candidate(cand):
                return cand
    return None

def extract_claim_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 300) -> Optional[str]:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max(1, pages_to_scan)]
        for img in pages:
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Extract the claim number from this image. Output only the claim number or nothing."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is the claim number?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]}
                ]
            )
            claim = response.choices[0].message.content.strip()
            if _valid_claim_candidate(claim):
                return _clean_claim(claim)
            txt = ocr_text_fast(img, psm=6)
            c = extract_claim_from_text(txt)
            if c: return c
    except Exception as e:
        logger.warning(f"Claim first-pages OCR error: {e}")
    return None

# =========================================
# Key extraction using GPT-4o vision
# =========================================
def extract_keys_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 300) -> Dict[str, Any]:
    extracted = {
        "vin": None,
        "claim_number": None,
        "vehicle_description": None,
        "mileage": None,
        "labor_rate": None,
        "tax_rate": None
    }
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:pages_to_scan]
        for img in pages:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": 'Extract key fields from this estimate page as JSON: {"vin": "17-char VIN or null", "claim_number": "claim # or null", "vehicle_description": "the full description of the vehicle including year, make, model, trim, engine, color or null if not found", "mileage": "the odometer reading as string without commas or null", "labor_rate": "rate like $60.00 /hr or null", "tax_rate": "rate like 8.7500% or null"}.'},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract the fields."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]}
                ]
            )
            try:
                data = json.loads(response.choices[0].message.content)
                for key, value in data.items():
                    if value and not extracted[key]:
                        if key == "vin":
                            extracted[key] = normalize_vin(value)
                        elif key == "claim_number":
                            extracted[key] = _clean_claim(value)
                        elif key == "mileage":
                            extracted[key] = re.sub(r'[^\d]', '', value) if value else None
                        elif key == "labor_rate":
                            if re.search(r'\$\d+\.?\d*\s*/\s*hr', value, re.I):
                                extracted[key] = value
                        elif key == "tax_rate":
                            if re.search(r'\d+\.?\d*%', value):
                                extracted[key] = value
                        else:
                            extracted[key] = value
            except Exception as e:
                logger.warning(f"JSON parse error in key extraction: {e}")
            if all(extracted.values()):
                break
    except Exception as e:
        logger.error(f"Key extraction error: {e}")
    return extracted

# =========================================
# Vehicle & tax/parts helpers
# =========================================
MAKE_FIX = {
    "nessan": "Nissan","nisaan": "Nissan","nissan": "Nissan","toy0ta": "Toyota",
    "chevroler": "Chevrolet","cheverolet": "Chevrolet","chevrolet": "Chevrolet",
    "chev": "Chevrolet","chev.": "Chevrolet","chev," : "Chevrolet"
}
def normalize_vehicle_str(s: str) -> str:
    if not s: return s
    s2 = s
    for wrong, right in MAKE_FIX.items():
        s2 = re.sub(rf'\b{re.escape(wrong)}\b', right, s2, flags=re.IGNORECASE)
        s2 = re.sub(rf'\b{re.escape(wrong.upper())}\b', right, s2)
    s2 = re.sub(r'\s{2,}', ' ', s2).replace(' ,', ',')
    return s2.strip()

def extract_vehicle_from_text(text: str) -> Optional[str]:
    match = re.search(r'(?is)\b(\d{4})\s+([a-z0-9]+)\s+([a-z0-9\s]+?)(?=vin|license|odometer|\Z)', text)
    if match:
        year = match.group(1)
        make = match.group(2)
        model = match.group(3).strip()
        miles_match = re.search(r'(?is)odometer[\s:]*([\d,]+)', text)
        miles = miles_match.group(1) if miles_match else ""
        return normalize_vehicle_str(f"{year} {make} {model}, {miles} miles")
    return None

def parse_year_miles(text: str) -> Tuple[Optional[int], Optional[int]]:
    year_match = re.search(r'(?is)(vehicle|production date|year)[\s:]*(\d{4})', text)
    year = int(year_match.group(2)) if year_match else None
    if not year:
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        year = int(year_match.group(0)) if year_match and 1900 < int(year_match.group(0)) < 2100 else None
    miles_match = re.search(r'(?is)odometer[\s:]*([\d,]+)', text)
    miles_str = miles_match.group(1) if miles_match else None
    miles = int(re.sub(r'[,\.]', '', miles_str)) if miles_str else None
    return year, miles

def labor_rates_present_any(text: str) -> bool:
    return bool(re.search(r'(?i)\$\d+\.?\d*\s*/\s*hr|labor rate|body labor.*\$\d+', text))

def taxes_present(text: str) -> bool:
    return bool(re.search(r'(?i)tax.*@\s*\d+\.?\d*\s*%|sales tax.*%', text))

def non_oem_used(text: str) -> bool:
    return bool(re.search(r'(?i)\b(A/M|aftermarket|used|recond|non-oem)\b', text))

# =========================================
# Photo classification and required checks
# =========================================
def classify_photo(img_bytes: bytes) -> str:
    try:
        buf = io.BytesIO()
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "Classify this auto claim photo. Possible categories: vin_plate, odometer, license_plate, front_left_corner, front_right_corner, rear_left_corner, rear_right_corner, other_damage, unknown. Output only the category."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
        )
        category = response.choices[0].message.content.strip().lower()
        return category
    except Exception as e:
        logger.warning(f"Photo classification error: {e}")
        return "unknown"

def check_required_photos(image_blobs: List[Tuple[str, bytes]], combined_text: str) -> List[str]:
    present = {"four_corners": False, "vin": False, "odometer": False, "plate": False}
    corner_count = 0
    for _, b in image_blobs:
        cat = classify_photo(b)
        if cat == "vin_plate":
            present["vin"] = True
        elif cat == "odometer":
            present["odometer"] = True
        elif cat == "license_plate":
            present["plate"] = True
        elif "corner" in cat:
            corner_count += 1
    if corner_count >= 4:
        present["four_corners"] = True
    missing = [k for k, v in present.items() if not v]
    return missing

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for _, b in image_blobs:
        cat = classify_photo(b)
        if cat == "odometer":
            try:
                buf = io.BytesIO()
                Image.open(io.BytesIO(b)).convert("RGB").save(buf, format="JPEG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "Extract the odometer reading as integer string from this image, ignore other numbers."},
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}
                    ]
                )
                odo = response.choices[0].message.content.strip()
                if re.match(r'\d{3,6}(?:,\d{3})*', odo):
                    return odo
            except Exception as e:
                logger.warning(f"Odometer extraction error: {e}")
    return None

# =========================================
# Contact sheets
# =========================================
def make_contact_sheets_compact(image_blobs: List[Tuple[str, bytes]], max_sheets=3, cols=6, padding=6, base_thumb_w=320, jpeg_quality=68) -> List[Tuple[str, bytes]]:
    out = []
    images = []
    for _, b in image_blobs:
        try:
            images.append(Image.open(io.BytesIO(b)).convert("RGB"))
        except:
            pass
    for i in range(0, len(images), cols * 3):  # 3 rows per sheet
        sheet_images = images[i:i + cols * 3]
        if not sheet_images:
            break
        thumb_w = base_thumb_w
        aspect_ratios = [img.height / img.width for img in sheet_images]
        avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
        thumb_h = int(thumb_w * avg_aspect)
        sheet_w = cols * thumb_w + (cols + 1) * padding
        rows = math.ceil(len(sheet_images) / cols)
        sheet_h = rows * thumb_h + (rows + 1) * padding
        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
        for j, img in enumerate(sheet_images):
            thumb = img.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = padding + (j % cols) * (thumb_w + padding)
            y = padding + (j // cols) * (thumb_h + padding)
            sheet.paste(thumb, (x, y))
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=jpeg_quality)
        out.append((f"sheet_{len(out)+1}.jpg", buf.getvalue()))
        if len(out) >= max_sheets:
            break
    return out

# =========================================
# Estimate items and consistency
# =========================================
def extract_estimate_items(text: str) -> List[Dict]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract the repair items from this auto repair estimate text as JSON list of objects with keys: side ('front', 'rear', 'left', 'right', 'unspecified'), part (the part name), op ('repl', 'repair', 'other'), note (additional details like 'used', 'a/m', 'recond' or empty). Be accurate to the estimate content."},
            {"role": "user", "content": text}
        ]
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Estimate items extraction error: {e}")
        return []

def compare_estimate_with_photos(est_items: List[Dict], images_for_vision: List[Dict]) -> Dict:
    if not est_items:
        return {"overall": "No items extracted from estimate."}
    prompt = "Compare this estimate items to the photos. For each item, check if damage is evident in photos. Also list extra damages in photos not in estimate, and items not evident.\nItems: " + json.dumps(est_items)
    messages = [
        {"role": "system", "content": "Output JSON: {'per_item': list of {item_idx, photo_evidence: bool, confidence: float 0-1, note: str}, 'not_in_photos': list str, 'extra_damage_in_photos': list str, 'overall': str}"},
        {"role": "user", "content": [{"type": "text", "text": prompt}] + images_for_vision}
    ]
    response = client.chat.completions.create(model=MODEL, messages=messages)
    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Consistency comparison error: {e}")
        return {"overall": "Consistency review failed."}

# =========================================
# Summary and score helpers
# =========================================
def build_summary_markdown(missing_photos, combined_text, client_rules, require_oem, non_oem_flag):
    photos_lines = ["- All required photo types present (four corners, VIN, odometer, plate)."] if not missing_photos else [f"- Missing: {', '.join(missing_photos)}."]
    labor_lines = ["- Labor rates found on estimate."] if labor_rates_present_any(combined_text) else ["- Labor rates missing or not clearly listed."]
    taxes_lines = ["- Tax rate found per client rules."] if taxes_present(combined_text) else ["- Tax rate not found per client rules."]
    parts_lines = []
    if require_oem:
        parts_lines.append("- Compliant: OEM parts only for ≤ 2 years or ≤ 24k miles." if not non_oem_flag else "- Non-compliance: non-OEM parts on ≤ 2 years or ≤ 24k miles.")
    else:
        parts_lines.append("- Parts appear OEM or not flagged as non-OEM." if not non_oem_flag else "- Non-OEM parts noted; verify client rules allow on this vehicle.")
    client_lines = ["- Apply client-required documentation (labor rates, photos, taxes) where applicable."]
    notes_lines = ["- Ensure estimate notes clearly explain damage appraisal per client requirements."]
    sections = [
        "### Required Photos", *photos_lines,
        "### Labor Rates", *labor_lines,
        "### Taxes", *taxes_lines,
        "### Parts Compliance", *parts_lines,
        "### Client Rules Adherence", *client_lines,
        "### Additional Notes", *notes_lines
    ]
    return "\n".join(sections)

def check_labor_and_tax_score(combined_text, client_rules):
    adj = 0
    if not labor_rates_present_any(combined_text):
        adj -= 25
    if not taxes_present(combined_text):
        adj -= 25
    return adj

# =========================================
# PDF helpers
# =========================================
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12); pdf.cell(0, 8, txt=title, ln=True); pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10); pdf.multi_cell(0, 6, f"{key}: {val}")

# =========================================
# Routes
# =========================================
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
    first_pdf_bytes: Optional[bytes] = None

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_blobs.append((name, raw))
        elif name.endswith(".pdf"):
            embedded_txt = extract_text_from_pdf_embedded(raw)
            if embedded_txt.strip():
                texts.append(embedded_txt)
            else:
                texts.append(extract_text_from_pdf(io.BytesIO(raw), max_ocr_pages=8, dpi=200))

            if first_pdf_bytes is None:
                first_pdf_bytes = raw

            looks_like_estimate = bool(re.search(r'\bclaim\b', embedded_txt or "", re.IGNORECASE) and
                                       re.search(r'\bvin\b', embedded_txt or "", re.IGNORECASE))
            if not looks_like_estimate:
                harvested = harvest_photos_from_pdf(raw, max_pages=16, dpi=130)
                for hname, hbytes in harvested:
                    image_blobs.append((hname, hbytes))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {f.filename}")

    combined_text = "\n".join(texts)

    keys = extract_keys_from_pdf_first_pages(first_pdf_bytes) if first_pdf_bytes else {}

    # Fallback extractions if GPT failed
    if not keys.get('vehicle_description'):
        match = re.search(r'(?is)\b(\d{4})\s+([a-z0-9]+)\s+([a-z0-9\s]+?)(?=vin|license|odometer|\Z)', combined_text)
        if match:
            keys['vehicle_description'] = f"{match.group(1)} {match.group(2)} {match.group(3)}"
    if not keys.get('mileage'):
        match = re.search(r'(?is)odometer[\s:]*([\d,]+)', combined_text)
        if match:
            keys['mileage'] = re.sub(r'[,\.]', '', match.group(1))
    if not keys.get('labor_rate'):
        match = re.search(r'(?i)@\s*\$\s*(\d+\.?\d*)\s*/hr', combined_text)
        if match:
            keys['labor_rate'] = f"${match.group(1)} /hr"
    if not keys.get('tax_rate'):
        match = re.search(r'(?i)@\s*(\d+\.?\d*)%', combined_text)
        if match:
            keys['tax_rate'] = f"{match.group(1)}%"

    missing_photos = check_required_photos(image_blobs, combined_text)

    vin_est = keys.get("vin") or extract_vin_from_text(combined_text) or (extract_vin_from_pdf_first_pages(first_pdf_bytes, 4, 300) if first_pdf_bytes else None)
    claim_number = keys.get("claim_number") or extract_claim_from_text(combined_text) or (extract_claim_from_pdf_first_pages(first_pdf_bytes, 4, 300) if first_pdf_bytes else None)
    claim_number = claim_number or "N/A"

    vin_photo = extract_vin_from_photos(image_blobs)
    if vin_est and vin_photo:
        vin_verification = "Match" if vin_est == vin_photo else f"No Match (photo shows {vin_photo})"
    elif vin_est and not vin_photo:
        vin_verification = "VIN photo not found"
    elif not vin_est and vin_photo:
        vin_verification = "VIN not found in estimate"
    else:
        vin_verification = "VIN unavailable"
    vin_final = vin_est or "N/A"

    vehicle_desc = keys.get("vehicle_description") or extract_vehicle_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    est_items = extract_estimate_items(combined_text)
    contact_sheets = make_contact_sheets_compact(image_blobs, max_sheets=3, cols=6, padding=6, base_thumb_w=320, jpeg_quality=68)
    images_for_vision: List[Dict[str, Any]] = []
    for name, blob in contact_sheets:
        b64 = base64.b64encode(blob).decode("utf-8")
        images_for_vision.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    year, miles = parse_year_miles(combined_text)
    year = int(keys.get("vehicle_description", "").split()[0]) if not year and keys.get("vehicle_description") else year
    miles = int(keys.get("mileage")) if not miles and keys.get("mileage") else miles
    now_year = datetime.datetime.now().year
    age_years = (now_year - year) if year else None
    require_oem = (age_years is not None and age_years <= 2) or (miles is not None and miles <= 24000)
    non_oem_flag = non_oem_used(combined_text)

    # Verify non-OEM with client rules if applicable
    allow_non_oem = True
    if not require_oem and non_oem_flag and client_rules:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": f"Based on the client rules, are non-OEM parts allowed for a vehicle that is {age_years} years old with {miles} miles? Answer with 'yes' or 'no' only."},
                    {"role": "user", "content": client_rules}
                ]
            )
            allow_non_oem = response.choices[0].message.content.strip().lower() == 'yes'
        except Exception as e:
            logger.warning(f"Client rules verification error: {e}")

    summary_md = build_summary_markdown(missing_photos, combined_text, client_rules, require_oem, non_oem_flag)

    labor_present = bool(keys.get("labor_rate")) or labor_rates_present_any(combined_text)
    tax_present = bool(keys.get("tax_rate")) or taxes_present(combined_text)
    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    parts_adj = -25 if (require_oem and non_oem_flag) or (not require_oem and non_oem_flag and not allow_non_oem) else 0
    computed = max(0, 100 + labor_tax_adj + photo_adj + parts_adj)
    authoritative_score = computed

    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"File Number: {file_number}")
    pdf.multi_cell(0, 6, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 6, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Claim #: {claim_number}")
    pdf.multi_cell(0, 6, f"VIN: {vin_final}")
    pdf.multi_cell(0, 6, f"VIN Photo Verification: {vin_verification}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    if odo_photos: pdf.multi_cell(0, 6, f"Odometer (from photos): {odo_photos}")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4); pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, f"**Audit Results: {authoritative_score}%**")
    pdf.ln(1); pdf.multi_cell(0, 6, summary_md)

    pdf.ln(4); pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")
    if consistency.get("per_item"):
        for it in consistency["per_item"][:60]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            try: conf = float(it.get("confidence", 0))
            except Exception: conf = 0.0
            conf_txt = f"{round(conf*100)}%"
            line = f"- {it.get('side','unspecified').title()} {it.get('part','component')} · {it.get('op','op')} → Photo: {ev} ({conf_txt}); {it.get('note','')}"
            pdf.multi_cell(0, 6, line)
    else:
        pdf.multi_cell(0, 6, "Per-item comparison unavailable.")

    if consistency.get("not_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Items Estimated but Not Evident in Photos")
        for raw in consistency["not_in_photos"][:30]: pdf.multi_cell(0, 6, f"- {raw}")

    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:30]: pdf.multi_cell(0, 6, f"- {d}")

    pdf.ln(2); pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
        logger.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"; msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_number}
VIN: {vin_final}
VIN Photo Verification: {vin_verification}
Vehicle: {vehicle_desc}

Compliance Score: {authoritative_score}%

AI Review Summary:
Audit Results: {authoritative_score}%

{summary_md}
"""
        msg.set_content(email_body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR"); smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing): {e}")

    return {
        "gpt_output": f"Audit Results: {authoritative_score}%\n\n{summary_md}",
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin_final,
        "vin_photo_verification": vin_verification,
        "score": f"{authoritative_score}%",
        "consistency_review": consistency
    }

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"; file_name = f"{client_name}.docx"; file_path = os.path.join(rules_dir, file_name)
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
























