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
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, Image
from openai import OpenAI

# =========================================
# PDF storage: save to /tmp with filename {file_number}.pdf
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
# OpenAI client (gpt-4o) — only for the vision comparison JSON
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
# OCR helpers (fast)
# =========================================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr_text_fast(img: Image.Image, psm: int = 6) -> str:
    try:
        proc = preprocess_image(img)
        return pytesseract.image_to_string(proc, lang="eng", config=f"--psm {psm} --oem 1")
    except Exception as e:
        logger.warning(f"OCR fast error: {e}")
        return ""

def extract_text_from_pdf(file_like: io.BytesIO, max_ocr_pages: int = 8, dpi: int = 140) -> str:
    """OCR only the first pages for speed. Most estimate details are up front."""
    try:
        file_like.seek(0)
        pages = convert_from_bytes(file_like.read(), dpi=dpi)
        text_output = ""
        for i, img in enumerate(pages, 1):
            if i > max_ocr_pages:
                break
            page_text = ocr_text_fast(img, psm=6)
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

# =========================================
# Harvest photos from PDFs (no OCR, fast heuristic)
# =========================================
def _page_var(img: Image.Image) -> float:
    g = img.convert("L")
    return ImageStat.Stat(g).var[0]

def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int = 20, dpi: int = 135) -> List[Tuple[str, bytes]]:
    """Return pages that look like photo pages. Fast variance heuristic; no OCR."""
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
    s = s.strip().upper().replace(" ", "")
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

# ---------- Stronger VIN parsing (estimate only) ----------
VIN_LABEL = re.compile(r'(?i)\bV[\W_]*I[\W_]*N\b')  # matches VIN / V I N / V.I.N
VIN_PHRASE = re.compile(r'(?i)\bVehicle\s*Identification\s*Number\b')

def extract_vin_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # 1) Look for VIN label/phrase and capture nearby 17-char token (same line or next ~100 chars)
    for m in VIN_LABEL.finditer(text):
        window = text[m.end(): m.end()+140]
        cands = re.findall(r'([A-HJ-NPR-Z0-9]{17})', window)
        vin = best_vin_candidate(cands)
        if vin: return vin
    for m in VIN_PHRASE.finditer(text):
        window = text[m.end(): m.end()+160]
        cands = re.findall(r'([A-HJ-NPR-Z0-9]{17})', window)
        vin = best_vin_candidate(cands)
        if vin: return vin
    # 2) As a fallback, accept any 17-char candidate ONLY if within +/-120 chars of VIN label/phrase
    positions = [m.start() for m in VIN_LABEL.finditer(text)] + [m.start() for m in VIN_PHRASE.finditer(text)]
    if positions:
        for m in re.finditer(r'([A-HJ-NPR-Z0-9]{17})', text):
            if any(abs(m.start() - p) <= 120 for p in positions):
                vin = best_vin_candidate([m.group(1)])
                if vin: return vin
    return None

def extract_vin_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 10, dpi: int = 220) -> Optional[str]:
    """High-res OCR of the first few estimate pages to grab VIN if bulk OCR missed it."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max(1, pages_to_scan)]
        for img in pages:
            txt = ocr_text_fast(img, psm=6)
            v = extract_vin_from_text(txt)
            if v: return v
    except Exception as e:
        logger.warning(f"VIN first-pages OCR error: {e}")
    return None

# =========================================
# Claim extraction (robust + hi-res pages fallback; estimate only)
# =========================================
CLAIM_PATTERNS = [
    r'(?i)\bclaim\s*(?:#|no\.?|number|num)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/\.]{2,30})',
    r'(?i)\bloss\s*(?:#|no\.?|number|num)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/\.]{2,30})',
    r'(?i)\bfile\s*(?:#|no\.?|number|num)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/\.]{2,30})',
    r'(?i)\bref(?:erence)?\s*(?:#|no\.?|number|num)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/\.]{2,30})',
    r'(?i)\bassignment\s*(?:#|no\.?|number|num)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-\/\.]{2,30})',
]
CLAIM_WORD_FUZZY = re.compile(r'(?i)\bC[l1I][aA][iI1][mMnN]\b')

def extract_claim_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in CLAIM_PATTERNS:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip().strip('.').strip('-')
            # must contain at least one digit; allow all-digit or mixed
            if re.search(r'\d', cand):
                return cand
    # fuzzy key then capture next token
    for m in CLAIM_WORD_FUZZY.finditer(text or ""):
        tail = text[m.end(): m.end()+100]
        m2 = re.search(r'[:#\-\s]*([A-Z0-9][A-Z0-9\-\/\.]{2,30})', tail, re.IGNORECASE)
        if m2:
            cand = m2.group(1).strip().strip('.').strip('-')
            if re.search(r'\d', cand):
                return cand
    return None

def extract_claim_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 10, dpi: int = 220) -> Optional[str]:
    """High-res OCR of the first few estimate pages for Claim/Loss/File #."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max(1, pages_to_scan)]
        for img in pages:
            txt = ocr_text_fast(img, psm=6)
            c = extract_claim_from_text(txt)
            if c: return c
    except Exception as e:
        logger.warning(f"Claim first-pages OCR error: {e}")
    return None

# =========================================
# Field extraction & tax/parts signals (vehicle string normalization)
# =========================================
MAKE_FIX = {
    "nessan": "Nissan",
    "nisaan": "Nissan",
    "nissan": "Nissan",
    "toy0ta": "Toyota",
    "chevroler": "Chevrolet",
    "cheverolet": "Chevrolet",
}
def normalize_vehicle_str(s: str) -> str:
    if not s: return s
    s2 = s
    for wrong, right in MAKE_FIX.items():
        s2 = re.sub(rf'\b{re.escape(wrong)}\b', right, s2, flags=re.IGNORECASE)
    s2 = re.sub(r'\s{2,}', ' ', s2).replace(' ,', ',')
    return s2.strip()

def extract_vehicle_from_text(text: str) -> Optional[str]:
    m1 = re.search(r"\b(20\d{2})\s+([A-Za-z]{3,})\s+([A-Za-z0-9\-]{2,})", text or "")
    m2 = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text or "", re.IGNORECASE)
    if m1:
        year, make, model = m1.groups()
        miles = m2.group(1) if m2 else "Mileage unknown"
        out = f"{year} {make} {model}, {miles} miles"
        return normalize_vehicle_str(out)
    return None

def parse_year_miles(text: str) -> Tuple[Optional[int], Optional[int]]:
    year = None
    miles = None
    m_year = re.search(r"\b(20\d{2})\b", text or "")
    if m_year:
        try: year = int(m_year.group(1))
        except: year = None
    m_mi = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text or "", re.IGNORECASE)
    if m_mi:
        try: miles = int(m_mi.group(1).replace(",", ""))
        except: miles = None
    return year, miles

def taxes_present(text: str) -> bool:
    return re.search(r'tax[^\n]{0,50}(\d{1,3}\s*%|\$\s*\d+(\.\d{2})?)', text or "", re.IGNORECASE) is not None

# ===== Parts detection tightened: require op + aftermarket flag + known panel on SAME LINE
PART_FLAGS = r'(?:\bA/M\b|\bAFTER\s*MARKET\b|\bAFTERMARKET\b|\bLKQ\b|\bRECOND(?:ITIONED)?\b|\bCAPA\b|\bALT[-\s]*OE\b|\bREMAN(?:UFACTURED)?\b)'
OPS_TOK = re.compile(r'\b(REPL(?:ACE)?|R&R|R & R|R&I|R & I|REPAIR|REFINISH|PAINT)\b', re.IGNORECASE)
PANELS = [
    "bumper","fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
    "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
    "wheel","tire","pillar","garnish","molding","fog lamp","reinforcement","cover"
]
PANELS_U = [p.upper() for p in PANELS]

def non_oem_used(text: str) -> bool:
    lines = (text or "").splitlines()
    for line in lines:
        l = line.strip().upper()
        if not l:
            continue
        if OPS_TOK.search(l) and re.search(PART_FLAGS, l, re.IGNORECASE) and any(p in l for p in PANELS_U):
            return True
    if re.search(r'parts\s+presented\s+are\s+OEM[-\s]*parts', text or "", re.IGNORECASE):
        return False
    return False

# =========================================
# Photo parsing: VIN/ODO detection (for verification & presence only)
# =========================================
def _is_exterior_by_edges(img: Image.Image) -> bool:
    g = img.convert("L")
    var = ImageStat.Stat(g).var[0]
    edges = g.filter(ImageFilter.FIND_EDGES)
    evar = ImageStat.Stat(edges).var[0]
    return (var > 140 and evar > 400)

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    """Use photo VIN ONLY for verification (never to populate the VIN field)."""
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            img = img.copy()
            img.thumbnail((1600, 1600))
            txt = ocr_text_fast(img, psm=7)
            cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", txt.upper())
            vin = best_vin_candidate(cands)
            if vin:
                return vin
        except Exception as e:
            logger.warning(f"VIN photo OCR error ({name}): {e}")
    return None

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            img = img.copy()
            img.thumbnail((1400, 1400))
            txt = ocr_text_fast(img, psm=7)
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", txt, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception as e:
            logger.warning(f"Odometer photo OCR error ({name}): {e}")
    return None

def _sample_for_plate_ocr(image_blobs: List[Tuple[str, bytes]], k: int = 6) -> List[Tuple[str, bytes]]:
    if len(image_blobs) <= k:
        return image_blobs
    pairs = []
    for name, blob in image_blobs:
        h = hashlib.md5(blob).hexdigest()
        pairs.append((int(h[:8], 16), (name, blob)))
    pairs.sort()
    return [p[1] for p in pairs[:k]]

def check_required_photos(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> List[str]:
    """Required: four corners, odometer, VIN, license plate."""
    required = ["four corners", "odometer", "vin", "license plate"]
    present = set()
    txt = (ocr_text or "").lower()

    vin_text = bool(re.search(r'\bvin\b', txt))
    odo_text = bool(re.search(r'\bodometer|mileage\b', txt))

    vin_photo = extract_vin_from_photos(image_blobs) is not None
    odo_photo = extract_odometer_from_photos(image_blobs) is not None

    if vin_text or vin_photo:
        present.add("vin")
    if odo_text or odo_photo:
        present.add("odometer")

    for name, blob in _sample_for_plate_ocr(image_blobs, k=6):
        try:
            img = Image.open(io.BytesIO(blob))
            img.thumbnail((1300, 1300))
            txtp = ocr_text_fast(img, psm=7)
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", txtp, re.IGNORECASE):
                present.add("license plate")
                break
        except Exception:
            pass

    exterior_hits = 0
    for name, blob in image_blobs[:40]:
        try:
            img = Image.open(io.BytesIO(blob))
            img.thumbnail((1600, 1600))
            if _is_exterior_by_edges(img):
                exterior_hits += 1
        except Exception:
            continue
    if exterior_hits >= 3:
        present.add("four corners")

    missing = [p for p in required if p not in present]
    logger.debug(f"Photo check → present={sorted(list(present))}, missing={missing}, ext_hits={exterior_hits}")
    return missing

# =========================================
# Contact-sheet builder (ALL photos included, ≤ 3 sheets adaptively)
# =========================================
def shrink_to_width(img: Image.Image, max_w: int) -> Image.Image:
    if img.width <= max_w:
        return img.convert("RGB")
    h = int(img.height * max_w / img.width)
    return img.convert("RGB").resize((max_w, h), Image.LANCZOS)

def make_contact_sheets_compact(
    image_blobs: List[Tuple[str, bytes]],
    max_sheets: int = 3,
    cols: int = 6,
    padding: int = 6,
    base_thumb_w: int = 320,
    jpeg_quality: int = 68
) -> List[Tuple[str, bytes]]:
    if not image_blobs:
        return []

    thumbs: List[Image.Image] = []
    for _, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            thumbs.append(shrink_to_width(img, base_thumb_w))
        except Exception:
            continue

    n = len(thumbs)
    per_sheet = max(1, math.ceil(n / max_sheets))
    rows = math.ceil(per_sheet / cols)

    def build_sheet(chunk: List[Image.Image], thumb_w: int) -> Image.Image:
        row_heights = []
        for r in range(rows):
            row_imgs = chunk[r*cols:(r+1)*cols]
            if not row_imgs: break
            row_heights.append(max(im.height for im in row_imgs))
        canvas_w = cols * thumb_w + (cols + 1) * padding
        canvas_h = sum(row_heights) + (len(row_heights) + 1) * padding
        sheet = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
        y = padding
        pos = 0
        for r, row_h in enumerate(row_heights):
            x = padding
            for c in range(cols):
                if pos >= len(chunk): break
                im = chunk[pos]
                if im.width != thumb_w:
                    h = int(im.height * (thumb_w / im.width))
                    im = im.resize((thumb_w, h), Image.LANCZOS)
                y_off = (row_h - im.height) // 2
                sheet.paste(im, (x, y + y_off))
                x += thumb_w + padding
                pos += 1
            y += row_h + padding
        return sheet

    sheets: List[Tuple[str, bytes]] = []
    idx = 0
    sheet_num = 1
    thumb_w = base_thumb_w

    while idx < n:
        chunk = thumbs[idx: idx + per_sheet]
        attempt = 0
        sheet_img = build_sheet(chunk, thumb_w)
        while sheet_img.height > 3600 and thumb_w > 160 and attempt < 3:
            thumb_w = int(thumb_w * 0.85)
            sheet_img = build_sheet(chunk, thumb_w)
            attempt += 1
        buf = io.BytesIO()
        sheet_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        sheets.append((f"contact-sheet-{sheet_num}.jpg", buf.getvalue()))
        sheet_num += 1
        idx += per_sheet

    return sheets

# =========================================
# Labor/tax compliance checks
# =========================================
def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    def has_rate(label: str) -> bool:
        pat = rf"{label}[^\n]{{0,120}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        return re.search(pat, text or "", re.IGNORECASE) is not None
    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor", "Frame Labor"]
    if not any(has_rate(lbl) for lbl in labels):
        adj -= 50
    if re.search(r"tax\s*(required|must|utilize|apply)", client_rules or "", re.IGNORECASE):
        if not taxes_present(text or ""):
            adj -= 25
    return adj

def labor_rates_present_any(text: str) -> bool:
    labels = ["Body", "Paint", "Mechanical", "Structural", "Frame", "Refinish", "Supplies"]
    return any(re.search(rf"{lbl}[^\n]{{0,120}}?\$\s*\d{{2,3}}", text or "", re.IGNORECASE) for lbl in labels)

# =========================================
# Estimate parsing (line items for comparison)
# =========================================
OPS = ["replace", "repair", "refinish", "r&i", "r & i", "align", "blend", "calibrate"]
def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for line in (text or "").splitlines():
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
# GPT compare: estimate ↔ photos (JSON only)
# =========================================
def compare_estimate_with_photos(items: List[Dict[str, str]],
                                 images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        "Return STRICT JSON ONLY per this schema: " + json.dumps(schema)
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
            max_tokens=420,
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
# Local narrative builder (fast, deterministic)
# =========================================
def build_summary_markdown(
    missing_photos: List[str],
    text: str,
    client_rules: str,
    require_oem: bool,
    non_oem_flag: bool
) -> str:
    if not missing_photos:
        photos_lines = ["- All required photo types present (four corners, VIN, odometer, plate)."]
    else:
        photos_lines = [f"- Missing: {', '.join(missing_photos)}."]

    if labor_rates_present_any(text):
        labor_lines = ["- Labor rates listed on estimate."]
    else:
        labor_lines = ["- Labor rates missing or not clearly listed."]

    if taxes_present(text):
        taxes_lines = ["- Tax rate present on estimate."]
    else:
        taxes_lines = ["- Tax rate not found per client rules."]

    parts_lines: List[str] = []
    if require_oem:
        if non_oem_flag:
            parts_lines.append("- Non-compliance: non-OEM parts on ≤ 2 years or ≤ 24k miles.")
        else:
            parts_lines.append("- Compliant: OEM parts only for ≤ 2 years or ≤ 24k miles.")
    else:
        if non_oem_flag:
            parts_lines.append("- Non-OEM parts noted; verify client rules allow on this vehicle.")
        else:
            parts_lines.append("- Parts appear OEM or not flagged as non-OEM.")

    client_lines = [
        "- Apply client-required documentation (labor rates, photos, taxes) where applicable."
    ]
    notes_lines = [
        "- Ensure estimate notes clearly explain damage appraisal per client requirements."
    ]

    sections = [
        "### Required Photos",
        *photos_lines,
        "### Labor Rates",
        *labor_lines,
        "### Taxes",
        *taxes_lines,
        "### Parts Compliance",
        *parts_lines,
        "### Client Rules Adherence",
        *client_lines,
        "### Additional Notes",
        *notes_lines,
    ]
    return "\n".join(sections)

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

    texts: List[str] = []
    image_blobs: List[Tuple[str, bytes]] = []
    first_pdf_bytes: Optional[bytes] = None  # estimate PDF for hi-res VIN/Claim fallback

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_blobs.append((name, raw))
        elif name.endswith(".pdf"):
            # Treat all PDFs as potential estimate sources for text OCR
            texts.append(extract_text_from_pdf(io.BytesIO(raw), max_ocr_pages=8, dpi=140))
            if first_pdf_bytes is None:
                first_pdf_bytes = raw  # used ONLY to re-OCR estimate pages for VIN/Claim
            # Also harvest photo-like pages for contact sheets
            harvested = harvest_photos_from_pdf(raw, max_pages=20, dpi=135)
            for hname, hbytes in harvested:
                image_blobs.append((hname, hbytes))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8", errors="ignore"))
        else:
            texts.append(f"⚠️ Skipped unsupported file: {f.filename}")

    combined_text = "\n".join(texts)

    # ----- Build compact contact sheets for GPT (ALL photos represented, ≤ 3 images sent)
    contact_sheets = make_contact_sheets_compact(
        image_blobs,
        max_sheets=3,
        cols=6,
        padding=6,
        base_thumb_w=320,
        jpeg_quality=68
    )
    images_for_vision: List[Dict[str, Any]] = []
    for name, blob in contact_sheets:
        b64 = base64.b64encode(blob).decode("utf-8")
        images_for_vision.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    # ----- REQUIRED PHOTOS (presence only)
    missing_photos = check_required_photos(image_blobs, combined_text)

    # ----- VIN & CLAIM: from estimate ONLY (with hi-res estimate pages fallback)
    vin_est = extract_vin_from_text(combined_text)
    if not vin_est and first_pdf_bytes:
        vin_est = extract_vin_from_pdf_first_pages(first_pdf_bytes, pages_to_scan=10, dpi=220)

    claim_number = extract_claim_from_text(combined_text)
    if not claim_number and first_pdf_bytes:
        claim_number = extract_claim_from_pdf_first_pages(first_pdf_bytes, pages_to_scan=10, dpi=220)
    claim_number = claim_number or "N/A"

    # ----- VIN photo verification (compare only; do NOT populate with photo)
    vin_photo = extract_vin_from_photos(image_blobs)  # may be None
    if vin_est and vin_photo:
        vin_verification = "Match" if vin_est == vin_photo else f"No Match (photo shows {vin_photo})"
    elif vin_est and not vin_photo:
        vin_verification = "VIN photo not found"
    elif not vin_est and vin_photo:
        vin_verification = "VIN not found in estimate"
    else:
        vin_verification = "VIN unavailable"

    vin_final = vin_est or "N/A"  # always from estimate only

    # ----- Vehicle & odo
    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    # ----- parse estimate items & compare to photos (vision uses contact sheets)
    est_items = extract_estimate_items(combined_text)
    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    # ----- rules for summary
    year, miles = parse_year_miles(combined_text)
    now_year = datetime.datetime.now().year
    age_years = (now_year - year) if year else None
    require_oem = (age_years is not None and age_years <= 2) or (miles is not None and miles <= 24000)
    non_oem_flag = non_oem_used(combined_text)

    # ===== Build deterministic narrative (FAST; no GPT)
    summary_md = build_summary_markdown(
        missing_photos=missing_photos,
        text=combined_text,
        client_rules=client_rules,
        require_oem=require_oem,
        non_oem_flag=non_oem_flag
    )

    # ===== Score calc
    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    computed = max(0, 100 + labor_tax_adj + photo_adj)
    authoritative_score = computed

    # =========================================
    # PDF build
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
    pdf.multi_cell(0, 6, f"File Number: {file_number}")
    pdf.multi_cell(0, 6, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 6, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Claim #: {claim_number}")
    pdf.multi_cell(0, 6, f"VIN: {vin_final}")
    pdf.multi_cell(0, 6, f"VIN Photo Verification: {vin_verification}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    if odo_photos:
        pdf.multi_cell(0, 6, f"Odometer (from photos): {odo_photos}")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, f"**Audit Results: {authoritative_score}%**")
    pdf.ln(1)
    pdf.multi_cell(0, 6, summary_md)

    # ======== Estimate ↔ Photos Consistency Review ========
    pdf.ln(4)
    pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")

    if consistency.get("per_item"):
        for it in consistency["per_item"][:60]:
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
        for raw in consistency["not_in_photos"][:30]:
            pdf.multi_cell(0, 6, f"- {raw}")

    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2)
        pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:30]:
            pdf.multi_cell(0, 6, f"- {d}")

    pdf.ln(2)
    pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    # Save PDF to /tmp with name {file_number}.pdf
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        logger.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # OPTIONAL email (unchanged)
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
VIN Photo Verification: {vin_verification}
Vehicle: {vehicle_desc}

Compliance Score: {authoritative_score}%

AI Review Summary:
Audit Results: {authoritative_score}%

{summary_md}
"""
        msg.set_content(email_body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
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


















