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

def extract_vin_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 170) -> Optional[str]:
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
# Claim extraction (robust; requires digits)
# =========================================
CLAIM_AFTER_LABEL = re.compile(
    r'(?is)\bclaim\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\. ]{2,60})'
)
ALT_CLAIM_LABELS = [
    re.compile(r'(?is)\bloss\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\. ]{2,60})'),
    re.compile(r'(?is)\bfile\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\. ]{2,60})'),
    re.compile(r'(?is)\bref(?:erence)?\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\. ]{2,60})'),
    re.compile(r'(?is)\bassignment\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\. ]{2,60})'),
]
_CLAIM_BLACKLIST = {"SERVICE", "SERVICES", "PHONE", "EMAIL", "FAX", "TOTAL", "POLICY"}

def _clean_claim(c: str) -> str:
    c = c.strip().strip(':').strip().strip('.').strip('-')
    c = c.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    return re.sub(r'\s+', '', c)

def _valid_claim_candidate(c: str) -> bool:
    if not c or len(c) < 3: return False
    if not re.search(r'\d', c): return False
    if c.upper() in _CLAIM_BLACKLIST: return False
    return True

def extract_claim_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for m in CLAIM_AFTER_LABEL.finditer(text):
        cand = _clean_claim(m.group(1))
        if _valid_claim_candidate(cand): return cand
    for pat in ALT_CLAIM_LABELS:
        for m in pat.finditer(text):
            cand = _clean_claim(m.group(1))
            if _valid_claim_candidate(cand): return cand
    return None

def extract_claim_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 170) -> Optional[str]:
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
# Vehicle & tax/parts helpers
# =========================================
MAKE_FIX = {
    "nessan": "Nissan","nisaan": "Nissan","nissan": "Nissan","niss": "Nissan","niss.": "Nissan",
    "toy0ta": "Toyota","chevroler": "Chevrolet","cheverolet": "Chevrolet"
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
    year = None; miles = None
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

PART_FLAGS = r'(?:\bA/M\b|\bAFTER\s*MARKET\b|\bAFTERMARKET\b|\bLKQ\b|\bRECOND(?:ITIONED)?\b|\bCAPA\b|\bALT[-\s]*OE\b|\bREMAN(?:UFACTURED)?\b)'
OPS_TOK = re.compile(r'\b(REPL(?:ACE)?|R&R|R & R|R&I|R & I|REPAIR|REFINISH|PAINT)\b', re.IGNORECASE)
PANELS = ["bumper","fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
          "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
          "wheel","tire","pillar","garnish","molding","fog lamp","reinforcement","cover"]
PANELS_U = [p.upper() for p in PANELS]

def non_oem_used(text: str) -> bool:
    lines = (text or "").splitlines()
    for line in lines:
        l = line.strip().upper()
        if not l: continue
        if OPS_TOK.search(l) and re.search(PART_FLAGS, l, re.IGNORECASE) and any(p in l for p in PANELS_U):
            return True
    if re.search(r'parts\s+presented\s+are\s+OEM[-\s]*parts', text or "", re.IGNORECASE):
        return False
    return False

# =========================================
# Photo parsing (VIN/ODO/plate presence)
# =========================================
def _is_exterior_by_edges(img: Image.Image) -> bool:
    g = img.convert("L")
    var = ImageStat.Stat(g).var[0]
    edges = g.filter(ImageFilter.FIND_EDGES)
    evar = ImageStat.Stat(edges).var[0]
    return (var > 140 and evar > 400)

# ---- FAST+ROBUST VIN from photos ----
def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    VIN_NEAR_LABEL = re.compile(r'(?i)\bV[\W_]*I[\W_]*N\b')
    VIN_17 = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')
    VIN_SEP_SEQ = re.compile(r'(?i)((?:[A-HJ-NPR-Z0-9][\s\.\-–—:_]){16}[A-HJ-NPR-Z0-9])')

    # FAST prescan of all images to find likely VIN candidates
    candidates: List[Tuple[str, bytes]] = []
    for name, blob in image_blobs:
        try:
            im = Image.open(io.BytesIO(blob)).convert("L")
            im.thumbnail((1100, 1100))
            txt = pytesseract.image_to_string(im, lang="eng", config="--psm 6 --oem 1")
            up = (txt or "").upper()
            if "VIN" in up or VIN_17.search(up) or VIN_SEP_SEQ.search(up):
                candidates.append((name, blob))
        except Exception:
            continue
    candidates = candidates[:6]  # limit heavy OCR

    def _variants(im: Image.Image) -> List[Image.Image]:
        im = im.copy()
        max_w = 2200
        if im.width < max_w:
            h = int(im.height * (max_w / im.width))
            im = im.resize((max_w, h), Image.LANCZOS)
        g = im.convert("L")
        return [
            ImageEnhance.Contrast(g).enhance(2.0),
            ImageEnhance.Sharpness(g).enhance(2.0),
            g.point(lambda p: 255 if p > 180 else 0, mode="1").convert("L"),
            ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3))),
        ]

    def _ocr_all(im: Image.Image) -> str:
        out = []
        for psm in (7, 6, 11):
            try:
                out.append(pytesseract.image_to_string(im, lang="eng", config=f"--psm {psm} --oem 1"))
            except Exception:
                pass
        return "\n".join([t for t in out if t])

    for name, blob in candidates:
        try:
            im = Image.open(io.BytesIO(blob))
            up_all = ""
            for var in _variants(im):
                up_all += "\n" + _ocr_all(var)
            up_all = up_all.upper()

            for m in VIN_NEAR_LABEL.finditer(up_all):
                window = up_all[m.end(): m.end() + 220]
                for mm in VIN_SEP_SEQ.finditer(window):
                    vin = normalize_vin(mm.group(1))
                    if vin and vin_checksum_ok(vin):
                        return vin
                vin = best_vin_candidate(re.findall(r'\b([A-HJ-NPR-Z0-9]{17})\b', window))
                if vin: return vin

            for mm in VIN_SEP_SEQ.finditer(up_all):
                vin = normalize_vin(mm.group(1))
                if vin and vin_checksum_ok(vin): return vin

            vin = best_vin_candidate(re.findall(r'\b([A-HJ-NPR-Z0-9]{17})\b', up_all))
            if vin: return vin
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

# ---- License plate OCR: scan ALL images robustly
PLATE_RX = re.compile(r'\b([A-Z0-9]{1,3}[-\s]?[A-Z0-9]{3,4}|[A-Z0-9]{5,8})\b')
def _plate_ocr_variants(img: Image.Image) -> str:
    def variants(im: Image.Image) -> List[Image.Image]:
        im = im.copy(); im.thumbnail((1600, 1600))
        g = im.convert("L")
        return [
            ImageEnhance.Contrast(g).enhance(1.8),
            ImageEnhance.Sharpness(g).enhance(1.8),
            ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3))),
            g.point(lambda p: 255 if p > 170 else 0, mode="1").convert("L"),
            g.point(lambda p: 255 if p > 190 else 0, mode="1").convert("L"),
        ]
    out = []
    for v in variants(im):
        for psm in (6, 7, 11):
            try:
                t = pytesseract.image_to_string(v, lang="eng", config=f"--psm {psm} --oem 1")
                if t: out.append(t)
            except Exception:
                pass
    return "\n".join(out)

def check_required_photos(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> List[str]:
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

    # License plate: scan ALL images (robust OCR)
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            txtp = _plate_ocr_variants(img)
            if re.search(r'(license|registration)\s*plate', txtp, re.IGNORECASE) or PLATE_RX.search(txtp):
                present.add("license plate")
                break
        except Exception:
            pass

    # Four corners heuristic (first 40 to bound cost)
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
# Contact sheets (unchanged behavior)
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
        y = padding; pos = 0
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
    idx = 0; sheet_num = 1; thumb_w = base_thumb_w
    while idx < n:
        chunk = thumbs[idx: idx + per_sheet]
        attempt = 0
        sheet_img = build_sheet(chunk, thumb_w)
        while sheet_img.height > 3600 and thumb_w > 160 and attempt < 3:
            thumb_w = int(thumb_w * 0.85)
            sheet_img = build_sheet(chunk, thumb_w); attempt += 1
        buf = io.BytesIO()
        sheet_img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        sheets.append((f"contact-sheet-{sheet_num}.jpg", buf.getvalue()))
        sheet_num += 1; idx += per_sheet
    return sheets

# =========================================
# Labor/tax compliance checks (aligned with display logic)
# =========================================
def labor_rates_present_any(text: str) -> bool:
    labels = ["Body", "Paint", "Mechanical", "Structural", "Frame", "Refinish", "Supplies"]
    return any(re.search(rf"{lbl}[^\n]{{0,120}}?\$\s*\d{{2,3}}", text or "", re.IGNORECASE) for lbl in labels)

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    if not labor_rates_present_any(text or ""):
        adj -= 50
    if re.search(r"tax\s*(required|must|utilize|apply|include)", (client_rules or ""), re.IGNORECASE):
        if not taxes_present(text or ""):
            adj -= 25
    return adj

# =========================================
# Client-guideline parsing & adherence (full review)
# =========================================
PHOTO_KEYWORDS = {
    "four corners": ["four corners", "4 corners", "four-corners"],
    "vin": ["vin", "v.i.n"],
    "odometer": ["odometer", "mileage"],
    "license plate": ["license plate", "registration plate", "plate photo"],
    "damage close-ups": ["close-up", "close ups", "closeups", "detail photos", "damage photos"],
    "interior": ["interior photo", "interior photos"],
}

def _mentions_any(s: str, needles: List[str]) -> bool:
    s2 = (s or "").lower()
    return any(n in s2 for n in needles)

def parse_client_rules(client_rules: str) -> Dict[str, Any]:
    cr = (client_rules or "").lower()
    photos_req = set()
    for key, variants in PHOTO_KEYWORDS.items():
        if _mentions_any(cr, variants):
            photos_req.add(key)

    require_labor = bool(re.search(r"\blabor rate[s]?\b|\brates\s+for\s+(?:body|paint|mechanical|frame)", cr))
    require_tax = bool(re.search(r"\bapply\s+tax\b|\btax\s+required\b|\binclude\s+tax\b", cr))
    require_market_doc = bool(re.search(r"\b(nada|kelley|kbb|black\s*book|retail|market\s+value)\b", cr))
    require_valuation_includes_tax = bool(re.search(r"\bvaluation\b.*\btax\b|\binclude\b.*\btax\b.*\bvaluation\b", cr, re.DOTALL))
    require_total_loss_decl = bool(re.search(r"\btotal\s+loss\b.*\bdeclare|\bdeclare\b.*\btotal\s+loss\b", cr, re.DOTALL))
    oem_recent = bool(re.search(r"(?:<=?|less\s+than|under)\s*2\s*year", cr)) or bool(re.search(r"(?:<=?|less\s+than|under)\s*24\s*[,k]*\s*mi", cr))

    # Prefer aftermarket/LKQ/recon regardless of year/miles (NEW)
    prefer_aftermarket = bool(re.search(
        r"(heavy\s+on\s+the\s+use\s+of\s+aftermarket|consider\s+.*aftermarket\s+.*before\s+(?:lkq|oem)|"
        r"utilize\s+(?:lkq|recon|aftermarket)\s+parts\s+regardless\s+of\s+year|regardless\s+of\s+year\s+or\s+mileage)",
        cr, re.DOTALL))

    return {
        "photos_required": photos_req,
        "require_labor_rates": require_labor,
        "require_tax": require_tax,
        "require_market_doc": require_market_doc,
        "require_valuation_includes_tax": require_valuation_includes_tax,
        "require_total_loss_decl": require_total_loss_decl,
        "oem_required_if_recent": oem_recent,
        "prefer_aftermarket": prefer_aftermarket,   # NEW
    }

def has_market_value_doc(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\b(nada|kelley|kbb|black\s*book)\b", t)) or \
           bool(re.search(r"\b(retail|market)\s+value\b", t))

def build_client_adherence_lines(
    guidelines: Dict[str, Any],
    missing_photos: List[str],
    text: str,
    year: Optional[int],
    miles: Optional[int],
    non_oem_flag: bool,
) -> List[str]:
    lines: List[str] = []
    req_photos = guidelines.get("photos_required", set())

    # Photos
    if req_photos:
        for p in sorted(req_photos):
            if p in missing_photos:
                lines.append(f"- Non-compliant: required photo '{p}' missing.")
            else:
                lines.append(f"- Compliant: required photo '{p}' provided.")
    else:
        if missing_photos:
            lines.append(f"- Non-compliant: missing required photo(s): {', '.join(missing_photos)}.")
        else:
            lines.append("- Compliant: required photo set present.")

    # Labor rates
    if guidelines.get("require_labor_rates"):
        if labor_rates_present_any(text):
            lines.append("- Compliant: labor rates listed on estimate.")
        else:
            lines.append("- Non-compliant: labor rates not listed per client rules.")

    # Taxes
    if guidelines.get("require_tax"):
        if taxes_present(text):
            lines.append("- Compliant: tax rate present per client rules.")
        else:
            lines.append("- Non-compliant: tax rate not found per client rules.")

    # Market/retail value doc
    if guidelines.get("require_market_doc"):
        if has_market_value_doc(text):
            lines.append("- Compliant: required retail/market value documentation present (e.g., NADA/KBB).")
        else:
            lines.append("- Non-compliant: required retail/market value documentation not found.")

    # Valuation includes tax
    if guidelines.get("require_valuation_includes_tax"):
        if re.search(r"valuation[^\n]{0,80}(tax|incl(?:uded)?|with\s+tax)", text, re.IGNORECASE):
            lines.append("- Compliant: valuation indicates tax inclusion.")
        elif taxes_present(text):
            lines.append("- Unable to verify: tax present but valuation line not explicit about inclusion.")
        else:
            lines.append("- Non-compliant: valuation tax inclusion not indicated.")

    # Total loss declaration
    if guidelines.get("require_total_loss_decl"):
        if re.search(r"\btotal\s+loss\b", text, re.IGNORECASE):
            lines.append("- Compliant: total loss declaration present.")
        else:
            lines.append("- Unable to verify: total loss declaration not found.")

    # Parts policy vs. vehicle age/mileage and client preference (UPDATED)
    prefer_aftermarket = bool(guidelines.get("prefer_aftermarket"))
    if prefer_aftermarket:
        # Client wants aftermarket/LKQ/recon first regardless of year/miles
        if non_oem_flag:
            lines.append("- Compliant: aftermarket/LKQ/recon parts used per client preference.")
        else:
            lines.append("- Non-compliant: OEM used; document why alternative parts were not utilized per client preference.")
    elif guidelines.get("oem_required_if_recent"):
        # OEM-only if recent rule
        recent = ((year is not None and (datetime.datetime.now().year - year) <= 2) or
                  (miles is not None and miles <= 24000))
        if recent:
            if non_oem_flag:
                lines.append("- Non-compliant: non-OEM parts used on a ≤2 years or ≤24k miles vehicle.")
            else:
                lines.append("- Compliant: OEM parts used per ≤2 years/≤24k miles rule.")
        else:
            # No special requirement—treat non-OEM neutrally unless other rules say otherwise
            if non_oem_flag:
                lines.append("- Non-OEM parts noted; verify usage aligns with remaining client rules.")
            else:
                lines.append("- Parts appear OEM; acceptable absent a client preference for alternatives.")
    else:
        # No explicit parts direction in rules; keep neutral
        if non_oem_flag:
            lines.append("- Non-OEM parts noted; verify client rules allow on this vehicle.")
        else:
            lines.append("- Parts appear OEM or not flagged as non-OEM.")

    return lines

# =========================================
# Estimate parsing
# =========================================
OP_RX = re.compile(r'\b(repl(?:ace)?|r&r|r & r|r&i|r & i|repair|refinish|paint|align|blend|calibrate)\b', re.I)
PANEL_RX = re.compile(r'\b(' + '|'.join(re.escape(p) for p in PANELS) + r')\b', re.I)
PRICE_OR_LABOR_RX = re.compile(r'(\$\s*\d|\bhrs?\s*@\s*\$|\brate\b)', re.I)

# ---- Side/part normalization helpers ---------------------------------
SIDE_PATTERNS = [
    (re.compile(r'\b(RR|R/R|RIGHT\s*REAR)\b', re.I), "right rear"),
    (re.compile(r'\b(LR|L/R|LEFT\s*REAR)\b', re.I), "left rear"),
    (re.compile(r'\b(RF|R/F|RIGHT\s*FRONT)\b', re.I), "right front"),
    (re.compile(r'\b(LF|L/F|LEFT\s*FRONT)\b', re.I), "left front"),
    (re.compile(r'\b(REAR)\b', re.I), "rear"),
    (re.compile(r'\b(FRONT)\b', re.I), "front"),
    (re.compile(r'\b(RIGHT|RH)\b', re.I), "right"),
    (re.compile(r'\b(LEFT|LH)\b', re.I), "left"),
]

def detect_side(raw: str) -> str:
    for rx, name in SIDE_PATTERNS:
        if rx.search(raw or ""):
            return name
    return "unspecified"

def canonicalize_part(raw_line: str, part: str) -> str:
    l = (raw_line or "").lower()
    p = (part or "").lower()
    if p == "cover" and "bumper" in l:
        return "bumper cover"
    if p in {"lamp", "taillamp", "tail lamp"} and ("tail" in l or "rear" in l):
        return "tail lamp"
    return part

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    """Strict block-based parser (prefers CCC-style line table)."""
    items: List[Dict[str, str]] = []
    in_lines = False
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue

        if re.search(r'\bLine\s+Oper\s+Description\b', line, re.I):
            in_lines = True
            continue
        if in_lines and re.search(r'\b(ESTIMATE\s+TOTALS|NOTES|REQUEST A SUPPLEMENT|ALTERNATE PARTS USAGE|RECALL INFO)\b', line, re.I):
            in_lines = False

        if not in_lines:
            continue

        l = line.lower()
        if OP_RX.search(l) and PANEL_RX.search(l) and PRICE_OR_LABOR_RX.search(l):
            op = OP_RX.search(l).group(1) if OP_RX.search(l) else "op"
            part = PANEL_RX.search(l).group(1) if PANEL_RX.search(l) else "component"
            part = canonicalize_part(raw, part)
            side = detect_side(raw)
            items.append({"op": op, "part": part, "side": side, "raw": raw})

    # de-dupe
    uniq, seen = [], set()
    for it in items:
        key = (it["op"].lower(), it["part"].lower(), it["side"].lower())
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

# --- Loose estimate scanner when strict table not found ----
LOOSE_OP = re.compile(r'\b(repair|repl(?:ace)?|r&r|r&i|refinish|paint|align|blend|calibrate|scan|clear)\b', re.I)
LOOSE_PART = re.compile(r'\b(' + '|'.join(re.escape(p) for p in PANELS + ["valance","finish panel","combo lamp","lamp","sensor","bracket"]) + r')\b', re.I)
SIDE_RX = re.compile(r'\b(left|lh|right|rh|rear|front)\b', re.I)

def extract_estimate_items_loose(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for raw in (text or "").splitlines():
        l = raw.lower()
        if LOOSE_OP.search(l) and LOOSE_PART.search(l):
            op = LOOSE_OP.search(l).group(1)
            part = LOOSE_PART.search(l).group(1)
            part = canonicalize_part(raw, part)
            side = detect_side(raw)
            items.append({"op": op, "part": part, "side": side, "raw": raw})

    # de-dupe & cap
    uniq, seen = [], set()
    for it in items:
        key = (it["op"].lower(), it["part"].lower(), it["side"].lower())
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq[:30]

# =========================================
# GPT compare (richer notes/overall)
# =========================================
def compare_estimate_with_photos(items: List[Dict[str, str]],
                                 images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    # same schema, but we’ll request richer content in 'note' and 'overall'
    schema = {"type":"object","properties":{
        "per_item":{"type":"array","items":{"type":"object","properties":{
            "op":{"type":"string"},"part":{"type":"string"},"side":{"type":"string"},
            "photo_evidence":{"type":"boolean"},"confidence":{"type":"number"},
            "note":{"type":"string"}},
            "required":["op","part","side","photo_evidence","confidence","note"]}},
        "not_in_photos":{"type":"array","items":{"type":"string"}},
        "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
        "overall":{"type":"string"}}, "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]}

    system = (
        "You are an auto-damage visual auditor. "
        "Given estimate line items and vehicle photos, decide for EACH item whether visible photo evidence exists. "
        "Treat generic terms like 'cover' as 'bumper cover' when the context suggests bumper work. "
        "Notes must be concise but substantive (1–2 full sentences), pointing to the visible area (e.g., 'RR bumper shows scuff/deformation'). "
        "The 'overall' must be a compact multi-sentence summary (3–5 sentences) describing what is supported vs not, "
        "any extra visible damage, and any items likely procedural/hidden (scans/R&I/clear, etc.). "
        "Return STRICT JSON ONLY per this schema: " + json.dumps(schema)
    )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": "Estimate items (normalized):\n" + json.dumps(items, ensure_ascii=False)}
    ]
    user_parts.extend(images_for_vision)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_parts}],
            max_tokens=450,
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
        return {"per_item": [],"not_in_photos": [],"extra_damage_in_photos": [],"overall": f"Comparison unavailable ({type(e).__name__})."}

# ---------- Text fallback ----------
POI_RX = re.compile(r'Point\s+of\s+Impact\s*:\s*([^\n]+)', re.I)

def build_text_consistency_review(items: List[Dict[str, str]], estimate_text: str) -> Tuple[List[str], str]:
    bullets: List[str] = []

    # Point of Impact
    poi = ""
    m = POI_RX.search(estimate_text or "")
    if m:
        poi = m.group(1).strip()
        bullets.append(f"- Point of impact per estimate: {poi}")

    # If items empty, try to recover with loose scan
    if not items:
        items = extract_estimate_items_loose(estimate_text)

    # Bucket items
    scope, refinish, procedural = [], [], []
    for it in items:
        op = (it.get("op","") or "").lower()
        part = (it.get("part","") or "").lower()
        side = it.get("side","unspecified")
        label_op = "Repair" if "repair" in op else "Replace" if "repl" in op else it.get("op","").title() or "Operation"
        pretty = f"{side.title()} {part}".strip().replace("Unspecified ","").replace("unspecified ","").title()
        entry = f"{pretty} — *{label_op}*"

        if "refinish" in op or "paint" in op or "blend" in op:
            refinish.append(entry)
        elif any(tok in op for tok in ["scan","calibrate","clear","r&r","r&i","align"]):
            procedural.append(entry)
        else:
            scope.append(entry)

    if scope:
        bullets.append("- Items in estimate (selected):")
        for s in scope[:12]:
            bullets.append(f"  • {s}")
    if refinish:
        bullets.append("- Refinish/Paint operations:")
        for r in refinish[:8]:
            bullets.append(f"  • {r}")
    if procedural:
        bullets.append("- Procedural/administrative operations:")
        for p in procedural[:10]:
            bullets.append(f"  • {p}")

    if not bullets:
        bullets.append("- No parseable line items found in the estimate text.")

    overall = "Estimate scope summarized above; use photos to validate visible exterior items and keep procedural lines as non-photo-verifiable."
    return bullets, overall

# =========================================
# Summary builder
# =========================================
def build_summary_markdown(
    missing_photos: List[str],
    text: str,
    client_rules: str,
    require_oem: bool,
    non_oem_flag: bool,
    client_lines_override: Optional[List[str]] = None,
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

    # Parts Compliance (respect client preference if present)
    parts_lines: List[str] = []
    prefer_aftermarket = bool(re.search(
        r"(heavy\s+on\s+the\s+use\s+of\s+aftermarket|consider\s+.*aftermarket\s+.*before\s+(?:lkq|oem)|"
        r"utilize\s+(?:lkq|recon|aftermarket)\s+parts\s+regardless\s+of\s+year|regardless\s+of\s+year\s+or\s+mileage)",
        (client_rules or "").lower(), re.DOTALL))

    if prefer_aftermarket:
        if non_oem_flag:
            parts_lines.append("- Compliant: aftermarket/LKQ/recon parts used per client preference.")
        else:
            parts_lines.append("- Non-compliant: OEM used; document why alternatives were not used per client preference.")
    else:
        if require_oem:
            parts_lines.append(
                "- Non-compliant: non-OEM parts on ≤ 2 years or ≤ 24k miles." if non_oem_flag
                else "- Compliant: OEM parts only for ≤ 2 years or ≤ 24k miles."
            )
        else:
            parts_lines.append(
                "- Non-OEM parts noted; verify client rules allow on this vehicle." if non_oem_flag
                else "- Parts appear OEM or not flagged as non-OEM."
            )

    client_lines = client_lines_override if client_lines_override else ["- Apply client-required documentation (labor rates, photos, taxes) where applicable."]
    notes_lines = ["- Ensure estimate notes clearly explain damage appraisal per client requirements."]

    sections = [
        "### Required Photos", *photos_lines,
        "### Labor Rates", *labor_lines,
        "### Taxes", *taxes_lines,
        "### Parts Compliance", *parts_lines,
        "### Client Rules Adherence", *client_lines,
        "### Additional Notes", *notes_lines,
    ]
    return "\n".join(sections)

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
    first_pdf_bytes: Optional[bytes] = None

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_blobs.append((name, raw))
        elif name.endswith(".pdf"):
            embedded_txt = extract_text_from_pdf_embedded(raw)
            if embedded_txt:
                texts.append(embedded_txt)
            texts.append(extract_text_from_pdf(io.BytesIO(raw), max_ocr_pages=8, dpi=140))
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

    # Contact sheets for vision step
    contact_sheets = make_contact_sheets_compact(image_blobs, max_sheets=3, cols=6, padding=6, base_thumb_w=320, jpeg_quality=68)
    images_for_vision: List[Dict[str, Any]] = []
    for name, blob in contact_sheets:
        b64 = base64.b64encode(blob).decode("utf-8")
        images_for_vision.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    # Required photos presence
    missing_photos = check_required_photos(image_blobs, combined_text)

    # VIN & Claim from estimate
    vin_est = extract_vin_from_text(combined_text) or (extract_vin_from_pdf_first_pages(first_pdf_bytes, 4, 170) if first_pdf_bytes else None)
    claim_number = extract_claim_from_text(combined_text) or (extract_claim_from_pdf_first_pages(first_pdf_bytes, 4, 170) if first_pdf_bytes else None)
    claim_number = claim_number or "N/A"

    # VIN photo verification
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

    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    # Parse estimate items (strict, then loose if needed)
    est_items = extract_estimate_items(combined_text)
    if not est_items:
        est_items = extract_estimate_items_loose(combined_text)

    # Vision compare (richer)
    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    year, miles = parse_year_miles(combined_text)
    now_year = datetime.datetime.now().year
    age_years = (now_year - year) if year else None
    require_oem = (age_years is not None and age_years <= 2) or (miles is not None and miles <= 24000)
    non_oem_flag = non_oem_used(combined_text)

    # Client guideline adherence lines
    guidelines = parse_client_rules(client_rules)
    client_lines = build_client_adherence_lines(
        guidelines=guidelines,
        missing_photos=missing_photos,
        text=combined_text,
        year=year,
        miles=miles,
        non_oem_flag=non_oem_flag,
    )

    summary_md = build_summary_markdown(
        missing_photos=missing_photos,
        text=combined_text,
        client_rules=client_rules,
        require_oem=require_oem,
        non_oem_flag=non_oem_flag,
        client_lines_override=client_lines,
    )

    # ----- Scoring -----
    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)

    # parts adjustment for client preference or OEM-when-recent
    parts_adj = 0
    prefer_aftermarket = bool(guidelines.get("prefer_aftermarket"))
    recent = ((year is not None and (now_year - year) <= 2) or (miles is not None and miles <= 24000))
    if prefer_aftermarket and not non_oem_flag:
        parts_adj -= 25
    elif guidelines.get("oem_required_if_recent") and recent and non_oem_flag:
        parts_adj -= 25

    computed = max(0, 100 + labor_tax_adj + photo_adj + parts_adj)
    authoritative_score = computed

    # Build PDF
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
        supported = 0
        total = len(consistency["per_item"])
        for it in consistency["per_item"][:80]:
            ev = bool(it.get("photo_evidence"))
            if ev: supported += 1
            try: conf = float(it.get("confidence", 0))
            except Exception: conf = 0.0
            conf_txt = f"{round(conf*100)}%"
            side = it.get('side','unspecified').title()
            part = it.get('part','component')
            op = it.get('op','op')
            note = it.get('note','').strip()
            pdf.multi_cell(0, 6, f"- {side} {part} · {op} → Photo: {'YES' if ev else 'NO'} ({conf_txt}); {note}")

        # Comparison summary block
        pdf.ln(2); pdf_add_section_title(pdf, "Comparison Summary")
        pdf.multi_cell(0, 6, f"Supported by photos: {supported} of {total}")
        if consistency.get("not_in_photos"):
            pdf.multi_cell(0, 6, f"Estimated but not evident: {', '.join(consistency['not_in_photos'][:12])}")
        if consistency.get("extra_damage_in_photos"):
            pdf.multi_cell(0, 6, f"Visible but not estimated: {', '.join(consistency['extra_damage_in_photos'][:12])}")
    else:
        # richer text fallback (multi-bullet)
        fb_bullets, fb_overall = build_text_consistency_review(est_items, combined_text)
        for b in fb_bullets:
            pdf.multi_cell(0, 6, b)
        consistency["fallback_text"] = fb_bullets
        consistency["overall"] = fb_overall

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

    # Email — enabled (no attachment)
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        smtp_from = os.getenv("SMTP_FROM", "info@nspxn.com")
        smtp_to = os.getenv("SMTP_TO", "info@nspxn.com")
        msg["From"] = smtp_from
        msg["To"] = smtp_to

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

        smtp_host = os.getenv("SMTP_HOST", "mail.tierra.net")
        smtp_port = int(os.getenv("SMTP_PORT", "465"))
        smtp_user = os.getenv("SMTP_USER", "info@nspxn.com")
        smtp_pass = os.getenv("SMTP_PASS")  # must be set in env

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as smtp:
                if smtp_user and smtp_pass:
                    smtp.login(smtp_user, smtp_pass)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
                smtp.ehlo()
                try:
                    smtp.starttls()
                    smtp.ehlo()
                except Exception:
                    logger.info("STARTTLS not supported or failed; continuing without TLS.")
                if smtp_user and smtp_pass:
                    smtp.login(smtp_user, smtp_pass)
                smtp.send_message(msg)

        logger.info("Email sent successfully (no attachment).")
    except Exception as e:
        logger.error(f"Email error: {e}")

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




























