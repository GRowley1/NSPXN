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
# OpenAI client (gpt-4o default; vision uses VISION_MODEL below)
# =========================================
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

# ------- Speed knobs (override with ENV to tune without redeploy) -------
FAST_MODE = os.getenv("FAST_MODE", "1") == "1"   # set FAST_MODE=0 to disable

# OCR/page controls
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "4" if FAST_MODE else "8"))
OCR_DPI = int(os.getenv("OCR_DPI", "120" if FAST_MODE else "140"))

# Contact sheet controls (this is the biggest latency lever)
CONTACT_MAX_SHEETS = int(os.getenv("CONTACT_MAX_SHEETS", "2" if FAST_MODE else "3"))
CONTACT_COLS = int(os.getenv("CONTACT_COLS", "7" if FAST_MODE else "6"))
CONTACT_THUMB_W = int(os.getenv("CONTACT_THUMB_W", "260" if FAST_MODE else "320"))
CONTACT_JPEG_QUALITY = int(os.getenv("CONTACT_JPEG_QUALITY", "55" if FAST_MODE else "68"))

# Make the contact sheets even lighter by default in FAST_MODE
if FAST_MODE:
    CONTACT_MAX_SHEETS = int(os.getenv("CONTACT_MAX_SHEETS", "1"))
    CONTACT_THUMB_W = int(os.getenv("CONTACT_THUMB_W", "220"))
    CONTACT_JPEG_QUALITY = int(os.getenv("CONTACT_JPEG_QUALITY", "50"))

# VIN scan controls
VIN_UPSCALE_WIDTH = int(os.getenv("VIN_UPSCALE_WIDTH", "2000" if FAST_MODE else "2400"))
VIN_MAX_PHOTOS_SCAN = int(os.getenv("VIN_MAX_PHOTOS_SCAN", "20" if FAST_MODE else "9999"))

# Vision model (faster but still good enough)
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini" if FAST_MODE else "gpt-4o")

# Time budget and OpenAI timeout
REQUEST_BUDGET_SECONDS = int(os.getenv("REQUEST_BUDGET_SECONDS", "65"))  # stay under proxy limits
OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "30"))

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

def extract_text_from_pdf(file_like: io.BytesIO, max_ocr_pages: int = None, dpi: int = None) -> str:
    """Faster OCR: fewer pages, lower DPI, single-PSM retry if needed."""
    max_ocr_pages = max_ocr_pages or OCR_MAX_PAGES
    dpi = dpi or OCR_DPI
    try:
        file_like.seek(0)
        pages = convert_from_bytes(file_like.read(), dpi=dpi)
        text_output = []
        for i, img in enumerate(pages, 1):
            if i > max_ocr_pages:
                break
            t = ocr_text_fast(img, psm=6)
            if len((t or "").strip()) < 20:
                t = ocr_text_fast(img, psm=3)
            if t:
                text_output.append(f"\n[Page {i}]\n{t}")
        return "".join(text_output)
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
# Photo harvesting (skip for obvious estimates)
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
# Vehicle & tax/parts helpers (robust vehicle parsing)
# =========================================
MAKE_ALIASES = {
    "CHEV": "Chevrolet", "CHEVROLET": "Chevrolet", "CHEVY": "Chevrolet",
    "MERCEDES-BENZ": "Mercedes-Benz", "MERCEDES": "Mercedes-Benz",
    "LAND ROVER": "Land Rover", "VW": "Volkswagen",
}
KNOWN_MAKES = [
    "ACURA","ALFA ROMEO","AUDI","BMW","BUICK","CADILLAC","CHEV","CHEVROLET","CHEVY",
    "CHRYSLER","DODGE","FIAT","FORD","GMC","HONDA","HYUNDAI","INFINITI","ISUZU",
    "JAGUAR","JEEP","KIA","LAND ROVER","LEXUS","LINCOLN","MAZDA","MERCEDES","MERCEDES-BENZ",
    "MINI","MITSUBISHI","NISSAN","PORSCHE","RAM","SAAB","SUBARU","SUZUKI","TESLA",
    "TOYOTA","VOLKSWAGEN","VW","VOLVO"
]
MAKE_RE = re.compile(
    r"\b(20\d{2}|\d{4})\s+(" + "|".join(re.escape(m) for m in KNOWN_MAKES) + r")\b[^\n]{0,60}",
    re.IGNORECASE
)

def _normalize_make(m: str) -> str:
    m = m.upper()
    return MAKE_ALIASES.get(m, m.title())

def normalize_vehicle_str(s: str) -> str:
    if not s: return s
    s2 = re.sub(r'\s{2,}', ' ', s).replace(' ,', ',')
    return s2.strip()

def extract_vehicle_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    best_line = None
    for m in MAKE_RE.finditer(text):
        year = m.group(1)
        make = _normalize_make(m.group(2))
        line_end = text.find("\n", m.start())
        if line_end == -1:
            line_end = m.end() + 60
        chunk = text[m.end():line_end]
        model = re.sub(r"[,/].*$", "", chunk).strip()
        model = re.sub(r"\b(vehicles?|contain|minor|turbocharged|diesel|indirect|fi|4dr?|4d|p/u)\b.*", "", model, flags=re.I)
        model = re.sub(r"\s{2,}", " ", model).strip(" -·")
        if not model:
            model = "Model"
        best_line = f"{year} {make} {model}"
        break
    if not best_line:
        return None
    m_mi = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    miles = m_mi.group(1) if m_mi else "unknown"
    return normalize_vehicle_str(f"{best_line}, {miles} miles")

def parse_year_miles(text: str) -> Tuple[Optional[int], Optional[int]]:
    year = None; miles = None
    m_year = re.search(r"\b(19|20)\d{2}\b", text or "")
    if m_year:
        try: year = int(m_year.group(0))
        except: year = None
    m_mi = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text or "", re.IGNORECASE)
    if m_mi:
        try: miles = int(m_mi.group(1).replace(",", ""))
        except: miles = None
    return year, miles

def taxes_present(text: str) -> bool:
    return re.search(r'tax[^\n]{0,50}(\d{1,3}\s*%|\$\s*\d+(\.\d{2})?)', text or "", re.IGNORECASE) is not None

# Parts token counting
PART_TOKEN_RX = re.compile(r'\b(A/M|AFTERMARKET|LKQ|RECOND(?:ITIONED)?|REMAN(?:UFACTURED)?|CAPA|NSF|ALT[-\s]*OE|OEM)\b', re.I)
def estimate_parts_mix(text: str) -> Dict[str, int]:
    counts = {"oem":0,"aftermarket":0,"lkq":0,"recon":0,"capa":0,"nsf":0,"alt_oe":0}
    for m in PART_TOKEN_RX.finditer(text or ""):
        tok = m.group(1).upper().replace("ALTOE","ALT OE")
        if tok in ("A/M","AFTERMARKET"): counts["aftermarket"] += 1
        elif tok == "LKQ": counts["lkq"] += 1
        elif tok.startswith("RECOND") or tok.startswith("REMAN"): counts["recon"] += 1
        elif tok == "CAPA": counts["capa"] += 1
        elif tok == "NSF": counts["nsf"] += 1
        elif "ALT" in tok and "OE" in tok: counts["alt_oe"] += 1
        elif tok == "OEM": counts["oem"] += 1
    return counts

# =========================================
# Photo parsing (VIN/ODO/plate presence)
# =========================================
def _is_exterior_by_edges(img: Image.Image) -> bool:
    g = img.convert("L")
    var = ImageStat.Stat(g).var[0]
    edges = g.filter(ImageFilter.FIND_EDGES)
    evar = ImageStat.Stat(edges).var[0]
    return (var > 140 and evar > 400)

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    VIN_NEAR_LABEL = re.compile(r'(?i)\bV[\W_]*I[\W_]*N\b')
    VIN_SEP_SEQ = re.compile(r'(?i)((?:[A-HJ-NPR-Z0-9][\s\.\-–—:_]){16}[A-HJ-NPR-Z0-9])')

    def variants(im: Image.Image) -> List[Image.Image]:
        base = im.convert("L")
        if base.width < VIN_UPSCALE_WIDTH:
            h = int(base.height * (VIN_UPSCALE_WIDTH / max(base.width, 1)))
            base = base.resize((VIN_UPSCALE_WIDTH, h), Image.LANCZOS)
        return [
            base,
            ImageEnhance.Contrast(base).enhance(2.0),
            ImageOps.autocontrast(base.filter(ImageFilter.MedianFilter(3))),
        ]

    def ocr_all(im: Image.Image) -> str:
        out = []
        for v in variants(im):
            for psm in (7, 6):  # fewer PSMs = faster
                try:
                    t = pytesseract.image_to_string(v, lang="eng", config=f"--psm {psm} --oem 1")
                    if t: out.append(t)
                except Exception:
                    pass
        return "\n".join(out).upper()

    scanned = 0
    for name, blob in image_blobs:
        if scanned >= VIN_MAX_PHOTOS_SCAN:
            break
        scanned += 1
        try:
            im = Image.open(io.BytesIO(blob))
        except Exception:
            continue
        up = ocr_all(im)

        for m in VIN_NEAR_LABEL.finditer(up):
            window = up[m.end(): m.end() + 240]
            for mm in VIN_SEP_SEQ.finditer(window):
                vin = normalize_vin(mm.group(1))
                if vin and vin_checksum_ok(vin):
                    return vin

        for mm in VIN_SEP_SEQ.finditer(up):
            vin = normalize_vin(mm.group(1))
            if vin and vin_checksum_ok(vin):
                return vin

        cands = re.findall(r'\b([A-HJ-NPR-Z0-9]{17})\b', up)
        vin = best_vin_candidate(cands)
        if vin: return vin

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
    for v in variants(img):
        for psm in (6, 7, 11):
            try:
                t = pytesseract.image_to_string(v, lang="eng", config=f"--psm {psm} --oem 1")
                if t: out.append(t)
            except Exception:
                pass
    return "\n".join(out)

# =========================================
# Required photos: PHOTOS-ONLY presence
# =========================================
def check_required_photos(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> List[str]:
    """
    Photo presence must be proven by photos only (no text fallback).
    Prevents contradictions like 'VIN photo not found' while claiming the VIN photo is present.
    """
    # Optional cap for presence checks (does NOT limit uploads)
    if FAST_MODE and len(image_blobs) > 60:
        image_blobs = image_blobs[:60]

    required = ["four corners", "odometer", "vin", "license plate"]
    present = set()

    # VIN by photo only
    if extract_vin_from_photos(image_blobs) is not None:
        present.add("vin")

    # Odometer by photo only
    if extract_odometer_from_photos(image_blobs) is not None:
        present.add("odometer")

    # License plate by photo OCR
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            txtp = _plate_ocr_variants(img)
            if re.search(r'(license|registration)\s*plate', txtp, re.IGNORECASE) or PLATE_RX.search(txtp):
                present.add("license plate")
                break
        except Exception:
            pass

    # Four corners heuristic by exterior-photo count
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
    logger.debug(f"[photos-only] present={sorted(list(present))}, missing={missing}, ext_hits={exterior_hits}")
    return missing

# =========================================
# Estimate: simple, BRIEF summary
# =========================================
POI_RX = re.compile(r'Point\s+of\s+Impact\s*:\s*([^\n]+)', re.I)
SUBTOTAL_RX = re.compile(r'\bSubtotal\b[^\n]*\s(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', re.I)
TOTAL_RX = re.compile(r'\bTotal\s+Cost\s+of\s+Repairs\b[^\n]*\s(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', re.I)
LABOR_RATE_RX = re.compile(r'\b(Body|Paint|Mechanical|Frame|Refinish|Supplies)\b[^\n]{0,120}?\$\s*\d{2,3}', re.I)

def build_estimate_brief(text: str) -> List[str]:
    bullets: List[str] = []
    if not text:
        return ["- Unable to read estimate text."]

    m = POI_RX.search(text)
    if m:
        bullets.append(f"- Point of Impact: {m.group(1).strip()}")

    counts = {
        "Replace": len(re.findall(r'\bRepl(?:ace)?\b', text, re.I)),
        "Repair": len(re.findall(r'\b(Rpr|Repair)\b', text, re.I)),
        "Refinish/Paint": len(re.findall(r'\b(Refinish|Paint|Blend)\b', text, re.I)),
        "R&I/R&R": len(re.findall(r'\b(R&I|R & I|R&R|R & R)\b', text, re.I)),
        "Scan/Calibrate": len(re.findall(r'\b(Scan|Calibrate)\b', text, re.I)),
    }
    ops_summary = ", ".join([f"{k} {v}" for k, v in counts.items() if v > 0]) or "No operations detected"
    bullets.append(f"- Operations: {ops_summary}")

    mix = estimate_parts_mix(text)
    mix_line = (f"OEM {mix.get('oem',0)}, Aftermarket {mix.get('aftermarket',0)}, "
                f"CAPA {mix.get('capa',0)}, LKQ {mix.get('lkq',0)}, Recon/Reman {mix.get('recon',0)}, "
                f"NSF {mix.get('nsf',0)}, ALT-OE {mix.get('alt_oe',0)}")
    bullets.append(f"- Parts mix: {mix_line}")

    bullets.append("- Labor rates: present" if LABOR_RATE_RX.search(text) else "- Labor rates: not clearly listed")
    bullets.append("- Sales tax: present" if taxes_present(text) else "- Sales tax: not found")

    t = TOTAL_RX.search(text)
    s = SUBTOTAL_RX.search(text)
    if t:
        bullets.append(f"- Total repairs: ${t.group(1)}")
    elif s:
        bullets.append(f"- Subtotal: ${s.group(1)}")

    return bullets[:8]

# =========================================
# GPT: Estimate ↔ Photos comparison (concise)
# =========================================
def compare_estimate_with_photos_brief(estimate_text: str,
                                       images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = {"type":"object","properties":{
        "per_item":{"type":"array","items":{"type":"object","properties":{
            "item":{"type":"string"},
            "photo_evidence":{"type":"boolean"},
            "confidence":{"type":"number"},
            "note":{"type":"string"}},
            "required":["item","photo_evidence","confidence","note"]}},
        "not_in_photos":{"type":"array","items":{"type":"string"}},
        "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
        "overall":{"type":"string"}}, "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]}

    system = (
        "You are an auto-damage visual auditor. "
        "Compare the provided ESTIMATE TEXT to the VEHICLE PHOTOS. "
        "Extract up to 8 of the most material scope items from the estimate (e.g., bumper cover repair, RR combo lamp replace, scans). "
        "For EACH selected item, decide if photos show evidence (YES/NO) and give a short note. "
        "List items estimated but not supported by photos, and visible damage not on the estimate. "
        "Write a crisp overall summary (3–4 sentences). "
        "Return STRICT JSON matching this schema: " + json.dumps(schema)
    )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": "ESTIMATE TEXT (OCR):\n" + (estimate_text or "")[:35000]}
    ]
    user_parts.extend(images_for_vision)

    try:
        rsp = client.with_options(timeout=OPENAI_REQUEST_TIMEOUT).chat.completions.create(
            model=VISION_MODEL,   # faster model for vision
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_parts}],
            max_tokens=400 if FAST_MODE else 500,
            temperature=0
        )
        txt = (rsp.choices[0].message.content or "").strip()
        txt = txt.removesuffix("```").removesuffix("```json").removeprefix("```json").strip()
        data = json.loads(txt)
        if not isinstance(data, dict) or "per_item" not in data:
            raise ValueError("JSON shape mismatch")
        return data
    except Exception as e:
        logger.error(f"Vision compare BRIEF JSON error: {type(e).__name__}: {e}")
        # FAST fallback: return an empty structured result so the PDF still completes quickly
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison unavailable (timeout or error)."}

# =========================================
# Client-guideline parsing & adherence
# =========================================
PHOTO_KEYWORDS = {
    "four corners": ["four corners", "4 corners", "four-corners"],
    "vin": ["vin", "v.i.n"],
    "odometer": ["odometer", "mileage"],
    "license plate": ["license plate", "registration plate", "plate photo"],
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
    require_tax = bool(re.search(r"\b(apply|include|utilize)\s+tax\b|\btax\s+(required|must)\b", cr))
    require_market_doc = bool(re.search(r"\b(nada|kelley|kbb|black\s*book|retail\s+value|market\s+value)\b", cr))
    require_valuation_includes_tax = bool(re.search(r"\bvaluation\b.*\btax\b|\binclude\b.*\btax\b.*\bvaluation\b", cr, re.DOTALL))
    require_total_loss_decl = bool(re.search(r"\b(total\s+loss).*(declare|declaration)|\bdeclare\b.*\btotal\s+loss\b", cr, re.DOTALL))

    oem_recent = bool(re.search(r"(?:<=?|less\s+than|under)\s*2\s*year", cr)) or bool(re.search(r"(?:<=?|less\s+than|under)\s*24\s*[,k]*\s*mi", cr))
    prefer_aftermarket = bool(re.search(
        r"(heavy\s+on\s+the\s+use\s+of\s+aftermarket|"
        r"consider\s+.*aftermarket\s+.*before\s+(?:lkq|oem)|"
        r"utilize\s+(?:lkq|recon|aftermarket)\s+parts\s+regardless\s+of\s+year|"
        r"regardless\s+of\s+year\s+or\s+mileage)",
        cr, re.DOTALL))

    return {
        "photos_required": photos_req,
        "require_labor_rates": require_labor,
        "require_tax": require_tax,
        "require_market_doc": require_market_doc,
        "require_valuation_includes_tax": require_valuation_includes_tax,
        "require_total_loss_decl": require_total_loss_decl,
        "oem_required_if_recent": oem_recent,
        "prefer_aftermarket": prefer_aftermarket,
    }

def labor_rates_present_any(text: str) -> bool:
    labels = ["Body", "Paint", "Mechanical", "Structural", "Frame", "Refinish", "Supplies"]
    return any(re.search(rf"{lbl}[^\n]{{0,120}}?\$\s*\d{{2,3}}", text or "", re.IGNORECASE) for lbl in labels)

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

    if guidelines.get("require_labor_rates"):
        if labor_rates_present_any(text):
            lines.append("- Compliant: labor rates listed on estimate.")
        else:
            lines.append("- Non-compliant: labor rates not listed per client rules.")

    if guidelines.get("require_tax"):
        if taxes_present(text):
            lines.append("- Compliant: tax rate present per client rules.")
        else:
            lines.append("- Non-compliant: tax rate not found per client rules.")

    if guidelines.get("require_market_doc"):
        if has_market_value_doc(text):
            lines.append("- Compliant: required retail/market value documentation present (e.g., NADA/KBB).")
        else:
            lines.append("- Non-compliant: required retail/market value documentation not found.")

    if guidelines.get("require_valuation_includes_tax"):
        if re.search(r"valuation[^\n]{0,80}(tax|incl(?:ued)?|with\s+tax)", text, re.IGNORECASE):
            lines.append("- Compliant: valuation indicates tax inclusion.")
        elif taxes_present(text):
            lines.append("- Unable to verify: tax present but valuation line not explicit about inclusion.")
        else:
            lines.append("- Non-compliant: valuation tax inclusion not indicated.")

    if guidelines.get("require_total_loss_decl"):
        if re.search(r"\btotal\s+loss\b", text, re.IGNORECASE):
            lines.append("- Compliant: total loss declaration present.")
        else:
            lines.append("- Unable to verify: total loss declaration not found.")

    prefer_aftermarket = bool(guidelines.get("prefer_aftermarket"))
    recent = ((year is not None and (datetime.datetime.now().year - year) <= 2) or
              (miles is not None and miles <= 24000))
    if prefer_aftermarket:
        if non_oem_flag:
            lines.append("- Compliant: aftermarket/LKQ/recon parts used per client preference.")
        else:
            lines.append("- Non-compliant: OEM used; document why alternative parts were not utilized per client preference.")
    else:
        if guidelines.get("oem_required_if_recent") and recent:
            if non_oem_flag:
                lines.append("- Non-compliant: non-OEM parts used on a ≤2 years or ≤24k miles vehicle.")
            else:
                lines.append("- Compliant: OEM parts used per ≤ 2 years/≤ 24k miles rule.")
        else:
            if non_oem_flag:
                lines.append("- Non-OEM parts noted; verify usage aligns with remaining client rules.")
            else:
                lines.append("- Parts appear OEM or not flagged as non-OEM.")

    return lines

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
        photos_lines = [f"- Missing: {missing_photos[0]}."] if len(missing_photos) == 1 else [f"- Missing: {', '.join(missing_photos)}."]

    labor_lines = ["- Labor rates listed on estimate."] if labor_rates_present_any(text) else ["- Labor rates missing or not clearly listed."]
    taxes_lines = ["- Tax rate present on estimate."] if taxes_present(text) else ["- Tax rate not found per client rules."]

    prefer_aftermarket = bool(re.search(
        r"(heavy\s+on\s+the\s+use\s+of\s+aftermarket|consider\s+.*aftermarket\s+.*before\s+(?:lkq|oem)|"
        r"utilize\s+(?:lkq|recon|aftermarket)\s+parts\s+regardless\s+of\s+year|regardless\s+of\s+year\s+or\s+mileage)",
        (client_rules or "").lower(), re.DOTALL))

    parts_lines: List[str] = []
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

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    if not labor_rates_present_any(text or ""):
        adj -= 50
    if re.search(r"tax\s*(required|must|utilize|apply|include)", (client_rules or ""), re.IGNORECASE):
        if not taxes_present(text or ""):
            adj -= 25
    return adj

# =========================================
# Helpers for contact sheets
# =========================================
def shrink_to_width(img: Image.Image, max_w: int) -> Image.Image:
    if img.width <= max_w:
        return img.convert("RGB")
    h = int(img.height * max_w / img.width)
    return img.convert("RGB").resize((max_w, h), Image.LANCZOS)

def make_contact_sheets_compact(
    image_blobs: List[Tuple[str, bytes]],
    max_sheets: int = None,
    cols: int = None,
    padding: int = 6,
    base_thumb_w: int = None,
    jpeg_quality: int = None
) -> List[Tuple[str, bytes]]:
    max_sheets = max_sheets or CONTACT_MAX_SHEETS
    cols = cols or CONTACT_COLS
    base_thumb_w = base_thumb_w or CONTACT_THUMB_W
    jpeg_quality = jpeg_quality or CONTACT_JPEG_QUALITY

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
    if n == 0:
        return []
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
# Utility: time budget
# =========================================
def time_budget_exceeded(start_ts: datetime.datetime, budget_s: int) -> bool:
    return (datetime.datetime.now() - start_ts).total_seconds() > budget_s

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
    start_ts = datetime.datetime.now()
    def stage(msg: str):
        dt = (datetime.datetime.now() - start_ts).total_seconds()
        logger.info(f"[T+{dt:0.2f}s] {msg}")

    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error": "Appraiser ID is required."})

    texts: List[str] = []
    image_blobs: List[Tuple[str, bytes]] = []
    first_pdf_bytes: Optional[bytes] = None

    # -------- Ingest uploads --------
    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_blobs.append((name, raw))
        elif name.endswith(".pdf"):
            embedded_txt = extract_text_from_pdf_embedded(raw)
            if embedded_txt:
                texts.append(embedded_txt)
            texts.append(extract_text_from_pdf(io.BytesIO(raw), max_ocr_pages=OCR_MAX_PAGES, dpi=OCR_DPI))
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

    stage("uploads received / OCR done")
    combined_text = "\n".join(texts)

    # -------- Contact sheets for GPT vision --------
    contact_sheets = make_contact_sheets_compact(
        image_blobs,
        max_sheets=CONTACT_MAX_SHEETS,
        cols=CONTACT_COLS,
        base_thumb_w=CONTACT_THUMB_W,
        jpeg_quality=CONTACT_JPEG_QUALITY
    )
    images_for_vision: List[Dict[str, Any]] = []
    for name, blob in contact_sheets:
        b64 = base64.b64encode(blob).decode("utf-8")
        images_for_vision.append({"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    stage("contact sheets built")

    # -------- Required photos (PHOTOS ONLY) --------
    missing_photos = check_required_photos(image_blobs, combined_text)

    # -------- VIN / Claim from estimate & photos --------
    vin_est = extract_vin_from_text(combined_text) or (extract_vin_from_pdf_first_pages(first_pdf_bytes, 4, 170) if first_pdf_bytes else None)
    claim_number = extract_claim_from_text(combined_text) or (extract_claim_from_pdf_first_pages(first_pdf_bytes, 4, 170) if first_pdf_bytes else None)
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

    stage("VIN/claim extracted")

    # -------- Vehicle + parts mix --------
    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)
    parts_mix = estimate_parts_mix(combined_text)
    non_oem_flag = (parts_mix.get("aftermarket",0)+parts_mix.get("lkq",0)+parts_mix.get("recon",0)+parts_mix.get("capa",0)+parts_mix.get("nsf",0)+parts_mix.get("alt_oe",0) > 0)

    # -------- GPT comparison with time budget --------
    if time_budget_exceeded(start_ts, REQUEST_BUDGET_SECONDS - 10):
        logger.warning("Skipping GPT vision compare due to time budget.")
        consistency = {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison skipped due to time budget."}
    else:
        consistency = compare_estimate_with_photos_brief(combined_text, images_for_vision)
    stage("GPT vision compare done (or skipped)")

    # -------- Scoring --------
    year, miles = parse_year_miles(combined_text)
    now_year = datetime.datetime.now().year
    recent_vehicle = ((year is not None and (now_year - year) <= 2) or (miles is not None and miles <= 24000))

    guidelines = parse_client_rules(client_rules)
    prefer_aftermarket = bool(guidelines.get("prefer_aftermarket"))
    require_oem_due_to_rules = bool(guidelines.get("oem_required_if_recent")) and recent_vehicle

    parts_noncompliant = False
    parts_reason = ""
    if prefer_aftermarket and not non_oem_flag:
        parts_noncompliant = True
        parts_reason = "Client prefers aftermarket/LKQ; OEM used without justification."
    elif require_oem_due_to_rules and non_oem_flag:
        parts_noncompliant = True
        parts_reason = "Non-OEM parts used on a ≤2 years or ≤24k miles vehicle."

    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    parts_adj = -25 if parts_noncompliant else 0

    computed_score = 100 + labor_tax_adj + photo_adj + parts_adj
    authoritative_score = max(0, min(100, computed_score))

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
        require_oem=require_oem_due_to_rules,
        non_oem_flag=non_oem_flag,
        client_lines_override=client_lines,
    )

    # ============== PDF ==============
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

    pdf.ln(4)
    pdf.set_font_size(12); pdf.cell(0, 8, txt="AI-4-IA Review Summary", ln=True); pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"**Audit Results: {authoritative_score}%**")
    if parts_noncompliant and parts_reason:
        pdf.multi_cell(0, 6, f"Deduction applied: -25% (Parts) – {parts_reason}")
    if missing_photos:
        pdf.multi_cell(0, 6, f"Deduction applied: -{25*len(missing_photos)}% (Missing photos)")
    if labor_tax_adj != 0:
        pdf.multi_cell(0, 6, f"Deduction applied: {labor_tax_adj}% (Labor/Tax rules)")
    pdf.ln(1); pdf.multi_cell(0, 6, summary_md)

    # Estimate Details (Brief)
    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0, 8, txt="Estimate Details (Brief)", ln=True); pdf.set_font_size(10)
    for line in build_estimate_brief(combined_text):
        pdf.multi_cell(0, 6, line)
    pm = parts_mix
    mix_line = (f"OEM: {pm.get('oem',0)} | Aftermarket: {pm.get('aftermarket',0)} | CAPA: {pm.get('capa',0)} | "
                f"LKQ: {pm.get('lkq',0)} | Recon/Reman: {pm.get('recon',0)} | NSF: {pm.get('nsf',0)} | ALT-OE: {pm.get('alt_oe',0)}")
    pdf.multi_cell(0, 6, f"Parts Mix: {mix_line}")

    # Estimate ↔ Photos Consistency Review
    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0, 8, txt="Estimate ↔ Photos Consistency Review", ln=True); pdf.set_font_size(10)
    if consistency.get("per_item"):
        for it in consistency["per_item"][:12]:
            ev = bool(it.get("photo_evidence"))
            try: conf = float(it.get("confidence", 0.0))
            except Exception: conf = 0.0
            conf_txt = f"{round(conf*100)}%"
            item = it.get("item","(item)")
            note = it.get("note","").strip()
            pdf.multi_cell(0, 6, f"- {item} → Photo: {'YES' if ev else 'NO'} ({conf_txt}); {note}")
        if consistency.get("not_in_photos"):
            pdf.ln(2); pdf.set_font_size(12); pdf.cell(0, 8, txt="Items Estimated but Not Evident in Photos", ln=True); pdf.set_font_size(10)
            for raw in consistency["not_in_photos"][:20]: pdf.multi_cell(0, 6, f"- {raw}")
        if consistency.get("extra_damage_in_photos"):
            pdf.ln(2); pdf.set_font_size(12); pdf.cell(0, 8, txt="Damage Visible in Photos but Missing on Estimate", ln=True); pdf.set_font_size(10)
            for d in consistency["extra_damage_in_photos"][:20]: pdf.multi_cell(0, 6, f"- {d}")
        pdf.ln(2); pdf.set_font_size(10); pdf.multi_cell(0, 6, f"Consistency Overall: {consistency.get('overall', '')}")
    else:
        pdf.multi_cell(0, 6, "- Comparison unavailable.")

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
        logger.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        logger.error(f"PDF write error: {e}")
    stage("PDF written")

    # -------- Email (unchanged: plain text body, no attachment) --------
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

Compliance Score / Audit Results: {authoritative_score}%

{summary_md}
"""
        msg.set_content(email_body)

        with smtplib.SMTP_SSL("mail.tierra.net", 465, timeout=20) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)

        logger.info("Email sent successfully (original settings, no attachment).")
    except Exception as e:
        logger.error(f"Email error: {e}")
    stage("email attempted")

    return {
        "gpt_output": f"Audit Results: {authoritative_score}%\n\n{summary_md}",
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin_final,
        "vin_photo_verification": vin_verification,
        "score": f"{authoritative_score}%",
        "consistency_review": consistency,
        "parts_mix": parts_mix
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





































