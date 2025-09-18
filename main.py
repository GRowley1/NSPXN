from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any, Callable
import os, re, io, base64, json, logging, asyncio, time, tempfile, subprocess, glob, math
from concurrent.futures import ThreadPoolExecutor

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, ImageFile
from openai import OpenAI

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================= SPEED / BEHAVIOR TUNABLES =======================
PDF_OCR_DPI_EST = int(os.getenv("PDF_OCR_DPI_EST", "160"))
PDF_OCR_DPI_TXT = int(os.getenv("PDF_OCR_DPI_TXT", "150"))
PDF_OCR_DPI_PH  = int(os.getenv("PDF_OCR_DPI_PH",  "140"))
MAX_TEXT_PAGES  = int(os.getenv("MAX_TEXT_PAGES",  "3"))     # quick skim
MAX_PHOTO_PAGES = int(os.getenv("MAX_PHOTO_PAGES", "36"))    # expanded to improve presence detection
MAX_VISION_IMGS = int(os.getenv("MAX_VISION_IMGS", "10"))    # images passed to vision compare
THREADS         = int(os.getenv("OCR_THREADS",     "4"))
OAI_MODEL       = os.getenv("OAI_MODEL", "gpt-4o-mini")
OAI_TIMEOUT_S   = float(os.getenv("OAI_TIMEOUT_S", "15"))
TIME_BUDGET_S   = float(os.getenv("TIME_BUDGET_S", "55"))    # sub-minute target
VISION_BATCH    = int(os.getenv("VISION_BATCH", "12"))        # items per batch for vision compare

# ======================= PDF storage =======================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

# ======================= Logging =======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ======================= OpenAI =======================
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client_fast = client.with_options(timeout=OAI_TIMEOUT_S)

# ======================= FastAPI =======================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com", "https://www.nspxn.com",
        "http://nspxn.com", "http://www.nspxn.com",
        "https://nspxn.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= Time budget helpers =======================
def t0_start() -> float: return time.monotonic()
def t_elapsed(t0: float) -> float: return time.monotonic() - t0
def time_left(t0: float) -> float: return max(0.0, TIME_BUDGET_S - t_elapsed(t0))
def nearly_out_of_time(t0: float, margin: float = 6.0) -> bool: return time_left(t0) <= margin

# ======================= OCR helpers =======================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr_image_quick(img: Image.Image, config: str = "--psm 6") -> str:
    return pytesseract.image_to_string(preprocess_image(img), lang="eng", config=config)

def ocr_pdf_first_page(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=PDF_OCR_DPI_EST)
    return ocr_image_quick(pages[0]) if pages else ""

def ocr_pdf_text_caps(pdf_bytes: bytes, max_pages: int) -> str:
    pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_TXT)
    buf, used = [], 0
    for i, p in enumerate(pages, 1):
        txt = ocr_image_quick(p)
        if len(txt.strip()) >= 25:
            buf.append(f"[Page {i}]\n{txt}")
            used += 1
        if used >= max_pages:
            break
    return "\n".join(buf)

def _page_has_tax(text: str) -> bool:
    return re.search(r"(sales\s*tax|tax)[^\n]{0,160}?(?:\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(?:\.\d{2})?)",
                     text, re.IGNORECASE) is not None

def _page_has_any_labor_rate(text: str) -> bool:
    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    for lbl in labels:
        pat = rf"{lbl}[^\n]{{0,200}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def ocr_pdf_scan_tax_labor_page(pdf_bytes: bytes, max_pages: int = 60) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_TXT)
        for i, p in enumerate(pages[:max_pages], 1):
            txt = ocr_image_quick(p)
            if _page_has_tax(txt) or _page_has_any_labor_rate(txt):
                return f"[Page {i}]\n{txt}"
    except Exception as e:
        logger.warning(f"scan_tax_labor_page error: {e}")
    return ""

# ===== Corner label helpers =====
CORNER_LABEL_PAT = re.compile(r'\b(?:left\s*front|right\s*front|left\s*rear|right\s*rear|lf|rf|lr|rr)\b', re.IGNORECASE)
def count_corner_labels(text: str) -> int:
    found = set()
    for m in re.finditer(CORNER_LABEL_PAT, text or ""):
        token = m.group(0).lower().replace(" ", "")
        if token in ("lf", "leftfront"): found.add("lf")
        elif token in ("rf", "rightfront"): found.add("rf")
        elif token in ("lr", "leftrear"): found.add("lr")
        elif token in ("rr", "rightrear"): found.add("rr")
    return len(found)

# ===== pdftotext helpers =====
def pdftotext_extract(pdf_bytes: bytes, first_page: int, last_page: int) -> str:
    try:
        with tempfile.TemporaryDirectory() as td:
            in_pdf = os.path.join(td, "in.pdf")
            with open(in_pdf, "wb") as f: f.write(pdf_bytes)
            out_txt = os.path.join(td, "out.txt")
            args = ["pdftotext", "-layout", "-f", str(first_page), "-l", str(last_page), in_pdf, out_txt]
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    except Exception as e:
        logger.info(f"pdftotext not available or failed (range): {e}")
    return ""

def pdftotext_extract_all(pdf_bytes: bytes) -> str:
    try:
        with tempfile.TemporaryDirectory() as td:
            in_pdf = os.path.join(td, "in.pdf")
            with open(in_pdf, "wb") as f: f.write(pdf_bytes)
            out_txt = os.path.join(td, "out.txt")
            args = ["pdftotext", "-layout", in_pdf, out_txt]
            subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    except Exception as e:
        logger.info(f"pdftotext not available or failed (full): {e}")
    return ""

# ===== Ensure PNG helper (prevents invalid_image_format) =====
def to_png_bytes(blob: bytes) -> Optional[bytes]:
    try:
        im = Image.open(io.BytesIO(blob))
        if im.mode not in ("RGB","RGBA","L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"to_png_bytes failed: {e}")
        return None

# ===== Extract embedded images quickly; normalize to PNG =====
def pdfimages_harvest(pdf_bytes: bytes, max_images: int = MAX_PHOTO_PAGES) -> List[Tuple[str, bytes, float]]:
    out: List[Tuple[str, bytes, float]] = []
    try:
        with tempfile.TemporaryDirectory() as td:
            in_pdf = os.path.join(td, "in.pdf")
            with open(in_pdf, "wb") as f: f.write(pdf_bytes)
            prefix = os.path.join(td, "img")
            subprocess.run(["pdfimages", "-j", in_pdf, prefix],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            files = sorted(glob.glob(prefix + "*"))
            files.sort(key=lambda p: os.path.getsize(p), reverse=True)
            for i, fp in enumerate(files[:max_images], 1):
                try:
                    with open(fp, "rb") as fh:
                        raw = fh.read()
                    png = to_png_bytes(raw)
                    if png:
                        out.append((f"pdfimg-{i}.png", png, float(os.path.getsize(fp))))
                except Exception:
                    continue
    except Exception as e:
        logger.info(f"pdfimages not available or failed: {e}")
    return out

# ===== Photo-like page harvest (render fallback) — keep VIN/ODO pages =====
def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int) -> List[Tuple[str, bytes, float]]:
    out: List[Tuple[str, bytes, float]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_PH)
        used = 0
        for i, page in enumerate(pages, 1):
            proc = preprocess_image(page)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            up = (ocr or "").upper()

            has_vin_cue = bool(re.search(r"\bVIN\b", up) or re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up))
            has_odo_cue = ("ODOMETER" in up or "ODO " in up or "MILEAGE" in up or "MPH" in up or "RPM" in up)

            corner_hits = count_corner_labels(ocr)
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            looks_like_photos = var > 120 or corner_hits >= 2 or "IMAGE REPORT" in up

            if looks_like_photos or has_vin_cue or has_odo_cue:
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                score = (corner_hits * 10 + var) + (50 if has_vin_cue else 0) + (40 if has_odo_cue else 0)
                out.append((f"pdf-p{i}.png", buf.getvalue(), score))
                used += 1
                if used >= max_pages:
                    break
    except Exception as e:
        logger.warning(f"harvest_photos_from_pdf error: {e}")
    return out

def ocr_pdf_items_wide_scan(pdf_bytes: bytes, limit_pages: int = 40, dpi: int = 180) -> str:
    out = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        for i, p in enumerate(pages[:limit_pages], 1):
            txt = ocr_image_quick(p)
            if len(txt.strip()) >= 20:
                out.append(f"[WideScan Page {i}]\n{txt}")
    except Exception as e:
        logger.warning(f"ocr_pdf_items_wide_scan error: {e}")
    return "\n".join(out)

# ======================= VIN utilities =======================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
_translit = {**{str(i): i for i in range(10)},
             **dict(A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8,
                    J=1, K=2, L=3, M=4, N=5, P=7, R=9,
                    S=2, T=3, U=4, V=5, W=6, X=7, Y=8, Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def normalize_vin(s: str) -> Optional[str]:
    s = s.strip().upper().replace(" ", "").replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

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
    for c in cands:
        vin = normalize_vin(c)
        if vin:
            return vin
    return None

# ======================= Field extraction =======================
MAKES = r"(?:Acura|Alfa(?:\s*Romeo)?|Audi|BMW|Buick|Cadillac|Chevrolet|Chevy|Chrysler|Dodge|Ferrari|Fiat|Ford|GMC|Genesis|Honda|Hyundai|Infiniti|Jaguar|Jeep|Kia|Lamborghini|Land\s*Rover|Lexus|Lincoln|Maserati|Mazda|Mercedes(?:-|\s*)Benz|Mini|Mitsubishi|Nissan|Porsche|Ram|Scion|Subaru|Suzuki|Tesla|Toyota|Volkswagen|VW|Volvo)"

def extract_claim_from_text(text: str) -> Optional[str]:
    for pat in [
        r"(?:^|\s)(?:Claim\s*(?:#|No\.?|Number)[:\s]*)\s*([A-Za-z0-9\-_/]+)",
        r"(?:^|\s)Claim\s*[:#]\s*([A-Za-z0-9\-_/]+)",
        r"(?:^|\s)File\s*(?:#|No\.?|Number)[:\s]*([A-Za-z0-9\-_/]+)"
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    label_block = re.findall(r"(?:^|\n).{0,60}VIN[:\s\-]*([A-HJ-NPR-Z0-9]{10,20}).*", text, re.IGNORECASE)
    if label_block:
        vin = best_vin_candidate(label_block)
        if vin: return vin
    candidates = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, re.IGNORECASE)
    return best_vin_candidate(candidates)

def extract_vehicle_line_from_first_page(first_page_text: str) -> Optional[str]:
    if not first_page_text:
        return None
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    for ln in lines:
        if re.search(rf"\b(19\d{{2}}|20\d{{2}})\b", ln) and re.search(rf"\b{MAKES}\b", ln, re.IGNORECASE):
            cleaned = re.sub(r"\s{2,}", " ", ln).strip()
            cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
            return cleaned
    return None

def _normalize_percent_str(pct_str: str) -> str:
    s = pct_str.strip().replace(" ", "").replace("%", "")
    try:
        v = float(s); return f"{v:g}%"
    except Exception:
        return pct_str.strip().rstrip("%") + "%"

def parse_tax_rate(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"(?i)(?:sales\s*tax|tax)[^\n]{0,160}?(\d{1,3}(?:\.\d+)?\s*%)", text)
    if m: return _normalize_percent_str(m.group(1))
    m2 = re.search(r"(?i)(?:sales\s*tax|tax)[^\n]{0,160}?\$\s*\d+(?:\.\d{2})?", text)
    if m2: return m2.group(0).strip()
    return None

def parse_labor_rates(text: str) -> Dict[str, str]:
    if not text: return {}
    labels = {
        "Body": r"Body\s*Labor",
        "Paint": r"Paint\s*Labor",
        "Mechanical": r"Mechanical\s*Labor",
        "Structural": r"Structural\s*Labor",
    }
    out: Dict[str, str] = {}
    for key, lbl_pat in labels.items():
        pat = rf"(?i){lbl_pat}[^\n]{{0,200}}?\$\s*(\d{{2,3}}(?:\.\d+)?)\s*(?:/hr|/hour|per\s*hour|hr)"
        m = re.search(pat, text)
        if m: out[key] = f"${m.group(1)}/hr"
    return out

# ======================= Photo presence (vision-verified, no assumptions) =======================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def _looks_like_door_label(text: str) -> bool:
    t = (text or "").upper()
    hits = 0
    for kw in ("MFD BY", "GENERAL MOTORS", "FORD MOTOR", "TOYOTA MOTOR", "GVWR", "GAWR",
               "THIS VEHICLE CONFORMS", "DATE", "TIRE", "RIM", "VIN"):
        if kw in t: hits += 1
    return hits >= 2

def detect_required_photo_presence(image_blobs: List[Tuple[str, bytes]]) -> Dict[str, bool]:
    """
    Two-pass OCR + rotation to avoid false 'missing' on VIN/ODO.
    """
    flags = {"four corners": False, "odometer": False, "vin": False, "license plate": False}
    ext_like = 0
    corner_hits = 0
    limit = min(len(image_blobs), 64)
    for name, blob in image_blobs[:limit]:
        try:
            base = Image.open(io.BytesIO(blob))
            for r in (0, 90, 180, 270):
                img = base.rotate(r, expand=True)
                proc = preprocess_image(img)
                text = pytesseract.image_to_string(proc, lang="eng", config="--psm 6")
                up = (text or "").upper()

                def mark_odo(u: str) -> bool:
                    return (re.search(r"\bODOMETER\b", u) or "ODO " in u or "MILEAGE" in u or
                            "MPH" in u or "RPM" in u or re.search(r"\b\d{4,7}\b", u))

                if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up) or "VIN" in up or _looks_like_door_label(text):
                    flags["vin"] = True
                if mark_odo(up):
                    flags["odometer"] = True

                if re.search(r"\b[A-Z0-9]{5,8}\b", up) or "CALIFORNIA" in up or "ARIZONA" in up or "NEVADA" in up:
                    flags["license plate"] = True

                if _image_is_exterior_wide(img):
                    ext_like += 1
                corner_hits += count_corner_labels(text)

                if flags["vin"] and flags["odometer"] and flags["license plate"]:
                    break  # early exit for this image
        except Exception as e:
            logger.warning(f"presence image error: {e}")
    if ext_like >= 2 or corner_hits >= 3:
        flags["four corners"] = True
    return flags

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    def ocr_variants(pil_img: Image.Image) -> List[str]:
        texts = []
        texts.append(pytesseract.image_to_string(preprocess_image(pil_img), lang="eng", config="--psm 7"))
        texts.append(pytesseract.image_to_string(preprocess_image(pil_img), lang="eng",
                                                 config="--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
        for thr in (180, 200, 220):
            g = pil_img.convert("L").point(lambda x: 255 if x > thr else 0, mode="1").convert("L")
            texts.append(pytesseract.image_to_string(g, lang="eng",
                         config="--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
        return texts

    def crop_near_vin_label(pil_img: Image.Image) -> List[Image.Image]:
        outs = []
        try:
            data = pytesseract.image_to_data(preprocess_image(pil_img), lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT)
            n = len(data.get("text", []))
            for i in range(n):
                if (data["text"][i] or "").strip().upper() == "VIN":
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    x1 = max(0, x + int(w * 0.9))
                    y1 = max(0, y - h)
                    x2 = min(pil_img.width, x1 + w * 15)
                    y2 = min(pil_img.height, y + int(h * 2.5))
                    outs.append(pil_img.crop((x1, y1, x2, y2)))
        except Exception:
            pass
        return outs

    found: List[str] = []
    limit = min(len(image_blobs), 64)
    for name, blob in image_blobs[:limit]:
        try:
            base = Image.open(io.BytesIO(blob)).convert("RGB")
            for r in (0, 90, 180, 270):
                img = base.rotate(r, expand=True)
                for txt in ocr_variants(img):
                    cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", txt.upper())
                    if cands: found.extend(cands)
                for crop in crop_near_vin_label(img):
                    for txt in ocr_variants(crop):
                        cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", txt.upper())
                        if cands: found.extend(cands)
        except Exception as e:
            logger.warning(f"VIN photo OCR error ({name}): {e}")

    return best_vin_candidate(found)

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    limit = min(len(image_blobs), 24)
    for name, blob in image_blobs[:limit]:
        try:
            img = Image.open(io.BytesIO(blob))
            for r in (0, 90, 180, 270):
                rot = img.rotate(r, expand=True)
                ocr = pytesseract.image_to_string(preprocess_image(rot), lang="eng", config="--psm 6")
                m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{4,7})\b\s*(?:mi|miles|km)?\b", ocr, re.IGNORECASE)
                if m: return m.group(1)
        except Exception as e:
            logger.warning(f"Odometer OCR ({name}): {e}")
    return None

# ======================= Labor/tax score =======================
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

# ======================= Estimate items (robust parser + LLM fallback) =======================
PANELS = [
    "front bumper cover","rear bumper cover","bumper cover","bumper",
    "fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
    "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
    "radiator","support","wheel","tire","pillar","garnish","molding","fog lamp",
    "reinforcement","valance","bracket","impact bar","condenser","condensor","core support"
]
OPS = [
    "replace","repair","refinish","align","blend","calibrate",
    "r&i","r & i","remove & install","remove and install","r&r","r & r","remove & replace","remove and replace"
]
OP_ALIASES = {
    "repl":"replace","rep":"repair","rpr":"repair","r&i":"r&i","r & i":"r&i",
    "r&r":"replace","r & r":"replace","remove & replace":"replace","remove and replace":"replace",
    "remove & install":"r&i","remove and install":"r&i","blend":"blend","refinish":"refinish",
    "align":"align","calibrate":"calibrate","replace":"replace","repair":"repair"
}
SIDE_TOKENS = {
    "lh":"left","rh":"right","lf":"left front","rf":"right front","lr":"left rear","rr":"right rear",
    "left":"left","right":"right","front":"front","rear":"rear"
}
SIDE_REGEX = r"\b(?:LH|RH|LF|RF|LR|RR|LEFT|RIGHT|FRONT|REAR)\b"

def _norm_side_from_text(t: str) -> str:
    m = re.search(SIDE_REGEX, t, flags=re.I)
    if not m: return "unspecified"
    return SIDE_TOKENS.get(m.group(0).lower(), "unspecified")

def _norm_op(token: str) -> Optional[str]:
    token = token.strip().lower()
    return OP_ALIASES.get(token, token if token in OPS else None)

def _find_part(segment: str) -> Optional[str]:
    seg = segment.lower()
    for p in sorted(PANELS, key=len, reverse=True):
        if p in seg:
            return p
    m = re.search(r"\bbumper(?:\s*cover)?\b|\bfender\b|\bdoor\b|\bhood\b|\bgrille\b|\b(head|tail)lamp\b|\bquarter\s*panel\b", seg)
    if m: return m.group(0)
    return None

BULLET_PAT = re.compile(
    r"^[\-\*\u2022]\s*(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace)\b[^\n]{0,120}$",
    re.IGNORECASE
)
REV_PAT = re.compile(
    r"(?:^|\s)(front|rear|left|right|lh|rh|lf|rf|lr|rr)?[^\n]{0,60}?"
    r"(bumper(?:\s*cover)?|fender|door|hood|grille|headlamp|headlight|taillamp|tail\s*lamp|quarter\s*panel|rocker|mirror|decklid|trunk|valance|bracket|reinforcement|core\s*support)"
    r"[^\n]{0,60}?(?:—|-|–|,)?\s*"
    r"(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace)\b",
    re.IGNORECASE
)

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 5: 
            continue
        l = line.lower()

        # Column style (line number + op + rest)
        m_col = re.search(r"^\s*(?:\d{1,4}[A-Z]?\s+)?([A-Za-z& ]{2,14})\s+(.+)$", line)
        if m_col:
            op_raw = m_col.group(1).strip()
            tail   = m_col.group(2).strip()
            op = _norm_op(op_raw)
            if op:
                part = _find_part(tail)
                if part:
                    side = _norm_side_from_text(tail)
                    items.append({"op": op, "part": part, "side": side, "raw": line})
                    continue

        # Natural phrase
        m_phrase = re.search(r"^(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace)\s+(.+)$", l, flags=re.I)
        if m_phrase:
            op = _norm_op(m_phrase.group(1))
            tail = m_phrase.group(2)
            if op:
                part = _find_part(tail)
                if part:
                    side = _norm_side_from_text(tail)
                    items.append({"op": op, "part": part, "side": side, "raw": line})
                    continue

        # "Part — Replace"
        m_rev = re.search(rf"(.*?)(?:[—\-–]|  +)\s*(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace)\b", l, flags=re.I)
        if m_rev:
            head = m_rev.group(1)
            op   = _norm_op(m_rev.group(2))
            if op:
                part = _find_part(head)
                if part:
                    side = _norm_side_from_text(head)
                    items.append({"op": op, "part": part, "side": side, "raw": line})
                    continue

        # Bulleted line (op only) — skip unless part detectable (rare)
        if BULLET_PAT.search(line):
            continue

        # Reversed order helper
        m2 = REV_PAT.search(line)
        if m2:
            side_txt = m2.group(1) or ""
            part_txt = m2.group(2) or ""
            op = _norm_op(m2.group(3) or "")
            if op:
                part = _find_part(part_txt)
                side = _norm_side_from_text(side_txt)
                if part:
                    items.append({"op": op, "part": part, "side": side, "raw": line})
                    continue

    # Deduplicate
    uniq, seen = [], set()
    for it in items:
        key = (it["op"], it["part"], it["side"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

def extract_estimate_items_llm(text: str) -> List[Dict[str, str]]:
    schema = {"type":"array","items":{"type":"object","properties":{
        "op":{"type":"string"},"part":{"type":"string"},"side":{"type":"string"},"raw":{"type":"string"}},
        "required":["op","part","side","raw"]}}
    sys = "Extract concise estimate line items (operation, part, and side) from the text. Return STRICT JSON only per this schema: " + json.dumps(schema)
    try:
        rsp = client_fast.chat.completions.create(
            model=os.getenv("OAI_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":sys},{"role":"user","content":text[:18000]}],
            temperature=0, max_tokens=550,
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned = []
            for it in data[:160]:
                cleaned.append({
                    "op": (it.get("op") or "").lower(),
                    "part": (it.get("part") or "").lower(),
                    "side": (it.get("side") or "unspecified").lower(),
                    "raw": it.get("raw") or f"{it.get('op','')} {it.get('part','')}".strip()
                })
            return [d for d in cleaned if d["op"] and d["part"]]
    except Exception as e:
        logger.error(f"LLM item extraction failed: {e}")
    return []

def _chunk_text(txt: str, size: int = 6000, overlap: int = 400) -> List[str]:
    if not txt: return []
    txt = txt.replace("\r", "")
    out: List[str] = []
    i, n = 0, len(txt)
    while i < n:
        j = min(n, i + size)
        out.append(txt[i:j])
        if j >= n: break
        i = i + size - overlap
    return out

def llm_extract_items_chunked(full_text: str, time_guard: Callable[[], bool]) -> List[Dict[str, str]]:
    chunks = _chunk_text(full_text, size=6000, overlap=400)[:8]
    merged: List[Dict[str, str]] = []
    seen = set()
    for ch in chunks:
        if time_guard():
            break
        items = extract_estimate_items_llm(ch)
        for it in items:
            key = (it["op"], it["part"], it["side"])
            if key not in seen:
                merged.append(it); seen.add(key)
    return merged

LAST_CHANCE_OP = re.compile(
    r"\b(REPL|R&R|R&I|REPAIR|REFINISH|BLEND|ALIGN|CALIBRATE)\b[^\n]{0,100}?\b("
    r"BUMPER(?:\s*COVER)?|FENDER|DOOR|HOOD|GRILLE|HEADLAMP|HEADLIGHT|TAILLAMP|TAIL\s*LAMP|"
    r"QUARTER\s*PANEL|ROCKER|MIRROR|DECKLID|TRUNK|VALANCE|BRACKET|REINFORCEMENT|CORE\s*SUPPORT"
    r")\b[^\n]{0,60}?(LH|RH|LF|RF|LR|RR|LEFT|RIGHT|FRONT|REAR)?",
    re.IGNORECASE
)

def extract_estimate_items_last_chance(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for m in LAST_CHANCE_OP.finditer(text or ""):
        op_raw, part_raw, side_raw = m.group(1) or "", m.group(2) or "", (m.group(3) or "unspecified")
        op = _norm_op(op_raw)
        part = _find_part(part_raw)
        side = _norm_side_from_text(side_raw)
        if op and part:
            items.append({"op": op, "part": part, "side": side, "raw": m.group(0).strip()})
    uniq, seen = [], set()
    for it in items:
        key = (it["op"], it["part"], it["side"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

# ===== Target the most likely estimate pages if initial parse fails (still fast) =====
def find_estimate_like_pages(full_text: str) -> List[int]:
    pages = []
    for m in re.finditer(r"\[Page\s+(\d+)\]", full_text):
        pg = int(m.group(1))
        win = full_text[m.end():m.end()+4000].upper()
        if any(k in win for k in ("ESTIMATE", "LINE", "OPERATION", "DESCRIPTION", "PART", "LABOR")):
            pages.append(pg)
    return list(dict.fromkeys(pages))[:10]

def ocr_specific_pages(pdf_bytes: bytes, pages: List[int], dpi: int = 180) -> str:
    if not pages: return ""
    out = []
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        for idx in pages:
            if 1 <= idx <= len(imgs):
                txt = ocr_image_quick(imgs[idx-1])
                if txt.strip():
                    out.append(f"[OCR Page {idx}]\n{txt}")
    except Exception as e:
        logger.warning(f"OCR specific pages error: {e}")
    return "\n".join(out)

# ======================= Vision compare =======================
def select_images_for_vision(image_blobs: List[Tuple[str, bytes]], max_imgs: int) -> List[Tuple[str, bytes]]:
    scored = []
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = preprocess_image(img)
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            text = pytesseract.image_to_string(proc, lang="eng")
            up = (text or "").upper()
            vin_hit = bool(re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up) or " VIN" in up)
            odo_hit = ("ODOMETER" in up or "ODO " in up or "MILEAGE" in up or "MPH" in up or "RPM" in up)
            score = var + (8 * count_corner_labels(text)) + (50 if vin_hit else 0) + (40 if odo_hit else 0)
            scored.append((score, name, blob))
        except Exception:
            continue
    scored.sort(reverse=True)
    return [(n, b) for _, n, b in scored[:max_imgs]]

def compare_estimate_with_photos(items: List[Dict[str, str]],
                                 images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = {"type":"object","properties":{
        "per_item":{"type":"array","items":{"type":"object","properties":{
            "op":{"type":"string"},"part":{"type":"string"},"side":{"type":"string"},
            "photo_evidence":{"type":"boolean"},"confidence":{"type":"number"},"note":{"type":"string"}},
            "required":["op","part","side","photo_evidence","confidence","note"]}},
        "not_in_photos":{"type":"array","items":{"type":"string"}},
        "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
        "overall":{"type":"string"}}, "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]}
    system = ("You are an auto-damage visual auditor. Given estimate line items and vehicle photos, "
              "decide for EACH item whether visible photo evidence exists. Hidden ops may not be visible. "
              "Return STRICT JSON ONLY per this schema:\n" + json.dumps(schema))
    user_parts: List[Dict[str, Any]] = [{"type":"text","text":"Estimate items:\n"+json.dumps(items, ensure_ascii=False)}]
    user_parts.extend(images_for_vision)
    try:
        rsp = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user_parts}],
            max_tokens=700, temperature=0
        )
        txt = (rsp.choices[0].message.content or "").strip()
        txt = txt.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(txt)
        if not isinstance(data, dict) or "per_item" not in data:
            raise ValueError("JSON shape mismatch")
        return data
    except Exception as e:
        logger.error(f"Vision compare JSON error: {type(e).__name__}: {e}")
        return {"per_item":[],"not_in_photos":[],"extra_damage_in_photos":[],"overall":f"Comparison unavailable ({type(e).__name__})."}

def compare_batched(items: List[Dict[str, str]],
                    images_for_vision: List[Dict[str, Any]],
                    batch_size: int) -> Dict[str, Any]:
    all_per, not_in, extra = [], [], []
    overalls = []
    if not items:
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "No estimate items provided."}
    total = len(items)
    batches = math.ceil(total / max(1, batch_size))
    for i in range(batches):
        chunk = items[i*batch_size:(i+1)*batch_size]
        if not chunk:
            continue
        res = compare_estimate_with_photos(chunk, images_for_vision)
        all_per.extend(res.get("per_item", []))
        not_in.extend(res.get("not_in_photos", []))
        extra.extend(res.get("extra_damage_in_photos", []))
        if res.get("overall"): overalls.append(res["overall"])
    overall = "; ".join(overalls[:3]) if overalls else "Batched comparison completed."
    uniq_not_in = list(dict.fromkeys(not_in))
    uniq_extra  = list(dict.fromkeys(extra))
    return {"per_item": all_per, "not_in_photos": uniq_not_in, "extra_damage_in_photos": uniq_extra, "overall": overall}

# ======================= PDF helpers =======================
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12); pdf.cell(0, 8, txt=title, ln=True); pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10); pdf.multi_cell(0, 6, f"{key}: {val}")

def build_brief_consistency_summary(cons: Dict[str, Any], items: List[Dict[str, str]]) -> str:
    n_items = len(items)
    n_supported = sum(1 for it in (cons.get("per_item") or []) if it.get("photo_evidence"))
    n_missing = sum(1 for it in (cons.get("per_item") or []) if not it.get("photo_evidence"))
    extra = cons.get("extra_damage_in_photos") or []
    not_seen = cons.get("not_in_photos") or []
    parts = []
    if n_items:
        parts.append(f"{n_supported}/{n_items} estimate items show visible support in photos; {n_missing} lack visible evidence.")
    if not_seen[:3]:
        parts.append("Not evident: " + "; ".join(not_seen[:3]) + ("" if len(not_seen) <= 3 else " …"))
    if extra[:3]:
        parts.append("Extra damage seen in photos: " + "; ".join(extra[:3]) + ("" if len(extra) <= 3 else " …"))
    if not parts:
        return "Per-item comparison unavailable."
    return " ".join(parts)

# ======================= Routes =======================
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(request: Request):
    """
    Comprehensive: FULL estimate text → item extraction → batched vision review.
    VIN is sourced ONLY from estimate (first page anchor), then verified by photo.
    Sub-minute run with strict time budget and fast paths.
    """
    t0 = t0_start()
    ctype = request.headers.get("content-type", "").lower()
    files_all: List[Tuple[str, bytes]] = []
    client_rules = ""
    file_number = ""
    ia_company = ""
    appraiser_id = ""

    # ---------- Accept multipart or JSON ----------
    try:
        if "multipart/form-data" in ctype:
            form = await request.form()
            client_rules = (form.get("client_rules") or "").strip()
            file_number  = (form.get("file_number")  or "").strip()
            ia_company   = (form.get("ia_company")   or "").strip()
            appraiser_id = (form.get("appraiser_id") or "").strip()
            for key in ("files", "files[]", "estimate", "photos", "guidelines"):
                for f in form.getlist(key):
                    if hasattr(f, "filename"):
                        raw = await f.read()
                        files_all.append(((f.filename or "upload").lower(), raw))
        elif "application/json" in ctype:
            payload = await request.json()
            client_rules = (payload.get("client_rules") or "").strip()
            file_number  = (payload.get("file_number")  or "").strip()
            ia_company   = (payload.get("ia_company")   or "").strip()
            appraiser_id = (payload.get("appraiser_id") or "").strip()
            for item in (payload.get("files") or []):
                if "b64" in item:
                    try:
                        b = base64.b64decode(item["b64"])
                        fname = (item.get("filename") or "upload").lower()
                        files_all.append((fname, b))
                    except Exception as e:
                        logger.warning(f"Bad base64 file: {e}")
                elif "url" in item:
                    try:
                        import httpx
                        r = httpx.get(item["url"], timeout=10)
                        r.raise_for_status()
                        fname = (item.get("filename") or os.path.basename(item["url"]) or "download").lower()
                        files_all.append((fname, r.content))
                    except Exception as e:
                        logger.warning(f"Fetch failed: {item.get('url')}: {e}")
        else:
            return JSONResponse(status_code=415, content={"error":"Unsupported Content-Type. Use multipart/form-data or application/json."})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad request body: {type(e).__name__}: {e}"})

    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error": "Appraiser ID is required."})
    if not files_all:
        return JSONResponse(status_code=400, content={"error": "No files uploaded. Send at least one estimate/photo/guideline file."})

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        first_page_texts: List[str] = []
        quick_text_chunks: List[str] = []
        full_text_chunks: List[str] = []
        image_blobs: List[Tuple[str, bytes]] = []
        pdf_photo_candidates: List[Tuple[str, bytes, float]] = []
        pdf_raws: List[bytes] = []

        async def handle_file(name: str, raw: bytes):
            if name.endswith((".jpg",".jpeg",".png",".webp",".gif")):
                png = to_png_bytes(raw)
                image_blobs.append((name if name.endswith(".png") else name.rsplit(".",1)[0]+".png",
                                    png if png else raw))
                return
            if name.endswith(".pdf"):
                pdf_raws.append(raw)

                # First page ONLY (anchor fields)
                txt_p1 = await loop.run_in_executor(pool, pdftotext_extract, raw, 1, 1)
                if txt_p1.strip():
                    first_page_texts.append(txt_p1)
                else:
                    first_txt = await loop.run_in_executor(pool, ocr_pdf_first_page, raw)
                    if first_txt: first_page_texts.append(first_txt)

                # QUICK skim (speed)
                txt_fast = await loop.run_in_executor(pool, pdftotext_extract, raw, 1, MAX_TEXT_PAGES)
                if txt_fast.strip():
                    quick_text_chunks.append(txt_fast)
                else:
                    if not nearly_out_of_time(t0, 10):
                        more_txt = await loop.run_in_executor(pool, ocr_pdf_text_caps, raw, MAX_TEXT_PAGES)
                        if more_txt: quick_text_chunks.append(more_txt)

                # FULL doc text (comprehensive)
                if not nearly_out_of_time(t0, 12):
                    full_txt = await loop.run_in_executor(pool, pdftotext_extract_all, raw)
                    if full_txt.strip():
                        full_text_chunks.append(full_txt)
                    elif not nearly_out_of_time(t0, 8):
                        full_ocr = await loop.run_in_executor(pool, ocr_pdf_items_wide_scan, raw, 40, 180)
                        if full_ocr: full_text_chunks.append(full_ocr)

                # Tax/Labor page assist
                if not nearly_out_of_time(t0, 12):
                    tax_labor_page = await loop.run_in_executor(pool, ocr_pdf_scan_tax_labor_page, raw, 60)
                    if tax_labor_page:
                        quick_text_chunks.append(tax_labor_page)

                # PHOTOS: embedded + render fallback
                if not nearly_out_of_time(t0, 10):
                    cand_fast = await loop.run_in_executor(pool, pdfimages_harvest, raw, MAX_PHOTO_PAGES)
                    pdf_photo_candidates.extend(cand_fast or [])
                if len(pdf_photo_candidates) < 2 and not nearly_out_of_time(t0, 8):
                    cand_render = await loop.run_in_executor(pool, harvest_photos_from_pdf, raw, MAX_PHOTO_PAGES)
                    pdf_photo_candidates.extend(cand_render or [])
                return

            if name.endswith(".docx"):
                try:
                    doc = await loop.run_in_executor(pool, Document, io.BytesIO(raw))
                    quick_text_chunks.append("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
                except Exception:
                    pass
            elif name.endswith(".txt"):
                try:
                    quick_text_chunks.append(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    pass

        await asyncio.gather(*(handle_file(n, b) for n, b in files_all))

    if pdf_photo_candidates:
        pdf_photo_candidates.sort(key=lambda t: t[2], reverse=True)
        keep = pdf_photo_candidates[:MAX_PHOTO_PAGES]
        for n, data, _ in keep: image_blobs.append((n, data))

    # ====== ID fields strictly from FIRST PAGE ======
    first_page_text = "\n".join(first_page_texts)
    quick_text = "\n".join(quick_text_chunks)
    full_text  = "\n".join(full_text_chunks) or quick_text

    # Presence flags (vision-verified)
    presence = detect_required_photo_presence(image_blobs)
    missing_photos = [k for k, v in presence.items() if not v]

    id_source_text = (first_page_text or "").strip() or (quick_text or "")
    claim_number = extract_claim_from_text(id_source_text) or "N/A"
    vin_est      = extract_vin_from_text(id_source_text)
    vehicle_desc = extract_vehicle_line_from_first_page(id_source_text) or "N/A"

    # VIN verify (estimate-only source; photo verification result)
    vin_photos_text = extract_vin_from_photos(image_blobs)
    if vin_est:
        if presence.get("vin"):
            if vin_photos_text:
                vin_verify_status = "MATCH" if normalize_vin(vin_est) == normalize_vin(vin_photos_text) else "MISMATCH"
            else:
                vin_verify_status = "VIN PHOTO PRESENT — TEXT UNREADABLE"
        else:
            vin_verify_status = "VIN PHOTO NOT FOUND"
    else:
        vin_verify_status = "VIN NOT FOUND IN ESTIMATE"

    vin_final_for_report = vin_est or "N/A"
    odo_photos_value = extract_odometer_from_photos(image_blobs)

    # ===== Estimate items (full text → robust, then chunked LLM, then OCR target pages, then last-chance) =====
    est_items = extract_estimate_items(full_text)

    if not est_items and not nearly_out_of_time(t0, 12):
        est_items = llm_extract_items_chunked(full_text, time_guard=lambda: nearly_out_of_time(t0, 8))

    # targeted OCR on likely estimate pages if still empty
    if not est_items and ('pdf_raws' in locals() and pdf_raws) and not nearly_out_of_time(t0, 9):
        # find likely pages from whatever text we have
        page_hints = find_estimate_like_pages(full_text or quick_text)
        # if we have no hints, try first 6 pages as common estimate region
        if not page_hints:
            page_hints = list(range(1, 7))
        targeted_ocr = ocr_specific_pages(pdf_raws[0], page_hints[:10], dpi=190)
        if targeted_ocr:
            est_items = extract_estimate_items(targeted_ocr)
            if not est_items and not nearly_out_of_time(t0, 6):
                est_items = llm_extract_items_chunked(targeted_ocr, time_guard=lambda: nearly_out_of_time(t0, 4))

    if not est_items and not nearly_out_of_time(t0, 4):
        est_items = extract_estimate_items_last_chance(full_text or quick_text)

    # ===== Vision compare (always PNG; batched) =====
    max_imgs = 4 if nearly_out_of_time(t0, 12) else MAX_VISION_IMGS
    chosen_images = select_images_for_vision(image_blobs, max_imgs=max_imgs)
    images_for_vision = []
    for _, b in chosen_images:
        png = to_png_bytes(b) or b
        images_for_vision.append({"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png).decode("utf-8")}})

    if images_for_vision and est_items:
        batch_size = max(6, min(VISION_BATCH, 12))
        if len(est_items) > batch_size and not nearly_out_of_time(t0, 10):
            consistency = compare_batched(est_items, images_for_vision, batch_size)
        else:
            consistency = compare_estimate_with_photos(est_items, images_for_vision)
    elif images_for_vision:
        consistency = {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "No estimate items parsed."}
    else:
        consistency = {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "No photos available for comparison."}

    # ===== Labor & Tax =====
    labor_rates = parse_labor_rates(full_text or quick_text)
    tax_rate    = parse_tax_rate(full_text or quick_text)
    labor_line = "None detected"
    if labor_rates:
        parts = []
        for key in ("Body","Paint","Mechanical","Structural"):
            if key in labor_rates: parts.append(f"{key} {labor_rates[key]}")
        if parts: labor_line = "; ".join(parts)
    tax_line = tax_rate or "Not found"

    # ===== Narrative & score =====
    photo_line = "None" if not missing_photos else ", ".join(missing_photos)

    facts_text = f"""FACTS to ground your evaluation (do not contradict):
- Required photo types missing: {photo_line if photo_line != "None" else "None (all required photo types present)"}
- Number of photos analyzed: {len(image_blobs)}
- VIN verification status (estimate vs photo): {vin_verify_status}
- Labor rates found: {labor_line}
- Tax rate found: {tax_line}
POLICY:
- Do NOT assume any photo is missing; rely on the presence results above.
- Use only the uploaded estimate and photos; do NOT cite external websites.
- Be concise and deterministic; no speculation.
"""

    system_prompt = f'''
You are an AI auto damage auditor. Evaluate STRICTLY by these rules:

- Start at 100% and deduct only for: labor (-50% if ALL sections missing), tax (-25% if rules require but not present), photos (-25% per missing type), parts (-25% if a 2024–2025 vehicle uses LKQ/AM in violation).
- Required photos: four corners, odometer, VIN, license plate.
- "Four corners" is satisfied if at least two exterior corner views are present OR multiple Image Report pages/corner labels are present.
- Do NOT assume total loss unless explicitly stated.
- If any labor rate is present (body OR paint OR mechanical OR structural), do NOT apply the -50% deduction.

Rules to follow from client:
{client_rules}
'''.strip()

    user_parts: List[Dict[str, Any]] = [{"type":"text","text":facts_text}]
    if full_text:
        user_parts.append({"type":"text","text":full_text[:18000]})

    max_tokens_summary = 450 if nearly_out_of_time(t0, 10) else 650
    try:
        rsp = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_parts}],
            max_tokens=max_tokens_summary, temperature=0
        )
        gpt_output = rsp.choices[0].message.content or "⚠️ GPT returned no output."
    except Exception as e:
        gpt_output = f"⚠️ AI review failed: {type(e).__name__}: {e}"

    score_ai = None
    for pat in [r"Total\s*Evaluation\s*[:\-]?\s*(\d{1,3})\s*%?",
                r"Final\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?",
                r"Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?"]:
        m = re.search(pat, gpt_output, re.IGNORECASE)
        if m: score_ai = int(m.group(1)); break

    labor_tax_adj = check_labor_and_tax_score(full_text or quick_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    computed = max(0, 100 + labor_tax_adj + photo_adj)
    authoritative_score = max(0, min(100, score_ai if score_ai is not None else computed))

    gpt_output_clean = re.sub(
        r'(?im)^(?:Final\s*Score|Compliance\s*Score|Total\s*Evaluation)\s*[:\-]?\s*\d{1,3}\s*%.*$',
        '',
        gpt_output
    ).strip()

    gpt_output_clean += f"\n\nVIN verification (estimate vs photo): {vin_verify_status}"
    gpt_output_clean += f"\nRequired photo verification (vision): {photo_line}"
    gpt_output_clean += f"\nLabor rates detected: {labor_line}"
    gpt_output_clean += f"\nTax Rate detected: {tax_line}"

    # ======================= PDF (layout unchanged; brief summary added) =======================
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
    pdf.multi_cell(0, 6, f"VIN: {vin_final_for_report}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    if odo_photos_value:
        pdf.multi_cell(0, 6, f"Odometer (from photos): {odo_photos_value}")
    elif presence.get("odometer"):
        pdf.multi_cell(0, 6, "Odometer (photo present): unreadable")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, gpt_output_clean)

    pdf.ln(4)
    pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")
    brief = build_brief_consistency_summary(consistency, est_items)
    pdf.multi_cell(0, 6, f"Brief Summary: {brief}")

    if consistency.get("per_item"):
        for it in consistency["per_item"][:80]:
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
        for raw in consistency["not_in_photos"][:40]: pdf.multi_cell(0, 6, f"- {raw}")

    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:40]: pdf.multi_cell(0, 6, f"- {d}")

    pdf.ln(2); pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # ======================= EMAIL (structure unchanged) =======================
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
VIN: {vin_final_for_report}
Vehicle: {vehicle_desc}

Compliance Score: {authoritative_score}%

AI Review Summary:
{gpt_output_clean}
"""
        msg.set_content(email_body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing): {e}")

    return {
        "gpt_output": gpt_output_clean,
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin_final_for_report,
        "score": f"{authoritative_score}%",
        "consistency_review": consistency
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})





















