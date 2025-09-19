from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any, Callable
import os, re, io, base64, json, logging, asyncio, time, tempfile, subprocess, glob
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
MAX_TEXT_PAGES  = int(os.getenv("MAX_TEXT_PAGES",  "3"))     # quick skim (sub-minute)
MAX_PHOTO_PAGES = int(os.getenv("MAX_PHOTO_PAGES", "36"))    # better presence detection
MAX_VISION_IMGS = int(os.getenv("MAX_VISION_IMGS", "10"))    # vision comparison
THREADS         = int(os.getenv("OCR_THREADS",     "4"))
OAI_MODEL       = os.getenv("OAI_MODEL", "gpt-4o-mini")
OAI_TIMEOUT_S   = float(os.getenv("OAI_TIMEOUT_S", "15"))
TIME_BUDGET_S   = float(os.getenv("TIME_BUDGET_S", "55"))    # sub-minute target
VISION_BATCH    = int(os.getenv("VISION_BATCH", "12"))

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

# ======================= OCR & text helpers =======================
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

# ======== Image format helpers (NO blanket PNG re-encode) ========
ACCEPTED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

def sniff_image_mime(blob: bytes) -> Optional[str]:
    try:
        im = Image.open(io.BytesIO(blob))
        fmt = (im.format or "").upper()
        if fmt == "JPEG": return "image/jpeg"
        if fmt == "PNG":  return "image/png"
        if fmt == "WEBP": return "image/webp"
        if fmt == "GIF":  return "image/gif"
        return None
    except Exception:
        return None

def ensure_openai_image(blob: bytes) -> Tuple[bytes, str]:
    """
    Return (bytes, mime). Pass-through for JPEG/PNG/WEBP/GIF.
    If not a supported/valid image, re-encode to JPEG.
    """
    mime = sniff_image_mime(blob)
    if mime in ACCEPTED_MIMES:
        return blob, mime
    # try to decode and re-encode to JPEG
    try:
        im = Image.open(io.BytesIO(blob))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        # last resort, return original with a safe default (JPEG conversion)
        try:
            im = Image.new("RGB", (8, 8), (0, 0, 0))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return buf.getvalue(), "image/jpeg"
        except Exception:
            return blob, "image/jpeg"

def make_data_url(blob: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(blob).decode('utf-8')}"

# ===== Extract embedded images via pdfimages; respect original format =====
def pdfimages_harvest(pdf_bytes: bytes, max_images: int = MAX_PHOTO_PAGES) -> List[Tuple[str, bytes, float, str]]:
    """
    Returns list of (name, bytes, size_bytes, mime). Keeps original format when possible.
    For pdfimages outputs like .jpg/.ppm/.pbm, we convert PPM/PBM to JPEG only.
    """
    out: List[Tuple[str, bytes, float, str]] = []
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
                    mime = sniff_image_mime(raw)
                    if mime not in ACCEPTED_MIMES:
                        # convert only if needed
                        raw, mime = ensure_openai_image(raw)
                    name_ext = {
                        "image/jpeg": f"pdfimg-{i}.jpg",
                        "image/png":  f"pdfimg-{i}.png",
                        "image/webp": f"pdfimg-{i}.webp",
                        "image/gif":  f"pdfimg-{i}.gif",
                    }[mime]
                    out.append((name_ext, raw, float(os.path.getsize(fp)), mime))
                except Exception:
                    continue
    except Exception as e:
        logger.info(f"pdfimages not available or failed: {e}")
    return out

# ===== Photo-like page harvest (render fallback) — not forced to PNG anymore =====
def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int) -> List[Tuple[str, bytes, float, str]]:
    out: List[Tuple[str, bytes, float, str]] = []
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
                # Save rendered page as JPEG (new image; no original to preserve)
                page.convert("RGB").save(buf, format="JPEG", quality=85)
                score = (corner_hits * 10 + var) + (50 if has_vin_cue else 0) + (40 if has_odo_cue else 0)
                out.append((f"pdf-p{i}.jpg", buf.getvalue(), score, "image/jpeg"))
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
    for raw in (ln.strip() for ln in first_page_text.splitlines() if ln.strip()):
        m_year = re.search(r"\b(19|20)\d{2}\b", raw)
        if not m_year:
            continue
        year_idx = m_year.start()
        tail = raw[year_idx:].strip()
        if not re.search(rf"\b{MAKES}\b", tail, re.IGNORECASE):
            continue
        tail = re.sub(r"\s{2,}", " ", tail)
        tail = re.sub(r"https?://\S+", "", tail).strip()
        return tail
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

def parse_mileage_from_text(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"(?i)(?:odometer|mileage|mi\.)\s*[:\-]?\s*(\d{1,3}(?:,\d{3})+|\d{4,7})", text)
    if m: return m.group(1)
    m2 = re.search(r"(?i)odometer\s*(?:in|out)?\s*[:\-]?\s*(\d{1,3}(?:,\d{3})+|\d{4,7})", text)
    if m2: return m2.group(1)
    return None

# ======================= Photo presence (VIN & ODO presence only) =======================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def detect_required_photo_presence(image_blobs: List[Tuple[str, bytes]]) -> Dict[str, bool]:
    flags = {"four corners": False, "odometer": False, "vin": False, "license plate": False}
    ext_like = 0
    corner_hits = 0

    VIN_DOOR_LABEL_CUES = ("MFD BY", "GENERAL MOTORS", "THIS VEHICLE CONFORMS", "GVWR", "GAWR", "VIN")
    ODO_PAT = re.compile(r"\b\d{3,7}\s*(?:mi|miles)\b", re.IGNORECASE)

    limit = min(len(image_blobs), 64)
    for name, blob in image_blobs[:limit]:
        try:
            base = Image.open(io.BytesIO(blob))
        except Exception:
            continue
        for r in (0, 90, 180, 270):
            try:
                img = base.rotate(r, expand=True)
                proc = preprocess_image(img)
                text = pytesseract.image_to_string(proc, lang="eng", config="--psm 6")
                up = (text or "").upper()

                if (re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up) or any(cue in up for cue in VIN_DOOR_LABEL_CUES)):
                    flags["vin"] = True
                if (ODO_PAT.search(text or "") or "ODOMETER" in up or "ODO " in up or "MILEAGE" in up or
                    "PRND" in up or "RPM" in up or "MPH" in up or "KM/H" in up):
                    flags["odometer"] = True
                if ("LICENSE" in up or "PLATE" in up or re.search(r"\b[A-Z0-9]{5,8}\b", up)):
                    flags["license plate"] = True

                if _image_is_exterior_wide(img):
                    ext_like += 1
                corner_hits += count_corner_labels(text)
            except Exception:
                continue

    if ext_like >= 2 or corner_hits >= 3:
        flags["four corners"] = True
    return flags

# ======================= Client guideline ingestion =======================
GUIDE_HINTS = ("guide", "guideline", "rules", "policy", "client", "requirements", "instruction")

def looks_like_guideline_name(filename: str) -> bool:
    fn = (filename or "").lower()
    return any(h in fn for h in GUIDE_HINTS)

def extract_text_from_pdf_bytes_all(pdf_bytes: bytes) -> str:
    txt = pdftotext_extract_all(pdf_bytes)
    if txt and txt.strip():
        return txt
    return ocr_pdf_items_wide_scan(pdf_bytes, limit_pages=50, dpi=170)

def append_client_rules_from_blob(name: str, raw: bytes, rules_parts: List[str]):
    try:
        if name.endswith(".pdf"):
            t = extract_text_from_pdf_bytes_all(raw)
            if t.strip(): rules_parts.append(t)
        elif name.endswith(".docx"):
            d = Document(io.BytesIO(raw))
            t = "\n".join(p.text for p in d.paragraphs if p.text.strip())
            if t.strip(): rules_parts.append(t)
        elif name.endswith(".txt"):
            t = raw.decode("utf-8", errors="ignore")
            if t.strip(): rules_parts.append(t)
    except Exception as e:
        logger.warning(f"guideline extract error ({name}): {e}")

# ======================= Estimate items (robust parser + LLM fallback) =======================
PANELS = [
    "front bumper cover","rear bumper cover","bumper cover","bumper",
    "fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
    "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
    "radiator","support","wheel","tire","pillar","garnish","molding","fog lamp",
    "reinforcement","valance","bracket","impact bar","condenser","condensor","core support",
    "fuel tank","battery","fuel system","radiator support","radiator"
]
OPS = [
    "replace","repair","refinish","align","blend","calibrate",
    "r&i","r & i","remove & install","remove and install",
    "r&r","r & r","remove & replace","remove and replace",
    "disconnect & reconnect","disconnect and reconnect","repl","d&r"
]
OP_ALIASES = {
    "repl":"replace","rep":"repair","rpr":"repair","r&i":"r&i","r & i":"r&i",
    "r&r":"replace","r & r":"replace","remove & replace":"replace","remove and replace":"replace",
    "remove & install":"r&i","remove and install":"r&i","blend":"blend","refinish":"refinish",
    "align":"align","calibrate":"calibrate","replace":"replace","repair":"repair",
    "disconnect & reconnect":"r&i","disconnect and reconnect":"r&i","d&r":"r&i"
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
    m = re.search(r"\bbumper(?:\s*cover)?\b|\bfender\b|\bdoor\b|\bhood\b|\bgrille\b|\b(head|tail)lamp\b|\bquarter\s*panel\b|\bbattery\b|\bfuel\s*tank\b|\bradiator\s*support\b|\bradiator\b", seg)
    if m: return m.group(0)
    return None

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 5:
            continue
        l = line.lower()

        m_col = re.search(r"^\s*(?:\d{1,4}[A-Z]?\s+)?([A-Za-z& ]{2,20})\s+(.+)$", line)
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

        m_phrase = re.search(r"^(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace|disconnect\s*&\s*reconnect|disconnect\s*and\s*reconnect|repl|r&i|r&r|d&r)\s+(.+)$", l, flags=re.I)
        if m_phrase:
            op = _norm_op(m_phrase.group(1))
            tail = m_phrase.group(2)
            if op:
                part = _find_part(tail)
                if part:
                    side = _norm_side_from_text(tail)
                    items.append({"op": op, "part": part, "side": side, "raw": line})
                    continue

        m_rev = re.search(rf"(.*?)(?:[—\-–]|  +)\s*(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace|disconnect\s*&\s*reconnect|disconnect\s*and\s*reconnect|repl|r&i|r&r|d&r)\b", l, flags=re.I)
        if m_rev:
            head = m_rev.group(1)
            op   = _norm_op(m_rev.group(2))
            if op:
                part = _find_part(head)
                if part:
                    side = _norm_side_from_text(head)
                    items.append({"op": op, "part": part, "side": side, "raw": line})
                    continue

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

# ======================= Vision compare (uses proper per-image MIME) =======================
def build_vision_payload(images: List[Tuple[str, bytes, str]]) -> List[Dict[str, Any]]:
    """
    images: list of (name, bytes, mime)
    """
    parts: List[Dict[str, Any]] = []
    for name, blob, mime in images[:MAX_VISION_IMGS]:
        safe_bytes, safe_mime = ensure_openai_image(blob)
        parts.append({"type": "image_url", "image_url": {"url": make_data_url(safe_bytes, safe_mime)}})
    return parts

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
        "Return STRICT JSON ONLY per this schema:\n" + json.dumps(schema)
    )

    user_parts: List[Dict[str, Any]] = [
        {"type": "text", "text": "Estimate items:\n" + json.dumps(items, ensure_ascii=False)}
    ]
    user_parts.extend(images_for_vision)

    try:
        rsp = client.chat.completions.create(
            model=OAI_MODEL,
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

# ======================= PDF helpers =======================
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12)
    pdf.cell(0, 8, txt=title, ln=True)
    pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"{key}: {val}")

# ======================= Routes =======================
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

    t0 = t0_start()
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=THREADS)

    # ----- Ingest everything (no filename allowlist; accept any form keys) -----
    texts: List[str] = []
    photos_for_presence: List[Tuple[str, bytes]] = []
    images_for_openai: List[Tuple[str, bytes, str]] = []  # (name, bytes, mime)
    rules_parts: List[str] = []

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()

        if name.endswith((".pdf",)):
            texts.append(ocr_pdf_first_page(raw))
            texts.append(ocr_pdf_text_caps(raw, MAX_TEXT_PAGES))
            tax_or_labor = ocr_pdf_scan_tax_labor_page(raw, 60)
            if tax_or_labor:
                texts.append(tax_or_labor)

            # harvest embedded images (respect original format)
            emb = pdfimages_harvest(raw, MAX_PHOTO_PAGES)
            for hname, hbytes, _sz, hmime in emb:
                photos_for_presence.append((hname, hbytes))
                images_for_openai.append((hname, hbytes, hmime))

            # photo-like rendered pages fallback
            rendered = harvest_photos_from_pdf(raw, max_pages=12)
            for rname, rbytes, _score, rmime in rendered:
                photos_for_presence.append((rname, rbytes))
                images_for_openai.append((rname, rbytes, rmime))

            # Also treat PDFs named like guidelines as rules
            if looks_like_guideline_name(name):
                rules_parts.append(extract_text_from_pdf_bytes_all(raw))

        elif name.endswith((".docx", ".txt")):
            if name.endswith(".docx"):
                try:
                    d = Document(io.BytesIO(raw))
                    texts.append("\n".join(p.text for p in d.paragraphs if p.text.strip()))
                except Exception as e:
                    logger.warning(f"DOCX error: {e}")
            else:
                texts.append(raw.decode("utf-8", errors="ignore"))

            if looks_like_guideline_name(name):
                append_client_rules_from_blob(name, raw, rules_parts)

        elif name.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            # pass-through; no force-PNG. Prepare mime later.
            mime = sniff_image_mime(raw) or "image/jpeg"
            photos_for_presence.append((name, raw))
            images_for_openai.append((name, raw, mime))

        else:
            # Unknown file types: attempt to treat as text
            try:
                texts.append(raw.decode("utf-8", errors="ignore"))
            except Exception:
                logger.info(f"Skipped unsupported file: {name}")

    combined_text = "\n".join(t for t in texts if t)

    # Merge client_rules form text with any attached rules docs
    effective_rules = (client_rules or "").strip()
    if rules_parts:
        effective_rules = (effective_rules + "\n\n" + "\n\n".join(rules_parts)).strip()

    # ----- Required photo presence (vision verified presence only)
    presence_flags = await loop.run_in_executor(pool, detect_required_photo_presence, photos_for_presence)
    missing_photos = [k for k, v in presence_flags.items() if not v]

    # ----- VIN + vehicle fields (VIN from estimate text only; verify via photo presence)
    vin_est = extract_vin_from_text(combined_text) or "N/A"
    vin_photo_present = presence_flags.get("vin", False)
    vin_verify_note = (
        "MATCH (photo present & readable)" if (vin_photo_present and vin_est != "N/A")
        else ("VIN PHOTO PRESENT—TEXT UNREADABLE" if vin_photo_present else "VIN PHOTO NOT FOUND")
    )

    claim_number = extract_claim_from_text(combined_text) or "N/A"
    first_page_text = combined_text.split("[Page 1]")[-1] if "[Page 1]" in combined_text else combined_text
    vehicle_line = extract_vehicle_line_from_first_page(first_page_text) or "N/A"
    labor_rates_page = parse_labor_rates(combined_text)
    tax_rate = parse_tax_rate(combined_text) or "Not detected"
    mileage_est = parse_mileage_from_text(combined_text) or "Not listed"

    # ----- Estimate items; fallback to LLM if regex misses and time allows
    est_items = extract_estimate_items(combined_text)
    if not est_items and not nearly_out_of_time(t0):
        est_items = llm_extract_items_chunked(combined_text, lambda: nearly_out_of_time(t0))

    # ----- Vision payload (per-image MIME, no PNG forcing)
    vision_payload = build_vision_payload(images_for_openai)

    # ----- Compare estimate ↔ photos (skip if out of time)
    if nearly_out_of_time(t0):
        consistency = {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Skipped due to time budget."}
    else:
        consistency = compare_estimate_with_photos(est_items, vision_payload)

    # ----- Scoring (kept same logic)
    photo_adj = -25 * len(missing_photos)
    labor_penalty = 0 if labor_rates_page else -50
    tax_penalty = 0
    if re.search(r"tax\s*(required|must|utilize|apply)", effective_rules, re.IGNORECASE):
        if "Not detected" in tax_rate:
            tax_penalty = -25
    authoritative_score = max(0, min(100, 100 + photo_adj + labor_penalty + tax_penalty))

    # ======================= PDF build (layout unchanged) =======================
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
    pdf.multi_cell(0, 6, f"VIN (from estimate): {vin_est}")
    pdf.multi_cell(0, 6, f"VIN verification (estimate vs photo): {vin_verify_note}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_line}")
    pdf.multi_cell(0, 6, f"Odometer (estimate): {mileage_est}")
    if labor_rates_page:
        pdf.multi_cell(0, 6, "Labor rates detected: " + ", ".join(f"{k} {v}" for k,v in labor_rates_page.items()))
    else:
        pdf.multi_cell(0, 6, "Labor rates detected: NONE")
    pdf.multi_cell(0, 6, f"Tax Rate detected: {tax_rate}")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    # Photos presence
    pdf.ln(4)
    pdf_add_section_title(pdf, "Required Photos Presence (vision-verified)")
    if missing_photos:
        pdf.multi_cell(0, 6, "Missing: " + ", ".join(missing_photos))
    else:
        pdf.multi_cell(0, 6, "All required photos present.")

    # ======== Estimate ↔ Photos Consistency Review ========
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

    # Save PDF
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
VIN (from estimate): {vin_est}
VIN verification (estimate vs photo): {vin_verify_note}
Vehicle: {vehicle_line}

Compliance Score: {authoritative_score}%
"""
        msg.set_content(email_body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing): {e}")

    return {
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_line,
        "vin_estimate": vin_est,
        "vin_verification": vin_verify_note,
        "score": f"{authoritative_score}%",
        "missing_photos": missing_photos,
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
























