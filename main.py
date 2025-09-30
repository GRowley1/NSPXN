
def scan_estimate_for_vin(pdf_bytes: bytes) -> Optional[str]:
    """Lightweight but robust VIN finder for the estimate PDF only.
    Strategy: read up to 12 text pages; if none found, OCR up to 8 pages @240 DPI.
    Prefer check-digit-valid VIN; else return the first normalized 17-char VIN seen."""
    
    text_12 = ""
    try:
        text_12 = fast_pdf_text(pdf_bytes, limit_pages=12)
    except Exception:
        text_12 = ""
    # First pass: strict VIN from text
    v = vin_from_text(text_12)
    if not v:
        # Try OCR (more pages, but capped)
        try:
            pages = convert_from_bytes(pdf_bytes, dpi=240)[:8]
        except Exception:
            pages = []
        ocr_all = []
        for im in pages:
            try:
                im = _pp(im)
                ocr_all.append(pytesseract.image_to_string(im, lang="eng", config="--psm 6"))
            except Exception:
                pass
        ocr_text = "\n".join(ocr_all)
        v = vin_from_text(text_12 + "\n" + ocr_text)
        if not v:
            # final relaxed: accept first 17-char VIN-like even if check-digit fails
            cands = re.findall(r"\b([A-HJ-NPR-Z0-9\-\s]{17,40})\b", text_12 + "\n" + ocr_text, flags=re.I)
            for c in cands:
                cc = re.sub(r"[^A-HJ-NPR-Z0-9]", "", c.upper()).replace("O","0").replace("I","1").replace("Q","0")
                if len(cc) == 17 and re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", cc):
                    return cc
            return None
    return v


def ocr_pages_for_vin(pdf_bytes: bytes, max_pages: int = 4, dpi: int = 250) -> str:
    """Lightweight multi-page OCR used only if VIN not found in text.
    Capped pages + dpi to keep latency low.
    """
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception:
        return ""
    out = []
    for im in pages:
        try:
            im = _pp(im)
            out.append(pytesseract.image_to_string(im, lang="eng", config="--psm 6"))
        except Exception:
            pass
    return "\n".join(out)
# --- Compatibility shim for legacy callsite ---
def extract_vehicle_line_from_first_page(text: str):
    """Best-effort vehicle line from the first-page text; returns None if not found.
    Prefer minimal logic; delegate to vehicle_from_text.
    """
    try:
        return vehicle_from_text(text)
    except Exception:
        return None

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
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, ImageFile, Image
from openai import OpenAI



# --- Compatibility shims (legacy call sites) ---
def extract_vin_from_text(text: str):
    try:
        return vin_from_text(text)
    except Exception:
        return None

def extract_vehicle_from_text(text: str):
    try:
        return vehicle_from_text(text)
    except Exception:
        return None

def extract_mileage_from_text(text: str):
    try:
        return mileage_from_text(text)
    except Exception:
        return None

# === NSPXN speed/robustness helpers (no behavior change) ===
from contextvars import ContextVar

_REQUEST_DEADLINE = ContextVar("NSPXN_REQUEST_DEADLINE", default=None)

def set_request_deadline(seconds: float):
    import time
    _REQUEST_DEADLINE.set(time.time() + float(seconds))

def time_left():
    import time
    dl = _REQUEST_DEADLINE.get()
    if not dl:
        return 999999.0
    return max(0.0, dl - time.time())

def over_budget(pad: float = 0.0):
    import time
    dl = _REQUEST_DEADLINE.get()
    if not dl:
        return False
    return (time.time() + float(pad)) > dl

# ---- lightweight perceptual aHash (no extra deps) for near-duplicate filtering ----
import io, base64
from PIL import Image

def _ahash_from_bytes(b: bytes) -> int:
    try:
        im = Image.open(io.BytesIO(b)).convert("L")
        im = im.resize((8, 8), Image.LANCZOS)
        pix = list(im.getdata())
        avg = sum(pix) / len(pix)
        bits = 0
        for i, v in enumerate(pix):
            if v >= avg:
                bits |= (1 << i)
        return bits
    except Exception:
        return -1

def _bytes_from_data_url(url: str):
    try:
        if url.startswith("data:image"):
            head, b64 = url.split(",", 1)
            return base64.b64decode(b64.encode())
    except Exception:
        pass
    return None

def dedup_image_parts_by_phash(image_parts: list, max_hamming: int = 5) -> list:
    """Removes near-duplicates among *data-URL* photos only. No image cap."""
    kept = []
    hashes: list[int] = []
    for part in image_parts or []:
        try:
            url = part.get("image_url", {}).get("url", "")
            b = _bytes_from_data_url(url)
            if not b:
                kept.append(part)
                continue
            h = _ahash_from_bytes(b)
            if h == -1:
                kept.append(part)
                continue
            dupe = False
            for h2 in hashes:
                x = (h ^ h2); d = 0
                while x:
                    d += 1; x &= x - 1
                if d <= max_hamming:
                    dupe = True
                    break
            if not dupe:
                hashes.append(h)
                kept.append(part)
        except Exception:
            kept.append(part)
    return kept
# === END NSPXN speed/robustness helpers ===


ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================= SPEED / BEHAVIOR TUNABLES =======================
PDF_OCR_DPI_EST = int(os.getenv("PDF_OCR_DPI_EST", "160"))
PDF_OCR_DPI_TXT = int(os.getenv("PDF_OCR_DPI_TXT", "150"))
PDF_OCR_DPI_PH  = int(os.getenv("PDF_OCR_DPI_PH",  "140"))
MAX_TEXT_PAGES  = int(os.getenv("MAX_TEXT_PAGES",  "3"))
MAX_PHOTO_PAGES = int(os.getenv("MAX_PHOTO_PAGES", "24"))
MAX_VISION_IMGS = int(os.getenv("MAX_VISION_IMGS", "8"))
THREADS         = int(os.getenv("OCR_THREADS",     "4"))
OAI_MODEL       = os.getenv("OAI_MODEL", "gpt-4o-mini")
OAI_TIMEOUT_S   = float(os.getenv("OAI_TIMEOUT_S", "15"))
TIME_BUDGET_S   = float(os.getenv("TIME_BUDGET_S", "55"))

MAX_TOTAL_IMAGE_BYTES = int(os.getenv("MAX_TOTAL_IMAGE_BYTES", str(3_000_000)))
MAX_PER_IMAGE_BYTES   = int(os.getenv("MAX_PER_IMAGE_BYTES",   str(350_000)))

PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

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

# ======================= Output sanitizers =======================
_NO_GPT_PAT = re.compile(
    r"(?is)"
    r"(?:\u26a0\ufe0f?\s*|&#9888;&#65039;\s*)?"
    r"(?:no\W*gp?t\W*output|gpt\W*returned\W*no\W*output|no\W*output\W*from\W*gpt|no\W*model\W*output|no\W*ai\W*output)"
)
def scrub_text(s: Any) -> Any:
    if isinstance(s, str):
        return _NO_GPT_PAT.sub("Comparison unavailable (fallback used).", s)
    return s
def sanitize_consistency(cons: Any) -> Dict[str, Any]:
    if not isinstance(cons, dict):
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison unavailable (fallback used)."}
    out = {
        "per_item": cons.get("per_item", []) or [],
        "not_in_photos": cons.get("not_in_photos", []) or [],
        "extra_damage_in_photos": cons.get("extra_damage_in_photos", []) or [],
        "overall": scrub_text(cons.get("overall", "") or "")
    }
    return out
def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return scrub_text(obj)
def _pre_sanitize_json_str(txt: str) -> str:
    return _NO_GPT_PAT.sub("Comparison unavailable (fallback used).", txt or "")
def _extract_json_fragment(txt: str) -> Optional[str]:
    txt = txt.strip()
    try:
        json.loads(txt); return txt
    except Exception:
        pass
    first_obj = txt.find("{"); last_obj = txt.rfind("}")
    first_arr = txt.find("["); last_arr = txt.rfind("]")
    cand = None
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        cand = txt[first_obj:last_obj+1]
    elif first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        cand = txt[first_arr:last_arr+1]
    return cand

# ======================= OCR helpers =======================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.75)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img
def _ocr_variants(img: Image.Image) -> str:
    texts = []
    for scale in (1.0, 1.5, 2.0):
        try:
            w, h = img.size
            big = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        except Exception:
            big = img
        for psm in ("--psm 6", "--psm 11"):
            try:
                txt = pytesseract.image_to_string(preprocess_image(big), lang="eng", config=psm)
                if txt and txt.strip():
                    texts.append(txt)
            except Exception:
                continue
    return "\n".join(texts)
def ocr_image_quick(img: Image.Image, config: str = "--psm 6") -> str:
    return pytesseract.image_to_string(preprocess_image(img), lang="eng", config=config)
def ocr_pdf_first_page(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=PDF_OCR_DPI_EST)
    return ocr_image_quick(pages[0]) if pages else ""
def ocr_pdf_text_caps(pdf_bytes: bytes, max_pages: int, t0: float) -> str:
    if nearly_out_of_time(t0, 10.0):
        return ""
    pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_TXT)
    buf, used = [], 0
    for i, p in enumerate(pages, 1):
        if used >= max_pages or nearly_out_of_time(t0, 8.0):
            break
        txt = ocr_image_quick(p)
        if len(txt.strip()) >= 25:
            buf.append(f"[Page {i}]\n{txt}")
            used += 1
    return "\n".join(buf)
def _page_has_tax(text: str) -> bool:
    return re.search(r"(sales\s*tax|tax)[^\n]{0,160}?(?:\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(?:\.\d{2})?)", text, re.IGNORECASE) is not None
def _page_has_any_labor_rate(text: str) -> bool:
    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    for lbl in labels:
        pat = rf"{lbl}[^\n]{{0,200}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False
def ocr_pdf_scan_tax_labor_page(pdf_bytes: bytes, max_pages: int, t0: float) -> str:
    try:
        if nearly_out_of_time(t0, 12.0):
            return ""
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_TXT)
        for i, p in enumerate(pages[:max_pages], 1):
            if nearly_out_of_time(t0, 10.0): break
            txt = ocr_image_quick(p)
            if _page_has_tax(txt) or _page_has_any_labor_rate(txt):
                return f"[Page {i}]\n{txt}"
    except Exception as e:
        logger.warning(f"scan_tax_labor_page error: {e}")
    return ""

# ===== pdftotext helpers =====
def pdftotext_extract_pages(pdf_bytes: bytes, first: int, last: int) -> str:
    try:
        with tempfile.TemporaryDirectory() as td:
            in_pdf = os.path.join(td, "in.pdf")
            with open(in_pdf, "wb") as f: f.write(pdf_bytes)
            out_txt = os.path.join(td, "out.txt")
            subprocess.run(["pdftotext", "-layout", "-f", str(first), "-l", str(last), in_pdf, out_txt], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    except Exception as e:
        logger.info(f"pdftotext page extract failed: {e}")
    return ""
def pdftotext_extract_all(pdf_bytes: bytes) -> str:
    try:
        with tempfile.TemporaryDirectory() as td:
            in_pdf = os.path.join(td, "in.pdf")
            with open(in_pdf, "wb") as f: f.write(pdf_bytes)
            out_txt = os.path.join(td, "out.txt")
            subprocess.run(["pdftotext", "-layout", in_pdf, out_txt], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out_txt):
                with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
    except Exception as e:
        logger.info(f"pdftotext not available or failed (full): {e}")
    return ""

# ======== Image format helpers (JPEG/PNG/WebP/GIF pass-through) ========
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
    mime = sniff_image_mime(blob)
    if mime in ACCEPTED_MIMES:
        return blob, mime
    try:
        im = Image.open(io.BytesIO(blob))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        im = Image.new("RGB", (8, 8), (0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue(), "image/jpeg"
def downscale_jpeg_to_max_bytes(blob: bytes, max_bytes: int) -> bytes:
    try:
        im = Image.open(io.BytesIO(blob)).convert("RGB")
    except Exception:
        return blob
    for q in (85, 80, 75, 70, 65, 60, 55):
        out = io.BytesIO(); im.save(out, format="JPEG", quality=q, optimize=True)
        b = out.getvalue()
        if len(b) <= max_bytes: return b
    w, h = im.size
    b = out.getvalue()
    while len(b) > max_bytes and min(w, h) > 800:
        w = int(w * 0.85); h = int(h * 0.85)
        im2 = im.resize((w, h), Image.LANCZOS)
        out = io.BytesIO(); im2.save(out, format="JPEG", quality=65, optimize=True)
        b = out.getvalue()
    return b
def make_data_url(blob: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(blob).decode('utf-8')}"

# ===== Extract embedded images via pdfimages; respect original format =====
def pdfimages_harvest(pdf_bytes: bytes, max_images: int, t0: float) -> List[Tuple[str, bytes, float, str]]:
    out: List[Tuple[str, bytes, float, str]] = []
    try:
        if nearly_out_of_time(t0, 15.0): return out
        with tempfile.TemporaryDirectory() as td:
            in_pdf = os.path.join(td, "in.pdf")
            with open(in_pdf, "wb") as f: f.write(pdf_bytes)
            prefix = os.path.join(td, "img")
            subprocess.run(["pdfimages", "-j", in_pdf, prefix], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            files = sorted(glob.glob(prefix + "*")); files.sort(key=lambda p: os.path.getsize(p), reverse=True)
            for i, fp in enumerate(files, 1):
                if i > max_images or nearly_out_of_time(t0, 12.0): break
                try:
                    with open(fp, "rb") as fh: raw = fh.read()
                    mime = sniff_image_mime(raw)
                    if mime not in ACCEPTED_MIMES:
                        raw, mime = ensure_openai_image(raw)
                    name_ext = {"image/jpeg": f"pdfimg-{i}.jpg",
                                "image/png":  f"pdfimg-{i}.png",
                                "image/webp": f"pdfimg-{i}.webp",
                                "image/gif":  f"pdfimg-{i}.gif"}[mime]
                    out.append((name_ext, raw, float(os.path.getsize(fp)), mime))
                except Exception:
                    continue
    except Exception as e:
        logger.info(f"pdfimages not available or failed: {e}")
    return out

# ===== Photo-like page harvest (render fallback) — saved as JPEG =====
def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int, t0: float) -> List[Tuple[str, bytes, float, str]]:
    out: List[Tuple[str, bytes, float, str]] = []
    try:
        if nearly_out_of_time(t0, 15.0): return out
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_PH)
        used = 0
        for i, page in enumerate(pages, 1):
            if used >= max_pages or nearly_out_of_time(t0, 12.0): break
            proc = preprocess_image(page)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            up = (ocr or "").upper()

            has_vin_cue = bool(re.search(r"\bVIN\b", up) or re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up))
            has_odo_cue = ("ODOMETER" in up or "ODO " in up or "MILEAGE" in up or "MPH" in up or "RPM" in up)
            corner_hits = sum(1 for _ in re.finditer(r'\b(?:LF|RF|LR|RR|LEFT|RIGHT|FRONT|REAR)\b', up))
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            looks_like_photos = var > 120 or corner_hits >= 2 or "IMAGE REPORT" in up

            if looks_like_photos or has_vin_cue or has_odo_cue:
                buf = io.BytesIO(); page.convert("RGB").save(buf, format="JPEG", quality=85)
                score = (corner_hits * 10 + var) + (50 if has_vin_cue else 0) + (40 if has_odo_cue else 0)
                out.append((f"pdf-p{i}.jpg", buf.getvalue(), score, "image/jpeg")); used += 1
    except Exception as e:
        logger.warning(f"harvest_photos_from_pdf error: {e}")
    return out

def ocr_pdf_items_wide_scan(pdf_bytes: bytes, limit_pages: int, dpi: int, t0: float) -> str:
    out = []
    try:
        if nearly_out_of_time(t0, 12.0): return ""
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        for i, p in enumerate(pages[:limit_pages], 1):
            if nearly_out_of_time(t0, 10.0): break
            txt = ocr_image_quick(p)
            if len(txt.strip()) >= 20: out.append(f"[WideScan Page {i}]\n{txt}")
    except Exception as e:
        logger.warning(f"ocr_pdf_items_wide_scan error: {e}")
    return "\n".join(out)

# ======================= VIN utilities =======================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
_translit = {**{str(i): i for i in range(10)},
             **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]
def normalize_vin(s: str) -> Optional[str]:
    s = s.strip().upper().replace(" ", "").replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s
def vin_checksum_ok(v: str) -> bool:
    if len(v) != 17: return False
    try:
        total = sum(_translit[ch] * _weights[i] for i, ch in enumerate(v))
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
def _strip_urlish_tail(s: str) -> str:
    if not s: return s
    orig = s; low = s.lower()
    markers = ["http", "www", ".com", ".net", ".org", ".io", ".co/", ".co ", "://", "jdpower", "kbb", "edmunds", "ccc"]
    cut = len(s)
    for m in markers:
        i = low.find(m)
        if i != -1: cut = min(cut, i)
    s = s[:cut]; s = re.sub(r"[•|,;:/\-]+$", "", s).strip()
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s or orig.strip()


def extract_claim_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # Flatten to handle label/token across line breaks
    flat = re.sub(r"[\r\n]+", " ", text)
    flat = re.sub(r"\s{2,}", " ", flat)

    LABELS = r"(?:Claim|Assignment|Reference|Ref|File|Loss|Case|Report|RO|Work\s*Order)"
    TOKEN  = r"([A-Za-z0-9][A-Za-z0-9\-_\/]*\d[A-Za-z0-9\-_\/]*)"

    # 1) Direct labeled extraction
    for pat in [
        rf"{LABELS}\s*(?:No\.?|Number|#)?\s*[:\-]?\s*{TOKEN}",
        rf"{LABELS}\s*[:\-]?\s*(?:No\.?|Number|#)?\s*{TOKEN}",
    ]:
        m = re.search(pat, flat, re.IGNORECASE)
        if m:
            cand = m.group(1).strip().strip(".:,;#")
            return cand

    # 2) Fallback: hyphenated long numeric near a label within 80 chars
    for m in re.finditer(r"\b\d{4,}-\d{5,}\b", flat):
        start, end = m.start(), m.end()
        window = flat[max(0, start-80):min(len(flat), end+80)]
        if re.search(LABELS, window, re.IGNORECASE):
            return m.group(0)

    # 3) Last resort: simple 'Claim :' variants from original logic
    for pat in [
        r"(?:^|\s)(?:Claim\s*(?:#|No\.?|Number)[:\s]*)\s*([A-Za-z0-9\-_/]+)",
        r"(?:^|\s)Claim\s*[:#]\s*([A-Za-z0-9\-_/]+)",
        r"(?:^|\s)File\s*(?:#|No\.?|Number)[:\s]*([A-Za-z0-9\-_/]+)"
    ]:
        m = re.search(pat, flat, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    return None

def extract_vehicle_line_global(full_text: str) -> Optional[str]:
    if not full_text: return None
    # Vehicle Description line
    m = re.search(r"(?im)^\s*vehicle\s*(?:description)?\s*:\s*(.+)$", full_text)
    if m:
        cand = _strip_urlish_tail(m.group(1).strip())
        if re.search(rf"\b(19|20)\d{2}\b", cand) or re.search(rf"\b{MAKES}\b", cand, re.I):
            return cand
    # Year/Make/Model triplet somewhere else
    y = re.search(r"(?i)\byear\b\s*[:\-]?\s*((?:19|20)\d{2})", full_text)
    mk = re.search(rf"(?i)\bmake\b\s*[:\-]?\s*({MAKES})", full_text, re.I)
    md = re.search(r"(?i)\bmodel\b\s*[:\-]?\s*([A-Za-z0-9\-/ ]{2,40})", full_text)
    if (y and mk and md):
        return _strip_urlish_tail(f"{y.group(1)} {mk.group(1)} {md.group(1)}")
    # Free-text global line
    for raw in (ln.strip() for ln in full_text.splitlines() if ln.strip()):
        m_year = re.search(r"\b(19|20)\d{2}\b", raw)
        if not m_year: continue
        year_idx = m_year.start(); tail = raw[year_idx:].strip()
        if re.search(rf"\b{MAKES}\b", tail, re.IGNORECASE):
            return _strip_urlish_tail(re.sub(r"https?://\S+", "", tail))
    return None

def _normalize_percent_str(pct_str: str) -> str:
    s = pct_str.strip().replace(" ", "").replace("%", "")
    try: return f"{float(s):g}%"
    except Exception: return pct_str.strip().rstrip("%") + "%"
def parse_tax_rate(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"(?i)(?:sales\s*tax|tax)[^\n]{0,160}?(\d{1,3}(?:\.\d+)?\s*%)", text)
    if m: return _normalize_percent_str(m.group(1))
    m2 = re.search(r"(?i)(?:sales\s*tax|tax)[^\n]{0,160}?\$\s*\d+(?:\.\d{2})?", text)
    if m2: return m2.group(0).strip()
    return None
def parse_labor_rates(text: str) -> Dict[str, str]:
    if not text: return {}
    labels = {"Body": r"Body\s*Labor","Paint": r"Paint\s*Labor","Mechanical": r"Mechanical\s*Labor","Structural": r"Structural\s*Labor"}
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

# ======================= Presence detection (+ vision fallback) =======================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def _classify_presence_with_vision(image_blobs: List[Tuple[str, bytes]], t0: float) -> Dict[str, bool]:
    if not image_blobs: return {}
    parts = []
    total = 0
    for name, blob in image_blobs[:6]:
        b, m = ensure_openai_image(blob)
        if m == "image/jpeg" and len(b) > MAX_PER_IMAGE_BYTES:
            b = downscale_jpeg_to_max_bytes(b, MAX_PER_IMAGE_BYTES)
        if total + len(b) > MAX_TOTAL_IMAGE_BYTES: break
        parts.append({"type":"image_url","image_url":{"url":make_data_url(b, m)}})
        total += len(b)
    schema = {"type":"object","properties":{"vin_present":{"type":"boolean"},"odometer_present":{"type":"boolean"},"license_plate_present":{"type":"boolean"}},"required":["vin_present","odometer_present","license_plate_present"]}
    sys = "Classify whether any of these images include: (1) a VIN door-jamb label/plate; (2) an odometer/cluster; (3) a license plate. Return STRICT JSON only."
    res = call_openai_json_sure(
        messages=[{"role":"system","content":sys},{"role":"user","content":[{"type":"text","text":"Return JSON per schema: "+json.dumps(schema)}, *parts]}],
        max_tokens=150, t0=t0, label="presence"
    )
    out = {}
    if isinstance(res, dict):
        out["vin"] = bool(res.get("vin_present"))
        out["odometer"] = bool(res.get("odometer_present"))
        out["license plate"] = bool(res.get("license_plate_present"))
    return out

def detect_required_photo_presence(image_blobs: List[Tuple[str, bytes]], t0: float) -> Dict[str, Any]:
    flags: Dict[str, Any] = {"four corners": False, "odometer": False, "vin": False, "license plate": False}
    ext_like = 0; corner_hits = 0
    VIN_DOOR_LABEL_CUES = ("MFD BY", "GENERAL MOTORS", "THIS VEHICLE CONFORMS", "GVWR", "GAWR", "VIN")
    ODO_CUES = ("ODOMETER", "ODO ", "MILEAGE", "MPH", "RPM", "PRND", "KM/H")
    vin_candidates: set[str] = set()

    limit = min(len(image_blobs), 40)
    for idx, (name, blob) in enumerate(image_blobs[:limit], 1):
        if nearly_out_of_time(t0, 10.0): break
        try:
            base = Image.open(io.BytesIO(blob))
        except Exception:
            continue

        name_low = (name or "").lower()
        if any(k in name_low for k in ("vin","doorjamb","door-jamb","door_jamb")):
            flags["vin"] = True
        if any(k in name_low for k in ("odo","odometer","cluster","speedo","dash")):
            flags["odometer"] = True
        if any(k in name_low for k in ("plate","license","tag")):
            flags["license plate"] = True

        for r in (0, 90, 180, 270):
            if nearly_out_of_time(t0, 8.0): break
            try:
                img = base.rotate(r, expand=True)
                text = _ocr_variants(img)
                up = (text or "").upper()
                if any(cue in up for cue in VIN_DOOR_LABEL_CUES) or re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up):
                    flags["vin"] = True
                for v in re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", up):
                    vnorm = normalize_vin(v) or ""
                    if vnorm: vin_candidates.add(vnorm)
                if any(c in up for c in ODO_CUES):
                    flags["odometer"] = True
                if "LICENSE" in up or "PLATE" in up or re.search(r"\b[A-Z0-9]{5,8}\b", up):
                    flags["license plate"] = True
                if _image_is_exterior_wide(img): ext_like += 1
                corner_hits += len(re.findall(r'\b(?:LF|RF|LR|RR|LEFT|RIGHT|FRONT|REAR)\b', up))
            except Exception:
                continue

    if ext_like >= 2 or corner_hits >= 3:
        flags["four corners"] = True

    # Vision fallback to lift recalls if any still missing
    need = [k for k, v in flags.items() if not v and k in ("vin", "odometer", "license plate")]
    if need and not nearly_out_of_time(t0, 3.0):
        vis = _classify_presence_with_vision(image_blobs[:10], t0)
        for k in ("vin", "odometer", "license plate"):
            if not flags[k] and vis.get(k) is True:
                flags[k] = True

    # stash candidates for VIN verification logic
    flags["_vin_candidates"] = list(vin_candidates)
    return flags

# ======================= Client guideline ingestion =======================
GUIDE_HINTS = ("guide", "guideline", "rules", "policy", "client", "requirements", "instruction")
def looks_like_guideline_name(filename: str) -> bool:
    fn = (filename or "").lower()
    return any(h in fn for h in GUIDE_HINTS)
def extract_text_from_pdf_bytes_all(pdf_bytes: bytes, t0: float) -> str:
    txt = pdftotext_extract_all(pdf_bytes)
    if txt and txt.strip(): return txt
    return ocr_pdf_items_wide_scan(pdf_bytes, limit_pages=30, dpi=170, t0=t0)
def append_client_rules_from_blob(name: str, raw: bytes, rules_parts: List[str], t0: float):
    try:
        if name.endswith(".pdf"):
            t = extract_text_from_pdf_bytes_all(raw, t0)
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

# ======================= Estimate items (regex + LLM fallback) =======================
PANELS = [
    "front bumper cover","rear bumper cover","bumper cover","bumper","fender","door","hood","grille",
    "headlamp","headlight","taillamp","tail lamp","quarter panel","rocker","roof","trunk","decklid","mirror",
    "apron","radiator support","radiator","support","wheel","tire","pillar","garnish","molding","fog lamp",
    "reinforcement","valance","bracket","impact bar","condenser","condensor","core support","fuel tank","battery",
    "fuel system","radiator support","radiator","trailer hitch"
]
OPS = [
    "replace","repair","refinish","align","blend","calibrate","r&i","r & i","remove & install","remove and install",
    "r&r","r & r","remove & replace","remove and replace","disconnect & reconnect","disconnect and reconnect","repl","d&r"
]
OP_ALIASES = {
    "repl":"replace","rep":"repair","rpr":"repair","r&i":"r&i","r & i":"r&i","r&r":"replace","r & r":"replace",
    "remove & replace":"replace","remove and replace":"replace","remove & install":"r&i","remove and install":"r&i",
    "blend":"blend","refinish":"refinish","align":"align","calibrate":"calibrate","replace":"replace","repair":"repair",
    "disconnect & reconnect":"r&i","disconnect and reconnect":"r&i","d&r":"r&i"
}
SIDE_TOKENS = {"lh":"left","rh":"right","lf":"left front","rf":"right front","lr":"left rear","rr":"right rear",
               "left":"left","right":"right","front":"front","rear":"rear"}
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
        if p in seg: return p
    m = re.search(r"\bbumper(?:\s*cover|(?:\s*ass(?:y|embly))?)?\b|\bfender\b|\bdoor\b|\bhood\b|\bgrille\b|\b(head|tail)lamp\b|\bquarter\s*panel\b|\brocker\b|\broof\b|\btrunk\b|\bdecklid\b|\bbattery\b|\bradiator\s*support\b|\bradiator\b|\btrailer\s*hitch\b", seg)
    return m.group(0) if m else None

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if len(line) < 5: continue
        l = line.lower()

        m_col = re.search(r"^\s*(?:\d{1,4}[A-Z]?\s+)?(?:\*{1,2}\s*)?([A-Za-z& ]{2,20})\s+(?:USED|A/M|CAPA|RECOND|LKQ|NSF|OEM|ALT\s*OEM)?\s*(.+)$", line)
        if m_col:
            op_raw = m_col.group(1).strip(); tail = m_col.group(2).strip()
            op = _norm_op(op_raw)
            if op:
                part = _find_part(tail)
                if part:
                    side = _norm_side_from_text(tail)
                    items.append({"op": op, "part": part, "side": side, "raw": line}); continue

        m_phrase = re.search(r"^(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace|disconnect\s*&\s*reconnect|disconnect\s*and\s*reconnect|repl|r&i|r&r|d&r)\s+(.+)$", l, flags=re.I)
        if m_phrase:
            op = _norm_op(m_phrase.group(1)); tail = m_phrase.group(2)
            if op:
                part = _find_part(tail)
                if part:
                    side = _norm_side_from_text(tail)
                    items.append({"op": op, "part": part, "side": side, "raw": line}); continue

        m_rev = re.search(rf"(.*?)(?:[—\-–]|  +)\s*(replace|repair|refinish|align|blend|calibrate|r\s*&\s*i|r\s*&\s*r|remove\s*&\s*install|remove\s*&\s*replace|remove\s*and\s*install|remove\s*and\s*replace|disconnect\s*&\s*reconnect|disconnect\s*and\s*reconnect|repl|r&i|r&r|d&r)\b", l, flags=re.I)
        if m_rev:
            head = m_rev.group(1); op = _norm_op(m_rev.group(2))
            if op:
                part = _find_part(head)
                if part:
                    side = _norm_side_from_text(head)
                    items.append({"op": op, "part": part, "side": side, "raw": line}); continue

    uniq, seen = [], set()
    for it in items:
        key = (it["op"], it["part"], it["side"])
        if key not in seen: uniq.append(it); seen.add(key)
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
            temperature=0, max_tokens=500,
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned = []
            for it in data[:140]:
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
        j = min(n, i + size); out.append(txt[i:j])
        if j >= n: break
        i = i + size - overlap
    return out
def llm_extract_items_chunked(full_text: str, time_guard: Callable[[], bool]) -> List[Dict[str, str]]:
    chunks = _chunk_text(full_text, size=6000, overlap=400)[:4]
    merged: List[Dict[str, str]] = []; seen = set()
    for ch in chunks:
        if time_guard(): break
        items = extract_estimate_items_llm(ch)
        for it in items:
            key = (it["op"], it["part"], it["side"])
            if key not in seen: merged.append(it); seen.add(key)
    return merged

# ======================= Vision helpers =======================
def build_vision_payload_capped(images: List[Tuple[str, bytes, str]]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []; total = 0
    for name, blob, mime in images[:MAX_VISION_IMGS]:
        safe_bytes, safe_mime = ensure_openai_image(blob)
        if safe_mime == "image/jpeg" and len(safe_bytes) > MAX_PER_IMAGE_BYTES:
            safe_bytes = downscale_jpeg_to_max_bytes(safe_bytes, MAX_PER_IMAGE_BYTES)
        if total + len(safe_bytes) > MAX_TOTAL_IMAGE_BYTES: break
        parts.append({"type": "image_url", "image_url": {"url": make_data_url(safe_bytes, safe_mime)}})
        total += len(safe_bytes)
    return parts

def call_openai_json_sure(messages: List[Dict[str, Any]], max_tokens: int, t0: float, label: str) -> Dict[str, Any]:
    for attempt in range(2):
        try:
            rsp = client_fast.chat.completions.create(
                model=OAI_MODEL,
                messages=messages + [{"role":"system","content":"Always respond with STRICT JSON only. Never include phrases like 'No GPT output'."}],
                max_tokens=max_tokens, temperature=0
            )
            txt = (rsp.choices[0].message.content or "").strip()
            if not txt: raise ValueError("empty_content")
            txt = _pre_sanitize_json_str(txt)
            if txt.startswith("```"): txt = txt.strip("`")
            try:
                return json.loads(txt)
            except Exception:
                frag = _extract_json_fragment(txt)
                if frag:
                    frag = _pre_sanitize_json_str(frag)
                    return json.loads(frag)
                raise
        except Exception as e:
            logger.error(f"OpenAI {label} attempt {attempt+1} failed: {type(e).__name__}: {e}")
            max_tokens = max(250, int(max_tokens * 0.7))
    return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison unavailable (fallback used)."}

def compare_estimate_with_photos(items: List[Dict[str, str]], images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "per_item": {"type":"array","items":{"type":"object","properties":{
                "op":{"type":"string"},"part":{"type":"string"},"side":{"type":"string"},
                "photo_evidence":{"type":"boolean"},"confidence":{"type":"number"},"note":{"type":"string"}},
                "required":["op","part","side","photo_evidence","confidence","note"]}},
            "not_in_photos":{"type":"array","items":{"type":"string"}},
            "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
            "overall":{"type":"string"}
        },
        "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]
    }
    system = (
        "You are an auto-damage visual auditor. "
        "Given estimate line items and vehicle photos, decide for EACH item whether visible photo evidence exists. "
        "Hidden ops (calibration, internal R&I) may not be visible → mark as no-evidence with a short note. "
        "Also list obvious damages seen in photos that are NOT in the estimate. "
        "If per_item would be empty, STILL provide a useful narrative in 'overall' explaining why, referencing image content. "
        "Return STRICT JSON ONLY per this schema:\n" + json.dumps(schema)
    )
    user_parts: List[Dict[str, Any]] = [{"type":"text","text":"Estimate items:\n"+json.dumps(items, ensure_ascii=False)}]
    user_parts.extend(images_for_vision)
    return call_openai_json_sure(
        messages=[{"role":"system","content":system},{"role":"user","content":user_parts}],
        max_tokens=900, t0=time.monotonic(), label="vision"
    )

# ======================= Guideline comparison (enhanced narrative) =======================
def parse_year_from_vehicle_line(vehicle_line: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", vehicle_line or ""); return int(m.group(0)) if m else None

def build_guideline_comparison(effective_rules: str, combined_text: str, vehicle_line: str,
                               mileage_est: str, vin_photo_present: bool, presence_flags: Dict[str, bool],
                               labor_rates: Dict[str, str], tax_rate: str, uploaded_names: List[str]) -> str:
    chk = lambda b: "✔️" if b else "❌"
    rules_low = (effective_rules or "").lower(); text_low = (combined_text or "").lower()
    names_low = [n.lower() for n in uploaded_names]

    # Accept any valuation doc as satisfying NADA requirement
    val_tokens = ["nada","valuation","ccc valuation","ccc vehicle valuation","acv","jd power","j.d. power","jdpower",
                  "kbb","kelley blue book","black book","edmunds"]
    nada_present = any(any(tok in n for tok in val_tokens) for n in names_low) or any(tok in text_low for tok in val_tokens)

    total_loss = bool(re.search(r"\btotal\s+loss\b", text_low))
    parts_tokens = ("lkq","recycled","used","aftermarket"," am ")
    parts_non_oem_used = any(tok in text_low for tok in parts_tokens)
    tow_storage_present = bool(re.search(r"\b(tow(ing)?|storage)\b", text_low))
    betterment_present = bool(re.search(r"(?m)^\s*(betterment|depreciation)\b.*\b(\$|\d)", text_low))

    photos_req = any(k in rules_low for k in ["photo","vin","odometer","license","four corners"])

    year = parse_year_from_vehicle_line(vehicle_line or ""); age_desc = ""
    if year:
        from datetime import datetime
        try:
            age = max(0, datetime.utcnow().year - year); age_desc = f"{age} years old"
        except Exception: pass

    lines: List[str] = []
    lines.append("✅ Compliance vs Client Guidelines")

    # Release / Disclosure
    lines.append("")
    lines.append("Release / Disclosure:")
    lines.append("✔️ Estimate includes standard disclosure; no owner/repair release noted (ok unless rules prohibit sends).")

    # Total loss
    lines.append("")
    lines.append("Total Loss Handling:")
    lines.append(f"{chk(True)} Total loss indicated; check CCC/salvage docs (not auto-evaluated)." if total_loss
                 else f"{chk(True)} Not a total loss. No CCC valuation or salvage bids required.")

    # Parts usage
    lines.append("")
    lines.append("Parts Usage:")
    if age_desc or mileage_est:
        age_miles = f"{age_desc}" + (f", {mileage_est} miles" if mileage_est and mileage_est != 'Not listed' else "")
        lines.append(f"{chk(True)} {age_miles} → {'Aftermarket/Recycled/Used parts present.' if parts_non_oem_used else 'No non-OEM parts flagged in estimate.'}")
    else:
        lines.append(f"{chk(not parts_non_oem_used)} Non-OEM parts present." if parts_non_oem_used else f"{chk(True)} No non-OEM parts flagged in estimate.")

    # Valuation/NADA
    lines.append("")
    lines.append("Valuation (NADA or Equivalent):")
    if "nada" in rules_low:
        lines.append(f"{chk(nada_present)} Guidelines: valuation report required (NADA/JD Power/KBB/CCC/ACV/etc.).")
        if not nada_present: lines.append("❌ Valuation report not found among uploaded files.")
    else:
        lines.append(f"{chk(True)} No specific valuation doc requirement found in provided rules.")

    # Photos
    lines.append("")
    lines.append("Photo Rules:")
    if photos_req:
        all_present = all(presence_flags.get(k, False) for k in ["four corners","vin","odometer","license plate"])
        lines.append(f"{chk(all_present)} Required photos include: four corners, VIN, odometer, plate.")
        miss = [k for k,v in presence_flags.items() if not v and not k.startswith("_")]
        if miss: lines.append(f"❌ Missing: {', '.join(miss)}")
        else: lines.append("✔️ Full required photo set present.")
    else:
        lines.append(f"{chk(True)} No explicit photo package requirement found in provided rules.")

    # Rates
    lines.append("")
    lines.append("Labor & Tax Rates:")
    lines.append(f"{chk(bool(labor_rates))} Labor rates detected: " + (", ".join(f"{k} {v}" for k,v in labor_rates.items()) if labor_rates else "NONE"))
    lines.append(f"{chk('Not detected' not in tax_rate)} Sales tax detected: {tax_rate}")

    # Tow/Storage
    lines.append("")
    lines.append("Tow/Storage:")
    lines.append(f"{chk(not tow_storage_present)} No tow/storage charges included." if not tow_storage_present else "❌ Tow/storage charges present; verify per rules.")

    # Betterment/Depreciation
    lines.append("")
    lines.append("Betterment/Depreciation:")
    lines.append("❌ Betterment/Depreciation applied—verify justification vs rules." if betterment_present else "✔️ None applied or not indicated in estimate text.")

    # Documentation
    lines.append("")
    lines.append("Documentation Requirements:")
    doc_req = any(k in rules_low for k in ["appraisal report","report notes","supplement notes"])
    if "nada" in rules_low or doc_req:
        ok_doc = nada_present or not ("nada" in rules_low)
        lines.append(f"{chk(ok_doc)} Core appraisal report/valuation/notes presence checked.")
        if "nada" in rules_low and not nada_present: lines.append("❌ Valuation report missing.")
    else:
        lines.append("✔️ No additional documentation requirements found in provided rules.")

    return "\n".join(lines)

# ======================= PDF helpers =======================
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12); pdf.cell(0, 8, txt=title, ln=True); pdf.set_font_size(10)
def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10); pdf.multi_cell(0, 6, f"{key}: {scrub_text(val)}")

# ======================= Helpers to select estimate PDF =======================
def select_estimate_pdf(all_uploads: List[Tuple[str, bytes]]) -> Optional[Tuple[str, bytes]]:
    for name, raw in all_uploads:
        if not name.lower().endswith(".pdf"):
            continue
        try:
            pages = convert_from_bytes(raw, first_page=1, last_page=1, dpi=140)
            if not pages:
                continue
            t = pytesseract.image_to_string(preprocess_image(pages[0]), lang="eng")
            up = t.upper()
            if ("ESTIMATE OF RECORD" in up or "OWNER:" in up) and ("JOB NUMBER" in up or "VEHICLE" in up):
                return (name, raw)
        except Exception:
            continue
    return None

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

    # ----- Ingest uploads once -----
    uploads: List[Tuple[str, bytes]] = []
    for f in files:
        raw = await f.read()
        uploads.append(((f.filename or "upload"), raw))

    texts: List[str] = []
    photos_for_presence: List[Tuple[str, bytes]] = []
    images_for_openai: List[Tuple[str, bytes, str]] = []
    rules_parts: List[str] = []
    uploaded_names: List[str] = [n for (n, _) in uploads]

    # Identify estimate PDF
    estimate_pdf = select_estimate_pdf(uploads)
    first_page_text_ocr = ""
    first_page_text_ptt = ""
    full_est_text_pdftotext = ""
    est_bytes = None

    if estimate_pdf:
        est_name, est_bytes = estimate_pdf
        first_page_text_ocr = ocr_pdf_first_page(est_bytes) or ""
        first_page_text_ptt = pdftotext_extract_pages(est_bytes, 1, 1) or ""
        full_est_text_pdftotext = pdftotext_extract_all(est_bytes)

        texts.append(ocr_pdf_text_caps(est_bytes, MAX_TEXT_PAGES, t0))
        tl = ocr_pdf_scan_tax_labor_page(est_bytes, 40, t0)
        if tl: texts.append(tl)

        # estimate-pdf images for presence + comparison
        emb = pdfimages_harvest(est_bytes, MAX_PHOTO_PAGES, t0)
        for hname, hbytes, _sz, hmime in emb:
            photos_for_presence.append((hname, hbytes))
            images_for_openai.append((hname, hbytes, hmime))
        rendered = harvest_photos_from_pdf(est_bytes, max_pages=8, t0=t0)
        for rname, rbytes, _score, rmime in rendered:
            photos_for_presence.append((rname, rbytes))
            images_for_openai.append((rname, rbytes, rmime))

    # Other uploads
    for name, raw in uploads:
        lname = name.lower()
        if estimate_pdf and name == estimate_pdf[0]:
            continue
        if lname.endswith(".pdf"):
            texts.append(ocr_pdf_text_caps(raw, MAX_TEXT_PAGES, t0))
            if looks_like_guideline_name(lname):
                rules_parts.append(extract_text_from_pdf_bytes_all(raw, t0))
            emb = pdfimages_harvest(raw, MAX_PHOTO_PAGES, t0)
            for hname, hbytes, _sz, hmime in emb:
                photos_for_presence.append((hname, hbytes))
                images_for_openai.append((hname, hbytes, hmime))
            rendered = harvest_photos_from_pdf(raw, max_pages=6, t0=t0)
            for rname, rbytes, _score, rmime in rendered:
                photos_for_presence.append((rname, rbytes))
                images_for_openai.append((rname, rbytes, rmime))
        elif lname.endswith((".docx", ".txt")):
            if lname.endswith(".docx"):
                try:
                    d = Document(io.BytesIO(raw))
                    texts.append("\n".join(p.text for p in d.paragraphs if p.text.strip()))
                except Exception as e:
                    logger.warning(f"DOCX error: {e}")
            else:
                try:
                    texts.append(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
            if looks_like_guideline_name(lname):
                append_client_rules_from_blob(name, raw, rules_parts, t0)
        elif lname.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            mime = sniff_image_mime(raw) or "image/jpeg"
            photos_for_presence.append((name, raw))
            images_for_openai.append((name, raw, mime))
        else:
            try:
                texts.append(raw.decode("utf-8", errors="ignore"))
            except Exception:
                logger.info(f"Skipped unsupported file: {name}")

    combined_text = "\n".join(t for t in texts if t)
    effective_rules = (client_rules or "").strip()
    if rules_parts:
        effective_rules = (effective_rules + "\n\n" + "\n\n".join(rules_parts)).strip()

    # Reorder: user photos first
    def is_user_photo(n: str) -> bool:
        n = (n or "").lower()
        return any(n.endswith(ext) for ext in (".jpg",".jpeg",".png",".webp",".gif"))
    photos_for_presence = [(n,b) for (n,b) in photos_for_presence if is_user_photo(n)] + \
                          [(n,b) for (n,b) in photos_for_presence if not is_user_photo(n)]
    images_for_openai = [(n,b,m) for (n,b,m) in images_for_openai if is_user_photo(n)] + \
                        [(n,b,m) for (n,b,m) in images_for_openai if not is_user_photo(n)]

    # Required photo presence
    presence_flags = await loop.run_in_executor(pool, detect_required_photo_presence, photos_for_presence, t0)
    missing_photos = [k for k in ("four corners","vin","odometer","license plate") if not presence_flags.get(k, False)]

    # First-page scope for VIN/Claim/Vehicle
    first_page_scope = (first_page_text_ptt + "\n" + first_page_text_ocr).strip()
    claim_number = extract_claim_from_text(first_page_scope) or "N/A"
    vin_est = extract_vin_from_text(first_page_scope) or "N/A"
    vehicle_line = extract_vehicle_line_from_first_page(first_page_scope) or None

    # Global fallback to avoid Vehicle: N/A
    if not vehicle_line:
        vehicle_line = extract_vehicle_line_global(full_est_text_pdftotext) or extract_vehicle_line_global(combined_text) or "Not listed"
    vehicle_line = _strip_urlish_tail(vehicle_line)

    # VIN verification (estimate vs photo) — never sourcing VIN from photo
    vin_photo_present = bool(presence_flags.get("vin"))
    vin_photo_vins = presence_flags.get("_vin_candidates", []) or []
    vin_verify_note = "VIN PHOTO NOT FOUND"
    if vin_photo_present:
        if vin_est != "N/A":
            if any(v == vin_est for v in vin_photo_vins):
                vin_verify_note = "MATCH"
            elif any(v for v in vin_photo_vins):
                vin_verify_note = "MISMATCH"
            else:
                vin_verify_note = "VIN PHOTO PRESENT—TEXT UNREADABLE"
        else:
            vin_verify_note = "Photos not provided"

    # Labor/Tax/Mileage
    basis_text = (full_est_text_pdftotext or "") + "\n" + first_page_scope
    labor_rates_page = parse_labor_rates(basis_text)
    tax_rate = parse_tax_rate(basis_text) or "Not detected"
    mileage_est = parse_mileage_from_text(basis_text) or "Not listed"

    # Estimate items — be aggressive so per-item comparison doesn't come back empty
    est_items = extract_estimate_items(full_est_text_pdftotext or "")
    if not est_items:
        est_items = extract_estimate_items(combined_text)

    # If still empty, widen OCR scan of estimate pages (fast)
    if not est_items and est_bytes and time_left(t0) > 6.0:
        wide_txt = ocr_pdf_items_wide_scan(est_bytes, limit_pages=8, dpi=165, t0=t0)
        est_items = extract_estimate_items(wide_txt)

    # Final fallback: LLM chunked extraction from all text we have
    if not est_items and time_left(t0) > 5.0:
        est_items = llm_extract_items_chunked((full_est_text_pdftotext or "") + "\n" + combined_text, lambda: time_left(t0) <= 3.5)

    # Vision payload + comparison
    if images_for_openai:
        subset = images_for_openai[:MAX_VISION_IMGS]
        image_parts = []

        # Remove near-duplicate data-URL photos (no image cap)
        image_parts = dedup_image_parts_by_phash(image_parts)
        for (n,b,m) in subset:
            bb, mm = ensure_openai_image(b)
            if mm == "image/jpeg" and len(bb) > MAX_PER_IMAGE_BYTES:
                bb = downscale_jpeg_to_max_bytes(bb, MAX_PER_IMAGE_BYTES)
            image_parts.append({"type":"image_url","image_url":{"url":make_data_url(bb, mm)}})
        consistency = compare_estimate_with_photos(est_items, image_parts)
    else:
        consistency = {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "No photos supplied."}

    consistency = sanitize_consistency(consistency)

    # If per-item is still empty, provide a clearer overall message instead of “No visible photo evidence…”
    if not consistency.get("per_item"):
        if images_for_openai and (full_est_text_pdftotext or combined_text):
            consistency["overall"] = "Per-item comparison unavailable. Estimate text or images did not yield discrete, mappable line items; recommend clearer estimate export or labeled photos."
        elif not images_for_openai:
            consistency["overall"] = "No photos supplied for visual confirmation."

    # Guidelines comparison (enhanced narrative)
    guideline_block = build_guideline_comparison(
        effective_rules=effective_rules,
        combined_text=combined_text,
        vehicle_line=vehicle_line,
        mileage_est=mileage_est,
        vin_photo_present=vin_photo_present,
        presence_flags={k:v for k,v in presence_flags.items() if not k.startswith("_")},
        labor_rates=labor_rates_page,
        tax_rate=tax_rate,
        uploaded_names=uploaded_names
    )

    # Scoring (unchanged)
    photo_adj = -25 * len(missing_photos)
    labor_penalty = 0 if labor_rates_page else -50
    tax_penalty = 0
    if re.search(r"tax\s*(required|must|utilize|apply)", effective_rules, re.IGNORECASE):
        if "Not detected" in tax_rate: tax_penalty = -25
    authoritative_score = max(0, min(100, 100 + photo_adj + labor_penalty + tax_penalty))

    # ======================= PDF build (layout unchanged) =======================
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0, 6, scrub_text(f"File Number: {file_number}"))
    pdf.multi_cell(0, 6, scrub_text(f"IA Company: {ia_company}"))
    pdf.multi_cell(0, 6, scrub_text(f"Appraiser ID #: {appraiser_id}"))
    pdf.ln(4)
    pdf.multi_cell(0, 6, scrub_text(f"Claim #: {claim_number}"))
    pdf.multi_cell(0, 6, scrub_text(f"VIN (from estimate): {vin_est}"))
    pdf.multi_cell(0, 6, scrub_text(f"VIN verification (estimate vs photo): {vin_verify_note}"))
    pdf.multi_cell(0, 6, scrub_text(f"Vehicle: {vehicle_line}"))
    pdf.multi_cell(0, 6, scrub_text(f"Odometer (estimate): {mileage_est}"))
    if labor_rates_page:
        pdf.multi_cell(0, 6, scrub_text("Labor rates detected: " + ", ".join(f"{k} {v}" for k,v in labor_rates_page.items())))
    else:
        pdf.multi_cell(0, 6, "Labor rates detected: NONE")
    pdf.multi_cell(0, 6, scrub_text(f"Tax Rate detected: {tax_rate}"))
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4); pdf_add_section_title(pdf, "Required Photos Presence (vision-verified)")
    if missing_photos:
        pdf.multi_cell(0, 6, scrub_text("Missing: " + ", ".join(missing_photos)))
    else:
        pdf.multi_cell(0, 6, "All required photos present.")

    # Time budget guard
    if over_budget(2.0):
        logger.warning('Skipping heavy step due to deadline budget.')
    pdf.ln(4); pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")
    if consistency.get("per_item"):
        for it in consistency["per_item"][:40]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            try: conf = float(it.get("confidence", 0))
            except Exception: conf = 0.0
            conf_txt = f"{round(conf*100)}%"
            line = f"- {it.get('side','unspecified').title()} {it.get('part','component')} · {it.get('op','op')} → Photo: {ev} ({conf_txt}); {it.get('note','')}"
            pdf.multi_cell(0, 6, scrub_text(line))
    else:
        pdf.multi_cell(0, 6, "Per-item comparison unavailable.")
    if consistency.get("not_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Items Estimated but Not Evident in Photos")
        for raw in consistency["not_in_photos"][:20]: pdf.multi_cell(0, 6, scrub_text(f"- {raw}"))
    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:20]: pdf.multi_cell(0, 6, scrub_text(f"- {d}"))
    pdf.ln(2); pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    pdf.ln(4); pdf_add_section_title(pdf, "Compliance vs Client Guidelines")
    for line in (guideline_block or "").splitlines():
        pdf.multi_cell(0, 6, scrub_text(line))

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
        logger.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # EMAIL (structure unchanged)
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

Compliance vs Client Guidelines
{guideline_block}
"""
        msg.set_content(scrub_text(email_body))
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing): {e}")

    # Build combined text block for UI — always non-empty
    if consistency.get("per_item"):
        yes_cnt = sum(1 for it in consistency["per_item"] if it.get("photo_evidence"))
        no_cnt  = sum(1 for it in consistency["per_item"] if not it.get("photo_evidence"))
        cons_head = f"Estimate ↔ Photos Summary: {yes_cnt} items with photo evidence, {no_cnt} without."
    else:
        cons_head = "Estimate ↔ Photos Summary: Per-item comparison unavailable."
    gpt_output = scrub_text(f"""{guideline_block}

{cons_head}
Overall: {consistency.get('overall','')}""").strip()
    if not gpt_output:
        gpt_output = "Automated review completed. See PDF sections above for guideline compliance and photo/estimate comparison."

    response_payload = {
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": {"line": vehicle_line},
        "vin_estimate": vin_est,
        "vin_verification": vin_verify_note,
        "score": f"{authoritative_score}%",
        "missing_photos": [m for m in missing_photos if not m.startswith("_")],
        "consistency_review": consistency,
        "guideline_comparison": guideline_block,
        "gpt_output": gpt_output
    }
    return sanitize_json(response_payload)

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
            return {"text": text}
        except Exception as e:
            logger.error(f"Client rules error: {str(e)}")
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        logger.error(f"Rules not found for client: {client_name}")
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})































