
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any, Callable
import os, re, io, base64, json, logging, asyncio, time, tempfile, subprocess, glob
from concurrent.futures import ThreadPoolExecutor

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, ImageFile, Image
from docx import Document
from openai import OpenAI

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================= CONFIG =======================
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

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

# ======================= TIME =======================
def t0_start() -> float: return time.monotonic()
def t_elapsed(t0: float) -> float: return time.monotonic() - t0
def time_left(t0: float) -> float: return max(0.0, TIME_BUDGET_S - t_elapsed(t0))
def nearly_out_of_time(t0: float, margin: float = 6.0) -> bool: return time_left(t0) <= margin

# ======================= IMAGE IO =======================
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

# ======================= SIMPLE OCR HELPERS =======================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.75)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr_image_quick(img: Image.Image, config: str = "--psm 6") -> str:
    return pytesseract.image_to_string(preprocess_image(img), lang="eng", config=config)

def ocr_pdf_first_page(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=PDF_OCR_DPI_EST)
    return ocr_image_quick(pages[0]) if pages else ""

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
        logger.info(f"pdftotext not available or failed: {e}")
    return ""

# ======================= VIN UTILITIES (bug-fixed) =======================
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
        if vin and vin_checksum_ok(vin):   # << fixed here
            return vin
    for c in cands:
        vin = normalize_vin(c)
        if vin:
            return vin
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    label_block = re.findall(r"(?:^|\\n).{0,60}VIN[:\\s\\-]*([A-HJ-NPR-Z0-9]{10,20}).*", text, re.IGNORECASE)
    if label_block:
        vin = best_vin_candidate(label_block)
        if vin: return vin
    candidates = re.findall(r"\\b([A-HJ-NPR-Z0-9]{17})\\b", text, re.IGNORECASE)
    return best_vin_candidate(candidates)

# ======================= VEHICLE / RATES PARSERS =======================
MAKES = r"(?:Acura|Alfa(?:\\s*Romeo)?|Audi|BMW|Buick|Cadillac|Chevrolet|Chevy|Chrysler|Dodge|Ferrari|Fiat|Ford|GMC|Genesis|Honda|Hyundai|Infiniti|Jaguar|Jeep|Kia|Lamborghini|Land\\s*Rover|Lexus|Lincoln|Maserati|Mazda|Mercedes(?:-|\\s*)Benz|Mini|Mitsubishi|Nissan|Porsche|Ram|Scion|Subaru|Suzuki|Tesla|Toyota|Volkswagen|VW|Volvo)"
def _strip_urlish_tail(s: str) -> str:
    if not s: return s
    orig = s; low = s.lower()
    markers = ["http", "www", ".com", ".net", ".org", ".io", ".co/", ".co ", "://", "jdpower", "kbb", "edmunds", "ccc"]
    cut = len(s)
    for m in markers:
        i = low.find(m)
        if i != -1: cut = min(cut, i)
    s = s[:cut]; s = re.sub(r"[•|,;:/\\-]+$", "", s).strip()
    s = re.sub(r"\\s{2,}", " ", s).strip()
    return s or orig.strip()

def extract_vehicle_line_from_first_page(first_page_text: str) -> Optional[str]:
    if not first_page_text: return None
    mveh = re.search(r"(?im)^.*\\bvehicle\\b\\s*:\\s*(.+)$", first_page_text)
    if mveh:
        line = _strip_urlish_tail(mveh.group(1).strip())
        if re.search(rf"\\b{MAKES}\\b", line, re.I):
            return line
    y = re.search(r"(?i)\\byear\\b\\s*[:\\-]?\\s*((?:19|20)\\d{2})", first_page_text)
    mk = re.search(rf"(?i)\\bmake\\b\\s*[:\\-]?\\s*({MAKES})", first_page_text, re.I)
    md = re.search(r"(?i)\\bmodel\\b\\s*[:\\-]?\\s*([A-Za-z0-9\\-/ ]{2,40})", first_page_text)
    if (y and mk and md):
        return _strip_urlish_tail(f"{y.group(1)} {mk.group(1)} {md.group(1)}")
    for raw in (ln.strip() for ln in first_page_text.splitlines() if ln.strip()):
        m_year = re.search(r"\\b(19|20)\\d{2}\\b", raw)
        if not m_year: continue
        year_idx = m_year.start(); tail = raw[year_idx:].strip()
        if not re.search(rf"\\b{MAKES}\\b", tail, re.IGNORECASE): continue
        tail = re.sub(r"\\s{2,}", " ", tail)
        tail = re.sub(r"https?://\\S+", "", tail).strip()
        return _strip_urlish_tail(tail)
    return None

def parse_tax_rate(text: str) -> Optional[str]:
    if not text: return None
    m = re.search(r"(?i)(?:sales\\s*tax|tax)[^\\n]{0,160}?(\\d{1,3}(?:\\.\\d+)?\\s*%)", text)
    if m: return f"{float(m.group(1).replace('%',''))}%"
    m2 = re.search(r"(?i)(?:sales\\s*tax|tax)[^\\n]{0,160}?\\$\\s*\\d+(?:\\.\\d{2})?", text)
    if m2: return m2.group(0).strip()
    return None

def parse_labor_rates(text: str) -> Dict[str, str]:
    if not text: return {}
    labels = {"Body": r"Body\\s*Labor","Paint": r"Paint\\s*Labor","Mechanical": r"Mechanical\\s*Labor","Structural": r"Structural\\s*Labor"}
    out: Dict[str, str] = {}
    for key, lbl_pat in labels.items():
        pat = rf"(?i){lbl_pat}[^\\n]{{0,200}}?\\$\\s*(\\d{{2,3}}(?:\\.\\d+)?)\\s*(?:/hr|/hour|per\\s*hour|hr)"
        m = re.search(pat, text)
        if m: out[key] = f"${m.group(1)}/hr"
    return out

# ======================= PRESENCE (OCR + VISION corners) =======================
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
                if txt and txt.strip(): texts.append(txt)
            except Exception: continue
    return "\\n".join(texts)

def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def _classify_presence_with_vision(image_blobs: List[Tuple[str, bytes]], t0: float) -> Dict[str, bool]:
    if not image_blobs: return {}
    parts = []
    total = 0
    for name, blob in image_blobs[:8]:
        b, m = ensure_openai_image(blob)
        if m == "image/jpeg" and len(b) > MAX_PER_IMAGE_BYTES:
            b = downscale_jpeg_to_max_bytes(b, MAX_PER_IMAGE_BYTES)
        if total + len(b) > MAX_TOTAL_IMAGE_BYTES:
            break
        parts.append({"type": "image_url", "image_url": {"url": make_data_url(b, m)}})
        total += len(b)
    schema = {
        "type": "object",
        "properties": {
            "vin_present": {"type": "boolean"},
            "odometer_present": {"type": "boolean"},
            "license_plate_present": {"type": "boolean"},
            "lf": {"type": "boolean"},
            "rf": {"type": "boolean"},
            "lr": {"type": "boolean"},
            "rr": {"type": "boolean"}
        },
        "required": ["vin_present", "odometer_present", "license_plate_present", "lf", "rf", "lr", "rr"]
    }
    sysmsg = (
        "Classify the set of images. Return STRICT JSON only per the schema. "
        "Definitions:\\n"
        "- vin_present: door-jamb label/plate or windshield VIN tag.\\n"
        "- odometer_present: instrument cluster with odometer mileage.\\n"
        "- license_plate_present: any exterior plate visible.\\n"
        "- lf/rf/lr/rr: exterior views clearly showing Left-Front, Right-Front, Left-Rear, Right-Rear corners.\\n"
        "A single photo can count for at most one corner (pick the best-fitting)."
    )
    try:
        rsp = client.chat.completions.create(
            model=OAI_MODEL,
            messages=[
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": [{"type": "text", "text": "Return JSON per schema:" + json.dumps(schema)}, *parts]},
            ],
            temperature=0,
            max_tokens=220,
        )
        raw = (rsp.choices[0].message.content or "").strip()
        if raw.startswith("```"): raw = raw.strip("`")
        data = json.loads(raw)
    except Exception as e:
        logger.error(f"vision presence fail: {e}")
        data = {}
    out = {"vin": False, "odometer": False, "license plate": False, "lf": False, "rf": False, "lr": False, "rr": False}
    if isinstance(data, dict):
        out["vin"] = bool(data.get("vin_present"))
        out["odometer"] = bool(data.get("odometer_present"))
        out["license plate"] = bool(data.get("license_plate_present"))
        out["lf"] = bool(data.get("lf"))
        out["rf"] = bool(data.get("rf"))
        out["lr"] = bool(data.get("lr"))
        out["rr"] = bool(data.get("rr"))
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
                if any(cue in up for cue in VIN_DOOR_LABEL_CUES) or re.search(r"\\b[A-HJ-NPR-Z0-9]{17}\\b", up):
                    flags["vin"] = True
                for v in re.findall(r"\\b([A-HJ-NPR-Z0-9]{17})\\b", up):
                    vnorm = normalize_vin(v) or ""
                    if vnorm: vin_candidates.add(vnorm)
                if any(c in up for c in ODO_CUES):
                    flags["odometer"] = True
                if "LICENSE" in up or "PLATE" in up or re.search(r"\\b[A-Z0-9]{5,8}\\b", up):
                    flags["license plate"] = True
                if _image_is_exterior_wide(img): ext_like += 1
                corner_hits += len(re.findall(r'\\b(?:LF|RF|LR|RR|LEFT|RIGHT|FRONT|REAR)\\b', up))
            except Exception:
                continue

    if ext_like >= 2 or corner_hits >= 3:
        flags["four corners"] = True

    # Vision fallback to lift recalls if any still missing
    need = [k for k, v in flags.items() if not v and k in ("vin", "odometer", "license plate")]
    vis = {}
    if (need or not flags["four corners"]) and not nearly_out_of_time(t0, 3.0):
        vis = _classify_presence_with_vision(image_blobs[:10], t0)
        for k in ("vin", "odometer", "license plate"):
            if not flags[k] and vis.get(k) is True:
                flags[k] = True
        corner_count = sum(1 for k in ("lf", "rf", "lr", "rr") if vis.get(k))
        if corner_count >= 3:
            flags["four corners"] = True

    flags["_vin_candidates"] = list(vin_candidates)
    return flags

# ======================= SIMPLE SELECTORS =======================
def select_estimate_pdf(all_uploads: List[Tuple[str, bytes]]) -> Optional[Tuple[str, bytes]]:
    for name, raw in all_uploads:
        if not name.lower().endswith(".pdf"):
            continue
        try:
            pages = convert_from_bytes(raw, first_page=1, last_page=1, dpi=140)
            if not pages: continue
            t = pytesseract.image_to_string(preprocess_image(pages[0]), lang="eng")
            up = (t or "").upper()
            if ("ESTIMATE" in up or "VEHICLE" in up):
                return (name, raw)
        except Exception:
            continue
    return None

# ======================= ROUTES =======================
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(""),
    ia_company: str = Form(""),
    appraiser_id: str = Form("")
):
    t0 = t0_start()
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=THREADS)

    uploads: List[Tuple[str, bytes]] = []
    for f in files:
        raw = await f.read()
        uploads.append(((f.filename or "upload"), raw))

    # Basic text
    texts: List[str] = []

    # Images for presence
    photo_blobs: List[Tuple[str, bytes]] = []
    for name, raw in uploads:
        if name.lower().endswith((".jpg",".jpeg",".png",".webp",".gif")):
            photo_blobs.append((name, raw))

    # Estimate OCR
    est = select_estimate_pdf(uploads)
    first_page_text = ""
    full_text = ""
    if est:
        first_page_text = ocr_pdf_first_page(est[1])
        full_text = pdftotext_extract_all(est[1])

    # VIN / Vehicle / Rates
    vin_est = extract_vin_from_text(first_page_text or full_text) or "N/A"
    vehicle_line = extract_vehicle_line_from_first_page(first_page_text or full_text) or "Not listed"
    labor_rates = parse_labor_rates(full_text or first_page_text or "")
    tax_rate = parse_tax_rate(full_text or first_page_text or "") or "Not detected"

    # Presence with vision 4-corners
    presence = await loop.run_in_executor(pool, detect_required_photo_presence, photo_blobs, t0)

    # VIN verify note
    vin_photo_present = bool(presence.get("vin"))
    vin_photo_vins = presence.get("_vin_candidates", []) or []
    if vin_photo_present:
        if vin_est != "N/A":
            if any(v == vin_est for v in vin_photo_vins):
                vin_verify_note = "MATCH"
            elif any(v for v in vin_photo_vins):
                vin_verify_note = "MISMATCH"
            else:
                vin_verify_note = "VIN PHOTO PRESENT—TEXT UNREADABLE"
        else:
            vin_verify_note = "VIN PHOTO PRESENT (no VIN on estimate)"
    else:
        vin_verify_note = "VIN PHOTO NOT FOUND"

    # Score (simple)
    missing = [k for k in ("four corners","vin","odometer","license plate") if not presence.get(k, False)]
    score = max(0, min(100, 100 - 25*len(missing) - (0 if labor_rates else 50) - (0 if "Not detected" not in tax_rate else 0)))

    # PDF
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.set_font("Arial", size=12)
    except Exception:
        pass
    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"File Number: {file_number}")
    pdf.multi_cell(0, 6, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 6, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(2)
    pdf.multi_cell(0, 6, f"VIN (from estimate): {vin_est}")
    pdf.multi_cell(0, 6, f"VIN verification (estimate vs photo): {vin_verify_note}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_line}")
    pdf.multi_cell(0, 6, f"Labor rates detected: " + (", ".join(f"{k} {v}" for k,v in labor_rates.items()) if labor_rates else "NONE"))
    pdf.multi_cell(0, 6, f"Tax Rate detected: {tax_rate}")
    pdf.multi_cell(0, 6, f"Compliance Score: {score}%")
    pdf.ln(3)
    pdf.multi_cell(0, 6, "Required Photos Presence (vision-verified)")
    if missing:
        pdf.multi_cell(0, 6, "Missing: " + ", ".join(missing))
    else:
        pdf.multi_cell(0, 6, "All required photos present.")
    out_pdf = os.path.join(PDF_DIR, f"{file_number or 'report'}.pdf")
    with open(out_pdf, "wb") as fh:
        fh.write(pdf.output(dest="S").encode("latin-1"))

    return FileResponse(out_pdf, media_type="application/pdf", filename=os.path.basename(out_pdf))
