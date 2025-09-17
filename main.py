from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
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
PDF_OCR_DPI_TXT = int(os.getenv("PDF_OCR_DPI_TXT", "140"))
PDF_OCR_DPI_PH  = int(os.getenv("PDF_OCR_DPI_PH",  "130"))
MAX_TEXT_PAGES  = int(os.getenv("MAX_TEXT_PAGES",  "3"))
MAX_PHOTO_PAGES = int(os.getenv("MAX_PHOTO_PAGES", "8"))
MAX_VISION_IMGS = int(os.getenv("MAX_VISION_IMGS", "6"))
THREADS         = int(os.getenv("OCR_THREADS",     "4"))
OAI_MODEL       = os.getenv("OAI_MODEL", "gpt-4o-mini")
OAI_TIMEOUT_S   = float(os.getenv("OAI_TIMEOUT_S", "15"))
TIME_BUDGET_S   = float(os.getenv("TIME_BUDGET_S", "55"))  # sub-minute target

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
def t0_start(): return time.monotonic()
def t_elapsed(t0): return time.monotonic() - t0
def time_left(t0): return max(0.0, TIME_BUDGET_S - t_elapsed(t0))
def nearly_out_of_time(t0, margin=6.0): return time_left(t0) <= margin

# ======================= OCR helpers (for estimate PDF only) =======================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageOps.autocontrast(img)
    return img

def ocr_image_quick(img: Image.Image, config="--psm 6") -> str:
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

def ocr_pdf_scan_tax_labor_page(pdf_bytes: bytes, max_pages: int = 30) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_TXT)
        for i, p in enumerate(pages, 1):
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

# ===== Fast text via pdftotext =====
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
        logger.info(f"pdftotext not available or failed: {e}")
    return ""

# ===== Ensure PNG helper =====
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

# ===== Photo-like page harvest (render fallback) =====
def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int) -> List[Tuple[str, bytes, float]]:
    out: List[Tuple[str, bytes, float]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_OCR_DPI_PH)
        used = 0
        for i, page in enumerate(pages, 1):
            proc = preprocess_image(page)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            corner_hits = count_corner_labels(ocr)
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            looks_like_photos = var > 120 or corner_hits >= 2 or "image report" in (ocr or "").lower()
            if looks_like_photos:
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                score = corner_hits * 10 + var
                out.append((f"pdf-p{i}.png", buf.getvalue(), score))
                used += 1
                if used >= max_pages:
                    break
    except Exception as e:
        logger.warning(f"harvest_photos_from_pdf error: {e}")
    return out

def ocr_pdf_items_wide_scan(pdf_bytes: bytes, limit_pages: int = 30, dpi: int = 120) -> str:
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

# ===== Extract key info from estimate text only =====
def extract_vin_from_estimate(text: str) -> Optional[str]:
    m = re.search(r'VIN:\s*([A-HJ-NPR-Z0-9]{17})', text, re.IGNORECASE)
    return m.group(1) if m else None

def extract_odometer_from_estimate(text: str) -> Optional[str]:
    m = re.search(r'Odometer:\s*([\d,]+)', text, re.IGNORECASE)
    return m.group(1).replace(',', '') if m else None

def extract_claim_from_estimate(text: str) -> Optional[str]:
    m = re.search(r'Claim #:\s*(\S+)', text, re.IGNORECASE)
    return m.group(1) if m else None

def extract_vehicle_from_estimate(text: str) -> str:
    m = re.search(r'VEHICLE\s+([\s\S]+?)\s*VIN:', text, re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"

# ===== New LLM-based photo type classification (no text extraction) =====
def detect_photo_type(blob: bytes) -> str:
    png = to_png_bytes(blob)
    if not png:
        return "other"
    base64_image = base64.b64encode(png).decode('utf-8')
    try:
        response = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[
                {"role": "system", "content": "Classify this photo based on visual content only, without extracting or reading any text. Types: 'odometer' (dashboard gauge cluster), 'vin' (vehicle identification label or plate), 'license plate' (car license plate), 'exterior corner' (vehicle exterior showing corners/bumpers), or 'other'. Return only the type."},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=20,
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.warning(f"LLM photo classify failed: {e}")
        return "other"

# ===== Detect photo presence =====
def detect_photo_presence(image_blobs: List[Tuple[str, bytes]]) -> Dict[str, bool]:
    presence = {"odometer": False, "vin": False, "license plate": False, "four corners": False}
    corner_count = 0
    for _, b in image_blobs:
        photo_type = detect_photo_type(b)
        if photo_type == 'odometer':
            presence["odometer"] = True
        elif photo_type == 'vin':
            presence["vin"] = True
        elif photo_type == 'license plate':
            presence["license plate"] = True
        elif photo_type == 'exterior corner':
            corner_count += 1
    presence["four corners"] = corner_count >= 2
    return presence

# ===== Visual confirmation without extraction =====
def confirm_vin_match(blob: bytes, vin_est: str) -> str:
    png = to_png_bytes(blob)
    if not png:
        return "NO PHOTO"
    base64_image = base64.b64encode(png).decode('utf-8')
    prompt = f"Visually inspect if the label in this photo shows a VIN that appears to match the length and format of a standard VIN. Do not read or extract the actual VIN. Answer 'MATCH' if it looks consistent with a VIN label, 'MISMATCH' if not, or 'NO VIN VISIBLE'."
    try:
        response = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM VIN confirm failed: {e}")
        return "ERROR"

def confirm_odometer_match(blob: bytes, odo_est: str) -> str:
    png = to_png_bytes(blob)
    if not png:
        return "NO PHOTO"
    base64_image = base64.b64encode(png).decode('utf-8')
    prompt = f"Visually inspect if the dashboard in this photo shows an odometer reading that appears to be a 6-digit number. Do not read or extract the actual number. Answer 'MATCH' if it looks consistent, 'MISMATCH' if not, or 'NO ODOMETER VISIBLE'."
    try:
        response = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM odo confirm failed: {e}")
        return "ERROR"

def confirm_vehicle_consistency(blob: bytes, vehicle_desc: str) -> str:
    png = to_png_bytes(blob)
    if not png:
        return "NO PHOTO"
    base64_image = base64.b64encode(png).decode('utf-8')
    prompt = f"Visually check if the vehicle or part shown matches a 2002 Chevrolet Silverado truck dashboard or label. Do not read text. Answer 'CONSISTENT', 'INCONSISTENT', or 'NO VEHICLE VISIBLE'."
    try:
        response = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"LLM vehicle confirm failed: {e}")
        return "ERROR"

# ===== Estimate items extraction =====
def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items = []
    lines = text.split('\n')
    current_section = ""
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\s+[A-Z]+\s+[A-Z]+$', line):
            current_section = line
            continue
        if re.match(r'^\d+\s', line):
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 3:
                item = {
                    "line": parts[0],
                    "oper": parts[1] if len(parts) > 1 else "",
                    "desc": parts[2] if len(parts) > 2 else "",
                    "part_num": parts[3] if len(parts) > 3 else "",
                    "qty": parts[4] if len(parts) > 4 else "",
                    "price": parts[5] if len(parts) > 5 else "",
                    "labor": parts[6] if len(parts) > 6 else "",
                    "paint": parts[7] if len(parts) > 7 else ""
                }
                items.append(item)
    return items

def extract_estimate_items_llm(text: str) -> List[Dict[str, str]]:
    prompt = f"""
Extract repair items from this estimate text. For each, use keys: 'line', 'oper', 'desc', 'part_num', 'qty', 'price', 'labor', 'paint'. Stick to text; no inventions. Output JSON array.
Text: {text[:4000]}
"""
    try:
        rsp = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0
        )
        output = rsp.choices[0].message.content
        return json.loads(output)
    except Exception as e:
        logger.warning(f"LLM estimate extract failed: {e}")
        return []

# ===== Parse labor and tax =====
def parse_labor_rates(text: str) -> Dict[str, str]:
    rates = {}
    labels = ["Body", "Paint", "Mechanical", "Structural"]
    for lbl in labels:
        m = re.search(rf"{lbl}\s*Labor[^\n]*?\$\s*(\d+\.\d+)\s*/hr", text, re.I)
        if m:
            rates[lbl] = f"${m.group(1)}/hr"
    return rates

def parse_tax_rate(text: str) -> Optional[str]:
    m = re.search(r'tax[^\n]*?(\d+\.\d+)%', text, re.I)
    return m.group(1) + "%" if m else None

# ===== Compare estimate with photos (visual description only) =====
def compare_estimate_with_photos(est_items: List[Dict[str, str]], images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = f"""
Visually describe any vehicle damage or parts in these photos without reading or extracting text. Then, for each estimate item, determine if the visible elements support the repair (YES/NO) with a note based on visual match only. Items: {json.dumps(est_items)}
Output JSON: {{"per_item": [{"item_desc": str, "photo_evidence": bool, "note": str}], "not_in_photos": [str], "extra_damage_in_photos": [str], "overall": str}}
"""
    try:
        rsp = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role": "system", "content": "Analyze visually only."}, {"role": "user", "content": [{"type": "text", "text": prompt}] + images_for_vision}],
            max_tokens=600,
            temperature=0
        )
        return json.loads(rsp.choices[0].message.content)
    except Exception as e:
        logger.warning(f"LLM compare failed: {e}")
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": "Comparison failed."}

# ===== Select images for vision =====
def select_images_for_vision(image_blobs: List[Tuple[str, bytes]], max_imgs: int) -> List[Tuple[str, bytes]]:
    return sorted(image_blobs, key=lambda x: len(x[1]), reverse=True)[:max_imgs]

# ===== Build brief summary =====
def build_brief_consistency_summary(consistency: Dict, est_items: List) -> str:
    if not consistency.get("per_item"):
        return "No items to compare."
    supported = sum(1 for it in consistency["per_item"] if it.get("photo_evidence"))
    total = len(est_items)
    return f"{supported}/{total} estimate items show visible support in photos; {total - supported} lack visible evidence."

# ===== Check labor and tax score =====
def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    if not _page_has_any_labor_rate(text):
        adj -= 50
    if not _page_has_tax(text):
        adj -= 25
    # Add client_rules logic if needed
    return adj

# ===== PDF helpers =====
def pdf_add_section_title(pdf, title):
    pdf.set_font_size(10)
    pdf.set_font("DejaVu", style="B")
    pdf.multi_cell(0, 6, title)
    pdf.set_font("DejaVu", style="")

def pdf_kv(pdf, key, value):
    pdf.multi_cell(0, 6, f"{key}: {value}")

# ===== Main endpoint (renamed to /vision-review) =====
@app.post("/vision-review")
async def process(request: Request):
    t0 = t0_start()
    data = await request.json()
    # Assume data has 'pdfs' (base64 list for estimate PDFs), 'images' (base64 list for photos), 'file_number', etc.
    pdf_raws = [base64.b64decode(p) for p in data.get("pdfs", [])]  # Estimate PDFs
    image_blobs = [(f"img{i}", base64.b64decode(img)) for i, img in enumerate(data.get("images", []))]  # Photos
    file_number = data.get("file_number", "8154702-0917-7")
    ia_company = data.get("ia_company", "SCA")
    appraiser_id = data.get("appraiser_id", "GRR")
    claim_number = extract_claim_from_estimate("") or data.get("claim_number", "RGBC2411013_V3")  # Will be extracted later
    vehicle_desc = data.get("vehicle_desc", "Used 2002 Chevrolet Silverado 2500 HD Pickup 3/4 Ton Extended Cab...")
    client_rules = data.get("client_rules", "")

    # ===== Build combined_text from estimate PDFs =====
    combined_text = ""
    for raw_pdf in pdf_raws:
        text = pdftotext_extract(raw_pdf, 1, 999)
        if text:
            combined_text += text + "\n"
        else:
            combined_text += ocr_pdf_text_caps(raw_pdf, MAX_TEXT_PAGES) + "\n"

    # ===== Extract from estimate =====
    vin_est = extract_vin_from_estimate(combined_text)
    odo_est = extract_odometer_from_estimate(combined_text)
    claim_est = extract_claim_from_estimate(combined_text)
    vehicle_desc = extract_vehicle_from_estimate(combined_text) or vehicle_desc

    # ===== Photo presence =====
    presence = detect_photo_presence(image_blobs)
    missing_photos = [p for p in ["odometer", "vin", "license plate", "four corners"] if not presence.get(p, False)]

    # ===== Visual comparisons =====
    vin_match = "NO VIN PHOTO"
    odo_match = "NO ODOMETER PHOTO"
    vehicle_consist = "NO PHOTO"
    for _, b in image_blobs:
        if detect_photo_type(b) == 'vin' and vin_est:
            vin_match = confirm_vin_match(b, vin_est)
        if detect_photo_type(b) == 'odometer' and odo_est:
            odo_match = confirm_odometer_match(b, odo_est)
        consist = confirm_vehicle_consistency(b, vehicle_desc)
        if consist == "CONSISTENT":
            vehicle_consist = "CONSISTENT"

    claim_match = "NOT VISIBLE"

    vin_verify_status = vin_match if vin_match != "NO VIN PHOTO" else "VIN PHOTO NOT FOUND"
    vin_final_for_report = vin_est or "N/A"

    # ===== Estimate items =====
    est_items = extract_estimate_items(combined_text)
    if not est_items and pdf_raws and not nearly_out_of_time(t0, 8):
        for raw_pdf in pdf_raws:
            extra_txt = await asyncio.to_thread(ocr_pdf_items_wide_scan, raw_pdf)
            if extra_txt:
                combined_text += "\n" + extra_txt
        est_items = extract_estimate_items(combined_text)
    if not est_items and not nearly_out_of_time(t0, 8):
        est_items = extract_estimate_items_llm(combined_text)

    # ===== Vision compare =====
    max_imgs = 4 if nearly_out_of_time(t0, 12) else MAX_VISION_IMGS
    chosen_images = select_images_for_vision(image_blobs, max_imgs=max_imgs)
    images_for_vision = []
    for _, b in chosen_images:
        png = to_png_bytes(b) or b
        images_for_vision.append({"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png).decode("utf-8")}})
    consistency = compare_estimate_with_photos(est_items, images_for_vision) if images_for_vision else {
        "per_item":[],"not_in_photos":[],"extra_damage_in_photos":[],"overall":"No photos available for comparison."
    }

    # ===== Labor & Tax =====
    labor_rates = parse_labor_rates(combined_text)
    tax_rate = parse_tax_rate(combined_text)
    labor_line = "None detected"
    if labor_rates:
        parts = [f"{key} {labor_rates[key]}" for key in ["Body","Paint","Mechanical","Structural"] if key in labor_rates]
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
    if combined_text:
        user_parts.append({"type":"text","text":combined_text})

    max_tokens_summary = 450 if nearly_out_of_time(t0, 10) else 600
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

    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
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

    # ===== PDF =====
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
    pdf.multi_cell(0, 6, f"Claim #: {claim_est}")
    pdf.multi_cell(0, 6, f"VIN: {vin_final_for_report}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, gpt_output_clean)

    pdf.ln(4)
    pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")
    brief = build_brief_consistency_summary(consistency, est_items)
    pdf.multi_cell(0, 6, f"Brief Summary: {brief}")

    if consistency.get("per_item"):
        for it in consistency["per_item"][:40]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            line = f"- {it.get('item_desc','unspecified')} → Photo: {ev}; {it.get('note','')}"
            pdf.multi_cell(0, 6, line)
    else:
        pdf.multi_cell(0, 6, "Per-item comparison unavailable.")

    if consistency.get("not_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Items Estimated but Not Evident in Photos")
        for raw in consistency["not_in_photos"][:20]: pdf.multi_cell(0, 6, f"- {raw}")

    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:20]: pdf.multi_cell(0, 6, f"- {d}")

    pdf.ln(2); pdf_kv(pdf, "Consistency Overall", consistency.get("overall", ""))

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # ===== EMAIL =====
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_est}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        email_body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_est}
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
        "claim_number": claim_est,
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















