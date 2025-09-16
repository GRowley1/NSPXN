from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from typing import List, Tuple, Optional, Dict, Any
import os
import re
import io
import base64
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
from openai import OpenAI

# ======================= SPEED TUNABLES (env overridable) =======================
PDF_OCR_DPI_EST = int(os.getenv("PDF_OCR_DPI_EST", "175"))   # estimate 1st page OCR
PDF_OCR_DPI_TXT = int(os.getenv("PDF_OCR_DPI_TXT", "160"))   # other text pages
PDF_OCR_DPI_PH  = int(os.getenv("PDF_OCR_DPI_PH",  "150"))   # photo harvest
MAX_TEXT_PAGES  = int(os.getenv("MAX_TEXT_PAGES",  "3"))     # max text OCR pages (incl. first)
MAX_PHOTO_PAGES = int(os.getenv("MAX_PHOTO_PAGES", "8"))     # max photo-like pages harvested from PDFs
MAX_VISION_IMGS = int(os.getenv("MAX_VISION_IMGS", "8"))     # max images sent to vision
THREADS         = int(os.getenv("OCR_THREADS",     "4"))     # OCR thread pool size
OAI_MODEL       = os.getenv("OAI_MODEL", "gpt-4o-mini")      # faster default model
OAI_TIMEOUT_S   = float(os.getenv("OAI_TIMEOUT_S", "25"))    # OpenAI timeout seconds
FAST_MODE_DEFAULT = os.getenv("FAST_MODE_DEFAULT", "1") == "1"  # default fast mode

# ======================= PDF storage (same as original) =======================
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

# ======================= OCR helpers =======================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = img.filter(ImageFilter.MedianFilter(3))
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
                page.save(buf, format="JPEG", quality=72)
                score = corner_hits * 10 + var
                out.append((f"pdf-p{i}.jpg", buf.getvalue(), score))
                used += 1
                if used >= max_pages:
                    break
    except Exception as e:
        logger.warning(f"harvest_photos_from_pdf error: {e}")
    return out

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
def extract_claim_from_text(text: str) -> Optional[str]:
    for pat in [
        r"(?:^|\s)(?:Claim\s*(?:#|No\.?|Number)[:\s]*)\s*([A-Za-z0-9\-]+)",
        r"(?:^|\s)Claim\s*[:#]\s*([A-Za-z0-9\-]+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    label_block = re.findall(r"(?:^|\n).{0,40}VIN[:\s\-]*([A-HJ-NPR-Z0-9]{10,20}).*", text, re.IGNORECASE)
    if label_block:
        vin = best_vin_candidate(label_block)
        if vin: return vin
    candidates = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, re.IGNORECASE)
    return best_vin_candidate(candidates)

def extract_vehicle_from_text(text: str) -> Optional[str]:
    m1 = re.search(r"\b(19\d{2}|20\d{2})\s+([A-Za-z]{3,})\s+([A-Za-z0-9\-]{2,})", text)
    m2 = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    if m1:
        year, make, model = m1.groups()
        miles = m2.group(1) if m2 else "Mileage unknown"
        return f"{year} {make} {model}, {miles} miles"
    return None

# ======================= Photo parsing & requirements =======================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = preprocess_image(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    found: List[str] = []
    for name, blob in image_blobs[:12]:
        try:
            base = Image.open(io.BytesIO(blob))
            for r in (0, 90, 180, 270):
                img = base.rotate(r, expand=True)
                ocr = pytesseract.image_to_string(preprocess_image(img), lang="eng", config="--psm 7")
                cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", ocr.upper())
                if cands: found.extend(cands)
        except Exception as e:
            logger.warning(f"VIN photo OCR error ({name}): {e}")
    return best_vin_candidate(found)

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs[:12]:
        try:
            img = Image.open(io.BytesIO(blob))
            ocr = pytesseract.image_to_string(preprocess_image(img), lang="eng", config="--psm 6")
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", ocr, re.IGNORECASE)
            if m: return m.group(1)
        except Exception as e:
            logger.warning(f"Odometer OCR ({name}): {e}")
    return None

def check_required_photos(image_blobs: List[Tuple[str, bytes]], _ignored_text: str = "") -> List[str]:
    """
    Vision-only verification of required photos.
    Required: four corners, odometer, vin, license plate
    """
    required = ["four corners", "odometer", "vin", "license plate"]
    present = set()

    ext_like = 0
    corner_label_hits = 0

    for name, blob in image_blobs[:16]:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = preprocess_image(img)
            ocr = pytesseract.image_to_string(proc, lang="eng", config="--psm 6")

            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr, re.IGNORECASE):
                present.add("vin")
            if re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(mi|miles|km)\b", ocr, re.IGNORECASE):
                present.add("odometer")
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", ocr, re.IGNORECASE):
                present.add("license plate")

            if _image_is_exterior_wide(img):
                ext_like += 1
            corner_label_hits += count_corner_labels(ocr)
        except Exception as e:
            logger.warning(f"Image parse error {name}: {e}")

    if ext_like >= 2 or corner_label_hits >= 3:
        present.add("four corners")

    missing = [p for p in required if p not in present]
    return missing

# ======================= Labor/tax (unchanged logic) =======================
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

# ======================= Estimate parsing =======================
PANELS = ["bumper","fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
          "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
          "wheel","tire","pillar","garnish","molding","fog lamp","reinforcement","cover"]
OPS = ["replace","repair","refinish","r&i","r & i","align","blend","calibrate"]

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for line in text.splitlines():
        l = line.strip().lower()
        if not l or len(l) < 6: continue
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

def extract_estimate_items_llm(text: str) -> List[Dict[str, str]]:
    """
    LLM fallback: extract (op, part, side) items from raw estimate text.
    Returns a list of {op, part, side, raw}
    """
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "op": {"type": "string"},
                "part": {"type": "string"},
                "side": {"type": "string"},
                "raw": {"type": "string"}
            },
            "required": ["op","part","side","raw"]
        }
    }
    sys = "Extract concise estimate line items (operation, part, and side) from the text. Return STRICT JSON only per this schema: " + json.dumps(schema)
    try:
        rsp = client_fast.chat.completions.create(
            model=os.getenv("OAI_MODEL","gpt-4o-mini"),
            messages=[{"role":"system","content":sys},{"role":"user","content":text[:18000]}],
            temperature=0,
            max_tokens=700,
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned = []
            for it in data[:80]:
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

# ======================= Vision compare (limit images to speed) =======================
def select_images_for_vision(image_blobs: List[Tuple[str, bytes]]) -> List[Tuple[str, bytes]]:
    scored = []
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = preprocess_image(img)
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            text = pytesseract.image_to_string(proc, lang="eng")
            score = var + (15 if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", text) else 0) + (8 if count_corner_labels(text) else 0)
            scored.append((score, name, blob))
        except Exception:
            continue
    scored.sort(reverse=True)
    return [(n, b) for _, n, b in scored[:MAX_VISION_IMGS]]

def compare_estimate_with_photos(items: List[Dict[str, str]],
                                 images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema = {
        "type":"object",
        "properties":{
            "per_item":{"type":"array","items":{
                "type":"object",
                "properties":{
                    "op":{"type":"string"},"part":{"type":"string"},"side":{"type":"string"},
                    "photo_evidence":{"type":"boolean"},"confidence":{"type":"number"},"note":{"type":"string"}
                },
                "required":["op","part","side","photo_evidence","confidence","note"]
            }},
            "not_in_photos":{"type":"array","items":{"type":"string"}},
            "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
            "overall":{"type":"string"}
        },
        "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]
    }
    system = ("You are an auto-damage visual auditor. Given estimate line items and vehicle photos, "
              "decide for EACH item whether visible photo evidence exists. Hidden ops may not be visible. "
              "Return STRICT JSON ONLY per this schema:\n" + json.dumps(schema))
    user_parts: List[Dict[str, Any]] = [{"type":"text","text":"Estimate items:\n"+json.dumps(items, ensure_ascii=False)}]
    user_parts.extend(images_for_vision)
    try:
        rsp = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user_parts}],
            max_tokens=900, temperature=0
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

# ======================= PDF helpers (UNCHANGED OUTPUT FORMAT) =======================
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12); pdf.cell(0, 8, txt=title, ln=True); pdf.set_font_size(10)

def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10); pdf.multi_cell(0, 6, f"{key}: {val}")

# ======================= Routes =======================
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(request: Request):
    """
    Robust handler that accepts both:
      - multipart/form-data  (preferred)
      - application/json     (fallback: files as base64 or URLs)
    Keeps the rest of the pipeline identical.
    """
    # ---------- Gather inputs (supports multipart or JSON) ----------
    ctype = request.headers.get("content-type", "").lower()
    files_all: List[Tuple[str, bytes]] = []
    client_rules = ""
    file_number = ""
    ia_company = ""
    appraiser_id = ""
    fast = None

    try:
        if "multipart/form-data" in ctype:
            form = await request.form()

            # Text fields
            client_rules = (form.get("client_rules") or "").strip()
            file_number  = (form.get("file_number")  or "").strip()
            ia_company   = (form.get("ia_company")   or "").strip()
            appraiser_id = (form.get("appraiser_id") or "").strip()
            fast         = form.get("fast")

            # Files – accept multiple keys commonly used by UIs
            for key in ("files", "files[]", "estimate", "photos", "guidelines"):
                for f in form.getlist(key):
                    if hasattr(f, "filename"):
                        raw = await f.read()
                        files_all.append(((f.filename or "upload").lower(), raw))

        elif "application/json" in ctype:
            payload = await request.json()

            # Text fields
            client_rules = (payload.get("client_rules") or "").strip()
            file_number  = (payload.get("file_number")  or "").strip()
            ia_company   = (payload.get("ia_company")   or "").strip()
            appraiser_id = (payload.get("appraiser_id") or "").strip()
            fast         = payload.get("fast")

            # Files: base64 or URL
            for item in (payload.get("files") or []):
                # b64 form: {"filename":"Est.pdf","b64":"..."}
                if "b64" in item:
                    try:
                        b = base64.b64decode(item["b64"])
                        fname = (item.get("filename") or "upload").lower()
                        files_all.append((fname, b))
                    except Exception as e:
                        logger.warning(f"Bad base64 file: {e}")

                # URL form: {"url":"https://.../file.pdf","filename":"Est.pdf"}
                elif "url" in item:
                    try:
                        import httpx
                        r = httpx.get(item["url"], timeout=15)
                        r.raise_for_status()
                        fname = (item.get("filename")
                                 or os.path.basename(item["url"]) or "download").lower()
                        files_all.append((fname, r.content))
                    except Exception as e:
                        logger.warning(f"Fetch failed: {item.get('url')}: {e}")

        else:
            return JSONResponse(
                status_code=415,
                content={
                    "error": "Unsupported Content-Type. Use multipart/form-data (recommended) or application/json."
                },
            )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Bad request body: {type(e).__name__}: {e}"})

    # ---------- Minimal validation (we keep your behavior) ----------
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error": "Appraiser ID is required."})
    if not files_all:
        return JSONResponse(
            status_code=400,
            content={"error": "No files uploaded. Send at least one estimate/photo/guideline file."},
        )

    # ---------- Existing pipeline below (unchanged) ----------
    is_fast = FAST_MODE_DEFAULT if fast is None else (str(fast) != "0")

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        text_chunks: List[str] = []
        image_blobs: List[Tuple[str, bytes]] = []
        pdf_photo_candidates: List[Tuple[str, bytes, float]] = []

        async def handle_file(name: str, raw: bytes):
            if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_blobs.append((name, raw))
            elif name.endswith(".pdf"):
                first_txt = await loop.run_in_executor(pool, ocr_pdf_first_page, raw)
                if first_txt:
                    text_chunks.append(first_txt)
                extra_pages = max(0, MAX_TEXT_PAGES - 1)
                if extra_pages > 0:
                    more_txt = await loop.run_in_executor(pool, ocr_pdf_text_caps, raw, extra_pages)
                    if more_txt:
                        text_chunks.append(more_txt)
                caps = MAX_PHOTO_PAGES if is_fast else MAX_PHOTO_PAGES * 2
                cand = await loop.run_in_executor(pool, harvest_photos_from_pdf, raw, caps)
                pdf_photo_candidates.extend(cand)
            elif name.endswith(".docx"):
                try:
                    doc = Document(io.BytesIO(raw))
                    text_chunks.append("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
                except Exception:
                    pass
            elif name.endswith(".txt"):
                try:
                    text_chunks.append(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    pass

        await asyncio.gather(*(handle_file(n, b) for n, b in files_all))

    if pdf_photo_candidates:
        pdf_photo_candidates.sort(key=lambda t: t[2], reverse=True)
        keep = pdf_photo_candidates[: (MAX_PHOTO_PAGES if is_fast else MAX_PHOTO_PAGES * 2)]
        for n, data, _ in keep:
            image_blobs.append((n, data))

    combined_text = "\n".join(text_chunks)

    # ===== Vision-only required photos check =====
    missing_photos = check_required_photos(image_blobs, combined_text)

    # ===== VIN handling (estimate primary; photo verification only) =====
    vin_est = extract_vin_from_text(combined_text)
    vin_photos = extract_vin_from_photos(image_blobs)
    vin_final = vin_est or "N/A"

    vin_match_status = "UNVERIFIED"
    if vin_est and vin_photos:
        vin_match_status = "MATCH" if normalize_vin(vin_est) == normalize_vin(vin_photos) else "MISMATCH"

    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    claim_number = extract_claim_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    # ===== Estimate items + fallback LLM extraction =====
    est_items = extract_estimate_items(combined_text)
    if not est_items:
        est_items = extract_estimate_items_llm(combined_text)

    chosen_images = select_images_for_vision(image_blobs)
    images_for_vision = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode("utf-8")},
        }
        for _, b in chosen_images
    ]

    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    # ===== AI narrative & scoring (unchanged) =====
    photo_line = "None" if not missing_photos else ", ".join(missing_photos)
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

    user_parts: List[Dict[str, Any]] = []
    if combined_text:
        user_parts.append({"type": "text", "text": combined_text})

    try:
        rsp = client_fast.chat.completions.create(
            model=OAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_parts[:1]}],
            max_tokens=600,
            temperature=0,
        )
        gpt_output = rsp.choices[0].message.content or "⚠️ GPT returned no output."
    except Exception as e:
        gpt_output = f"⚠️ AI review failed: {type(e).__name__}: {e}"

    score_ai = None
    for pat in [
        r"Total\s*Evaluation\s*[:\-]?\s*(\d{1,3})\s*%?",
        r"Final\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?",
        r"Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?",
    ]:
        m = re.search(pat, gpt_output, re.IGNORECASE)
        if m:
            score_ai = int(m.group(1))
            break

    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    computed = max(0, 100 + labor_tax_adj + photo_adj)
    authoritative_score = max(0, min(100, score_ai if score_ai is not None else computed))

    gpt_output_clean = re.sub(
        r"(?im)^(?:Final\s*Score|Compliance\s*Score|Total\s*Evaluation)\s*[:\-]?\s*\d{1,3}\s*%.*$",
        "",
        gpt_output,
    ).strip()
    gpt_output_clean += f"\n\nVIN verification (estimate vs photo): {vin_match_status}"
    gpt_output_clean += f"\nRequired photo verification (vision): {photo_line}"

    # ===== PDF & Email (unchanged structure) =====
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
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    if odo_photos:
        pdf.multi_cell(0, 6, f"Odometer (from photos): {odo_photos}")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, gpt_output_clean)

    pdf.ln(4)
    pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")
    if consistency.get("per_item"):
        for it in consistency["per_item"][:40]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            try:
                conf = float(it.get("confidence", 0))
            except Exception:
                conf = 0.0
            conf_txt = f"{round(conf * 100)}%"
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

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # EMAIL (unchanged)
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
        "vin": vin_final,
        "score": f"{authoritative_score}%",
        "consistency_review": consistency,
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})










