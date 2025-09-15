from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, base64, json, logging, math, datetime, hashlib, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat
from openai import OpenAI

# ---------------- Basic setup ----------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
logging.basicConfig(level=logging.DEBUG, filename="app.log", filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY not set")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nspxn.com","https://www.nspxn.com","http://nspxn.com",
                   "http://www.nspxn.com","https://nspxn.onrender.com"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------------- OCR helpers ----------------
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(3))
    return ImageOps.autocontrast(img)

def ocr_text_fast(img: Image.Image, psm: int = 6) -> str:
    try:
        return pytesseract.image_to_string(preprocess_image(img), lang="eng",
                                           config=f"--psm {psm} --oem 1")
    except Exception as e:
        logger.warning(f"OCR fast error: {e}"); return ""

def extract_text_from_pdf(pdf_io: io.BytesIO, max_ocr_pages: int = 8, dpi: int = 140) -> str:
    try:
        pdf_io.seek(0); pages = convert_from_bytes(pdf_io.read(), dpi=dpi)
        out = []
        for i, p in enumerate(pages, 1):
            if i > max_ocr_pages: break
            t = ocr_text_fast(p, psm=6)
            if len(t.strip()) < 20: t = ocr_text_fast(p, psm=3)
            if t.strip(): out.append(f"[Page {i}]\n{t}")
        return "\n".join(out)
    except Exception as e:
        logger.error(f"OCR error: {e}"); return ""

def extract_text_from_docx(file_like: io.BytesIO) -> str:
    try:
        doc = Document(file_like); return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.error(f"DOCX read error: {e}"); return ""

def extract_text_from_pdf_embedded(pdf_bytes: bytes) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:
        logger.debug(f"Embedded text extraction failed: {e}"); return ""

# ---------------- Photo harvest for non-estimates ----------------
def _page_var(img: Image.Image) -> float:
    return ImageStat.Stat(img.convert("L")).var[0]

def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int = 20, dpi: int = 135) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
        for i, page in enumerate(pages, 1):
            if _page_var(page) > 110:
                buf = io.BytesIO()
                page.convert("RGB").save(buf, format="JPEG", quality=82, optimize=True)
                out.append((f"pdf-photo-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        logger.warning(f"harvest_photos_from_pdf error: {e}")
    return out

# ---------------- VIN helpers ----------------
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
def normalize_vin(s: str) -> Optional[str]:
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", s.strip().upper()).replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s

_translit = {**{str(i): i for i in range(10)}, **dict(
    A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,
    S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def vin_checksum_ok(v: str) -> bool:
    if len(v) != 17: return False
    try:
        total = sum(_translit[ch]*_weights[i] for i, ch in enumerate(v))
        check = "X" if total % 11 == 10 else str(total % 11)
        return v[8] == check
    except Exception:
        return False

def best_vin_candidate(cands: List[str]) -> Optional[str]:
    for c in cands:
        vv = normalize_vin(c)
        if vv and vin_checksum_ok(vv): return vv
    return None

VIN_LABEL = re.compile(r'(?i)\bV[\W_]*I[\W_]*N\b')
VIN_PHRASE = re.compile(r'(?i)\bVehicle\s*Identification\s*Number\b')
VIN_SEP_SEQ = re.compile(r'(?i)((?:[A-HJ-NPR-Z0-9][\s\.\-–—:_]){16}[A-HJ-NPR-Z0-9])')

def _extract_vin_near_positions(text: str, positions: List[int], radius: int = 240) -> Optional[str]:
    for pos in positions:
        window = text[pos: pos + radius]
        for m in VIN_SEP_SEQ.finditer(window):
            vin = normalize_vin(m.group(1))
            if vin and vin_checksum_ok(vin): return vin
        vin = best_vin_candidate(re.findall(r'([A-HJ-NPR-Z0-9]{17})', window))
        if vin: return vin
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    if not text: return None
    pos = [m.end() for m in VIN_LABEL.finditer(text)] + [m.end() for m in VIN_PHRASE.finditer(text)]
    vin = _extract_vin_near_positions(text, pos, radius=240)
    if vin: return vin
    for m in VIN_SEP_SEQ.finditer(text):
        vin = normalize_vin(m.group(1)); if vin and vin_checksum_ok(vin): return vin
    return best_vin_candidate(re.findall(r'\b([A-HJ-NPR-Z0-9]{17})\b', text))

def extract_vin_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 170) -> Optional[str]:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max(1, pages_to_scan)]
        for img in pages:
            t = ocr_text_fast(img, psm=6)
            v = extract_vin_from_text(t)
            if v: return v
    except Exception as e:
        logger.warning(f"VIN first-pages OCR error: {e}")
    return None

# ---------------- Claim # extraction ----------------
CLAIM_AFTER_LABEL = re.compile(r'(?is)\bclaim\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})')
ALT_CLAIM_LABELS = [
    re.compile(r'(?is)\bloss\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
    re.compile(r'(?is)\bfile\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
    re.compile(r'(?is)\bref(?:erence)?\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
    re.compile(r'(?is)\bassignment\b\W{0,6}(?:#:?|no\.?|number)?\W{0,6}([A-Z0-9][A-Z0-9\-/\.]{2,60})'),
]
_CLAIM_BLACKLIST = {"SERVICE","SERVICES","PHONE","EMAIL","FAX","TOTAL","POLICY"}
def _clean_claim(c: str) -> str:
    c = c.strip().strip(':').strip('.').strip('-')
    c = c.replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
    c = c.replace("_",""); c = re.sub(r'\s+','', c); c = re.sub(r'(?:V\d+)$','', c, flags=re.I)
    return c
def _valid_claim_candidate(c: str) -> bool:
    return bool(c and len(c)>=3 and re.search(r'\d',c) and c.upper() not in _CLAIM_BLACKLIST)
def extract_claim_from_text(text: str) -> Optional[str]:
    if not text: return None
    for m in CLAIM_AFTER_LABEL.finditer(text):
        c = _clean_claim(m.group(1)); if _valid_claim_candidate(c): return c
    for pat in ALT_CLAIM_LABELS:
        for m in pat.finditer(text):
            c = _clean_claim(m.group(1)); if _valid_claim_candidate(c): return c
    return None
def extract_claim_from_pdf_first_pages(pdf_bytes: bytes, pages_to_scan: int = 4, dpi: int = 170) -> Optional[str]:
    try:
        for img in convert_from_bytes(pdf_bytes, dpi=dpi)[:max(1,pages_to_scan)]:
            t = ocr_text_fast(img, psm=6); c = extract_claim_from_text(t)
            if c: return c
    except Exception as e:
        logger.warning(f"Claim first-pages OCR error: {e}")
    return None

# ---------------- Vehicle / parts / taxes ----------------
MAKE_CANON = {
    "nessan":"Nissan","nisaan":"Nissan","nissan":"Nissan",
    "toy0ta":"Toyota","toyota":"Toyota",
    "chevroler":"Chevrolet","cheverolet":"Chevrolet","chevrolet":"Chevrolet",
    "chev":"Chevrolet","chev.":"Chevrolet",
    "honda":"Honda","ford":"Ford","hyundai":"Hyundai","kia":"Kia","mazda":"Mazda",
    "subaru":"Subaru","mercedes":"Mercedes","mercedes-benz":"Mercedes-Benz","bmw":"BMW","audi":"Audi",
    "volkswagen":"Volkswagen","vw":"Volkswagen","jeep":"Jeep","ram":"Ram","dodge":"Dodge","gmc":"GMC",
    "lexus":"Lexus","infiniti":"Infiniti","acura":"Acura","cadillac":"Cadillac","lincoln":"Lincoln",
    "buick":"Buick","volvo":"Volvo","porsche":"Porsche","mitsubishi":"Mitsubishi","mini":"Mini",
}
MAKE_RX = re.compile(r'\b(' + '|'.join(re.escape(k) for k in MAKE_CANON.keys()) + r')\b', re.I)

def normalize_vehicle_str(s: str) -> str:
    if not s: return s
    out = s
    for k,v in MAKE_CANON.items():
        out = re.sub(rf'\b{re.escape(k)}\b', v, out, flags=re.I)
    return re.sub(r'\s{2,}', ' ', out).strip()

def extract_vehicle_from_text(text: str) -> Optional[str]:
    """Pick a Year next to a real make; ignore junk like 'Estimate Provided'."""
    if not text: return None
    tokens = text.splitlines()
    best = None; best_dist = 9999
    for ln in tokens:
        y = re.search(r'\b(19|20)\d{2}\b', ln)
        m = MAKE_RX.search(ln)
        if not y or not m: continue
        # avoid lines like '2025 Estimate Provided'
        after = ln[m.end():m.end()+20].lower()
        if after.strip().startswith("estimate"): continue
        dist = abs(m.start() - y.start())
        if dist < best_dist:
            best_dist = dist
            year = re.search(r'(19|20)\d{2}', y.group(0)).group(0)
            make_raw = m.group(1)
            # model is next token(s)
            tail = ln[m.end():].strip()
            model = (tail.split()[0] if tail else "").strip(",.;:-")
            if not model or len(model) < 2: model = "Vehicle"
            vehicle = f"{year} {make_raw} {model}"
            best = normalize_vehicle_str(vehicle)
    # mileage
    miles = None
    mm = re.search(r'(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)', text, re.I)
    if mm: miles = mm.group(1)
    if best and miles: return f"{best}, {miles} miles"
    return best

def parse_year_miles(text: str) -> Tuple[Optional[int], Optional[int]]:
    y = re.search(r'\b(19|20)\d{2}\b', text or "")
    year = int(y.group(0)) if y else None
    m = re.search(r'(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)', text or "", re.I)
    miles = int(m.group(1).replace(",", "")) if m else None
    return year, miles

def taxes_present(text: str) -> bool:
    if not text: return False
    if re.search(r'\bSales?\s*Tax\b', text, re.I): return True
    return re.search(r'tax[^\n]{0,80}(\d{1,3}\s*%|\$\s*\d+(?:\.\d{2})?)', text, re.I) is not None

OPS_TOK = re.compile(r'\b(REPL(?:ACE)?|R&R|R & R|R&I|R & I|REPAIR|REFINISH|PAINT)\b', re.I)
PANELS = ["bumper","fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp","combo lamp",
          "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support","wheel","tire",
          "pillar","garnish","molding","fog lamp","reinforcement","cover","finish panel"]
PANELS_U = [p.upper() for p in PANELS]
PART_FLAGS = r'(?:\bA/M\b|\bAFTER\s*MARKET\b|\bAFTERMARKET\b|\bLKQ\b|\bRECOND(?:ITIONED)?\b|\bCAPA\b|\bALT[-\s]*OE\b|\bREMAN(?:UFACTURED)?\b)'

def non_oem_used(text: str) -> bool:
    for line in (text or "").splitlines():
        L = line.strip().upper()
        if OPS_TOK.search(L) and re.search(PART_FLAGS, L, re.I) and any(p in L for p in PANELS_U): return True
    if re.search(r'parts\s+presented\s+are\s+OEM[-\s]*parts', text or "", re.I): return False
    return False

# ---------------- Required photos (VIN/ODO must be from photos if photos exist) ----------------
PLATE_RX = re.compile(r'\b([A-Z0-9]{1,3}[-\s]?[A-Z0-9]{3,4}|[A-Z0-9]{5,8})\b')

def _is_exterior_by_edges(img: Image.Image) -> bool:
    g = img.convert("L"); var = ImageStat.Stat(g).var[0]
    evar = ImageStat.Stat(g.filter(ImageFilter.FIND_EDGES)).var[0]
    return var > 140 and evar > 400

def _plate_ocr_variants(img: Image.Image) -> str:
    img = img.copy(); img.thumbnail((1600,1600)); g = img.convert("L")
    variants = [
        ImageEnhance.Contrast(g).enhance(1.8),
        ImageEnhance.Sharpness(g).enhance(1.8),
        ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3))),
        g.point(lambda p: 255 if p>170 else 0, "1").convert("L"),
        g.point(lambda p: 255 if p>190 else 0, "1").convert("L"),
    ]
    out = []
    for v in variants:
        for psm in (6,7,11):
            try:
                t = pytesseract.image_to_string(v, lang="eng", config=f"--psm {psm} --oem 1")
                if t: out.append(t)
            except Exception: pass
    return "\n".join(out)

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for _, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob)); img.thumbnail((1400,1400))
            t = ocr_text_fast(img, psm=7)
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", t, re.I)
            if m: return m.group(1)
        except Exception: pass
    return None

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    """Scan up to 40 images (no prefilter) with multiple variants/PSMs; return first checksum-valid VIN."""
    VIN_17 = re.compile(r'\b([A-HJ-NPR-Z0-9]{17})\b')
    VIN_SEP = re.compile(r'(?i)((?:[A-HJ-NPR-Z0-9][\s\.\-–—:_]){16}[A-HJ-NPR-Z0-9])')
    blobs = image_blobs[:40]
    def variants(im: Image.Image) -> List[Image.Image]:
        im = im.copy()
        if im.width < 2200:
            im = im.resize((2200, int(im.height*2200/im.width)), Image.LANCZOS)
        g = im.convert("L")
        return [
            ImageEnhance.Contrast(g).enhance(2.0),
            ImageEnhance.Sharpness(g).enhance(2.0),
            g.point(lambda p: 255 if p>180 else 0, "1").convert("L"),
            ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3))),
        ]
    def ocr_all(im: Image.Image) -> str:
        out = []
        for psm in (7,6,11):
            try: out.append(pytesseract.image_to_string(im, lang="eng", config=f"--psm {psm} --oem 1"))
            except Exception: pass
        return "\n".join([t for t in out if t])
    for _, blob in blobs:
        try:
            im = Image.open(io.BytesIO(blob))
            big = ""
            for v in variants(im): big += "\n" + ocr_all(v)
            big = big.upper()
            for m in VIN_SEP.finditer(big):
                vin = normalize_vin(m.group(1)); if vin and vin_checksum_ok(vin): return vin
            vin = best_vin_candidate(VIN_17.findall(big))
            if vin: return vin
        except Exception as e:
            logger.warning(f"VIN photo OCR error: {e}")
    return None

def check_required_photos(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> List[str]:
    required = ["four corners","odometer","vin","license plate"]
    present = set()
    have_photos = len(image_blobs) > 0

    # VIN/ODO must come from photos when photos exist
    vin_photo = extract_vin_from_photos(image_blobs) is not None
    odo_photo = extract_odometer_from_photos(image_blobs) is not None
    if have_photos:
        if vin_photo: present.add("vin")
        if odo_photo: present.add("odometer")
    else:
        if re.search(r'\bvin\b', ocr_text or "", re.I): present.add("vin")
        if re.search(r'\bodometer|mileage\b', ocr_text or "", re.I): present.add("odometer")

    # License plate via OCR on a subset
    subset = image_blobs if len(image_blobs) <= 24 else [image_blobs[i] for i in range(0, len(image_blobs), max(1, len(image_blobs)//24))][:24]
    for _, blob in subset:
        try:
            img = Image.open(io.BytesIO(blob))
            t = _plate_ocr_variants(img)
            if re.search(r'(license|registration)\s*plate', t, re.I) or PLATE_RX.search(t):
                present.add("license plate"); break
        except Exception: pass

    # Four corners heuristic
    ext_hits = 0
    for _, blob in image_blobs[:40]:
        try:
            img = Image.open(io.BytesIO(blob)); img.thumbnail((1600,1600))
            if _is_exterior_by_edges(img): ext_hits += 1
        except Exception: pass
    if ext_hits >= 3: present.add("four corners")

    return [p for p in required if p not in present]

# ---------------- Labor & tax scoring ----------------
def labor_rates_present_any(text: str) -> bool:
    if not text: return False
    # e.g., "Body Labor 5.1 hrs @ $ 58.00 /hr" or "$58.00/hr" or "58 per hour"
    rate_rx = re.compile(r'\$\s*\d{2,3}(?:\.\d{2})?\s*/?\s*hr|\d{2,3}(?:\.\d{2})?\s*(?:per\s*hour|/hr)', re.I)
    labels = ["Body","Paint","Mechanical","Structural","Frame","Refinish","Supplies"]
    for lbl in labels:
        block = re.search(rf"{lbl}[^\n]{{0,120}}", text, re.I)
        if block and rate_rx.search(block.group(0)): return True
    return rate_rx.search(text) is not None

def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    if not labor_rates_present_any(text): adj -= 50
    if re.search(r"tax\s*(required|must|utilize|apply)|\bSales?\s*Tax\b", client_rules or "", re.I):
        if not taxes_present(text): adj -= 25
    return adj

# ---------------- Estimate parsing & compare ----------------
OPS = ["replace","repair","refinish","r&i","r & i","align","blend","calibrate"]
def extract_estimate_items(text: str) -> List[Dict[str,str]]:
    items = []
    for line in (text or "").splitlines():
        l = line.strip().lower()
        if not l or len(l)<6: continue
        if any(op in l for op in OPS) and any(p in l for p in PANELS):
            side="unspecified"
            if "left" in l or re.search(r"\blh\b", l): side="left"
            if "right" in l or re.search(r"\brh\b", l): side="right"
            op = next((op for op in OPS if op in l), "unspecified")
            part = next((p for p in PANELS if p in l), "component")
            items.append({"op":op,"part":part,"side":side,"raw":line.strip()})
    uniq, seen = [], set()
    for it in items:
        key=(it["op"],it["part"],it["side"])
        if key not in seen: uniq.append(it); seen.add(key)
    return uniq

def compare_estimate_with_photos(items: List[Dict[str,str]], images_for_vision: List[Dict[str,Any]]) -> Dict[str,Any]:
    schema = {"type":"object","properties":{
        "per_item":{"type":"array","items":{"type":"object","properties":{
            "op":{"type":"string"},"part":{"type":"string"},"side":{"type":"string"},
            "photo_evidence":{"type":"boolean"},"confidence":{"type":"number"},"note":{"type":"string"}},
            "required":["op","part","side","photo_evidence","confidence","note"]}},
        "not_in_photos":{"type":"array","items":{"type":"string"}},
        "extra_damage_in_photos":{"type":"array","items":{"type":"string"}},
        "overall":{"type":"string"}}, "required":["per_item","not_in_photos","extra_damage_in_photos","overall"]}
    system = ("You are an auto-damage visual auditor. "
              "Given estimate line items and vehicle photos, decide for EACH item whether photo evidence exists. "
              "Return STRICT JSON only per schema: " + json.dumps(schema))
    user_parts = [{"type":"text","text":"Estimate items:\n"+json.dumps(items, ensure_ascii=False)}] + images_for_vision
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":user_parts}],
            max_tokens=220, temperature=0
        )
        txt = (rsp.choices[0].message.content or "").strip()
        txt = txt.removeprefix("```json").removesuffix("```").strip()
        return json.loads(txt)
    except Exception as e:
        logger.error(f"Vision compare JSON error: {type(e).__name__}: {e}")
        return {"per_item": [],"not_in_photos": [],"extra_damage_in_photos": [],"overall": "Comparison unavailable."}

# ---------------- PDF small helpers ----------------
def pdf_add_section_title(pdf: FPDF, title: str):
    pdf.set_font_size(12); pdf.cell(0, 8, txt=title, ln=True); pdf.set_font_size(10)
def pdf_kv(pdf: FPDF, key: str, val: str):
    pdf.set_font_size(10); pdf.multi_cell(0, 6, f"{key}: {val}")

# ---------------- API ----------------
@app.get("/")
async def root(): return {"status":"ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(...),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...)
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error":"Appraiser ID is required."})

    texts: List[str] = []; image_blobs: List[Tuple[str, bytes]] = []; first_pdf: Optional[bytes] = None
    for f in files:
        raw = await f.read(); name = (f.filename or "upload").lower()
        if name.endswith((".jpg",".jpeg",".png",".webp")):
            image_blobs.append((name, raw))
        elif name.endswith(".pdf"):
            emb = extract_text_from_pdf_embedded(raw)
            texts.append(emb if emb.strip() else extract_text_from_pdf(io.BytesIO(raw), 8, 140))
            if first_pdf is None: first_pdf = raw
            looks_like_est = bool(re.search(r'\bclaim\b', emb or "", re.I) and re.search(r'\bvin\b', emb or "", re.I))
            if not looks_like_est:
                image_blobs.extend(harvest_photos_from_pdf(raw, max_pages=16, dpi=130))
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8", errors="ignore"))

    combined = "\n".join(texts)

    # contact sheets for vision
    def make_contact_sheets_compact(image_blobs, max_sheets=3, cols=6, padding=6, base_thumb_w=320, jpeg_quality=68):
        if not image_blobs: return []
        def shrink_to_width(img: Image.Image, max_w: int) -> Image.Image:
            if img.width <= max_w: return img.convert("RGB")
            h = int(img.height * max_w / img.width); return img.convert("RGB").resize((max_w, h), Image.LANCZOS)
        thumbs = []
        for _, b in image_blobs:
            try: thumbs.append(shrink_to_width(Image.open(io.BytesIO(b)), base_thumb_w))
            except Exception: pass
        n = len(thumbs); per = max(1, math.ceil(n/3)); rows = math.ceil(per/cols)
        def build(chunk, tw):
            row_heights=[]; 
            for r in range(rows):
                row = chunk[r*cols:(r+1)*cols]; 
                if not row: break
                row_heights.append(max(im.height for im in row))
            canvas_w = cols*tw + (cols+1)*padding
            canvas_h = sum(row_heights)+(len(row_heights)+1)*padding
            sheet = Image.new("RGB",(canvas_w,canvas_h),(245,245,245))
            y=padding; pos=0
            for rh in row_heights:
                x=padding
                for _ in range(cols):
                    if pos>=len(chunk): break
                    im = chunk[pos]
                    if im.width!=tw:
                        im = im.resize((tw,int(im.height*tw/im.width)), Image.LANCZOS)
                    yoff = (rh - im.height)//2
                    sheet.paste(im,(x,y+yoff)); x += tw+padding; pos += 1
                y += rh+padding
            return sheet
        sheets=[]; idx=0; tw=base_thumb_w; sn=1
        while idx<n:
            ch = thumbs[idx:idx+per]; img = build(ch, tw); tries=0
            while img.height>3600 and tw>160 and tries<3:
                tw=int(tw*0.85); img = build(ch, tw); tries+=1
            buf=io.BytesIO(); img.save(buf,"JPEG",quality=jpeg_quality,optimize=True)
            sheets.append((f"contact-sheet-{sn}.jpg", buf.getvalue())); sn+=1; idx+=per
        return sheets

    sheets = make_contact_sheets_compact(image_blobs)
    images_for_vision = [{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b).decode("utf-8")}} for _, b in sheets]

    # required photos
    missing_photos = check_required_photos(image_blobs, combined)

    # VIN & Claim (embedded-first)
    vin_est = extract_vin_from_text(combined) or (extract_vin_from_pdf_first_pages(first_pdf,4,170) if first_pdf else None)
    claim_number = extract_claim_from_text(combined) or (extract_claim_from_pdf_first_pages(first_pdf,4,170) if first_pdf else None)
    claim_number = claim_number or "N/A"

    vin_photo = extract_vin_from_photos(image_blobs)
    if vin_est and vin_photo:
        vin_verify = "Match" if vin_est == vin_photo else f"No Match (photo shows {vin_photo})"
    elif vin_est and not vin_photo:
        vin_verify = "VIN photo not found"
    elif not vin_est and vin_photo:
        vin_verify = "VIN not found in estimate"
    else:
        vin_verify = "VIN unavailable"
    vin_final = vin_est or "N/A"

    vehicle_desc = extract_vehicle_from_text(combined) or "N/A"
    year, miles = parse_year_miles(combined)
    now_year = datetime.datetime.now().year
    age_years = (now_year - year) if year else None
    require_oem = (age_years is not None and age_years <= 2) or (miles is not None and miles <= 24000)
    non_oem_flag = non_oem_used(combined)

    # summary text
    def section_summary():
        lines=[]
        # Photos
        if not missing_photos:
            lines += ["### Required Photos","- All required photo types present (four corners, VIN, odometer, plate)."]
        else:
            lines += ["### Required Photos", f"- Missing: {', '.join(missing_photos)}."]
        # Labor
        lines += ["### Labor Rates", "- Labor rates listed on estimate." if labor_rates_present_any(combined) else "- Labor rates missing or not clearly listed."]
        # Taxes
        lines += ["### Taxes", "- Tax rate present on estimate." if taxes_present(combined) else "- Tax rate not found per client rules"]
        # Parts
        if require_oem:
            lines += ["### Parts Compliance", "- Non-compliance: non-OEM parts on ≤ 2 years or ≤ 24k miles." if non_oem_flag else "- Compliant: OEM parts only for ≤ 2 years or ≤ 24k miles."]
        else:
            lines += ["### Parts Compliance", "- Non-OEM parts noted; verify client rules allow on this vehicle." if non_oem_flag else "- Parts appear OEM or not flagged as non-OEM."]
        # Client line kept generic per your direction
        lines += ["### Client Rules Adherence", "- Apply client-required documentation (labor rates, photos, taxes) where applicable."]
        return "\n".join(lines)
    summary_md = section_summary()

    # estimate items & consistency
    est_items = extract_estimate_items(combined)
    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    # scoring
    labor_tax_adj = check_labor_and_tax_score(combined, client_rules)
    photo_adj = -25 * len(missing_photos)
    parts_adj = -25 if (require_oem and non_oem_flag) else 0
    score = max(0, 100 + labor_tax_adj + photo_adj + parts_adj)

    # PDF
    pdf = FPDF(); pdf.add_page()
    try: pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception: pdf.set_font("Arial", size=11)
    pdf.cell(200,10,txt="NSPXN.com AI Review Report", ln=True, align="C"); pdf.set_font_size(10)
    for k,v in [("File Number",file_number),("IA Company",ia_company),("Appraiser ID #",appraiser_id),
                ("Claim #",claim_number),("VIN",vin_final),("VIN Photo Verification",vin_verify),
                ("Vehicle",vehicle_desc),("Compliance Score", f"{score}%")]:
        pdf.multi_cell(0,6,f"{k}: {v}")
    pdf.ln(4); pdf_add_section_title(pdf,"AI-4-IA Review Summary")
    pdf.multi_cell(0,6,f"**Audit Results: {score}%**"); pdf.ln(1); pdf.multi_cell(0,6,summary_md)
    pdf.ln(4); pdf_add_section_title(pdf,"Estimate ↔ Photos Consistency Review")
    if consistency.get("per_item"):
        for it in consistency["per_item"][:60]:
            ev = "YES" if it.get("photo_evidence") else "NO"
            conf = it.get("confidence",0); 
            try: conf = float(conf)
            except: conf = 0.0
            line = f"- {it.get('side','unspecified').title()} {it.get('part','component')} · {it.get('op','op')} → Photo: {ev} ({round(conf*100)}%); {it.get('note','')}"
            pdf.multi_cell(0,6,line)
    else:
        pdf.multi_cell(0,6,"Per-item comparison unavailable.")
    if consistency.get("not_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf,"Items Estimated but Not Evident in Photos")
        for raw in consistency["not_in_photos"][:30]: pdf.multi_cell(0,6,f"- {raw}")
    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf,"Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:30]: pdf.multi_cell(0,6,f"- {d}")
    pdf.ln(2); pdf_kv(pdf,"Consistency Overall", consistency.get("overall",""))

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path,"wb") as f: f.write(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # email (same behavior as your last working build: plain body, no attachment)
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"; msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_number}
VIN: {vin_final}
VIN Photo Verification: {vin_verify}
Vehicle: {vehicle_desc}

Compliance Score: {score}%

AI Review Summary:
Audit Results: {score}%

{summary_md}
"""
        msg.set_content(body)
        with smtplib.SMTP_SSL("mail.tierra.net",465) as smtp:
            smtp.login("info@nspxn.com","grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing): {e}")

    return {
        "gpt_output": f"Audit Results: {score}%\n\n{summary_md}",
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin": vin_final,
        "vin_photo_verification": vin_verify,
        "score": f"{score}%",
        "consistency_review": consistency
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    fn = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(fn):
        return FileResponse(path=fn, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"; fp = os.path.join(rules_dir, f"{client_name}.docx")
    if not os.path.exists(fp): return JSONResponse(status_code=404, content={"error":"Rules not found for this client."})
    try:
        doc = Document(fp); text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

























