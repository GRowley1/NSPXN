# main.py  —  Simplified, fast audit with VIN verification + cleaner photo checks
# Endpoints: 
#   POST /vision-review  (files[], client_rules, file_number, ia_company, appraiser_id)
#   GET  /download-pdf?file_number=123
#
# Notes:
# - Keeps your lean, <~2 min flow (one GPT pass).
# - Adds VIN verify (estimate vs photos) + stronger required-photo presence.
# - Produces a narrative similar to the “Comprehensive Audit…” sample you pasted.

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, base64, json, logging

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, Image
from openai import OpenAI

# =========================================
# PDF storage: save to /tmp but filename stays {file_number}.pdf
# =========================================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

# =========================================
# Logging
# =========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================
# OpenAI client
# =========================================
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.getenv("OAI_MODEL", "gpt-4o-mini")  # fast + good enough for this audit

# =========================================
# FastAPI app + CORS
# =========================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com", "https://www.nspxn.com",
        "http://nspxn.com",  "http://www.nspxn.com",
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
    img = ImageEnhance.Contrast(img).enhance(1.9)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr_page(img: Image.Image, psm: str = "--psm 6") -> str:
    try:
        return pytesseract.image_to_string(preprocess_image(img), lang="eng", config=psm)
    except Exception:
        return pytesseract.image_to_string(preprocess_image(img), lang="eng", config="--psm 3")

def extract_text_from_pdf(file_like: io.BytesIO, dpi: int = 180, max_pages: int = 40) -> str:
    try:
        file_like.seek(0)
        images = convert_from_bytes(file_like.read(), dpi=dpi)[:max_pages]
        out = []
        for i, img in enumerate(images, 1):
            txt = ocr_page(img)
            if len((txt or "").strip()) >= 25:
                out.append(f"\n[Page {i}]\n{txt}")
        return "".join(out)
    except Exception as e:
        logger.warning(f"OCR error: {e}")
        return ""

def extract_text_from_docx(file_like: io.BytesIO) -> str:
    try:
        doc = Document(file_like)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.warning(f"DOCX read error: {e}")
        return ""

# =========================================
# Photo harvest + corner cues
# =========================================
CORNER_LABEL_PAT = re.compile(
    r'\b(?:left\s*front|right\s*front|left\s*rear|right\s*rear|lf|rf|lr|rr)\b',
    re.IGNORECASE
)

def count_corner_labels(text: str) -> int:
    found = set()
    for m in re.finditer(CORNER_LABEL_PAT, text or ""):
        token = m.group(0).lower().replace(" ", "")
        if token in ("lf", "leftfront"): found.add("lf")
        elif token in ("rf", "rightfront"): found.add("rf")
        elif token in ("lr", "leftrear"): found.add("lr")
        elif token in ("rr", "rightrear"): found.add("rr")
    return len(found)

def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int = 20) -> List[Tuple[str, bytes]]:
    """
    Convert PDF pages to images and return (name, jpeg_bytes) for pages that look like photo pages.
    - Prefer pages whose OCR contains 'Image Report' OR corner labels.
    - Fallback: visually rich pages (variance threshold).
    """
    out: List[Tuple[str, bytes]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)[:max_pages]
        for i, page in enumerate(pages, 1):
            proc = preprocess_image(page)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            is_img_report = "image report" in (ocr or "").lower()
            corner_hits = count_corner_labels(ocr)
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            looks_like_photos = var > 120 or corner_hits >= 2
            if is_img_report or looks_like_photos:
                buf = io.BytesIO()
                page.convert("RGB").save(buf, format="JPEG", quality=85)
                tag = "imgrep" if is_img_report else ("corner" if corner_hits else "pdfphoto")
                out.append((f"pdf-{tag}-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        logger.info(f"harvest_photos_from_pdf error: {e}")
    return out

# =========================================
# VIN utilities
# =========================================
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

# =========================================
# Field extraction from estimate text
# =========================================
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

def extract_vehicle_from_text(text: str) -> Optional[str]:
    # Try the CCC header pattern first
    m_year = re.search(r"\b(19|20)\d{2}\b", text)
    if m_year:
        # Grab the line that mentions VEHICLE or a line with Make/Model around it
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if "VEHICLE" in ln.upper() or re.search(r"\b(19|20)\d{2}\b", ln):
                # Clean trailing site/url clutter
                s = re.sub(r"https?://\S+", "", ln)
                s = re.sub(r"\s{2,}", " ", s).strip()
                if len(s) >= 10:
                    return s
    # Fallback: Year Make Model, Miles
    m1 = re.search(r"\b(20\d{2}|19\d{2})\s+([A-Za-z]{3,})\s+([A-Za-z0-9\-/]{2,})", text)
    m2 = re.search(r"(?:Odometer|Mileage)\s*[:\-]?\s*([\d,]+)", text, re.IGNORECASE)
    if m1:
        year, make, model = m1.groups()
        miles = m2.group(1) if m2 else "Mileage unknown"
        return f"{year} {make} {model}, {miles} miles"
    return None

# =========================================
# Photo parsing & required photo presence (CLEANER)
# =========================================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    proc = preprocess_image(img)
    text = pytesseract.image_to_string(proc, lang="eng")
    var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
    return len(text.strip()) < 10 and var > 150

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> List[str]:
    rots = [0, 90, 180, 270]
    found: List[str] = []
    for name, blob in image_blobs:
        try:
            base = Image.open(io.BytesIO(blob))
            for r in rots:
                img = base.rotate(r, expand=True)
                proc = preprocess_image(img)
                for psm in ("--psm 7", "--psm 6", "--psm 11"):
                    ocr = pytesseract.image_to_string(proc, lang="eng", config=psm)
                    for cand in re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", (ocr or "").upper()):
                        if normalize_vin(cand):
                            found.append(normalize_vin(cand))
        except Exception:
            continue
    # return unique list in order found
    seen, uniq = set(), []
    for v in found:
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    return uniq

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = preprocess_image(img)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", ocr, re.IGNORECASE)
            if m: return m.group(1)
        except Exception:
            continue
    return None

def detect_required_photo_presence(image_blobs: List[Tuple[str, bytes]], est_text: str) -> Dict[str, bool]:
    """
    Returns flags for: four corners, odometer, vin, license plate
    Heuristics:
      - Exterior-wide detection
      - OCR cues
      - Image-report/corner pages from PDF harvest
      - Corner labels in estimate text
    """
    flags = {"four corners": False, "odometer": False, "vin": False, "license plate": False}
    ext_like = 0
    corner_page_count = sum(1 for n,_ in image_blobs if "corner" in n.lower())
    imgrep_count = sum(1 for n,_ in image_blobs if "imgrep" in n.lower())
    corner_labels_in_text = count_corner_labels(est_text or "")

    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            if _image_is_exterior_wide(img):
                ext_like += 1
            proc = preprocess_image(img)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            up = (ocr or "").upper()
            # VIN cue
            if "VIN" in up or re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", up):
                flags["vin"] = True
            # Odometer cue
            if any(k in up for k in ("ODOMETER", "MILEAGE", "MPH", "RPM", "KM/H", "PRND")) or re.search(r"\b\d{2,6}\b\s*(MI|MILES|KM)\b", up):
                flags["odometer"] = True
            # Plate cue (len 5–8 mix, tolerant)
            if "LICENSE" in up or "PLATE" in up or re.search(r"\b[A-Z0-9]{5,8}\b", up):
                flags["license plate"] = True
        except Exception:
            continue

    if ext_like >= 2 or imgrep_count >= 2 or (corner_page_count + corner_labels_in_text) >= 3:
        flags["four corners"] = True

    return flags

# =========================================
# Estimate items (simple parse) + GPT compare
# =========================================
PANELS = [
    "bumper","fender","door","hood","grille","headlamp","headlight","taillamp","tail lamp",
    "quarter panel","rocker","roof","trunk","decklid","mirror","apron","radiator support",
    "wheel","tire","pillar","garnish","molding","fog lamp","reinforcement","cover","finish panel","combo lamp"
]
OPS = ["replace","repair","refinish","r&i","r & i","align","blend","calibrate","repl","rpr","r&r"]

def extract_estimate_items(text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for line in (text or "").splitlines():
        l = line.strip().lower()
        if not l or len(l) < 6: continue
        if any(op in l for op in OPS) and any(p in l for p in PANELS):
            side = "unspecified"
            if "left" in l or re.search(r"\blh\b", l): side = "left"
            if "right" in l or re.search(r"\brh\b", l): side = "right"
            op = next((op for op in OPS if op in l), "unspecified").replace("r&r","replace").replace("r & i","r&i")
            panel = next((p for p in PANELS if p in l), "component")
            items.append({"op": op, "part": panel, "side": side, "raw": line.strip()})
    uniq, seen = [], set()
    for it in items:
        key = (it["op"], it["part"], it["side"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

def compare_estimate_with_photos(items: List[Dict[str, str]], images_for_vision: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        "Also list obvious damages seen in photos that are NOT in the estimate. "
        "Return STRICT JSON ONLY per this schema:\n" + json.dumps(schema)
    )

    user_parts: List[Dict[str, Any]] = [{"type": "text", "text": "Estimate items:\n" + json.dumps(items, ensure_ascii=False)}]
    user_parts.extend(images_for_vision)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system},{"role": "user", "content": user_parts}],
            max_tokens=1000, temperature=0
        )
        txt = (rsp.choices[0].message.content or "").strip()
        txt = txt.removeprefix("```json").removesuffix("```").strip()
        data = json.loads(txt)
        if not isinstance(data, dict) or "per_item" not in data:
            raise ValueError("JSON shape mismatch")
        return data
    except Exception as e:
        logger.error(f"Vision compare JSON error: {type(e).__name__}: {e}")
        return {"per_item": [], "not_in_photos": [], "extra_damage_in_photos": [], "overall": f"Comparison unavailable ({type(e).__name__})."}

# =========================================
# Labor/tax compliance checks (simple, fast)
# =========================================
def check_labor_and_tax_score(text: str, client_rules: str) -> int:
    adj = 0
    def has_rate(label: str) -> bool:
        pat = rf"{label}[^\n]{{0,140}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        return re.search(pat, text, re.IGNORECASE) is not None
    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    if not any(has_rate(lbl) for lbl in labels):
        adj -= 50
    if re.search(r"tax\s*(required|must|utilize|apply)", client_rules, re.IGNORECASE):
        if not re.search(r"(sales\s*tax|tax)[^\n]{0,120}?(\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(\.\d{2})?)", text, re.IGNORECASE):
            adj -= 25
    return adj

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

    # ----- read uploads
    texts: List[str] = []
    image_blobs: List[Tuple[str, bytes]] = []
    images_for_vision: List[Dict[str, Any]] = []

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_blobs.append((name, raw))
            b64 = base64.b64encode(raw).decode("utf-8")
            images_for_vision.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        elif name.endswith(".pdf"):
            texts.append(extract_text_from_pdf(io.BytesIO(raw)))
            # also harvest photo-like pages from PDF so photo checks can see them
            for hname, hbytes in harvest_photos_from_pdf(raw):
                image_blobs.append((hname, hbytes))
                b64 = base64.b64encode(hbytes).decode("utf-8")
                images_for_vision.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        elif name.endswith(".docx"):
            texts.append(extract_text_from_docx(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8", errors="ignore"))
        else:
            # ignore other binary types quietly
            try:
                texts.append(raw.decode("utf-8", errors="ignore"))
            except Exception:
                pass

    combined_text = "\n".join(texts)

    # ----- presence checks + VIN verify
    presence = detect_required_photo_presence(image_blobs, combined_text)
    missing_photos = [k for k, v in presence.items() if not v]

    vin_est = extract_vin_from_text(combined_text)  # VIN only from the estimate text
    vin_cands_photos = extract_vin_from_photos(image_blobs)  # VINs seen in VIN photos (door/windshield labels)
    vin_verify_note = "VIN PHOTO NOT FOUND"
    if presence.get("vin"):
        if vin_est:
            if any(v == vin_est for v in vin_cands_photos):
                vin_verify_note = "MATCH"
            elif any(vin_cands_photos):
                vin_verify_note = "MISMATCH"
            else:
                vin_verify_note = "VIN PHOTO PRESENT—TEXT UNREADABLE"
        else:
            vin_verify_note = "VIN PHOTO PRESENT (no VIN in estimate text)"
    vin_final_display = vin_est or (vin_cands_photos[0] if vin_cands_photos else "N/A")

    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    claim_number = extract_claim_from_text(combined_text) or "N/A"
    odo_photos = extract_odometer_from_photos(image_blobs)

    # ----- parse estimate items & compare to photos
    est_items = extract_estimate_items(combined_text)
    consistency = compare_estimate_with_photos(est_items, images_for_vision)

    # ----- narrative prompt (structured like your sample)
    photo_hint = f"\n\nMISSING PHOTOS: {', '.join(missing_photos) if missing_photos else 'None'}"
    system_prompt = f"""
You are an AI auto damage auditor. Write a concise, professional report titled:
"Comprehensive Audit of Estimate and Photos Comparison".

Style requirements:
- Use short sections with headings that mirror: Client Quick Summary, Fatal Errors, Photo Rules, Parts/Tax/Labor, Estimate↔Photos Comparison, Summary.
- Be definitive but avoid guessing. Do not state total loss unless the estimate says so.
- If registration photo or windshield VIN is missing, call it out.
- If tax rate and labor rates appear, mark compliant (do not over-explain).
- Keep it tight and readable (bullets + short sentences). No fluff.

Input rules from client (verbatim may be partial):
{client_rules}

Now analyze the provided estimate text and the attached photos.
""".strip()

    user_parts: List[Dict[str, Any]] = []
    if combined_text:
        user_parts.append({"type": "text", "text": combined_text + photo_hint})
    if images_for_vision:
        user_parts.extend(images_for_vision)

    try:
        gpt = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_parts}],
            max_tokens=1000, temperature=0
        )
        gpt_output = (gpt.choices[0].message.content or "").strip()
    except Exception as e:
        gpt_output = f"Automated narrative unavailable ({type(e).__name__})."

    # ----- SCORE: single authoritative number used everywhere
    labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
    photo_adj = -25 * len(missing_photos)
    authoritative_score = max(0, min(100, 100 + labor_tax_adj + photo_adj))

    # Clean any stray score lines from GPT block so PDF shows just narrative
    gpt_output_clean = re.sub(
        r'(?im)^(?:Final\s*Score|Compliance\s*Score|Total\s*Evaluation)\s*[:\-]?\s*\d{1,3}\s*%.*$',
        '',
        gpt_output
    ).strip()

    # =========================================
    # PDF build
    # =========================================
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"File Number: {file_number}")
    pdf.multi_cell(0, 6, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 6, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Claim #: {claim_number}")
    pdf.multi_cell(0, 6, f"VIN (from estimate): {vin_est or 'N/A'}")
    pdf.multi_cell(0, 6, f"VIN verification (estimate vs photo): {vin_verify_note}")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    if odo_photos:
        pdf.multi_cell(0, 6, f"Odometer (from photos): {odo_photos}")
    pdf.multi_cell(0, 6, f"Compliance Score: {authoritative_score}%")

    pdf.ln(4)
    pdf_add_section_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0, 6, gpt_output_clean or "No narrative generated.")

    # ======== Estimate ↔ Photos Consistency Review ========
    pdf.ln(4); pdf_add_section_title(pdf, "Estimate ↔ Photos Consistency Review")
    if consistency.get("per_item"):
        for it in consistency["per_item"][:40]:
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
        for raw in consistency["not_in_photos"][:20]: pdf.multi_cell(0, 6, f"- {raw}")
    if consistency.get("extra_damage_in_photos"):
        pdf.ln(2); pdf_add_section_title(pdf, "Damage Visible in Photos but Missing on Estimate")
        for d in consistency["extra_damage_in_photos"][:20]: pdf.multi_cell(0, 6, f"- {d}")

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
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
VIN (from estimate): {vin_est or 'N/A'}
VIN verification (estimate vs photo): {vin_verify_note}
Vehicle: {vehicle_desc}

Compliance Score: {authoritative_score}%

Summary:
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
        "vin_estimate": vin_est or "N/A",
        "vin_verification": vin_verify_note,
        "score": f"{authoritative_score}%",
        "presence": presence,
        "consistency_review": consistency
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(path=pdf_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})










