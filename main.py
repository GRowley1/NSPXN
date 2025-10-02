
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, base64, json, logging, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, Image

from openai import OpenAI

# =========================================
# Config
# =========================================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("ai4ia")

MODEL = os.getenv("OAI_MODEL", "gpt-4o")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))

if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    raise RuntimeError("❌ OPENAI_API_KEY is not set")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =========================================
# FastAPI
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
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img

def pdf_text_ocr(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 20) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception as e:
        log.warning(f"pdf to image failed: {e}")
        return ""
    out = []
    for i, pg in enumerate(pages, 1):
        try:
            txt = pytesseract.image_to_string(_pp(pg), lang="eng", config="--psm 6")
            if txt.strip():
                out.append(f"[Page {i}]\n{txt}")
        except Exception as e:
            log.warning(f"OCR page {i} failed: {e}")
    return "\n\n".join(out)

def docx_text(blob: bytes) -> str:
    try:
        d = Document(io.BytesIO(blob))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        log.warning(f"DOCX read error: {e}")
        return ""

# =========================================
# Photo harvesting & corner labels
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
    out: List[Tuple[str, bytes]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)[:max_pages]
        for i, page in enumerate(pages, 1):
            proc = _pp(page)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            is_img_report = "image report" in (ocr or "").lower()
            corner_hits = count_corner_labels(ocr)
            var = ImageStat.Stat(proc).var[0] if proc.mode == "L" else sum(ImageStat.Stat(proc).var)/3
            looks_like_photos = var > 120 or corner_hits >= 2
            if is_img_report or looks_like_photos:
                buf = io.BytesIO()
                page.save(buf, format="JPEG", quality=85)
                tag = "imgrep" if is_img_report else ("corner" if corner_hits else "pdfphoto")
                out.append((f"pdf-{tag}-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        log.warning(f"harvest photos error: {e}")
    return out

# =========================================
# VIN utilities
# =========================================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

_trans = {**{str(i): i for i in range(10)},
          **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def normalize_vin(s: str) -> Optional[str]:
    s = (s or "").upper().replace(" ", "")
    s = s.replace("O", "0").replace("I", "1").replace("Q", "0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

def vin_checksum_ok(v: str) -> bool:
    try:
        tot = sum(_trans[ch]*_w[i] for i,ch in enumerate(v))
        chk = tot % 11
        return v[8] == ("X" if chk==10 else str(chk))
    except Exception:
        return False

def best_vin(cands: List[str]) -> Optional[str]:
    for c in cands:
        v = normalize_vin(c)
        if v and vin_checksum_ok(v):
            return v
    for c in cands:
        v = normalize_vin(c)
        if v:
            return v
    return None

# =========================================
# Field extractors
# =========================================
def extract_vin_from_text(text: str) -> Optional[str]:
    if not text: return None
    lab = re.findall(r"\bVIN\s*[:#\-]?\s*([A-HJ-NPR-Z0-9]{10,20})", text, re.IGNORECASE)
    if lab:
        v = best_vin(lab)
        if v: return v
    cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, re.IGNORECASE)
    return best_vin(cands)

def extract_claim_from_text(text: str) -> Optional[str]:
    if not text: return None
    pats = [
        r"Claim\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
        r"Assignment\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
        r"Reference\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            tok = m.group(1).strip().rstrip(".,;")
            if re.search(r"\d", tok):
                return tok
    return None

MAKE_MAP = {
    "CHEV":"Chevrolet","CHEVY":"Chevrolet","MB":"Mercedes-Benz","MERCEDES":"Mercedes-Benz",
    "VW":"Volkswagen","VOLKS":"Volkswagen","GMC":"GMC","TOYOTA":"Toyota","HONDA":"Honda",
    "NISSAN":"Nissan","HYUNDAI":"Hyundai","KIA":"Kia","FORD":"Ford","DODGE":"Dodge","RAM":"Ram",
    "SUBARU":"Subaru","MAZDA":"Mazda","BMW":"BMW","AUDI":"Audi","JEEP":"Jeep"
}
VEH_STOP = set("""VIN WB ODOMETER MILEAGE ENGINE L TURBO SUPERCHARGED DIESEL GASOLINE
EXT INTR EXTERIOR INTERIOR COLOR COLORS TRIM TRANS TRANSMISSION 2D 4D P/U WB" WB’ WB’ WB”""".split())

def extract_vehicle_from_text(text: str) -> Optional[str]:
    if not text: return None
    cand = None
    for line in text.splitlines():
        ln = line.strip()
        m = re.match(r"^(19|20)\d{2}\s+([A-Za-z]{3,})\s+(.+)$", ln)
        if m:
            if cand is None or len(ln) > len(cand):
                cand = ln
    if not cand:
        return None
    parts = cand.split()
    year = parts[0]
    make_raw = parts[1].upper()
    make = MAKE_MAP.get(make_raw, parts[1].capitalize())
    model_tokens = []
    for tok in parts[2:]:
        t = re.sub(r"[^\w\-\"’”/]", "", tok).upper()
        if t in VEH_STOP or re.match(r"^\d+(\.\d+)?L$", t):
            break
        if t in ("VIN:", "VIN"):
            break
        model_tokens.append(tok)
        if len(" ".join(model_tokens)) > 80:
            break
    model = " ".join(model_tokens).strip() or "Vehicle"
    return f"{year} {make} {model}"

# =========================================
# Photo parsing & requirements
# =========================================
def _image_is_exterior_wide(img: Image.Image) -> bool:
    processed = _pp(img)
    text = pytesseract.image_to_string(processed, lang="eng")
    var = ImageStat.Stat(processed).var[0] if processed.mode == "L" else sum(ImageStat.Stat(processed).var)/3
    return len(text.strip()) < 10 and var > 150

def extract_vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    rots = [0, 90, 180, 270]
    found: List[str] = []
    for name, blob in image_blobs:
        try:
            base = Image.open(io.BytesIO(blob))
            for r in rots:
                img = base.rotate(r, expand=True)
                proc = _pp(img)
                for psm in ("--psm 7", "--psm 6", "--psm 11"):
                    ocr = pytesseract.image_to_string(proc, lang="eng", config=psm)
                    cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", ocr.upper())
                    if cands:
                        found.extend(cands)
        except Exception as e:
            log.warning(f"VIN photo OCR error ({name}): {e}")
    return best_vin(found)

def extract_odometer_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = _pp(img)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", ocr, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception as e:
            log.warning(f"Odometer photo OCR error ({name}): {e}")
    return None

def check_required_photos(image_blobs: List[Tuple[str, bytes]], ocr_text: str) -> List[str]:
    required = ["four corners", "odometer", "vin", "license plate"]
    present = set()
    txt = (ocr_text or "").lower()

    if any(k in txt for k in ["odometer", "mileage photo", "dashboard mileage"]):
        present.add("odometer")
    if any(k in txt for k in ["vin", "vehicle identification number", "vin photo"]):
        present.add("vin")
    if any(k in txt for k in ["license plate", "registration plate"]):
        present.add("license plate")

    ext_like = 0
    for name, blob in image_blobs:
        try:
            img = Image.open(io.BytesIO(blob))
            proc = _pp(img)
            ocr = pytesseract.image_to_string(proc, lang="eng")
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", ocr, re.IGNORECASE):
                present.add("vin")
            if re.search(r"\d{1,3}(,\d{3})*\s*(miles|km)", ocr, re.IGNORECASE):
                present.add("odometer")
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", ocr, re.IGNORECASE):
                present.add("license plate")
            if _image_is_exterior_wide(img):
                ext_like += 1
        except Exception as e:
            log.warning(f"Image parse error {name}: {e}")

    imgrep_count = sum(1 for n, _ in image_blobs if "imgrep" in n.lower())
    corner_page_count = sum(1 for n, _ in image_blobs if "corner" in n.lower())
    corner_labels_in_text = count_corner_labels(ocr_text)

    if ext_like >= 2 or imgrep_count >= 2 or (corner_page_count + corner_labels_in_text) >= 3:
        present.add("four corners")

    missing = [p for p in required if p not in present]
    log.debug(f"Photo check present={sorted(list(present))}, missing={missing}")
    return missing

# =========================================
# Labor/tax checks
# =========================================
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

# =========================================
# PDF helpers
# =========================================
def pdf_title(pdf: FPDF, title: str):
    pdf.set_font_size(12)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font_size(10)

# =========================================
# Routes
# =========================================
@app.get("/")
async def ok():
    return {"ok": True}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(...),
    ai_intent: str = Form("comprehensive")
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error":"Appraiser ID is required."})

    # ---- read uploads
    texts: List[str] = []
    images: List[Tuple[str, bytes]] = []
    vision_imgs: List[Dict[str, Any]] = []

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            t = pdf_text_ocr(raw, dpi=200, max_pages=12)
            texts.append(t)
            for nm, jb in harvest_photos_from_pdf(raw):
                images.append((nm, jb))
                vision_imgs.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(jb).decode("utf-8")}})
        elif name.endswith(".docx"):
            texts.append(docx_text(raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp")):
            images.append((name, raw))
            vision_imgs.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(raw).decode("utf-8")}})
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8","ignore"))
        else:
            pass

    combined_text = "\n".join(texts)

    # ----- determine intent/mode -----
    intent = (ai_intent or "").strip().lower()
    supplement_mode = intent in (
        "invoices_with_photos",
        "supplement",
        "supplement_with_photos",
        "invoices",
        "supplement↔invoices"
    )

    # ---- extract fields
    vin_est = extract_vin_from_text(combined_text)
    vin_photo = extract_vin_from_photos(images)
    vin_final = vin_est or vin_photo or "N/A"

    claim_number = extract_claim_from_text(combined_text) or "N/A"
    vehicle_desc = extract_vehicle_from_text(combined_text) or "N/A"
    odo_photo = extract_odometer_from_photos(images)

    missing_photos = check_required_photos(images, combined_text)

    # ---- VIN verification
    if images:
        if vin_est and vin_photo:
            vin_verify = "MATCH" if normalize_vin(vin_est)==normalize_vin(vin_photo) else "MISMATCH"
        elif vin_est and not vin_photo:
            vin_verify = "NOT VERIFIED"
        else:
            vin_verify = "NOT VERIFIED"
    else:
        vin_verify = "PHOTOS NOT PROVIDED"

    # ---- GPT narrative
    if supplement_mode:
        sys = "Supplement/invoices vs estimate (and photos if provided). Only use text provided; never infer. If no invoices detected, say so plainly. Do NOT require VIN/registration/odometer photos in this mode."
    else:
        sys = (
            "You are an auto-claims appraisal assistant. Be concise and specific. "
            "Where you cannot verify, say 'Not found in provided documents'. "
            "Always end with a single line: Final Evaluation: NN%."
        )

    user_parts: List[Dict[str, Any]] = []
    user_parts.append({"type":"text","text": f"ESTIMATE/CONTENT (OCR):\n{combined_text[:12000]}"})
    if images:
        user_parts.extend(vision_imgs)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content": sys},
                {"role":"user","content": user_parts}
            ],
            max_tokens=900,
            temperature=0
        )
        gpt_out = (rsp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"GPT error: {e}")
        gpt_out = "AI narrative unavailable."

    # ----- SCORE: single authoritative number used everywhere -----
    if supplement_mode:
        score_ai = None
        m = re.search(r"(Final|Total|Compliance)\s*(Score|Evaluation)?\s*[:\-]?\s*(\d{1,3})\s*%?", gpt_out, re.IGNORECASE)
        if m:
            try:
                score_ai = int(m.group(3))
            except Exception:
                score_ai = None
        labor_tax_adj = 0
        photo_adj = 0
        computed = 100
        authoritative_score = max(0, min(100, score_ai if score_ai is not None else computed))
    else:
        score_ai = None
        for pat in [
            r"Total\s*Evaluation\s*[:\-]?\s*(\d{1,3})\s*%?",
            r"Final\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?",
            r"Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%?",
        ]:
            m = re.search(pat, gpt_out, re.IGNORECASE)
            if m:
                score_ai = int(m.group(1))
                break

        labor_tax_adj = check_labor_and_tax_score(combined_text, client_rules)
        photo_adj = -25 * len(missing_photos)
        computed = max(0, 100 + labor_tax_adj + photo_adj)
        authoritative_score = max(0, min(100, score_ai if score_ai is not None else computed))

    # Clean AI output of duplicate score lines
    gpt_output_clean = re.sub(
        r'(?im)^(?:Final\s*Score|Compliance\s*Score|Total\s*Evaluation)\s*[:\-]?\s*\d{1,3}\s*%.*$',
        '',
        gpt_out
    ).strip()

    # =========================================
    # PDF build
    # =========================================
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(0, 10, "NSPXN.com AI Review Report", ln=True, align="C")
    pdf.set_font_size(10); pdf.ln(3)
    pdf.multi_cell(0,6,f"File Number: {file_number}")
    pdf.multi_cell(0,6,f"IA Company: {ia_company}")
    pdf.multi_cell(0,6,f"Appraiser ID #: {appraiser_id}")

    # Dynamic request type label
    request_type_label = {
        "guidelines_only": "Guidelines → Estimate (no photos)",
        "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
        "photos_only": "Photos Only: Compare to Estimate",
        "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
        "docs_checklist": "Documentation Checklist"
    }.get(ai_intent, "Comprehensive: Guidelines + Estimate + Photos (with VIN check)")
    pdf.multi_cell(0,6,f"Request Type: {request_type_label}")

    pdf.ln(2)
    pdf.multi_cell(0,6,f"Claim #: {claim_number}")
    pdf.multi_cell(0,6,f"VIN (from estimate/photos): {vin_final}")
    pdf.multi_cell(0,6,f"VIN verification (estimate vs photo): {vin_verify}")
    pdf.multi_cell(0,6,f"Vehicle: {vehicle_desc}")
    if odo_photo:
        pdf.multi_cell(0,6,f"Odometer (from photos): {odo_photo}")
    pdf.multi_cell(0,6,f"Compliance Score: {authoritative_score}%")

    pdf.ln(3)
    pdf_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0,6, gpt_output_clean or "No narrative.")

    # Save PDF
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        log.warning(f"PDF write error: {e}")

    # Email
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_number}
VIN: {vin_final}
Vehicle: {vehicle_desc}

Compliance Score: {authoritative_score}%

AI Summary:
{gpt_output_clean}
""")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.error(f"Email error: {e}")

    return {
        "file_number": file_number,
        "claim_number": claim_number,
        "vin": vin_final,
        "vin_verification": vin_verify,
        "vehicle": vehicle_desc,
        "compliance_score": authoritative_score,
        "missing_photos": missing_photos,
        "gpt_output": gpt_output_clean,
        "pdf_url": f"/download-pdf?file_number={file_number}",
        "pdf_filename": f"{file_number}.pdf"
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = re.sub(r"[^\w.\-]+","-", file_number).strip("-_.")
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})
