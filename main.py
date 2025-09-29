
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, json, logging, base64, smtplib, zipfile, time
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from openai import OpenAI

# ----------------- Config -----------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ai4ia-lite")

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))  # seconds

# OpenAI client with sane defaults
if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("OPENAI_API_KEY not set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=OPENAI_TIMEOUT)
MODEL_PRIMARY = os.getenv("OAI_MODEL", "gpt-4o-mini")
MODEL_FALLBACK = "gpt-3.5-turbo"

# Whitelisted request types
INTENTS = {
    "guidelines_only": "Guidelines → Estimate (no photos)",
    "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
    "photos_only": "Photos Only: Compare to Estimate",
    "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
    "docs_checklist": "Documentation Checklist",
}

# ----------------- App -----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com","https://www.nspxn.com",
        "http://nspxn.com","http://www.nspxn.com",
        "https://nspxn.onrender.com",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------
def _pp(img):
    """Light image preproc for OCR fallback."""
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.9)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def fast_pdf_text(pdf_bytes: bytes, limit_pages: Optional[int] = None) -> str:
    out = []
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = reader.pages
        if limit_pages:
            pages = pages[:limit_pages]
        for i, page in enumerate(pages, 1):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                out.append(f"[Page {i}]\n{t}")
    except Exception as e:
        log.warning(f"PyPDF2 extract failed: {e}")
    return "\n\n".join(out)

def quick_ocr_text(pdf_bytes: bytes, max_pages: int = 4, dpi: int = 240) -> str:
    """Very shallow OCR fallback to recover VIN/Claim quickly for scanned estimates."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
        out = []
        for i, p in enumerate(pages, 1):
            txt = pytesseract.image_to_string(_pp(p), lang="eng", config="--psm 6")
            if txt.strip():
                out.append(f"[OCR Page {i}]\n{txt}")
        return "\n\n".join(out)
    except Exception as e:
        log.warning(f"OCR fallback failed: {e}")
        return ""

def ocr_docx_text(file_like: io.BytesIO) -> str:
    try:
        doc = Document(file_like)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        log.warning(f"DOCX read error: {e}")
        return ""

# ---- Extractors ----
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
_trans = {**{str(i): i for i in range(10)},
          **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]
VIN_TIGHT = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
VIN_RELAX = re.compile(r"(?:V\.?I\.?N\.?|VIN|Vehicle\s+Identification\s+Number)\b[^A-Z0-9]{0,20}((?:[A-HJ-NPR-Z0-9][\s\-]*){17})", re.IGNORECASE)

def _norm_vin(s: str) -> Optional[str]:
    s = (s or "").upper()
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", s).replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s

def _vin_ok(v: str) -> bool:
    try:
        total = sum(_trans[ch] * _w[i] for i, ch in enumerate(v))
        check = total % 11
        return v[8] == ("X" if check == 10 else str(check))
    except Exception:
        return False

def vin_from_text(text: str) -> Optional[str]:
    cands = [m.group(1) for m in VIN_RELAX.finditer(text or "")] + VIN_TIGHT.findall(text or "")
    seen = set()
    uniq = []
    for c in cands:
        v = _norm_vin(c)
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    for v in uniq:
        if _vin_ok(v): return v
    return uniq[0] if uniq else None

MAKE_MAP = {"NISSAN":"Nissan","CHEV":"Chevrolet","CHEVY":"Chevrolet","TOYOTA":"Toyota","FORD":"Ford","HONDA":"Honda","HYUNDAI":"Hyundai","KIA":"Kia","BMW":"BMW","MERCEDES":"Mercedes-Benz","MB":"Mercedes-Benz","VW":"Volkswagen","VOLKS":"Volkswagen","SUBARU":"Subaru","MAZDA":"Mazda","DODGE":"Dodge"}
STOP = {"GASOLINE","DIESEL","HYBRID","ELECTRIC","BLACK","WHITE","BLUE","RED","SILVER","GRAY","GREY","4D","2D","SED","SDN","SUV","COUPE","HATCH","TRUCK","WAGON","AWD","FWD","RWD","2.5L","3.5L","L","GDI","DIRECT","INJECTION","TURBO","PAINT","CLEAR","COAT","COLOR"}

def vehicle_from_text(text: str) -> Optional[str]:
    lines = [re.sub(r"\s{2,}", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        if re.search(r"^\s*(19|20)\d{2}\b", ln) and not re.search(r"\b(AM|PM)\b", ln):
            toks = ln.split(); year = toks[0]; tail = toks[1:]
            keep = []
            for t in tail:
                raw = re.sub(r"[^\w\-]", "", t).upper()
                if raw in STOP or raw in ("A/M","OEM"): break
                keep.append(t)
                if len(keep) >= 4: break
            if keep:
                mk = MAKE_MAP.get(keep[0].upper(), keep[0].capitalize())
                return " ".join([year, mk] + keep[1:])
    return None

def mileage_from_text(text: str) -> Optional[str]:
    for p in [r"(?:Odometer|Odo|Mileage|Miles)\s*[:\-]?\s*([\d,]{2,7})\b", r"\b([\d,]{2,7})\s*(?:mi|miles)\b"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None

def claim_from_text(text: str) -> Optional[str]:
    pats = [
        r"(?:Carrier|Insurance|Insurer)?\s*Claim\s*(?:No\.?|Number|#)\s*[: ]\s*([A-Za-z0-9\-_\\/]{5,25})",
        r"(?:Shop|Body\s*Shop)\s*Claim\s*(?:No\.?|Number|#)\s*[: ]\s*([A-Za-z0-9\-_\\/]{5,25})",
        r"(?:SCA|IA)\s*Claim\s*(?:No\.?|Number|#)\s*[: ]\s*([A-Za-z0-9\-_\\/]{5,25})",
        r"(?:Assignment|Reference|Ref)\s*(?:No\.?|Number|#)\s*[: ]\s*([A-Za-z0-9\-_\\/]{5,25})",
        r"Claim\s*[:#]\s*([A-Za-z0-9\-_\\/]{5,25})",
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip().rstrip(".:,;")
    m2 = re.search(r"Claim[^A-Za-z0-9]{0,20}([A-Za-z0-9\-_\\/]{5,25})", text or "", re.IGNORECASE)
    if m2: return m2.group(1).strip().rstrip(".:,;")
    return None

def extract_days(text: str) -> Optional[int]:
    m = re.search(r"Days?\s*to\s*Repair\s*[:\-]?\s*([0-9]+)", text or "", re.IGNORECASE)
    try: return int(m.group(1)) if m else None
    except: return None

def openai_chat(messages, max_tokens=900):
    # retry with backoff; client-level timeout already set
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL_PRIMARY,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
        except Exception as e:
            s = str(e).lower()
            if "429" in s or "rate" in s or "timeout" in s:
                time.sleep(1.25 * (attempt + 1))
                continue
            break
    try:
        return client.chat.completions.create(
            model=MODEL_FALLBACK,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0,
        )
    except Exception as e:
        log.error(f"OpenAI fallback failed: {e}")
        return None

def strip_photo_sections(text: str) -> str:
    if not text: return text
    patterns = [
        r"(?is)^\s*#{1,6}\s*Client\s*Photo\s*Rules.*?(?=^\s*#{1,6}\s|\Z)",
        r"(?is)^\s*#{1,6}\s*Estimate.?↔.?Photos\s*Comparison.*?(?=^\s*#{1,6}\s|\Z)",
        r"(?is)^\s*#{1,6}\s*Photos?\s*(?:Provided|Coverage|Summary).*(?=^\s*#{1,6}\s|\Z)",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    text = re.sub(r"(?im)^\s*[-•].*photo.*$", "", text)
    return text.strip()



# ---- Minimal helper: parse compliance score from narrative ----
_SCORE_PATTERNS = [
    re.compile(r'(?i)Final\s*(?:Evaluation|Score)\s*[:\-]?\s*(\d{1,3})\s*%'),
    re.compile(r'(?i)Compliance\s*Score\s*[:\-]?\s*(\d{1,3})\s*%'),
    re.compile(r'(?<!\d)(\d{1,3})\s*%(?!\d)'),
]

def extract_score(narrative: str) -> str:
    s = narrative or ""
    for pat in _SCORE_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                val = int(m.group(1))
                val = max(0, min(100, val))
                return f"{val}%"
            except Exception:
                continue
    return "N/A"
# ----------------- API -----------------
@app.get("/")
async def root():
    return {"status":"ok",
        "compliance_score": extract_score(gpt_output)
    }

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...),
    ai_intent: str = Form("guidelines_only")
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error":"Appraiser ID is required."})

    intent = ai_intent if ai_intent in INTENTS else "guidelines_only"
    request_type_label = INTENTS.get(intent, intent)
    log.info(f"Intent={intent} ({request_type_label})")

    # Partition uploads (PDF/IMG/DOC/TXT/ZIP)
    pdfs: List[Tuple[str, bytes]] = []
    images: List[Tuple[str, bytes]] = []
    docs: List[str] = []
    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            pdfs.append((name, raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp")):
            images.append((name, raw))
        elif name.endswith(".docx"):
            docs.append(ocr_docx_text(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            docs.append(raw.decode("utf-8", errors="ignore"))
        elif name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for zi in zf.infolist():
                        if zi.is_dir():
                            continue
                        zname = zi.filename.lower()
                        zdata = zf.read(zi)
                        if zname.endswith(".pdf"):
                            pdfs.append((zname, zdata))
                        elif zname.endswith((".jpg",".jpeg",".png",".webp")):
                            images.append((zname, zdata))
                        elif zname.endswith(".docx"):
                            docs.append(ocr_docx_text(io.BytesIO(zdata)))
                        elif zname.endswith(".txt"):
                            docs.append(zdata.decode("utf-8", errors="ignore"))
            except Exception as e:
                log.warning(f"ZIP parse error for {name}: {e}")

    photos_present = len(images) > 0

    # Pick primary estimate PDF (prefer name with est/estimate)
    def pick_estimate_pdf(pdfs_list):
        if not pdfs_list: return None
        for nm, blob in pdfs_list:
            if "est" in nm or "estimate" in nm:
                return (nm, blob)
        return pdfs_list[0]

    est_pdf = pick_estimate_pdf(pdfs)

    # Estimate text: text-first; shallow OCR fallback only if text layer empty/short
    limit = 12 if intent == "comprehensive" else 6
    est_text = ""
    if est_pdf:
        est_text = fast_pdf_text(est_pdf[1], limit_pages=limit)
        if len(est_text.strip()) < 80:
            log.info("Text layer thin — using shallow OCR fallback (max 4 pages).")
            est_text = quick_ocr_text(est_pdf[1], max_pages=4, dpi=240)
    if not est_text and docs:
        est_text = "\n\n".join(docs)

    # Client rules text from form field only
    rules_text = (client_rules or "")

    # Extract identifiers from estimate text
    vin = vin_from_text(est_text) or "N/A"
    vehicle = vehicle_from_text(est_text) or "N/A"
    mileage = mileage_from_text(est_text)
    claim = claim_from_text(est_text) or "N/A"
    days_reported = extract_days(est_text)

    facts = {
        "vin": vin,
        "vehicle": vehicle,
        "claim": claim,
        "mileage_present": bool(mileage),
        "days_reported": days_reported,
        "photos_present": photos_present
    }

    # Build images payload (to GPT) only if photos present
    def vision_images(max_imgs=16):
        out=[]
        for i, (_, blob) in enumerate(images[:max_imgs]):
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        return out

    # ----------------- Intent routing (strict) -----------------
    gpt_output = ""
    if intent == "guidelines_only":
        system = (
            "Auto-damage compliance auditor. Compare client guidelines against the ESTIMATE content only. "
            "Do NOT restate the guidelines; output compliance decisions with short justifications from the estimate text. "
            f"PhotosPresent={photos_present}. If false, do NOT mention photos.\n"
            "- Tight bullets; no fluff.\n"
            "Sections:\n"
            "1) Client Quick Summary (2 bullets)\n"
            "2) Checklist — Guidelines vs Estimate (each rule: [Compliant|Non-compliant|Not found] — reason)\n"
            "3) Summary & Next Steps (1–2 bullets)\n"
            + json.dumps(facts, indent=2)
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(rules_text or "")[:10000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:12000]},
        ]
        rsp = openai_chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=900)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()
        if not photos_present: gpt_output = strip_photo_sections(gpt_output)

    elif intent == "comprehensive":
        system = (
            "Comprehensive audit: compare client guidelines to ESTIMATE, and compare ESTIMATE to PHOTOS (if photos present). "
            "Do NOT restate the guidelines; give compliance decisions with brief citations from estimate/photos. "
            "Numbered sections; hyphen bullets; no emojis.\n"
            f"PhotosPresent={photos_present}. If false, omit photo-dependent sections.\n"
            + json.dumps(facts, indent=2)
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(rules_text or "")[:9000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:12000]},
        ]
        if photos_present: user.extend(vision_images())
        rsp = openai_chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=1200)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()
        if not photos_present: gpt_output = strip_photo_sections(gpt_output)

    elif intent == "photos_only":
        if not photos_present:
            gpt_output = "No photos were provided with this request."
        else:
            system = "Compare ESTIMATE to PHOTOS only. Sections: Photo Coverage, Visible Damage vs Estimate, Discrepancies, Summary."
            user = [{"type":"text","text":"ESTIMATE TEXT:\n"+(est_text or "")[:8000]}] + vision_images()
            rsp = openai_chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=800)
            gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()

    elif intent == "invoices_with_photos":
        invoices_text = ""
        for nm, raw in pdfs:
            if any(k in nm for k in ("invoice","receipt","supplement")):
                invoices_text += fast_pdf_text(raw, limit_pages=5) or quick_ocr_text(raw, max_pages=2)
        system = (
            f"PhotosPresent={photos_present}. If false, omit photo-related sections.\n"
            "Audit whether the supplement/estimate is substantiated by invoices and, if present, by photos. "
            "Sections: Invoices Summary, Support vs Estimate Lines, (Photo Corroboration), Missing Documentation, Summary."
        )
        user = [
            {"type":"text","text":"ESTIMATE TEXT:\n"+(est_text or "")[:6000]},
            {"type":"text","text":"\n\nINVOICES TEXT:\n"+(invoices_text or '')[:6000]},
        ]
        if photos_present: user.extend(vision_images())
        rsp = openai_chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=900)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()
        if not photos_present: gpt_output = strip_photo_sections(gpt_output)

    elif intent == "docs_checklist":
        system = (
            "Documentation checklist only. State if the estimate includes each required doc mentioned in the client guidelines. "
            "Mark 'missing' if not found. Be terse; bullets only."
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\n"+(rules_text or "")[:6000]},
            {"type":"text","text":"\n\nESTIMATE TEXT:\n"+(est_text or "")[:8000]},
        ]
        rsp = openai_chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=500)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()

    # ----------------- PDF (shell unchanged) -----------------
    pdf = FPDF(); pdf.add_page()
    pdf.set_font("Arial", size=11)

    pdf.cell(200,10,"NSPXN.com AI Review Report",ln=True,align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0,6,f"File Number: {file_number}")
    pdf.multi_cell(0,6,f"IA Company: {ia_company}")
    pdf.multi_cell(0,6,f"Request Type: {request_type_label}")
    pdf.multi_cell(0,6,f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0,6,f"Claim #: {claim}")
    pdf.multi_cell(0,6,f"VIN (from estimate): {vin}")
    vin_line = "Included in narrative" if (intent == "comprehensive" and photos_present) else ("Photos not provided" if intent == "comprehensive" else "Not requested")
    pdf.multi_cell(0,6,f"VIN verification (estimate vs photo): {vin_line}")
    pdf.multi_cell(0,6,f"Vehicle: {vehicle}")
    if mileage: pdf.multi_cell(0,6,f"Odometer (from estimate): {mileage}")
    if days_reported is not None:
        pdf.multi_cell(0,6,f"Days to Repair (reported): {days_reported}")
    score_txt = extract_score(gpt_output)
    pdf.multi_cell(0,6,f"Compliance Score: {score_txt}")

    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0,8,"AI-4-IA Review Summary",ln=True)
    pdf.set_font_size(10); pdf.multi_cell(0,6,gpt_output or "No narrative generated.")

    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0,8,"Estimate ↔ Photos Consistency Review",ln=True)
    pdf.set_font_size(10)
    if intent in ("comprehensive","photos_only","invoices_with_photos") and photos_present:
        pdf.multi_cell(0,6,"Included in narrative above (single-pass review).")
    else:
        pdf.multi_cell(0,6,"Not requested or no photos provided.")

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        with open(pdf_path,"wb") as f: f.write(pdf.output(dest="S").encode("latin-1"))
        log.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        log.error(f"PDF write error: {e}")

    # ----------------- Email (shell unchanged) -----------------
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Request Type: {request_type_label}

Claim #: {claim}
VIN (from estimate): {vin}
VIN verification (estimate vs photo): {vin_line}
Vehicle: {vehicle}
{('Odometer (from estimate): ' + mileage) if mileage else ''}
{('Days to Repair (reported): ' + str(days_reported)) if days_reported is not None else ''}

Summary:
{gpt_output}
"""
        msg.set_content(body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logging.getLogger("ai4ia-lite").warning(f"Email send error (continuing): {e}")

    return {
        "request_type": request_type_label,
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim,
        "vehicle": vehicle,
        "vin_estimate": vin,
        "vin_verification": vin_line,
        "odometer_estimate": mileage or "Not documented",
        "days_to_repair": days_reported or "Not documented",
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})
