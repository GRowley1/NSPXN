
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import os, re, io, base64, json, logging, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from openai import OpenAI

# ==========================
# Config
# ==========================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ai4ia")

MODEL = os.getenv("OAI_MODEL", "gpt-4o")
if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    raise RuntimeError("❌ OPENAI_API_KEY is not set")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _safe_name(s: str) -> str:
    import re as _re
    return _re.sub(r"[^\w.\-]+", "-", (s or "").strip()).strip("-_.")

# ==========================
# App + CORS
# ==========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com","https://www.nspxn.com","http://nspxn.com","http://www.nspxn.com",
        "https://nspxn.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Minimal OCR helpers
# ==========================
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img

def pdf_text_ocr(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 12) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception as e:
        log.warning(f"pdf->image failed: {e}")
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
        log.warning(f"docx read error: {e}")
        return ""

# =========================================
# Client Rules endpoint (unchanged logic from last build)
# =========================================
@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = os.getenv("CLIENT_RULES_DIR", "client_rules")
    file_name = f"{client_name}.docx"
    file_path = os.path.join(rules_dir, file_name)
    if os.path.exists(file_path):
        try:
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return {"text": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})

# =========================================
# VIN helpers with checksum + OCR-ambiguity repair
# =========================================
def _clean(s: str) -> str:
    return (s or "").replace("\r", "")

def extract_claim_number(text: str) -> Optional[str]:
    t = _clean(text)
    pats = [
        r"Claim\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
        r"Assignment\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
        r"Reference\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
        r"Claim\s*(?:No\.?|Number|#)?\s*[:\-]?\s*\n\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
    ]
    for p in pats:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;")
    return None

VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
_trans = {**{str(i): i for i in range(10)},
          **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def _vin_checksum_ok(v: str) -> bool:
    try:
        tot = sum(_trans[ch] * _w[i] for i, ch in enumerate(v))
        chk = tot % 11
        return v[8] == ("X" if chk == 10 else str(chk))
    except Exception:
        return False

def _normalize_vin_basic(s: str) -> Optional[str]:
    s = (s or "").upper().replace(" ", "")
    s = s.replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

_swaps = {'S':['S','5'], '5':['5','S'], 'B':['B','8'], '8':['8','B'], 'Z':['Z','2'], '2':['2','Z']}
def _vin_ambiguous_variants(v: str):
    cands = ['']
    for ch in v:
        opts = _swaps.get(ch, [ch])
        cands = [p + o for p in cands for o in opts]
        if len(cands) > 128:
            cands = cands[:128]
    return cands

def _canon_vin(v: Optional[str]) -> Optional[str]:
    """Return a checksum-valid VIN by exploring ambiguous swaps; else None."""
    if not v: return None
    base = _normalize_vin_basic(v)
    if not base: return None
    if _vin_checksum_ok(base): return base
    for var in _vin_ambiguous_variants(base):
        if len(var)==17 and all(ch in VIN_ALLOWED for ch in var) and _vin_checksum_ok(var):
            return var
    return None

def _vin_candidates_from_text(t: str) -> list:
    t = (t or "").upper()
    raw_cands = []
    raw_cands.extend(re.findall(r"\bVIN\s*[:#\-]?\s*([A-HJ-NPR-Z0-9]{10,20})", t, re.IGNORECASE))
    raw_cands.extend(re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", t, re.IGNORECASE))
    out = []
    for raw in raw_cands:
        can = _canon_vin(raw)
        if can:
            out.append(can)
    # dedupe preserving order
    seen = set(); uniq = []
    for v in out:
        if v not in seen:
            seen.add(v); uniq.append(v)
    return uniq

def extract_vin(text: str) -> Optional[str]:
    c = _vin_candidates_from_text(text)
    return c[0] if c else None

def extract_all_vins(text: str) -> list:
    return _vin_candidates_from_text(text)

MAKE_MAP = {
    "CHEV":"Chevrolet","CHEVY":"Chevrolet","MB":"Mercedes-Benz","MERCEDES":"Mercedes-Benz",
    "VW":"Volkswagen","VOLKS":"Volkswagen","GMC":"GMC","TOYOTA":"Toyota","HONDA":"Honda",
    "NISSAN":"Nissan","HYUNDAI":"Hyundai","KIA":"Kia","FORD":"Ford","DODGE":"Dodge","RAM":"Ram",
    "SUBARU":"Subaru","MAZDA":"Mazda","BMW":"BMW","AUDI":"Audi","JEEP":"Jeep"
}
def extract_vehicle(text: str) -> Optional[str]:
    t = _clean(text)
    m = re.search(r"Vehicle\s*[:\-]\s*(.+)", t, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if 2 < len(val) < 120: return val
    best = None
    for ln in t.splitlines():
        ln = ln.strip()
        m2 = re.match(r"^(19|20)\d{2}\s+([A-Za-z]{3,})\s+(.+)$", ln)
        if m2:
            if best is None or len(ln) > len(best): best = ln
    if best:
        parts = best.split()
        year = parts[0]
        mk = MAKE_MAP.get(parts[1].upper(), parts[1].capitalize())
        model = " ".join(parts[2:]).strip()
        return f"{year} {mk} {model}"
    return None

def extract_odometer_estimate(text: str) -> Optional[str]:
    t = _clean(text)
    lab = re.search(r"Odometer\s*(?:Reading|)\s*[:\-]?\s*([\d,]{2,7})\s*(?:mi|miles|km)?", t, re.IGNORECASE)
    if lab:
        return lab.group(1)
    m = re.search(r"Mileage\s*[:\-]?\s*([\d,]{2,7})", t, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

def _ocr_image_to_texts(im: Image.Image) -> list:
    texts = []
    for angle in (0, 90):
        try:
            rim = im.rotate(angle, expand=True) if angle else im
            txt = pytesseract.image_to_string(_pp(rim), lang="eng", config="--psm 6")
            if txt.strip():
                texts.append(txt)
        except Exception:
            pass
    return texts

# ==========================
# Routes
# ==========================
@app.get("/")
async def ok():
    return {"ok": True}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("comprehensive")
):
    texts: List[str] = []
    photo_texts: List[str] = []
    images_b64: List[Dict[str, Any]] = []
    max_imgs = {"invoices_with_photos":2, "supplement":2, "photos_only":6, "comprehensive":6}.get(ai_intent, 0)

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            texts.append(pdf_text_ocr(raw))
        elif name.endswith(".docx"):
            texts.append(docx_text(raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp")) and max_imgs>0:
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((1280,1280))
                for t in _ocr_image_to_texts(im):
                    photo_texts.append(t)
                b = io.BytesIO()
                im.save(b, format="JPEG", quality=72, optimize=True)
                b64 = base64.b64encode(b.getvalue()).decode("utf-8")
            except Exception:
                b64 = base64.b64encode(raw).decode("utf-8")
            images_b64.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}})
            if len(images_b64) >= max_imgs:
                pass
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8","ignore"))

    combined_text = "\n".join(texts)[:8000]
    photos_text = "\n".join(photo_texts)[:4000]

    # Pre-extract (now canonicalized)
    pre_claim = extract_claim_number(combined_text) or "N/A"
    pre_vin_est = extract_vin(combined_text)  # already canonicalized with checksum
    pre_vehicle = extract_vehicle(combined_text) or "N/A"
    pre_odo = extract_odometer_estimate(combined_text) or "N/A"

    photo_vins = extract_all_vins(photos_text)  # canonicalized list
    vin_est_c = _canon_vin(pre_vin_est) if pre_vin_est else None
    if vin_est_c:
        if photo_vins:
            vin_verify = "MATCH" if vin_est_c in photo_vins else "MISMATCH"
        else:
            vin_verify = "MISMATCH"
    else:
        vin_verify = "MISMATCH"

    vin_display = vin_est_c or (photo_vins[0] if photo_vins else "N/A")

    intent_labels = {
        "guidelines_only": "Guidelines → Estimate (no photos)",
        "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
        "photos_only": "Photos Only: Compare to Estimate",
        "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
        "supplement": "Supplement ↔ Invoices (+ Photos)",
        "docs_checklist": "Documentation Checklist"
    }
    req_label = intent_labels.get(ai_intent, intent_labels["comprehensive"])

    SYSTEM = (
        "You are an auto-claims appraisal assistant. "
        "Return ONLY valid JSON (no backticks)."
    )

    MODE_DETAILS = {
        "guidelines_only": "### Overview\n### Guidelines Compliance\n### Missing or Issues\nEnd with: Final Evaluation: NN%",
        "photos_only": "### Overview\n### Damage Consistency vs Estimate\n### Required Photos Check\nEnd with: Final Evaluation: NN%",
        "comprehensive": "### Overview\n### Estimate Integrity\n### Photo Evidence Mapping\n### VIN Verification\n### Missing or Issues\nEnd with: Final Evaluation: NN%",
        "invoices_with_photos": "### Supplement Overview\n### Invoice vs Estimate — Line-Item Deltas\n### Photo Evidence Mapping\n### Missing or Unclear Evidence\nEnd with: Final Evaluation: NN%",
        "supplement": "### Supplement Overview\n### Invoice vs Estimate — Line-Item Deltas\n### Photo Evidence Mapping\n### Missing or Unclear Evidence\nEnd with: Final Evaluation: NN%",
        "docs_checklist": "### Documentation Checklist\n### Missing Items\nEnd with: Final Evaluation: NN%"
    }

    JSON_KEYS = ["file_number","request_type","claim_number","vin","vin_verification","vehicle","odometer_estimate_only","compliance_score","summary_brief","summary_markdown"]

    mode_detail = MODE_DETAILS.get(ai_intent, MODE_DETAILS["comprehensive"])
    preface = (
        f"Mode: {ai_intent}. {mode_detail}\n"
        f"Return JSON with keys exactly: {JSON_KEYS}.\n"
        "Rules: VIN verification is already determined outside the model; DO NOT change it. Compliance Score integer 0-100.\n"
        "Provide summary_brief (<=280 chars, plain text). Provide summary_markdown (full detail).\n\n"
        f"REQUEST TYPE: {req_label}\n\nCLIENT RULES (if provided):\n{client_rules[:2500]}\n\n"
        f"PRE-EXTRACTED: {{'claim_number': '{pre_claim}', 'vin': '{vin_display}', 'vehicle': '{pre_vehicle}', 'odometer_estimate_only': '{pre_odo}', 'vin_verification': '{vin_verify}'}}\n\n"
        f"ESTIMATE/CONTENT (OCR):\n{combined_text}\n\n"
        f"PHOTO OCR:\n{photos_text}\n\n"
        f"NOTE: The next {len(images_b64)} images are Photo 1..{len(images_b64)}."
    )

    user_parts = [{"type":"text","text": preface}]
    if images_b64:
        user_parts.extend(images_b64)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},{"role":"user","content": user_parts}],
            max_tokens=800 if ai_intent in ("invoices_with_photos","supplement","comprehensive") else 550,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"OpenAI error or JSON parse error: {e}")
        data = {}

    # Build summaries with consistent VIN verification block and prevent contradictions
    summary_brief = data.get("summary_brief") or "Summary unavailable."
    summary_full = data.get("summary_markdown") or "AI analysis unavailable."
    # Remove any existing "### VIN Verification" section to avoid conflict
    summary_full = re.sub(r"###\s*VIN Verification.*?(?=\n### |\Z)", "", summary_full, flags=re.IGNORECASE | re.DOTALL).strip()
    vin_block = "### VIN Verification\n- Result: {res}\n- Estimate VIN: {v_est}\n- Photo VIN(s): {v_ph}\n".format(
        res=vin_verify,
        v_est=(vin_est_c or "N/A"),
        v_ph=(", ".join(photo_vins) if photo_vins else "N/A")
    )
    summary_full = vin_block + "\n" + summary_full

    safe = {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": data.get("claim_number") or pre_claim,
        "vin": vin_display,
        "vin_verification": vin_verify,
        "vehicle": data.get("vehicle") or pre_vehicle,
        "odometer_estimate_only": data.get("odometer_estimate_only") or pre_odo,
        "compliance_score": data.get("compliance_score") if isinstance(data.get("compliance_score"), int) else 100,
        "summary_brief": summary_brief,
        "summary_markdown": summary_full
    }

    # Build PDF
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(0, 10, "NSPXN.com AI Review Report", ln=True, align="C")
    pdf.set_font_size(10); pdf.ln(3)
    def mc(s: str): pdf.multi_cell(0,6,s)

    mc(f"File Number: {file_number}")
    mc(f"IA Company: {ia_company}")
    mc(f"Appraiser ID #: {appraiser_id}")
    mc(f"Request Type: {req_label}")
    mc(f"Claim #: {safe['claim_number']}")
    mc(f"VIN (from estimate/photos): {safe['vin']}")
    mc(f"VIN verification (estimate vs photo): {safe['vin_verification']}")
    mc(f"Vehicle: {safe['vehicle']}")
    mc(f"Odometer (from estimate): {safe['odometer_estimate_only']}")
    mc(f"Compliance Score: {safe['compliance_score']}%")

    pdf.ln(3)
    mc("AI-4-IA Review Summary")
    mc((safe["summary_markdown"] or "No summary.").strip())

    # Save with sanitized name
    safe_file = _safe_name(file_number)
    pdf_path = os.path.join(PDF_DIR, f"{safe_file}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1","ignore")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
    except Exception as e:
        log.warning(f"PDF write error: {e}")

    # Email
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {safe['claim_number']}"
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Request Type: {req_label}
Claim #: {safe['claim_number']}
VIN (from estimate/photos): {safe['vin']}
VIN verification (estimate vs photo): {safe['vin_verification']}
Vehicle: {safe['vehicle']}
Odometer (from estimate): {safe['odometer_estimate_only']}
Compliance Score: {safe['compliance_score']}%

AI-4-IA Review Summary
{safe['summary_markdown']}
""")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.error(f"Email error: {e}")

    return {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": safe["claim_number"],
        "vin": safe["vin"],
        "vin_verification": safe["vin_verification"],
        "vehicle": safe["vehicle"],
        "odometer_estimate_only": safe["odometer_estimate_only"],
        "compliance_score": safe["compliance_score"],
        "summary_brief": safe["summary_brief"],
        "summary_markdown": safe["summary_markdown"],
        "web_summary": safe["summary_brief"],   # alias for older UI
        "gpt_output": safe["summary_markdown"], # alias for older UI
        "pdf_url": f"/download-pdf?file_number={safe_file}",
        "pdf_filename": f"{safe_file}.pdf"
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = _safe_name(file_number)
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    # backward compatibility: try raw name if exists
    raw_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(raw_path):
        return FileResponse(path=raw_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})
