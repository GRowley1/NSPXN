
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
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
# Legacy Client Rules endpoint (JSON)
# =========================================
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
            return JSONResponse(status_code=500, content={"error": str(e)})
    else:
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})

# =========================================
# Deterministic pre-extraction (to boost reliability)
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

def normalize_vin(v: str) -> Optional[str]:
    if not v: return None
    v = v.upper().replace(" ", "").replace("O","0").replace("I","1").replace("Q","0")
    if not re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", v): return None
    return v

def extract_vin(text: str) -> Optional[str]:
    t = _clean(text).upper()
    # Try explicit label first
    lab = re.findall(r"\bVIN\s*[:#\-]?\s*([A-HJ-NPR-Z0-9]{10,20})", t, re.IGNORECASE)
    for c in lab:
        v = normalize_vin(c)
        if v: return v
    # Then any 17-char candidate
    cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", t, re.IGNORECASE)
    for c in cands:
        v = normalize_vin(c)
        if v: return v
    return None

MAKE_MAP = {
    "CHEV":"Chevrolet","CHEVY":"Chevrolet","MB":"Mercedes-Benz","MERCEDES":"Mercedes-Benz",
    "VW":"Volkswagen","VOLKS":"Volkswagen","GMC":"GMC","TOYOTA":"Toyota","HONDA":"Honda",
    "NISSAN":"Nissan","HYUNDAI":"Hyundai","KIA":"Kia","FORD":"Ford","DODGE":"Dodge","RAM":"Ram",
    "SUBARU":"Subaru","MAZDA":"Mazda","BMW":"BMW","AUDI":"Audi","JEEP":"Jeep"
}
def extract_vehicle(text: str) -> Optional[str]:
    t = _clean(text)
    # Look for a "Vehicle:" line first
    m = re.search(r"Vehicle\s*[:\-]\s*(.+)", t, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if len(val) > 2 and len(val) < 120: return val
    # Else a "YEAR MAKE MODEL" line
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
    # Sometimes it's "Mileage"
    m = re.search(r"Mileage\s*[:\-]?\s*([\d,]{2,7})", t, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

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
    appraiser_id: str = Form(...),
    ai_intent: str = Form("comprehensive")
):
    # Ingest: gather text + a few compressed images (for speed)
    texts: List[str] = []
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

    combined_text = "\n".join(texts)[:8000]  # leave a bit more room for field search

    # Deterministic pre-extract to boost reliability
    pre = {
        "claim_number": extract_claim_number(combined_text) or "N/A",
        "vin": extract_vin(combined_text) or "N/A",
        "vehicle": extract_vehicle(combined_text) or "N/A",
        "odometer_estimate_only": extract_odometer_estimate(combined_text) or "N/A",
    }

    # Use GPT for EVERYTHING else; also cross-check and improve pre-extracted fields
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
        "Given the OCR'ed text and up to a few images, do ALL analysis based on the request type. "
        "Never invent data. If a field is not present, return 'N/A'. "
        "VIN Verification must be one of: MATCH, MISMATCH, or NOT VERIFIED. "
        "Compliance Score must be an integer 0-100. "
        "Return ONLY valid JSON and nothing else."
    )

    SUPP_DETAILS = (
        "When the request type is a supplement (invoices_with_photos or supplement), "
        "produce a detailed markdown summary with these sections:\n\n"
        "### Supplement Overview\n"
        "- High-level summary of the supplement scope and totals.\n\n"
        "### Invoice vs Estimate — Line-Item Deltas\n"
        "Provide a compact table with columns: Part/Operation | Qty | $Estimate | $Invoice | Δ (±) | Rationale.\n"
        "Use item descriptions and amounts extracted verbatim from the provided text when possible.\n\n"
        "### Photo Evidence Mapping\n"
        "Assume attached images are ordered as Photo 1..N in the same order provided. "
        "List each key added/changed item and reference the most relevant Photo #(s) if any cues from the text/photos suggest it. "
        "If ambiguous, say 'No clear photo evidence'.\n\n"
        "### Missing or Unclear Evidence\n"
        "List any items where invoices or estimate references are incomplete or not found.\n\n"
        "End with a single line exactly in this format: Final Evaluation: NN%"
    )

    PER_INTENT = {
        "guidelines_only": "Ignore photos. Analyze estimate text against provided client rules (if any).",
        "photos_only": "Ignore guidelines. Compare photos to estimate text for damage consistency and completeness.",
        "invoices_with_photos": "Analyze supplement invoices against estimate text. Include photos only if relevant. No photo/NADA/Advisor deductions.",
        "supplement": "Analyze supplement invoices against estimate text. Include photos only if relevant. No photo/NADA/Advisor deductions.",
        "docs_checklist": "Create a documentation checklist status strictly from the provided text.",
        "comprehensive": "Analyze guidelines + estimate + photos. Verify VIN between estimate and photos if possible."
    }
    mode_tip = PER_INTENT.get(ai_intent, PER_INTENT["comprehensive"])

    JSON_KEYS = ["file_number","request_type","claim_number","vin","vin_verification","vehicle","odometer_estimate_only","compliance_score","summary_brief","summary_markdown"]

    preface = f"Mode: {ai_intent}. {mode_tip}\n"
    if ai_intent in ("invoices_with_photos","supplement"):
        preface += SUPP_DETAILS + "\n\n"
    preface += (
        f"Return JSON with keys exactly: {JSON_KEYS}.\n"
        "Rules: VIN verification must be one of: MATCH, MISMATCH, NOT VERIFIED; Compliance Score integer 0-100; "
        "Populate all fields ONLY from the provided content; If missing, use 'N/A'.\n"
        "Provide summary_brief as a single short paragraph (<=280 chars), plain text (no bullets/markdown).\n"
        "Provide summary_markdown as the full detailed write‑up.\n\n"
        f"REQUEST TYPE: {req_label}\n\nCLIENT RULES (if provided):\n{client_rules[:2500]}\n\n"
        f"PRE-EXTRACTED (use to cross-check, correct if wrong):\n{json.dumps(pre)}\n\n"
        f"ESTIMATE/CONTENT (OCR):\n{combined_text}\n\n"
        f"NOTE: The next {len(images_b64)} images (if any) are the photos in order: Photo 1..Photo {len(images_b64)}."
    )

    user_parts: List[Dict[str, Any]] = [{"type":"text","text": preface}]
    if images_b64:
        user_parts.extend(images_b64)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},{"role":"user","content": user_parts}],
            max_tokens=700 if ai_intent in ("invoices_with_photos","supplement") else 500,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"OpenAI error or JSON parse error: {e}")
        data = {}

    # Fill with pre-extracted defaults if GPT missed them
    safe = {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": data.get("claim_number") or pre["claim_number"],
        "vin": data.get("vin") or pre["vin"],
        "vin_verification": data.get("vin_verification") or "NOT VERIFIED",
        "vehicle": data.get("vehicle") or pre["vehicle"],
        "odometer_estimate_only": data.get("odometer_estimate_only") or pre["odometer_estimate_only"],
        "compliance_score": data.get("compliance_score") if isinstance(data.get("compliance_score"), int) else 100,
        "summary_brief": data.get("summary_brief") or "Summary unavailable.",
        "summary_markdown": data.get("summary_markdown") or "AI analysis unavailable."
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
