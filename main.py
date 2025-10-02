\
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os, io, re, json, base64, logging, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes   # for PDFs only
import pytesseract                          # for PDFs only
from PIL import Image

from openai import OpenAI

# -----------------------
# Basic setup
# -----------------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
CLIENT_RULES_DIR = os.getenv("CLIENT_RULES_DIR", "client_rules")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("nspxn-ultralite")

MODEL = os.getenv("OAI_MODEL", "gpt-4o")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _safe(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "-", (s or "").strip()).strip("-_.")

def _as_int(val, default=100):
    try:
        if isinstance(val, (int, float)): return int(val)
        if isinstance(val, str):
            m = re.search(r"\d+", val)
            if m: return int(m.group(0))
        return default
    except Exception:
        return default

# -----------------------
# App + CORS
# -----------------------
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

# -----------------------
# Fast OCR (PDFs only). Photos go straight to GPT.
# -----------------------
def pdf_to_text_fast(blob: bytes, max_pages: int = 5, dpi: int = 150) -> str:
    try:
        pages = convert_from_bytes(blob, dpi=dpi)[:max_pages]
    except Exception as e:
        log.warning(f"pdf2image failed: {e}")
        return ""
    txts = []
    for i, im in enumerate(pages, 1):
        try:
            t = pytesseract.image_to_string(im, lang="eng", config="--psm 6")
            if t.strip():
                txts.append(f"[Page {i}]\n{t}")
        except Exception as e:
            log.warning(f"tesseract page {i}: {e}")
    return "\n\n".join(txts)

# -----------------------
# Client rules loader — EXACT docx by name (as before)
# -----------------------
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    base = client_name.strip()
    candidates = []
    if base.lower().endswith(".docx"):
        candidates.append(os.path.join(CLIENT_RULES_DIR, base))
        candidates.append(os.path.join(CLIENT_RULES_DIR, base[:-5] + ".DOCX"))
    else:
        candidates.append(os.path.join(CLIENT_RULES_DIR, base + ".docx"))
        candidates.append(os.path.join(CLIENT_RULES_DIR, base + ".DOCX"))

    path = _first_existing(candidates)
    if not path:
        try:
            for fn in os.listdir(CLIENT_RULES_DIR):
                if fn.lower().endswith(".docx") and os.path.splitext(fn)[0].lower() == base.lower():
                    path = os.path.join(CLIENT_RULES_DIR, fn)
                    break
        except FileNotFoundError:
            return JSONResponse(status_code=404, content={"error":"Rules directory not found","dir":CLIENT_RULES_DIR})

    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})

    try:
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unable to read rules: {e}"})

# -----------------------
# Minimal pre-extractors (just to avoid 'N/A' when the data exists)
# -----------------------
def pre_claim_number(text: str) -> Optional[str]:
    pats = [
        r"Claim\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
        r"Assignment\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
        r"Reference\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-_\/]+)",
    ]
    for p in pats:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(".,;")
    return None

def pre_estimate_vin(text: str) -> Optional[str]:
    m = re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", text)
    return m.group(0) if m else None

# -----------------------
# Main review: GPT does the work; we enforce only 2 things
#   1) VIN verification cannot be NOT VERIFIED (MATCH only if estimate VIN exists)
#   2) Summary must not list photos generically; we sanitize if GPT does
# -----------------------
@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("comprehensive")
):
    # Collect minimal text (PDFs/docx) + send photos as images
    ocr_chunks = []
    image_parts: List[Dict[str,Any]] = []
    photo_count = 0
    max_imgs = {"photos_only":6,"comprehensive":6,"supplement":4,"invoices_with_photos":4,"guidelines_only":0,"docs_checklist":6}.get(ai_intent, 4)

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            ocr_chunks.append(pdf_to_text_fast(raw))
        elif name.endswith(".docx"):
            try:
                txt = "\n".join([p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()])
            except Exception:
                txt = ""
            ocr_chunks.append(txt)
        elif name.endswith((".jpg",".jpeg",".png",".webp")) and photo_count < max_imgs:
            photo_count += 1
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((1400,1400))
                b = io.BytesIO(); im.save(b, format="JPEG", quality=70, optimize=True)
                raw = b.getvalue()
            except Exception:
                pass
            b64 = base64.b64encode(raw).decode("utf-8")
            image_parts.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}})
        elif name.endswith(".txt"):
            ocr_chunks.append(raw.decode("utf-8","ignore"))

    content_text = "\n\n".join(ocr_chunks)[:16000]

    # Pre-extract to avoid "N/A" when info is present
    claim_guess = pre_claim_number(content_text) or "N/A"
    vin_guess   = pre_estimate_vin(content_text) or "N/A"
    has_estimate_vin = vin_guess != "N/A"

    SYSTEM = (
        "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
        "Keep header and summary consistent. "
        "VIN handling: read VIN from the estimate (OCR text) and photos; correct OCR slips (S↔5, B↔8, Z↔2, O→0, I→1, Q→0). "
        "VIN verification rules: MATCH only if an estimate VIN is present in the OCR text and equals a VIN visible in the photos; "
        "otherwise MISMATCH. Do not output NOT VERIFIED. "
        "Provide very detailed, structured analysis per request type."
    )

    DETAIL_TEMPLATES = {
        "guidelines_only": (
            "### Overview\n"
            "- Brief context of the estimate and scope.\n\n"
            "### Guidelines Compliance\n"
            "| Check | Expected | Observed | Pass/Fail | Notes |\n"
            "|---|---:|---:|:--:|---|\n"
            "Include labor rates, refinish/overlap, materials, tax handling, OEM procedures, sublet docs.\n\n"
            "### Missing / Issues\n"
            "- Bulleted list of missing docs or ambiguities.\n\n"
            "End with: Final Evaluation: NN%"
        ),
        "photos_only": (
            "### Overview\n"
            "- Photo coverage, clarity, and labeling.\n\n"
            "### Damage Consistency vs Estimate\n"
            "| Area/Part | Est. Op | Photo Ref(s) | Consistent? | Notes |\n"
            "|---|---|---|:--:|---|\n\n"
            "### Required Photos Check\n"
            "- VIN, plate, odometer, 4 corners, close-ups; mark any missing.\n\n"
            "End with: Final Evaluation: NN%"
        ),
        "comprehensive": (
            "### Overview\n"
            "- Estimate scope + photo coverage.\n\n"
            "### Estimate Integrity\n"
            "| Topic | Finding | Impact |\n"
            "|---|---|---|\n"
            "Labor/materials/tax/refinish overlap/sublet/OEM procedures.\n\n"
            "### Photo Evidence Mapping\n"
            "| Line Item / Part | Photo Ref(s) | Rationale |\n"
            "|---|---|---|\n\n"
            "### VIN Verification\n"
            "- State MATCH/MISMATCH and list the VIN(s) read.\n\n"
            "### Missing / Issues\n"
            "- Bulleted list.\n\n"
            "End with: Final Evaluation: NN%"
        ),
        "supplement": (
            "### Supplement Overview\n"
            "- Totals, scope, and reasons for changes.\n\n"
            "### Invoice vs Estimate — Deltas\n"
            "| Part/Operation | Qty | $Estimate | $Invoice | Δ (±) | Evidence (Photo/Doc) | Rationale |\n"
            "|---|---:|---:|---:|---:|---|---|\n\n"
            "### Missing or Unclear Evidence\n"
            "- Bulleted list.\n\n"
            "End with: Final Evaluation: NN%"
        ),
        "invoices_with_photos": (
            "### Supplement Overview\n"
            "- Totals, scope, and reasons for changes.\n\n"
            "### Invoice vs Estimate — Deltas\n"
            "| Part/Operation | Qty | $Estimate | $Invoice | Δ (±) | Evidence (Photo/Doc) | Rationale |\n"
            "|---|---:|---:|---:|---:|---|---|\n\n"
            "### Missing or Unclear Evidence\n"
            "- Bulleted list.\n\n"
            "End with: Final Evaluation: NN%"
        ),
        "docs_checklist": (
            "### Documentation Checklist\n"
            "| Item | Present? | Notes |\n"
            "|---|:--:|---|\n"
            "Estimate, invoices, photos (VIN/plate/odometer/4-corners/damage close‑ups), registration, policy docs.\n\n"
            "### Missing Items\n"
            "- Bulleted list.\n\n"
            "End with: Final Evaluation: NN%"
        )
    }

    req_label = {
        "guidelines_only": "Guidelines → Estimate (no photos)",
        "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
        "photos_only": "Photos Only: Compare to Estimate",
        "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
        "supplement": "Supplement ↔ Invoices (+ Photos)",
        "docs_checklist": "Documentation Checklist"
    }.get(ai_intent, "Comprehensive: Guidelines + Estimate + Photos (with VIN check)")

    USER_TEXT = (
        f"REQUEST TYPE: {ai_intent}. Use layout below and return JSON only.\n"
        "Keys (exact): ['file_number','request_type','claim_number','vin','vin_verification','vehicle','odometer_estimate_only','compliance_score','summary_brief','summary_markdown']\n"
        f"Set request_type='{req_label}'.\n\n"
        "CLIENT RULES (text if provided):\n" + client_rules[:2500] + "\n\n"
        "ESTIMATE/CONTENT (OCR text from uploads):\n" + content_text + "\n\n"
        f"There are {len(image_parts)} photos attached (order = Photo 1..{len(image_parts)}).\n"
        "IMPORTANT OUTPUT RULES:\n"
        "- Do NOT dump a numbered list of photos. Only reference Photo # when used as evidence in a table/section.\n"
        "- If no estimate VIN is present in OCR text, set vin_verification=MISMATCH and explicitly say 'No estimate VIN present' in the VIN section.\n"
        "- Provide rich, structured detail per the template below.\n\n"
        "DETAIL LAYOUT for this request type:\n" + DETAIL_TEMPLATES.get(ai_intent, DETAIL_TEMPLATES["comprehensive"]) + "\n"
    )

    user_parts: List[Dict[str,Any]] = [{"type":"text","text": USER_TEXT}]
    if image_parts: user_parts.extend(image_parts)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},{"role":"user","content": user_parts}],
            max_tokens=1100,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"LLM failure or JSON parse error: {e}")
        data = {}

    # Fill with pre-extracted hints if GPT left fields blank
    vin_model = str(data.get("vin") or "").strip().upper()
    claim_model = str(data.get("claim_number") or "").strip()
    vehicle_model = str(data.get("vehicle") or "N/A")
    odo_model = str(data.get("odometer_estimate_only") or "N/A")

    vin_final = vin_model or vin_guess
    claim_final = claim_model or claim_guess

    # Minimal guard: forbid NOT VERIFIED; force MISMATCH when no estimate VIN
    vin_ver = (data.get("vin_verification") or "").upper()
    if vin_ver != "MATCH":
        vin_ver = "MISMATCH"
    if vin_ver == "MATCH" and not has_estimate_vin:
        vin_ver = "MISMATCH"

    summary_brief = str(data.get("summary_brief") or "Summary unavailable.")
    summary_full = str(data.get("summary_markdown") or "AI analysis unavailable.")

    # Sanitize out any generic "### Photos" list
    summary_full = re.sub(r"(?is)###\s*Photos\s*.*?(?=\n###|\Z)", "", summary_full).strip()

    # Prepend canonical VIN block to avoid contradictions
    vin_block = "### VIN Verification\n- Result: {res}\n- Estimate VIN present in OCR: {present}\n".format(
        res=vin_ver,
        present=("Yes" if has_estimate_vin else "No")
    )
    summary_full = (vin_block + "\n" + summary_full).strip()

    # Build final result
    result = {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": claim_final or "N/A",
        "vin": vin_final or "N/A",
        "vin_verification": vin_ver,
        "vehicle": vehicle_model,
        "odometer_estimate_only": odo_model,
        "compliance_score": _as_int(data.get("compliance_score"), 100),
        "summary_brief": summary_brief,
        "summary_markdown": summary_full
    }

    # PDF build (fixed header)
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)
    pdf.cell(0,10,"NSPXN.com AI Review Report", ln=True, align="C")
    pdf.set_font_size(10); pdf.ln(3)
    def mc(s): pdf.multi_cell(0,6,s)
    mc(f"File Number: {file_number}")
    mc(f"IA Company: {ia_company}")
    mc(f"Appraiser ID #: {appraiser_id}")
    mc(f"Request Type: {req_label}")
    mc(f"Claim #: {result['claim_number']}")
    mc(f"VIN (from estimate/photos): {result['vin']}")
    mc(f"VIN verification (estimate vs photo): {result['vin_verification']}")
    mc(f"Vehicle: {result['vehicle']}")
    mc(f"Odometer (from estimate): {result['odometer_estimate_only']}")
    mc(f"Compliance Score: {result['compliance_score']}%")
    pdf.ln(3); mc("AI-4-IA Review Summary"); mc(result["summary_markdown"].strip())

    safe_file = _safe(file_number); pdf_path = os.path.join(PDF_DIR, f"{safe_file}.pdf")
    try:
        data_bytes = pdf.output(dest="S").encode("latin-1","ignore")
        with open(pdf_path,"wb") as f: f.write(data_bytes)
    except Exception as e:
        log.warning(f"PDF write error: {e}")

    # Email (Tierra.net SMTP)
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {result['claim_number']}"
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Request Type: {req_label}
Claim #: {result['claim_number']}
VIN (from estimate/photos): {result['vin']}
VIN verification (estimate vs photo): {result['vin_verification']}
Vehicle: {result['vehicle']}
Odometer (from estimate): {result['odometer_estimate_only']}
Compliance Score: {result['compliance_score']}%

AI-4-IA Review Summary
{result['summary_markdown']}
""")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.error(f"Email error: {e}")

    return {
        **result,
        "web_summary": result["summary_brief"],
        "gpt_output": result["summary_markdown"],
        "pdf_url": f"/download-pdf?file_number={safe_file}",
        "pdf_filename": f"{safe_file}.pdf"
    }

# -----------------------
# PDF download
# -----------------------
@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = _safe(file_number)
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    raw_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(raw_path):
        return FileResponse(path=raw_path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})
