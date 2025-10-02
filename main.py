\
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
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
log = logging.getLogger("ultralite")

MODEL = os.getenv("OAI_MODEL", "gpt-4o")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def _safe(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "-", (s or "").strip()).strip("-_.")

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
def pdf_to_text_fast(blob: bytes, max_pages: int = 6, dpi: int = 150) -> str:
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
# Robust client rules loader (exact .docx for selected carrier, with fuzzy match)
# -----------------------
def _norm(s:str)->str: return re.sub(r"[^a-z0-9]+","", (s or "").lower())

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    if not os.path.isdir(CLIENT_RULES_DIR):
        return JSONResponse(status_code=404, content={"error":"Rules directory not found","dir":CLIENT_RULES_DIR})
    want = _norm(client_name)
    cands = []
    for root, _, files in os.walk(CLIENT_RULES_DIR):
        for fn in files:
            if fn.lower().endswith(".docx"):
                full = os.path.join(root, fn)
                base = os.path.splitext(fn)[0]
                cands.append((_norm(base), base, full))
    if not cands:
        return JSONResponse(status_code=404, content={"error":"No .docx rules found","dir":CLIENT_RULES_DIR})

    # exact normalized
    for n, base, full in cands:
        if n == want:
            try:
                txt = "\n".join([p.text for p in Document(full).paragraphs if p.text.strip()])
                return {"text": txt, "file": os.path.relpath(full, CLIENT_RULES_DIR)}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
    # prefix/contains
    for n, base, full in cands:
        if n.startswith(want) or want.startswith(n):
            try:
                txt = "\n".join([p.text for p in Document(full).paragraphs if p.text.strip()])
                return {"text": txt, "file": os.path.relpath(full, CLIENT_RULES_DIR)}
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e)})
    # similarity
    best = sorted([(len(os.path.commonprefix([want, n])), base, full) for n,base,full in cands], reverse=True)
    if best and best[0][0] >= 3:
        _, base, full = best[0]
        try:
            txt = "\n".join([p.text for p in Document(full).paragraphs if p.text.strip()])
            return {"text": txt, "file": os.path.relpath(full, CLIENT_RULES_DIR)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=404, content={"error":"Rules not found for this client.","tried":client_name})

# -----------------------
# Main review: let GPT do almost all the work
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
    # Collect minimal text (PDFs) + send photos as images
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

    content_text = "\n\n".join(ocr_chunks)[:12000]

    # System + prompt
    SYSTEM = (
        "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
        "You must extract the requested fields precisely and keep header and summary consistent. "
        "VIN handling: read VIN from the estimate text and photos; correct OCR slips (S↔5, B↔8, Z↔2, O→0, I→1, Q→0); "
        "if two VINs conflict, choose the checksum‑valid VIN; if both valid but different, use the estimate VIN. "
        "VIN verification must be 'MATCH' or 'MISMATCH' only. "
        "Compliance Score must be an integer 0–100. "
    )

    INTENT_HELP = {
        "guidelines_only": "Analyze estimate vs client guidelines only (no photos).",
        "photos_only": "Analyze photos only vs the estimate content; map damage to operations.",
        "comprehensive": "Analyze guidelines + estimate + photos, including VIN verification from images.",
        "supplement": "Analyze supplement/invoices with photos; show deltas vs estimate.",
        "invoices_with_photos": "Analyze invoices with photos; show deltas vs estimate.",
        "docs_checklist": "Audit presence/quality of required documents/photos."
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
        f"REQUEST TYPE: {ai_intent} — {INTENT_HELP.get(ai_intent,'comprehensive review')}.\n"
        "Return JSON with exactly these keys:\n"
        "['file_number','request_type','claim_number','vin','vin_verification','vehicle','odometer_estimate_only','compliance_score','summary_brief','summary_markdown']\n"
        f"Use request_type='{req_label}'.\n\n"
        "CLIENT RULES (verbatim text if provided):\n"
        + client_rules[:2500] + "\n\n"
        "ESTIMATE/CONTENT (OCR from uploads):\n" + content_text + "\n\n"
        f"There are {len(image_parts)} photos attached (order = Photo 1..{len(image_parts)}). "
        "Use them for VIN reading and evidence mapping.\n"
        "Constraints:\n"
        "- VIN must be 17 chars uppercase (checksum-valid if possible).\n"
        "- vin_verification must be MATCH or MISMATCH.\n"
        "- summary_brief: one short paragraph (<=280 chars), plain text.\n"
        "- summary_markdown: full write-up, include a '### VIN Verification' subsection that repeats vin and result."
    )

    user_parts: List[Dict[str,Any]] = [{"type":"text","text": USER_TEXT}]
    if image_parts: user_parts.extend(image_parts)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},{"role":"user","content": user_parts}],
            max_tokens=900,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"LLM failure or JSON parse error: {e}")
        data = {}

    # Safe defaults + minor sanitation
    def _iget(x, dflt): 
        try:
            return int(x)
        except Exception:
            return dflt

    result = {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": str(data.get("claim_number") or "N/A"),
        "vin": str(data.get("vin") or "N/A").upper(),
        "vin_verification": "MATCH" if str(data.get("vin_verification") or "").upper() == "MATCH" else "MISMATCH",
        "vehicle": str(data.get("vehicle") or "N/A"),
        "odometer_estimate_only": str(data.get("odometer_estimate_only") or "N/A"),
        "compliance_score": _iget(data.get("compliance_score"), 100),
        "summary_brief": str(data.get("summary_brief") or "Summary unavailable."),
        "summary_markdown": str(data.get("summary_markdown") or "AI analysis unavailable.")
    }

    # PDF build (unchanged header block you requested)
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
