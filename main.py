
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
                out.append(f"[Page {i}]\\n{txt}")
        except Exception as e:
            log.warning(f"OCR page {i} failed: {e}")
    return "\\n\\n".join(out)

def docx_text(blob: bytes) -> str:
    try:
        d = Document(io.BytesIO(blob))
        return "\\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        log.warning(f"docx read error: {e}")
        return ""

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
    max_imgs = {"invoices_with_photos":2, "supplement":2, "photos_only":4, "comprehensive":6}.get(ai_intent, 0)

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

    combined_text = "\\n".join(texts)[:6000]  # trim for speed

    # Use GPT for EVERYTHING: extract fields + compute score + build summary
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

    PER_INTENT = {
        "guidelines_only": "Ignore photos. Analyze estimate text against provided client rules (if any).",
        "photos_only": "Ignore guidelines. Compare photos to estimate text for damage consistency and completeness.",
        "invoices_with_photos": "Analyze supplement invoices against estimate text. Include photos only if relevant. No photo/NADA/Advisor deductions.",
        "supplement": "Analyze supplement invoices against estimate text. Include photos only if relevant. No photo/NADA/Advisor deductions.",
        "docs_checklist": "Create a documentation checklist status strictly from the provided text.",
        "comprehensive": "Analyze guidelines + estimate + photos. Verify VIN between estimate and photos if possible."
    }
    mode_tip = PER_INTENT.get(ai_intent, PER_INTENT["comprehensive"])

    JSON_KEYS = ["file_number","request_type","claim_number","vin","vin_verification","vehicle","odometer_estimate_only","compliance_score","summary_markdown"]

    user_parts: List[Dict[str, Any]] = [
        {"type":"text","text": (
            f"Mode: {ai_intent}. {mode_tip}\n"
            f"Return JSON with keys exactly: {JSON_KEYS}.\n"
            "Rules: VIN verification must be one of: MATCH, MISMATCH, NOT VERIFIED; Compliance Score integer 0-100; "
            "Populate all fields ONLY from the provided content; If missing, use 'N/A'.\n\n"
            f"REQUEST TYPE: {req_label}\n\nCLIENT RULES (if provided):\n{client_rules[:2500]}\n\n"
            f"ESTIMATE/CONTENT (OCR):\n{combined_text}"
        )}
    ]
    if images_b64:
        user_parts.extend(images_b64)

    import json as _json
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},{"role":"user","content": user_parts}],
            max_tokens=450,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = _json.loads(raw)
    except Exception as e:
        log.error(f"OpenAI error or JSON parse error: {e}")
        data = {
            "file_number": file_number,
            "request_type": req_label,
            "claim_number": "N/A",
            "vin": "N/A",
            "vin_verification": "NOT VERIFIED",
            "vehicle": "N/A",
            "odometer_estimate_only": "N/A",
            "compliance_score": 100,
            "summary_markdown": "AI analysis unavailable."
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
    mc(f"Claim #: {data.get('claim_number','N/A')}")
    mc(f"VIN (from estimate/photos): {data.get('vin','N/A')}")
    mc(f"VIN verification (estimate vs photo): {data.get('vin_verification','NOT VERIFIED')}")
    mc(f"Vehicle: {data.get('vehicle','N/A')}")
    mc(f"Odometer (from estimate): {data.get('odometer_estimate_only','N/A')}")
    mc(f"Compliance Score: {data.get('compliance_score', 'N/A')}%")

    pdf.ln(3)
    mc("AI-4-IA Review Summary")
    mc((data.get("summary_markdown","") or "No summary.").strip())

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
        msg["Subject"] = f"AI-4-IA Review: {data.get('claim_number','N/A')}"
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Request Type: {req_label}
Claim #: {data.get('claim_number','N/A')}
VIN (from estimate/photos): {data.get('vin','N/A')}
VIN verification (estimate vs photo): {data.get('vin_verification','NOT VERIFIED')}
Vehicle: {data.get('vehicle','N/A')}
Odometer (from estimate): {data.get('odometer_estimate_only','N/A')}
Compliance Score: {data.get('compliance_score','N/A')}%

AI-4-IA Review Summary
{data.get('summary_markdown','')}
""")
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.error(f"Email error: {e}")

    return {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": data.get("claim_number","N/A"),
        "vin": data.get("vin","N/A"),
        "vin_verification": data.get("vin_verification","NOT VERIFIED"),
        "vehicle": data.get("vehicle","N/A"),
        "odometer_estimate_only": data.get("odometer_estimate_only","N/A"),
        "compliance_score": data.get("compliance_score", "N/A"),
        "summary_markdown": data.get("summary_markdown",""),
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
