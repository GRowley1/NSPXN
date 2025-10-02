\
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os, io, re, json, base64, logging, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
from PIL import Image

from openai import OpenAI

# -----------------------
# Minimal setup
# -----------------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
CLIENT_RULES_DIR = os.getenv("CLIENT_RULES_DIR", "client_rules")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("nspxn-min")

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
# EXACT client rules loader (no fuzz)
# -----------------------
@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    base = client_name.strip()
    if not base.lower().endswith(".docx"):
        base = base + ".docx"
    path = os.path.join(CLIENT_RULES_DIR, base)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})
    try:
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return {"text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unable to read rules: {e}"})

# -----------------------
# Vision Review — GPT does EVERYTHING.
# No VIN checks, no coercions, no sanitizing. We just relay inputs and render outputs.
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
    # Collect content with minimal processing: turn PDFs to images; pass photos as images.
    parts: List[Dict[str, Any]] = []
    MAX_IMAGES = 12  # cap to keep requests reasonable
    used = 0

    text_concat = []  # if .txt or .docx present, include their text as a text block

    for f in files:
        raw = await f.read()
        fname = (f.filename or "upload").lower()
        if fname.endswith(".pdf") and used < MAX_IMAGES:
            try:
                pages = convert_from_bytes(raw, dpi=150)
                for im in pages[:MAX_IMAGES - used]:
                    b = io.BytesIO()
                    im.save(b, format="JPEG", quality=70, optimize=True)
                    b64 = base64.b64encode(b.getvalue()).decode("utf-8")
                    parts.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}})
                    used += 1
            except Exception as e:
                log.warning(f"pdf2image failed: {e}")
        elif fname.endswith((".jpg",".jpeg",".png",".webp")) and used < MAX_IMAGES:
            try:
                im = Image.open(io.BytesIO(raw)).convert("RGB")
                im.thumbnail((1400,1400))
                b = io.BytesIO(); im.save(b, format="JPEG", quality=70, optimize=True)
                raw = b.getvalue()
            except Exception:
                pass
            b64 = base64.b64encode(raw).decode("utf-8")
            parts.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}})
            used += 1
        elif fname.endswith(".docx"):
            try:
                text = "\n".join([p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()])
                if text.strip(): text_concat.append(text[:8000])
            except Exception:
                pass
        elif fname.endswith(".txt"):
            try:
                text_concat.append(raw.decode("utf-8","ignore")[:8000])
            except Exception:
                pass

    if text_concat:
        parts.insert(0, {"type":"text","text":"\n\n".join(text_concat)})

    SYSTEM = (
        "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
        "Populate these exact keys: "
        "['file_number','request_type','claim_number','vin','vin_verification','vehicle','odometer_estimate_only','compliance_score','summary_brief','summary_markdown'] . "
        "Do NOT invent fields. Keep header values and summary consistent. "
        "Write a rich, detailed 'summary_markdown' tailored to the request type."
    )

    REQ_LABELS = {
        "guidelines_only": "Guidelines → Estimate (no photos)",
        "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
        "photos_only": "Photos Only: Compare to Estimate",
        "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
        "supplement": "Supplement ↔ Invoices (+ Photos)",
        "docs_checklist": "Documentation Checklist"
    }
    req_label = REQ_LABELS.get(ai_intent, "Comprehensive: Guidelines + Estimate + Photos (with VIN check)")

    DETAIL_TEMPLATES = {
        "guidelines_only": (
            "### Overview\n"
            "### Guidelines Compliance (table)\n"
            "### Missing / Issues\n"
            "Final Evaluation: NN%"
        ),
        "photos_only": (
            "### Overview\n"
            "### Damage Consistency vs Estimate (table with Photo refs)\n"
            "### Required Photos Check\n"
            "Final Evaluation: NN%"
        ),
        "comprehensive": (
            "### Overview\n"
            "### Estimate Integrity (table)\n"
            "### Photo Evidence Mapping (table with Photo refs)\n"
            "### VIN Verification\n"
            "### Missing / Issues\n"
            "Final Evaluation: NN%"
        ),
        "supplement": (
            "### Supplement Overview\n"
            "### Invoice vs Estimate — Deltas (table with $Estimate/$Invoice/Δ/Evidence/Rationale)\n"
            "### Missing or Unclear Evidence\n"
            "Final Evaluation: NN%"
        ),
        "invoices_with_photos": (
            "### Supplement Overview\n"
            "### Invoice vs Estimate — Deltas (table with $Estimate/$Invoice/Δ/Evidence/Rationale)\n"
            "### Missing or Unclear Evidence\n"
            "Final Evaluation: NN%"
        ),
        "docs_checklist": (
            "### Documentation Checklist (matrix)\n"
            "### Missing Items\n"
            "Final Evaluation: NN%"
        )
    }

    prompt_text = (
        f"REQUEST TYPE: {ai_intent} — Use request_type='{req_label}'.\n\n"
        "CLIENT RULES (verbatim text if provided by the UI):\n" + (client_rules[:2500] if client_rules else "") + "\n\n"
        "IMPORTANT:\n"
        "- Use the images and any provided text to extract values.\n"
        "- compliance_score may be a number like 95 or a string like '95%'; do not include extra keys.\n"
        "- summary_brief <= 280 chars (plain text). summary_markdown = full write-up using the template below.\n\n"
        "DETAIL LAYOUT:\n" + DETAIL_TEMPLATES.get(ai_intent, DETAIL_TEMPLATES["comprehensive"]) + "\n"
    )

    user_parts: List[Dict[str,Any]] = [{"type":"text","text": prompt_text}]
    if parts: user_parts.extend(parts)

    # Call GPT and trust the output
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": user_parts}],
            max_tokens=1100,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"LLM failure or JSON parse error: {e}")
        return JSONResponse(status_code=500, content={"error":"Model output could not be parsed as JSON."})

    # Build final result — NO transformations; just defaults if missing
    def _get(k): 
        v = data.get(k)
        return "" if v is None else str(v)

    result = {
        "file_number": file_number,
        "request_type": _get("request_type") or req_label,
        "claim_number": _get("claim_number"),
        "vin": _get("vin"),
        "vin_verification": _get("vin_verification"),
        "vehicle": _get("vehicle"),
        "odometer_estimate_only": _get("odometer_estimate_only"),
        "compliance_score": _get("compliance_score"),
        "summary_brief": _get("summary_brief"),
        "summary_markdown": _get("summary_markdown"),
    }

    # PDF — print values exactly as GPT returned (no % added, no edits)
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
    mc(f"Request Type: {result['request_type']}")
    mc(f"Claim #: {result['claim_number']}")
    mc(f"VIN (from estimate/photos): {result['vin']}")
    mc(f"VIN verification (estimate vs photo): {result['vin_verification']}")
    mc(f"Vehicle: {result['vehicle']}")
    mc(f"Odometer (from estimate): {result['odometer_estimate_only']}")
    mc(f"Compliance Score: {result['compliance_score']}")
    pdf.ln(3); mc("AI-4-IA Review Summary"); mc((result["summary_markdown"] or "").strip())

    safe_file = _safe(file_number); pdf_path = os.path.join(PDF_DIR, f"{safe_file}.pdf")
    try:
        data_bytes = pdf.output(dest="S").encode("latin-1","ignore")
        with open(pdf_path,"wb") as f: f.write(data_bytes)
    except Exception as e:
        log.warning(f"PDF write error: {e}")

    # Email (Tierra.net SMTP) — same exact values
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {result['claim_number'] or file_number}"
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}
Request Type: {result['request_type']}
Claim #: {result['claim_number']}
VIN (from estimate/photos): {result['vin']}
VIN verification (estimate vs photo): {result['vin_verification']}
Vehicle: {result['vehicle']}
Odometer (from estimate): {result['odometer_estimate_only']}
Compliance Score: {result['compliance_score']}

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
