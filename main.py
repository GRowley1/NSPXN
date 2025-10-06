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

# Optional HEIC/HEIF support if available
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()  # type: ignore
except Exception:
    pass

from openai import OpenAI

# -----------------------
# Minimal setup
# -----------------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
CLIENT_RULES_DIR = os.getenv("CLIENT_RULES_DIR", "client_rules")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("nspxn-nologic")

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
# Prompt steering only (no backend logic)
# -----------------------
DETAIL_TEMPLATES = {
    "guidelines_only": (
        "## Inputs Used\n"
        "- List the exact files and sections used.\n\n"
        "## Executive Summary\n"
        "- 2–4 bullets on overall compliance and key risks.\n\n"
        "## Guidelines Compliance\n"
        "| Check | Guideline / Source | Estimate Evidence | Pass/Fail | Impact | Notes |\n"
        "|---|---|---|:--:|---|---|\n"
        "Include labor rates, refinish/overlap, materials, tax handling, OEM procedures, sublet docs.\n\n"
        "## Missing / Issues\n"
        "- Label severity: **High / Med / Low** with a one-line fix.\n\n"
        "## Final Evaluation\n"
        "- Compliance Score: NN%\n"
        "- One-sentence justification."
    ),
    "photos_only": (
        "## Inputs Used\n"
        "- State exact photos referenced (Photo #) and any text files.\n\n"
        "## Executive Summary\n"
        "- Coverage sufficiency + biggest gaps.\n\n"
        "## Damage Consistency vs Estimate\n"
        "| Area / Part | Observed Damage (Photo #) | Likely Operation | Consistent? | Notes |\n"
        "|---|---|---|:--:|---|\n\n"
        "## Required Photos Check\n"
        "- VIN, plate, odometer, 4 corners, damage close-ups; mark **Present / Missing / Unclear**.\n\n"
        "## Final Evaluation\n"
        "- Coverage Adequacy: Low/Med/High and 1–2 action items."
    ),
    "comprehensive": (
        "## Inputs Used\n"
        "- Enumerate files seen; reference estimate pages and Photo # where applicable.\n\n"
        "## Executive Summary\n"
        "- 2–4 bullets: overall integrity, major deltas, risk items.\n\n"
        "## Key Identifiers\n"
        "- Claim #, VIN(s) seen, Year/Make/Model (only if clearly supported; else N/A).\n\n"
        "## Estimate Integrity\n"
        "| Topic | Evidence (estimate page/section) | Finding | Impact |\n"
        "|---|---|---|---|\n"
        "Labor & materials, refinish overlap, sublet, OEM procedures, tax/markup.\n\n"
        "## Photo Evidence Mapping\n"
        "| Line / Part | Photo # | What the photo shows | Consistent? | Notes |\n"
        "|---|---|---|:--:|---|\n\n"
        "## VIN Verification\n"
        "- Label one of **MATCH / MISMATCH / NOT VERIFIED** and explicitly list: estimate VIN vs photo VIN(s). If any piece is unreadable, say so.\n\n"
        "## Missing / Issues\n"
        "- High/Med/Low with recommended fix.\n\n"
        "## Final Evaluation\n"
        "- Compliance Score: NN% with one-sentence rationale."
    ),
    "supplement": (
        "## Inputs Used\n"
        "- List invoices, estimate pages, and Photo # used as evidence.\n\n"
        "## Supplement Overview\n"
        "- What changed vs original estimate and why (diagnostics, teardown, hidden damage, OEM reqs).\n\n"
        "## Invoice vs Estimate — Deltas\n"
        "| Part / Operation | Qty | $Estimate | $Invoice | Δ (±) | Evidence (Photo/Doc) | Rationale | Disposition |\n"
        "|---|---:|---:|---:|---:|---|---|---|\n"
        "Use **Photo #** and file names where applicable. Disposition = Approve / Question / Reject.\n\n"
        "## Pricing & Labor Reasonableness\n"
        "- Labor rates, blend/refinish logic, materials, markups, taxes — cite where seen.\n\n"
        "## Missing or Unclear Evidence\n"
        "- List exact proof needed (photo of XYZ, line-level invoice, OEM procedure page).\n\n"
        "## Final Evaluation\n"
        "- Net Δ total (sum of line deltas). Decision summary in 1–2 sentences."
    ),
    "invoices_with_photos": (
        "## Inputs Used\n"
        "- Enumerate invoices and Photo # referenced.\n\n"
        "## Supplement Overview\n"
        "- Brief rationale for invoice items vs estimate scope.\n\n"
        "## Invoice vs Estimate — Deltas\n"
        "| Part / Operation | Qty | $Estimate | $Invoice | Δ (±) | Evidence (Photo/Doc) | Rationale | Disposition |\n"
        "|---|---:|---:|---:|---:|---|---|---|\n\n"
        "## Missing or Unclear Evidence\n"
        "- Exactly what proof is missing.\n\n"
        "## Final Evaluation\n"
        "- Net Δ total and recommendation."
    ),
    "docs_checklist": (
        "## Inputs Used\n"
        "- List all files and pages that were checked.\n\n"
        "## Documentation Checklist\n"
        "| Item | Present? | Evidence (file/page or Photo #) | Notes |\n"
        "|---|:--:|---|---|\n"
        "Estimate, invoices, VIN/plate/odometer photos, 4 corners, close-ups, OEM procedures, sublet docs, registration.\n\n"
        "## Missing Items\n"
        "- Enumerate with the exact artifact needed and purpose.\n\n"
        "## Final Evaluation\n"
        "- Readiness: Ready / Needs Addl Docs — plus next steps."
    ),
}

GLOBAL_RULES = (
    "WRITING & SOURCING RULES:\n"
    "- Do NOT guess vehicle make/model; only state it when clearly supported by estimate text or a legible VIN/badge. Otherwise use 'N/A'.\n"
    "- Never dump a generic list of photos. Only cite Photo # when it backs a specific claim or line item.\n"
    "- If a value cannot be confirmed from inputs, set it to 'N/A' and say why.\n"
    "- Keep currency with $ and thousands separators when you report amounts. Show Δ as Invoice - Estimate with sign.\n"
    "- Keep tables compact and only include rows that have evidence.\n"
    "- summary_brief must be ≤ 280 chars, plain text. The full report goes in summary_markdown.\n"
)


# -----------------------
# Vision Review — GPT does EVERYTHING (we just relay + render)
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
    parts: List[Dict[str, Any]] = []
    files_seen: List[str] = []
    MAX_IMAGES = 24
    used = 0

    # Gather inputs: PDFs -> images; images unchanged; docx/txt -> text
    for f in files:
        raw = await f.read()
        fname = f.filename or "upload"
        low = fname.lower()

        if low.endswith(".pdf"):
            try:
                pages = convert_from_bytes(raw, dpi=120)  # speed-friendly
                files_seen.append(f"{fname} (pdf, {len(pages)} page(s))")
                for im in pages[:MAX_IMAGES - used]:
                    b = io.BytesIO()
                    im.save(b, format="JPEG", quality=65, optimize=True)
                    b64 = base64.b64encode(b.getvalue()).decode("utf-8")
                    parts.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+b64}})
                    used += 1
            except Exception as e:
                log.warning(f"pdf2image failed: {e}")
                files_seen.append(f"{fname} (pdf, could not be converted)")

        elif low.endswith((".jpg",".jpeg",".png",".webp",".heic",".heif")) and used < MAX_IMAGES:
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
            files_seen.append(f"{fname} (photo)")

        elif low.endswith(".docx"):
            try:
                text = "\n".join([p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()])
            except Exception:
                text = ""
            if text.strip():
                parts.insert(0, {"type":"text","text": text[:10000]})
                files_seen.append(f"{fname} (docx text included)")
            else:
                files_seen.append(f"{fname} (docx, no readable text)")

        elif low.endswith(".txt"):
            try:
                text = raw.decode("utf-8","ignore")[:10000]
                parts.insert(0, {"type":"text","text": text})
                files_seen.append(f"{fname} (txt included)")
            except Exception:
                files_seen.append(f"{fname} (txt, read error)")
        else:
            files_seen.append(f"{fname} (unsupported type)")

    SYSTEM = (
        "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
        "Populate exactly these keys: "
        "['file_number','request_type','claim_number','vin','vin_verification','vehicle','odometer_estimate_only','compliance_score','summary_brief','summary_markdown'] . "
        "Do NOT invent fields. Keep header values and summary consistent. "
        "Write a rich, detailed 'summary_markdown' tailored to the request type. "
        "If you cannot determine a field from inputs, set it to 'N/A' rather than guessing."
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

    # Force the "Inputs Used" echo so we can see what GPT actually referenced
    prompt_text = (
        f"REQUEST TYPE: {ai_intent} — Use request_type='{req_label}'.\n\n"
        "FILES SEEN (include this list verbatim in your '## Inputs Used' section):\n- "
        + ("\n- ".join(files_seen) if files_seen else "none") + "\n\n"
        "CLIENT RULES (verbatim text if provided by the UI):\n" + (client_rules[:2500] if client_rules else "") + "\n\n"
        "DETAIL LAYOUT:\n" + DETAIL_TEMPLATES.get(ai_intent, DETAIL_TEMPLATES["comprehensive"]) + "\n\n"
        + GLOBAL_RULES +
        "Return JSON with the exact keys listed above. "
        "summary_brief must be <= 280 chars (plain text). summary_markdown = full write-up using the layout."
    )

    user_parts: List[Dict[str,Any]] = [{"type":"text","text": prompt_text}]
    if parts: user_parts.extend(parts)

    # Call GPT and relay output as-is
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": user_parts}],
            max_tokens=1400,
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
        "request_type": _get("request_type"),
        "claim_number": _get("claim_number"),
        "vin": _get("vin"),
        "vin_verification": _get("vin_verification"),
        "vehicle": _get("vehicle"),
        "odometer_estimate_only": _get("odometer_estimate_only"),
        "compliance_score": _get("compliance_score"),
        "summary_brief": _get("summary_brief"),
        "summary_markdown": _get("summary_markdown"),
    }

    # PDF — print values exactly as GPT returned
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
    pdf.ln(3); mc("AI-4-IA Review Summary"); mc((result["summary_markdown"] or '').strip())

    safe_file = _safe(file_number); pdf_path = os.path.join(PDF_DIR, f"{safe_file}.pdf")
    try:
        data_bytes = pdf.output(dest="S").encode("latin-1","ignore")
        with open(pdf_path,"wb") as f: f.write(data_bytes)
    except Exception as e:
        log.warning(f"PDF write error: {e}")

    # Email — same exact values
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
        log.error(f"Email error: {e}" )

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
