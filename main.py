from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os, io, re, json, base64, logging, smtplib, zipfile
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

# --- PII Redaction (Presidio) ---
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig  # important for anonymizer API

# -----------------------
# Minimal setup
# -----------------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
CLIENT_RULES_DIR = os.getenv("CLIENT_RULES_DIR", "client_rules")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("nspxn")

MODEL = os.getenv("OAI_MODEL", "gpt-4.1")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY missing")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --------------------------------
# Presidio: Analyzer/Anonymizer (preserve VIN & Claim #)
# --------------------------------
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

VIN_PATTERN = r"\b([A-HJ-NPR-Z0-9]{17})\b"
vin_recognizer = PatternRecognizer(
    supported_entity="VIN",
    patterns=[Pattern(name="vin-17", regex=VIN_PATTERN, score=0.8)],
)

CLAIM_PATTERN = r"\b(?:(?:Claim|CLM|Clm)\s*#?\s*[:\-]?\s*)?([A-Z0-9]{5,}[A-Z0-9\-]{0,})\b"
claim_recognizer = PatternRecognizer(
    supported_entity="CLAIM_NUMBER",
    patterns=[Pattern(name="claim-generic", regex=CLAIM_PATTERN, score=0.6)],
)

analyzer.registry.add_recognizer(vin_recognizer)
analyzer.registry.add_recognizer(claim_recognizer)

# Only redact these entity types (we PRESERVE VIN & CLAIM_NUMBER)
REDACT_ENTITY_TYPES = {
    "PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN",
    "CREDIT_CARD", "IBAN_CODE", "LOCATION", "NRP", "ORGANIZATION",
    "DATE_TIME", "IP_ADDRESS", "CRYPTO", "MEDICAL_LICENSE", "URL"
}

def _filter_results(results: list[RecognizerResult]) -> list[RecognizerResult]:
    return [r for r in results if r.entity_type in REDACT_ENTITY_TYPES]

def redact_text_preserve_vin_claim(text: str) -> str:
    """Mask PII while keeping VIN and Claim # intact. Fail-open to avoid 500s."""
    if not text:
        return text
    results = analyzer.analyze(text=text, language="en")
    to_mask = _filter_results(results)
    if not to_mask:
        return text
    try:
        return anonymizer.anonymize(
            text=text,
            analyzer_results=to_mask,
            operators={"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
        ).text
    except Exception as e:
        log.warning(f"Presidio anonymizer failed, passing text through. Error: {e}")
        return text

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
# GPT prompt steering (ONLY 3 intents)
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
    "comprehensive": (
        "## Inputs Used\n"
        "- Enumerate files seen; reference **Estimate page/line** and **Photo #** where applicable.\n\n"
        "## Executive Summary\n"
        "- 3–6 bullets: integrity of estimate, major deviations vs client rules, and any photo mismatches.\n\n"
        "## Client Guidelines Cross-Check\n"
        "**Use ONLY the topics below. If a topic has no rule in client_rules, mark it as Not Applicable (N/A).**\n"
        "Allowed topics: Labor Rates; Refinish/Overlap; Paint Materials; OEM Procedures; Sublet; Tax/Markup.\n"
        "- For each row, quote ≤12-word snippet **directly** about that topic. If no on-topic rule exists, set Rule='No rule provided', Source='N/A', Status='N/A', Impact='N/A', Fix='N/A'.\n"
        "- Status must be one of: **Compliant / Non-compliant / Not Evidenced / N/A**.\n"
        "- Estimate evidence must cite page/line; Photo corroboration must cite Photo # or 'N/A'.\n"
        "- Do NOT use unrelated lines (e.g., 'Repairable Damage Images', '3 Salvage Bids') for these topics.\n\n"
        "| Topic | Rule (short, ≤12 words) | Source (doc/page/section) | Estimate evidence (page/line) | Photo corroboration (Photo #) | Status | Impact | Required fix |\n"
        "|---|---|---|---|---|:--:|:--:|---|\n"
        "| Labor Rates |  |  |  |  |  |  |  |\n"
        "| Refinish/Overlap |  |  |  |  |  |  |  |\n"
        "| Paint Materials |  |  |  |  |  |  |  |\n"
        "| OEM Procedures |  |  |  |  |  |  |  |\n"
        "| Sublet |  |  |  |  |  |  |  |\n"
        "| Tax/Markup |  |  |  |  |  |  |  |\n\n"
        "## Photo ↔ Estimate Crosswalk\n"
        "Only list lines/parts that appear in the estimate; if a photo shows a part with no matching estimate line, mark 'Non-evidenced' and note it.\n\n"
        "| Line / Part | Photo # | What the photo shows | Consistent with estimate? | Notes |\n"
        "|---|---|---|:--:|---|\n\n"
        "## Required Evidence Presence\n"
        "| Item | Status |\n"
        "|---|---|\n"
        "| Odometer Photo | <Present / Present — not clearly legible / Missing> |\n"
        "| Registration Photo | <Present / Present — not clearly legible / Missing> |\n\n"
        "## VIN & Identifiers Verification\n"
        "- State one of **MATCH / MISMATCH / NOT VERIFIED** and explicitly list: estimate VIN vs photo VIN(s). If any piece is unreadable, say so.\n\n"
        "## Missing Evidence & Documentation\n"
        "- High / Medium / Low severity with one-line remediation (e.g., missing OEM doc, unclear photo of part).\n\n"
        "## Final Evaluation\n"
        "- **Compliance Score: NN%** and a one-sentence rationale.\n"
        "- **Next Actions:** 1–3 succinct steps to resolve gaps (e.g., request clearer photo of LH rail, attach OEM doc page X)."
    ),
    "damage_report_from_photos": (
        "# AI-4-IA Damage Report\n"
        "Create a concise, professional damage report **based only on the provided photos (and any optional text)**. Follow the provided sample style exactly.\n\n"
        "## Inputs Used\n"
        "- List exact Photo #s and any text used.\n\n"
        "## Quick Stats\n"
        "- Claim # (if visible): <value or N/A>\n"
        "- File # (echo from request): <value or N/A>\n"
        "- Odometer (if visible): <value or N/A>\n"
        "- Primary Impact: <area(s)>\n"
        "- Secondary Impact: <area(s) or 'None observed'>\n\n"
        "## Damage Summary\n"
        "- 6–12 bullets: **panel/part** + **condition** (dent/crease/scrape/misalignment) + **suggested op** (repair/replace/refinish/blend). Always reference **Photo #** when applicable.\n\n"
        "## Estimated Repair Costs\n"
        "  - Body Labor: <hrs> hr @ $<rate>/hr .......... $<amount>\n"
        "  - Paint Labor: <hrs> hr @ $<rate>/hr .......... $<amount>\n"
        "  - Paint Materials: <hrs> hr @ $<rate>/hr ...... $<amount>\n"
        "  - Parts: <brief list> .... $<amount>\n"
        "  - Subtotal ........................................ $<amount>\n"
        "  - Sales Tax (<rate>%) ............................... $<amount>\n"
        "  - Total Estimated Cost ...................... $<amount> ±<variance>%\n\n"
        "## Fraud & Authenticity Check\n"
        "- Summarize any inconsistencies between photos, timestamps, or visible identifiers (VIN/badges). If none, say so.\n\n"
        "## Conclusion\n"
        "- 1–2 sentences summarizing repairability and scope.\n"
    ),
}

ALLOWED_INTENTS = {"guidelines_only","comprehensive","damage_report_from_photos"}

GLOBAL_RULES = (
    "WRITING & SOURCING RULES:\n"
    "- Use ONLY what is visible/legible in provided inputs.\n"
    "- If a value cannot be confirmed, set it to 'N/A' and say why.\n"
    "- For the Damage Report mode, do NOT include estimate compliance boilerplate.\n"
    "- summary_brief must be ≤ 280 chars, plain text. The full report goes in summary_markdown.\n"
)

FRAUD_GUIDE = (
    "FRAUD DETECTION TASK:\n"
    "- Provide a section named 'Fraud & Authenticity Check' for Damage Report mode and 'Fraud Detection' for others.\n"
    "- Screen for reused/stock photos, metadata/date inconsistencies, VIN/odometer mismatches, image tampering clues, mismatched badges/colors.\n"
    "- If nothing material is found, say so.\n"
)

# -----------------------
# Supported file types
# -----------------------
SUPPORTED_IMAGE_EXTS = (".jpg",".jpeg",".png",".webp",".heic",".heif")
SUPPORTED_TEXT_EXTS = (".txt",)
SUPPORTED_DOCX_EXTS = (".docx",)
SUPPORTED_PDF_EXTS = (".pdf",)

# -----------------------
# Helpers to add parts from bytes
# -----------------------
def _image_part_from_bytes(raw: bytes) -> Dict[str, Any]:
    b64 = base64.b64encode(raw).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}

def _add_bytes(parts: List[Dict[str,Any]], files_seen: List[str], raw: bytes, fname: str, used: int, max_images: int) -> int:
    low = fname.lower()
    if low.endswith(SUPPORTED_PDF_EXTS) and used < max_images:
        try:
            pages = convert_from_bytes(raw, dpi=220)
            files_seen.append(f"{fname} (pdf, {len(pages)} page(s))")
            for im in pages[:max_images - used]:
                b = io.BytesIO()
                im.save(b, format="JPEG", quality=65, optimize=True)
                parts.append(_image_part_from_bytes(b.getvalue()))
                used += 1
        except Exception as e:
            logging.warning(f"pdf2image failed for {fname}: {e}")
            files_seen.append(f"{fname} (pdf, could not be converted)")
    elif low.endswith(SUPPORTED_IMAGE_EXTS) and used < max_images:
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            im.thumbnail((1400,1400))
            b = io.BytesIO(); im.save(b, format="JPEG", quality=70, optimize=True)
            raw = b.getvalue()
        except Exception:
            pass
        parts.append(_image_part_from_bytes(raw))
        used += 1
        files_seen.append(f"{fname} (photo)")
    elif low.endswith(SUPPORTED_DOCX_EXTS):
        try:
            text = "\n".join([p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()])
        except Exception:
            text = ""
        if text.strip():
            parts.insert(0, {"type":"text","text": text[:10000]})
            files_seen.append(f"{fname} (docx text included)")
        else:
            files_seen.append(f"{fname} (docx, no readable text)")
    elif low.endswith(SUPPORTED_TEXT_EXTS):
        try:
            text = raw.decode("utf-8","ignore")[:10000]
        except Exception:
            text = ""
        if text.strip():
            parts.insert(0, {"type":"text","text": text})
            files_seen.append(f"{fname} (txt included)")
        else:
            files_seen.append(f"{fname} (txt, empty)")
    else:
        files_seen.append(f"{fname} (unsupported type)")
    return used

# -----------------------
# Vision Review — GPT does EVERYTHING (ZIP supported)
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
    MAX_IMAGES = 32
    used = 0

    # Anti-zipbomb guardrails
    MAX_ZIP_FILES = 100
    MAX_ENTRY_SIZE = 15 * 1024 * 1024  # 15 MB

    for f in files:
        raw = await f.read()
        fname = f.filename or "upload"
        low = fname.lower()

        if low.endswith(".zip"):
            try:
                zf = zipfile.ZipFile(io.BytesIO(raw))
            except Exception as e:
                files_seen.append(f"{fname} (zip, unreadable: {e})")
                continue
            members = [zi for zi in zf.infolist() if not zi.is_dir()]
            if len(members) > MAX_ZIP_FILES:
                files_seen.append(f"{fname} (zip, too many entries: {len(members)})")
                members = members[:MAX_ZIP_FILES]
            for zi in members:
                inner_name = zi.filename
                if ".." in inner_name or inner_name.startswith(("/", "\\")):
                    files_seen.append(f"{fname}::{inner_name} (skipped unsafe path)"); continue
                if zi.file_size > MAX_ENTRY_SIZE:
                    files_seen.append(f"{fname}::{inner_name} (skipped >15MB)"); continue
                try:
                    data = zf.read(zi)
                except Exception as e:
                    files_seen.append(f"{fname}::{inner_name} (read error: {e})"); continue
                used = _add_bytes(parts, files_seen, data, f"{fname}::{inner_name}", used, MAX_IMAGES)
        else:
            used = _add_bytes(parts, files_seen, raw, fname, used, MAX_IMAGES)

    # Lock to 3 intents only
    if ai_intent not in ALLOWED_INTENTS:
        ai_intent = "comprehensive"

    # Labels to freeze request_type exactly as dropdown
    REQ_LABELS = {
        "guidelines_only": "Guidelines → Estimate (no photos)",
        "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
        "damage_report_from_photos": "Create a Damage Report from Photos",
    }
    req_label = REQ_LABELS.get(ai_intent, "Comprehensive: Guidelines + Estimate + Photos (with VIN check)")
    log.info(f"ai_intent received: {ai_intent} -> using label: {req_label}")

    # Keys expected back
    KEYS = [
        "file_number","request_type","claim_number","vin","vin_verification","vehicle",
        "odometer_estimate_only","compliance_score","summary_brief","summary_markdown",
        "fraud_markdown","primary_impact","secondary_impact","estimated_costs_markdown","conclusion"
    ]

    # ---- Intent-aware SYSTEM prompt ----
    SYSTEM_BASE = (
        "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
        "Populate exactly these keys (always include all, use 'N/A' when not applicable): "
        "['file_number','request_type','claim_number','vin','vin_verification','vehicle',"
        "'odometer_estimate_only','compliance_score','summary_brief','summary_markdown',"
        "'fraud_markdown','primary_impact','secondary_impact','estimated_costs_markdown','conclusion']. "
        "Use evidence only from inputs. Avoid guessing. "
        "summary_brief must be <= 280 chars (plain text)."
    )
    SYSTEM_COMPREHENSIVE_EXTRA = (
        "\nEXTRA RULES FOR COMPREHENSIVE:\n"
        "- In summary_markdown, include ALL sections defined by the Comprehensive template, "
        "with the two tables (**Client Guidelines Cross-Check** and **Photo ↔ Estimate Crosswalk**) rendered as Markdown tables.\n"
        "- Only use these topics in Client Guidelines Cross-Check: Labor Rates; Refinish/Overlap; Paint Materials; OEM Procedures; Sublet; Tax/Markup.\n"
        "- For each topic, quote a ≤12-word snippet that is explicitly about that topic. "
        "If no explicit rule exists in client_rules, set Rule='No rule provided' and Status='N/A'.\n"
        "- Reject snippets that are unrelated (e.g., 'Repairable Damage Images', '3 Salvage Bids', 'Clean Retail Value').\n"
        "- Estimate evidence must cite page/line; Photo corroboration must cite Photo # or 'N/A'.\n"
        "- Do not mark 'Missing' if there is any plausible visual/text cue; prefer 'Present — not clearly legible'.\n"
        "- Re-scan every PDF page and all photos before concluding 'Missing'.\n"
    )
    SYSTEM_OTHER_EXTRA = (
        "\nIf request_type is 'Create a Damage Report from Photos', ignore estimate/compliance details; "
        "focus ONLY on the Damage Report sections (Quick Stats, Damage Summary, Estimated Repair Costs, "
        "Fraud & Authenticity Check, Conclusion). For other request types, write a standard narrative "
        "under summary_markdown."
    )

    SYSTEM = SYSTEM_BASE + (SYSTEM_COMPREHENSIVE_EXTRA if ai_intent == "comprehensive" else "") + SYSTEM_OTHER_EXTRA

    prompt_text = (
        f"REQUEST TYPE SELECTED (exact): '{req_label}'. Use this exact string in 'request_type'.\n\n"
        "FILES SEEN (echo verbatim in '## Inputs Used'):\n- "
        + ("\n- ".join(files_seen) if files_seen else "none") + "\n\n"
        "CLIENT RULES (if provided; else blank):\n" + (client_rules[:2000] if client_rules else "") + "\n\n"
        "DETAIL LAYOUT:\n" + DETAIL_TEMPLATES.get(ai_intent, DETAIL_TEMPLATES["comprehensive"]) + "\n\n"
        + GLOBAL_RULES + "\n\n" + FRAUD_GUIDE
    )

    # Extra nudge for Comprehensive
    if ai_intent == "comprehensive":
        prompt_text += (
            "\n\nUploader note: Odometer and Registration photos were provided. "
            "If you cannot clearly read them, report 'Present — not clearly legible' rather than 'Missing'."
            "\n\n### Extra detail requirements for Comprehensive\n"
            "- In 'estimated_costs_markdown', break out Body Labor, Paint Labor, Paint Materials, Parts, Sublet, Tax, "
            "and show 1–2 sentence rationale.\n"
            "- In 'fraud_markdown', list any inconsistencies as bullets with the evidence reference (Photo # / Estimate page/line). "
            "If none, state 'No material inconsistencies found.'\n"
    )


    # Build user parts (redact PII in any free text, but keep VIN/Claim #)
    safe_user_parts: List[Dict[str,Any]] = []
    redaction_success = False

    try:
        red_prompt = redact_text_preserve_vin_claim(prompt_text)
        redaction_success = True
    except Exception as e:
        log.warning(f"Redaction failed on prompt_text: {e}")
        red_prompt = prompt_text

    safe_user_parts.append({"type": "text", "text": red_prompt})

    if parts:
        for p in parts:
            if p.get("type") == "text" and isinstance(p.get("text"), str):
                try:
                    red_txt = redact_text_preserve_vin_claim(p["text"])
                    redaction_success = True
                except Exception as e:
                    log.warning(f"Redaction failed on a text part: {e}")
                    red_txt = p["text"]
                safe_user_parts.append({"type": "text", "text": red_txt})
            else:
                safe_user_parts.append(p)

    # Simple status string for PDF/JSON
    redaction_status = "Redacted PII: Successful ✅" if redaction_success else "Redacted PII: Not Applied"

    # Token budget by intent
    MAX_TOKENS_BY_INTENT = {
        "comprehensive": 1800,
        "guidelines_only": 1500,
        "damage_report_from_photos": 1300
    }
    max_tokens = MAX_TOKENS_BY_INTENT.get(ai_intent, 1700)

    # Call GPT and parse JSON
    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": safe_user_parts}],
            max_tokens=max_tokens,
            temperature=0
        )
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removesuffix("```").removeprefix("```json").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"LLM failure or JSON parse error: {e}")
        return JSONResponse(status_code=500, content={"error":"Model output could not be parsed as JSON."})

    def _get(k):
        v = data.get(k)
        return "" if v is None else str(v)

    # Freeze request_type to the dropdown label
    result = {
        "file_number": file_number,
        "request_type": req_label,
        "claim_number": _get("claim_number"),
        "vin": _get("vin"),
        "vin_verification": _get("vin_verification"),
        "vehicle": _get("vehicle"),
        "odometer_estimate_only": _get("odometer_estimate_only"),
        "compliance_score": _get("compliance_score"),
        "summary_brief": _get("summary_brief"),
        "summary_markdown": _get("summary_markdown"),
        "fraud_markdown": _get("fraud_markdown"),
        "primary_impact": _get("primary_impact"),
        "secondary_impact": _get("secondary_impact"),
        "estimated_costs_markdown": _get("estimated_costs_markdown"),
        "conclusion": _get("conclusion"),
        "redaction_status": redaction_status,
    }

    # -----------------------
    # PDF — separate file for Damage Report, classic for others
    # -----------------------
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    def mc(s): pdf.multi_cell(0,6,s)

    if ai_intent == "damage_report_from_photos":
        # Title only; sample-style layout
        pdf.cell(0,10,"AI-4-IA Damage Report", ln=True, align="C")
        pdf.set_font_size(10); pdf.ln(3)

        mc(f"Claim #: {result['claim_number'] or 'N/A'}    File #: {file_number or 'N/A'}")
        if result["odometer_estimate_only"] or result["primary_impact"]:
            mc(f"Odometer: {result['odometer_estimate_only'] or 'N/A'}    Primary Impact: {result['primary_impact'] or 'N/A'}")
        if result["secondary_impact"]:
            mc(f"Secondary Impact: {result['secondary_impact']}")

        # Simple confirmation line
        mc(result["redaction_status"])

        pdf.ln(2); mc("Damage Summary"); mc((result["summary_markdown"] or "N/A").strip())
        pdf.ln(2); mc("Estimated Repair Costs"); mc((result["estimated_costs_markdown"] or "N/A").strip())
        pdf.ln(2); mc("Fraud & Authenticity Check"); mc((result["fraud_markdown"] or 'N/A').strip())
        pdf.ln(2); mc("Conclusion"); mc((result["conclusion"] or 'N/A').strip())

        safe_file = _safe(file_number)
        pdf_filename = f"AI_Damage_Report_{safe_file}.pdf"
    else:
        # Classic NSPXN header
        pdf.cell(0,10,"NSPXN.com AI Review Report", ln=True, align="C")
        pdf.set_font_size(10); pdf.ln(3)
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

        # Simple confirmation line in classic report
        mc(result["redaction_status"])

        pdf.ln(3); mc("AI-4-IA Review Summary"); mc((result["summary_markdown"] or '').strip())
        pdf.ln(3); mc("Fraud Detection"); mc((result["fraud_markdown"] or 'N/A').strip())

        safe_file = _safe(file_number)
        pdf_filename = f"{safe_file}.pdf"

    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    try:
        data_bytes = pdf.output(dest="S").encode("latin-1","ignore")
        with open(pdf_path,"wb") as f: f.write(data_bytes)
    except Exception as e:
        logging.warning(f"PDF write error: {e}")

    # -----------------------
    # Email — minimal mirror
    # -----------------------
    try:
        msg = EmailMessage()
        if ai_intent == "damage_report_from_photos":
            subj = f"AI Damage Report: {file_number or ''} {result['claim_number'] or ''}".strip()
            msg.set_content(f"""AI-4-IA Damage Report

Claim #: {result['claim_number'] or 'N/A'}    File #: {file_number or 'N/A'}
Odometer: {result['odometer_estimate_only'] or 'N/A'}    Primary Impact: {result['primary_impact'] or 'N/A'}
Secondary Impact: {result['secondary_impact'] or 'N/A'}

{result['redaction_status']}

Damage Summary
{result['summary_markdown'] or 'N/A'}

Estimated Repair Costs
{result['estimated_costs_markdown'] or 'N/A'}

Fraud & Authenticity Check
{result['fraud_markdown'] or 'N/A'}

Conclusion
{result['conclusion'] or 'N/A'}
""")
        else:
            subj = f"AI-4-IA Review: {result['claim_number'] or file_number}"
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

{result['redaction_status']}

AI-4-IA Review Summary
{result['summary_markdown']}

Fraud Detection
{result['fraud_markdown']}
""")
        msg["Subject"] = subj
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logging.error(f"Email error: {e}" )

    # Provide direct URL (filename for damage-report, file_number for others)
    if ai_intent == "damage_report_from_photos":
        pdf_url = f"/download-pdf?filename={pdf_filename}"
    else:
        pdf_url = f"/download-pdf?file_number={safe_file}"

    return {
        **result,
        "web_summary": result["summary_brief"],
        "gpt_output": result["summary_markdown"],
        "pdf_url": pdf_url,
        "pdf_filename": pdf_filename
    }

# -----------------------
# PDF download (supports explicit filename for damage report)
# -----------------------
@app.get("/download-pdf")
async def download_pdf(file_number: Optional[str] = None, filename: Optional[str] = None):
    # If an explicit filename is provided, serve it verbatim
    if filename:
        safe = _safe(filename)
        path = os.path.join(PDF_DIR, safe)
        if os.path.exists(path):
            return FileResponse(path=path, media_type="application/pdf", filename=safe)
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Back-compat: file_number param (classic behavior)
    if not file_number:
        return JSONResponse(status_code=400, content={"detail": "Missing query param 'filename' or 'file_number'"})

    safe_num = _safe(file_number)

    # 1) Classic report name: <FileNumber>.pdf
    classic_path = os.path.join(PDF_DIR, f"{safe_num}.pdf")
    if os.path.exists(classic_path):
        return FileResponse(path=classic_path, media_type="application/pdf", filename=f"{safe_num}.pdf")

    # 2) Damage Report name: AI_Damage_Report_{safe_num}.pdf
    dmg_name = f"AI_Damage_Report_{safe_num}.pdf"
    dmg_path = os.path.join(PDF_DIR, dmg_name)
    if os.path.exists(dmg_path):
        return FileResponse(path=dmg_path, media_type="application/pdf", filename=dmg_name)

    # 3) As last resort, try raw (unsanitized) number for legacy writes
    raw_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(raw_path):
        return FileResponse(path=raw_path, media_type="application/pdf", filename=f"{file_number}.pdf")

    return JSONResponse(status_code=404, content={"detail": "Not Found"})


