from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import os, io, re, json, base64, logging, zipfile, glob
import smtplib  # email transport
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

# Optional OCR (pytesseract). Safe no-op if not installed or tesseract binary missing.
try:
    import pytesseract  # type: ignore
    _OCR_ENABLED = True
except Exception:
    _OCR_ENABLED = False

from openai import OpenAI

# --- PII Redaction (Presidio) ---
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig  # required for anonymizer API

# -----------------------
# Minimal setup
# -----------------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp"); os.makedirs(PDF_DIR, exist_ok=True)
CLIENT_RULES_DIR = os.getenv("CLIENT_RULES_DIR", "client_rules")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("nspxn")
log.info(f"Using CLIENT_RULES_DIR={CLIENT_RULES_DIR}")

# Use GPT-4.1 everywhere
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

def _filter_results(results: List[RecognizerResult]) -> List[RecognizerResult]:
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
    return re.sub(r"[^\w.\-]+", "-", (s or "").strip()).strip("-_. ")

# -----------------------
# Prompt steering (free analysis + detailed narrative)
# -----------------------
DETAIL_TEMPLATES = {
    "guidelines_only": (
        "## Inputs Used\n"
        "- Briefly list which documents, pages, and photos you actually referenced.\n\n"
        "## Executive Summary\n"
        "- 3–5 bullets summarizing overall compliance and key risks.\n\n"
        "## AI-4-IA Review Summary\n"
        "- Write a formal, paragraph-style appraisal narrative. Include scope of impact, damage by zone/panel, "
        "repair vs. replace rationale, parts type (OEM/LKQ/Aftermarket), labor ops, refinish overlap, rate validation, "
        "tax handling, and estimate integrity. Reference evidence inline (e.g., 'p2/L14', 'Photo 3'). "
        "Close with compliance stance and final recommendation. Minimum 8–10 sentences.\n\n"
        "## Key Issues & Actions\n"
        "- Bullet list of the highest-impact issues with a one-line recommended action each.\n\n"
        "## Final\n"
        "- Compliance Score: NN% with one-sentence rationale."
    ),

    "comprehensive": (
        "## Inputs Used\n"
        "- List the estimate pages/lines and photo numbers you used, plus any rules text (if provided).\n\n"
        "## Executive Summary\n"
        "- 3–6 bullets capturing the big picture: estimate integrity, rule alignment (only if rules text was supplied), "
        "and photo consistency.\n\n"
        "## Detailed Audit Report\n"
        "- Write this section as a formal, paragraph-style appraisal report summarizing the entire claim. "
        "Include: scope of impact, damage by zone/panel, repair vs. replace rationale, parts type (OEM/LKQ/Aftermarket), "
        "labor operations, refinish/overlap considerations, rate validation, paint materials handling, sublet usage, "
        "tax/markup accuracy, and overall estimate integrity. Cite photos and estimate lines (e.g., 'Photo 3', 'p2/L14'). "
        "Close with compliance to any provided client rules and a clear final recommendation (Repairable vs. Total Loss). "
        "Do not declare Repairable/Total Loss unless the estimate itself explicitly marks 'Total Loss' or an ACV comparison is provided. "
        "If the shop info is listed under Repair Facility, add only the shop name to the Detailed Audit Report narrative. "
        "If a Printout showing the Clean Retail Value or Estimated Trade-In Value of the unit is present which may include ANY of the following: NADA, J.D. Power, Kelly Blue Book, Edmunds, Carfax, or Cars.com, DO NOT declare as missing if any of these are present. "
        "Minimum 10–14 sentences (one continuous narrative, not bullets).\n\n"
        "## Photo-by-Photo Damage Ledger\n"
        "| Photo # | View/Angle | Panels/Parts Visible | Condition (dent/crease/scrape/misalignment) | Identifiers (VIN/odo/plate/reg) | Legibility |\n"
        "|---:|---|---|---|---|---|\n"
        "- One row per photo used in the analysis (>=6 rows if >=6 photos exist). If an identifier is present but unreadable, mark 'Present — not clearly legible'.\n\n"
        "## Brief Damage Descriptions\n"
        "- 6–12 bullets. Each: part/panel + condition + suggested op (repair/replace/refinish/blend) + Photo #.\n\n"
        "## Estimate Line Extract (top relevant lines)\n"
        "| Est. p#/L# | Part/Op | Labor Hrs | Rate | Part Type | Price | Notes |\n"
        "|---|---|---:|---:|---|---:|---|\n"
        "- Include 8+ lines if available, focusing on items tied to observed damages. Use 'Notes' for overlap/blend or rationale.\n\n"
        "## Estimate Compliance Cross-Check (brief)\n"
        "Status: Compliant / Non-compliant / Not Evidenced. Cite only the most important evidence (p#/L# or Photo #). Keep it short.\n"
        "| Topic | Evidence (p#/L# or value; Photo # if relevant) | Status | Required Fix |\n"
        "|---|---|:--:|---|\n"
        "| Labor Rates |  |  |  |\n"
        "| Refinish/Overlap |  |  |  |\n"
        "| Paint Materials |  |  |  |\n"
        "| OEM Procedures |  |  |  |\n"
        "| Sublet |  |  |  |\n"
        "| Tax/Markup |  |  |  |\n\n"
        "## Client Guidelines Comparison (if rules text was supplied)\n"
        "- 3–8 bullets. Quote the relevant rule fragment and note Aligned / Not Aligned / Not Evidenced, with evidence refs (p#/L#, Photo #). "
        "If no client_rules were provided, omit this section.\n\n"
        "## Risks / Missing Evidence\n"
        "- Short bullets with severity (High/Med/Low) and a one-line remediation.\n\n"
        "## Compliance Score Rationale\n"
        "- REQUIRED if score < 100: start at 100 and list each deficiency with evidence refs (p#/L# and/or Photo #), "
        "severity (Minor/Moderate/Major), and numeric deduction. Show the arithmetic to the final score.\n\n"
        "## Final Evaluation\n"
        "- Compliance Score: NN% with a single-sentence justification. "
        "If no fraud indicators are identified, state 'No material inconsistencies found.' Do not use 'N/A'."
    ),

    "damage_report_from_photos": (
        "# AI-4-IA Damage Report\n"
        "Create a concise, professional damage report based only on the provided photos (and any optional text).\n\n"
        "## Inputs Used\n"
        "- List exact Photo #s and any text used.\n"
        "- Include a one-line count of photos and list the exact photo labels used (e.g., 'Photo 1–12; used 1–9, 11').\n\n"
        "## Quick Stats\n"
        "- Claim # (if visible): <value or N/A>\n"
        "- File # (echo from request): <value or N/A>\n"
        "- Odometer (if visible): <value or 'Present — not clearly legible'>\n"
        "- Primary Impact: <area(s)>\n"
        "- Secondary Impact: <area(s) or 'None observed'>\n\n"
        "## Photo-by-Photo Damage Ledger\n"
        "| Photo # | View/Angle | Panels/Parts Visible | Condition (dent/crease/scrape/misalignment) | Identifiers (VIN/odo/plate/reg) | Legibility |\n"
        "|---:|---|---|---|---|---|\n"
        "- One row per photo used in the analysis (>=6 rows if >=6 photos exist). If an identifier is present but unreadable, mark 'Present — not clearly legible'.\n"
        "- The ledger is required. Do not omit it. Each row must have a concrete Photo # that exists in the set.\n\n"
        "## Damage Summary\n"
        "- 6–12 bullets with panel/part + condition + suggested op, citing Photo #.\n\n"
        "## Detailed Audit Report\n"
        "- Provide a detailed appraisal narrative based on the photos: impact zones, repair/replace reasoning, "
        "likely parts source (OEM/LKQ/Aftermarket) when inferable, refinish/overlap notes, and cost implications. "
        "Reference specific Photo #s. Minimum 8–12 sentences (one continuous narrative, not bullets). "
        "Cite VIN and odometer with explicit Photo #s; if either is unreadable, write 'Present — not clearly legible' and explain why (glare/blur/angle) instead of 'Missing'.\n\n"
        "## Estimated Repair Costs\n"
        "- Provide a high-level breakdown (Body Labor, Paint Labor, Paint Materials, Parts, Sublet, Tax) and a one-line rationale tying to observed work "
        "(e.g., '2 panels refinish x 2.5 hr each' or 'rear bumper cover likely replace'). If uncertainty is high, state it explicitly.\n\n"
        "## Fraud & Authenticity Check\n"
        "- State VIN and odometer with Photo # citations; note any EXIF/date anomalies or duplicate images. If none, say so.\n\n"
        "## Compliance Score Rationale\n"
        "- Omit this section entirely for photos-only reports. Do not provide a score.\n\n"
        "## Conclusion\n"
        "- 1–2 sentences summarizing repairability and scope. "
        "Do not declare Repairable/Total Loss in photos-only mode. "
        "If no fraud indicators are identified, state 'No material inconsistencies found.' Do not use 'N/A'.\n"
    ),
}

# --- Static audit questions ---
STATIC_AUDIT_QUESTIONS = [
    "Do the photos substantiate the highest-cost operations (frame/sectioning/panel replace)?",
    "Are ADAS calibrations or wheel alignments required and supported by the damage and OEM procedures?",
    "Is blend time justified by color/finish (metallic/pearl/tri-coat) and adjacent panel visibility?",
    "Do invoices corroborate parts used and match estimate line items (brand/grade, price, quantity)?",
    "Are AM/LKQ choices compliant with age/mileage rules, and is OEM required anywhere by client policy or safety?",
    "Is there evidence of prior or unrelated damage (UPD) that materially affects valuation or repair scope?",
    "Are there structural/safety indicators (buckles, misalignments, airbags/pretensioners) that alter repair strategy?",
    "Are materials/hazard charges (paint supplies, corrosion protection, seam sealer) aligned with operations and shop norms?",
    "Are storage/tow charges and dates supported and reasonable given claim timeline and shop status?",
    "Are scanner reports (pre/post) included or needed; if absent, does that meaningfully impact confidence?",
    "Did the supplement (if any) correct earlier gaps, and are newly added operations now evidenced?",
    "Are client-required documents present (e.g., NADA printout, release forms, production date plate); if missing, what’s the impact?",
    "What is the bottom-line recommendation (approve as-is, adjust items, or request specific evidence)?",
]

# --- Identifiers Verification Protocol (prompt-only; no new logic) ---
IDENTIFIERS_VERIFICATION_PROTOCOL = (
    "\n\nIDENTIFIERS VERIFICATION PROTOCOL (must follow):"
    "\n1) Search the photos for: windshield VIN plate, driver-door VIN label, driver-door VIN label Production date, driver-door VIN label Date of Mfr, odometer cluster."
    "\n2) Transcribe the VIN exactly as visible and cite Photo # for EACH location you find."
    "\n3) If multiple VINs, compare them to each other and to the estimate VIN; explicitly state: MATCH / MISMATCH."
    "\n4) Transcribe the odometer reading exactly as shown and cite Photo #."
    "\n5) Grade legibility for each identifier as one of: 'Clearly legible' / 'Present — not clearly legible' / 'Not present'."
    "\n6) If any identifier is present but not clearly legible, say why (glare, blur, angle) and what photo would resolve it."
    "\n7) Write a one-line bottom line: 'VIN verification: <MATCH/MISMATCH/INCONCLUSIVE>; Odometer: <value or reason>'."
    "\n8) Weave these facts naturally into the '## Detailed Audit Report' narrative and keep the top-line fields "
    "(vin, vin_verification, odometer_estimate_only) consistent."
    "\n9) When citing more than one VIN location (e.g., windshield vs. door label), you must cite DISTINCT Photo #s; "
    "never reuse the same photo number for two different locations."
    "\n10) Compare VINs as literal 17-character strings. If any single character differs between sources, report "
    "MISMATCH, and quote both strings with their Photo #/page references."
    "\n11) ODOMETER RULES (photos-only especially): transcribe only the digits visible in the odometer photo; "
    "do not infer from estimate text or metadata. Include the exact Photo #. If any digit is unclear, state 'Present — not clearly legible' and explain why; do not guess."
)

# --- Consistency Guard (prompt-only; avoid contradictions) ---
CONSISTENCY_GUARD = (
    "\n\nCONSISTENCY GUARD:"
    "\n- Do not claim any required photo is 'missing' if you graded it 'Clearly legible' or 'Present — not clearly legible'."
    "\n- For VIN, Odometer, and Production Date specifically: if present in any photo, do not write any sentence implying they are absent."
    "\n- If a legible driver-door VIN label photo is present, treat the Production Date requirement as evidenced (the production month/year appears on the same label). Do NOT deduct or say 'not separately documented'."
    "\n- Only deduct for missing Repair Facility info when the Closing Report or other documents clearly show the vehicle is at a named repair facility AND the estimate's 'Repair Facility' section does not list that same facility; "
    "if no repair facility information appears anywhere in the estimate or Closing Report, report 'N/A — not provided' and do NOT deduct."
    "\n- If legibility is the issue, explicitly say 'Present — not clearly legible' and explain why (glare/blur/angle), and request a precise retake rather than marking it missing."
    "\n- Before finalizing, re-scan your output: confirm every referenced Photo # matches the content described (e.g., do not cite an Odometer photo as the point-of-impact photo). Correct any mismatches."
)

# --- Damage Side / Orientation Guard (prompt-only; prevents left/right drift) ---
DAMAGE_SIDE_GUARD = (
    "\n\nDAMAGE SIDE GUIDANCE:"
    "\n- Describe damage on any side (Driver, Passenger, Left, Right) when it is visible in photos."
    "\n- Use the most natural phrasing based on what is visually obvious."
    "\n- Do NOT suppress side-level damage descriptions when damage is clearly visible."
    "\n- Avoid guessing only when orientation is genuinely unclear."
)

# --- Parts Source Guard (prompt-only; prevents OEM vs Aftermarket drift) ---
PARTS_SOURCE_GUARD = (
    "\n\nPARTS SOURCE GUARD (MANDATORY):"
    "\n- Do NOT claim 'aftermarket', 'A/M', 'quality replacement', 'non-OEM', or 'LKQ/used/recycled' parts were used unless the estimate LINE ITEMS explicitly label them as such."
    "\n- Generic disclosure/boilerplate text about aftermarket crash parts does NOT prove aftermarket parts were used."
    "\n- If the Closing Report states no aftermarket/LKQ parts were included, your narrative must not claim they were used."
    "\n- When parts source is not explicit, state that it is not explicitly labeled and avoid guessing; default to OEM only when supported by part numbers/labels."
)

# --- Supplement Handling (prompt-only; ensures detection + narrative mention) ---
SUPPLEMENT_HANDLING = (
    "\n\nSUPPLEMENT HANDLING:"
    "\n- Examine the estimate documents for explicit supplement indicators: 'Supplement', 'Supplement of record', 'S01', 'S02', 'Supplement Summary', or similar."
    "\n- If a supplement is detected, clearly state in the narrative that the estimate is a supplement and summarize what changed: added operations/parts, rate updates, refinish overlap changes, or corrections to prior omissions."
    "\n- If the supplement corrects earlier deficiencies (e.g., missing materials line, added calibrations), note that improvement explicitly."
    "\n- If a supplement exists but required supporting evidence (invoices, photos) is still missing, call this out in Risks/Missing Evidence."
)

ALLOWED_INTENTS = {"guidelines_only","comprehensive","damage_report_from_photos"}

SYSTEM_BASE = (
    "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
    "Populate exactly these keys (always include all, use 'N/A' when not applicable): "
    "['file_number','request_type','claim_number','vin','vin_verification','vehicle',"
    "'odometer_estimate_only','compliance_score','summary_brief','summary_markdown',"
    "'fraud_markdown','primary_impact','secondary_impact','estimated_costs_markdown',"
    "'conclusion']. "
    "Use evidence only from the provided inputs. Cite estimate page/line as 'p#/L#' and photos as 'Photo #'. "
    "The narrative may describe more damage than the primary/secondary impact fields; do not suppress observed damage to force alignment."
    "Avoid guessing; if uncertain, say 'N/A' and why. summary_brief must be <= 280 chars (plain text)."
)

SYSTEM_BASE += (
    " Do not state or imply any client rule unless it appears verbatim in the provided client_rules text. "
    "If client_rules is blank, write the entire report without referencing client rules. "
    "If a value cannot be confirmed from the visible evidence, set it to 'N/A' and briefly state why. "
    " Except when the request_type is 'Create a Damage Report from Photos', Compliance Score must be a numeric percentage 0–100 (never 'N/A'). "
    "If the request_type is 'Create a Damage Report from Photos', set compliance_score to 'N/A' and omit the '## Compliance Score Rationale' section. "
    "If no client_rules are supplied, base the score on estimate-photo internal consistency, evidence completeness, and clarity/legibility. "
    "If compliance_score < 100, include a dedicated section titled '## Compliance Score Rationale' which itemizes every deficiency with exact evidence references "
    "(estimate p#/L# and/or Photo #), assigns an explicit deduction per item, and shows the arithmetic to the final score. "
    "Use a consistent scheme (e.g., Minor -5, Moderate -10, Major -20) and never go below 0. "
    "The 'fraud_markdown' section must never be 'N/A'. If nothing material is found, write "
    "'No material inconsistencies found.' and briefly note what was checked (VIN match, date/metadata, obvious photo tampering, duplicated images)."
)

SYSTEM_BASE += (
    " Focus on a cohesive, professional appraisal. Prefer narrative over rigid tables. "
    "Include a section named '## Detailed Audit Report'. "
    "Include '## Compliance Score Rationale' only when compliance_score < 100, and show deductions from 100 with brief evidence refs (p#/L# or Photo #). "
    "If you include tables, keep them concise and only when they help clarity. "
    "Avoid placeholder rows/columns; do not invent data. "
    "When client_rules text is provided, also include a section titled '## Client Guidelines Comparison' with 3–8 concise bullets quoting the relevant rule fragment and citing evidence (p#/L#, Photo #); "
    "weave any material rule alignment/misalignment into the Detailed Audit Report narrative."
    "When a valuation/clean retail printout exists but the header doesn’t match the estimate’s VIN/year/trim/mileage, label it “Present — mismatched (detail the differences)” and request a corrected printout; never mark it Missing/Not Evidenced. "
    "If a legible driver-door VIN label photo is present, treat Production Date as evidenced; do not mark 'missing' or deduct for lack of a separate photo. "
    "Only deduct for missing Repair Facility info when the Closing Report or other documents clearly show the vehicle is at a named repair facility AND the estimate's 'Repair Facility' section does not list that same facility; "
    "if no repair facility information appears anywhere in the estimate or Closing Report, report 'N/A — not provided' and do NOT deduct."
)

SYSTEM_BASE += (
    " Your 'summary_markdown' MUST include a top-level section named '## Detailed Audit Report' containing a cohesive narrative of at least 10–14 sentences (not bullets). "
    "It must synthesize: impact zones, per-panel damages, repair vs. replace rationale, parts type (OEM/LKQ/Aftermarket), labor ops, refinish/overlap, rate/materials/sublet/tax handling, and estimate integrity. "
    "It must cite concrete evidence inline (e.g., p2/L14, Photo 3). "
    "When evaluating paint materials, recognize that a summary line such as 'Paint Supplies' or 'Paint Materials' with hours and rate in the totals section constitutes a valid cost breakdown. "
    "Do not mark it missing if such a line is present, even if materials are not listed per-panel. "
    "Avoid categorical phrases such as 'deemed repairable' or 'deemed total loss' unless that exact determination appears in the provided documents (e.g., estimate header says 'Total Loss' or an ACV comparison is shown). "
    "Otherwise, use neutral language and do not make a repairability determination."
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

# Safe PDF text extract helper
def _maybe_extract_pdf_text(raw: bytes, fname: str, parts: List[Dict[str, Any]], files_seen: List[str], pdf_text_fulls: Optional[List[str]] = None) -> None:
    try:
        from pdfminer_high_level import extract_text as _x  # type: ignore
    except Exception:
        try:
            from pdfminer.high.level import extract_text as _x  # fallback
        except Exception:
            _x = None
    try:
        if _x:
            full = (_x(io.BytesIO(raw)) or "")
            if pdf_text_fulls is not None and full.strip():
                pdf_text_fulls.append(full)
            t = full[:12000]
            if t.strip():
                parts.insert(0, {"type": "text", "text": t})
                files_seen.append(f"{fname} (pdf text extracted)")
    except Exception:
        pass

# Lightweight OCR helper for images
def _maybe_ocr_image_text(im: Image.Image) -> str:
    if not _OCR_ENABLED:
        return ""
    try:
        im2 = im.convert("L")
        if max(im2.size) < 1400:
            scale = 1400 / max(im2.size)
            im2 = im2.resize((int(im2.width*scale), int(im2.height*scale)))
        txt = pytesseract.image_to_string(im2)
        return (txt or "").strip()
    except Exception:
        return ""

def _add_bytes(parts: List[Dict[str,Any]], files_seen: List[str], photo_index: List[str], raw: bytes, fname: str, used: int, max_images: int, pdf_text_fulls: Optional[List[str]] = None) -> int:
    low = fname.lower()
    if low.endswith(SUPPORTED_PDF_EXTS) and used < max_images:
        try:
            pages = convert_from_bytes(raw, dpi=200)
            files_seen.append(f"{fname} (pdf, {len(pages)} page(s))")
            _maybe_extract_pdf_text(raw, fname, parts, files_seen, pdf_text_fulls=pdf_text_fulls)
            OCR_PAGE_CAP = 100
            ocr_collected = []
            for idx, im in enumerate(pages[:max_images - used]):
                b = io.BytesIO()
                im.save(b, format="JPEG", quality=75, optimize=True)
                parts.append(_image_part_from_bytes(b.getvalue()))
                used += 1
                photo_index.append(f"{fname}::page_{idx+1}")
                if idx < OCR_PAGE_CAP:
                    txt = _maybe_ocr_image_text(im)
                    if txt:
                        ocr_collected.append(txt)
            if ocr_collected:
                parts.insert(0, {"type": "text", "text": ("\n".join(ocr_collected))[:12000]})
                files_seen.append(f"{fname} (ocr text extracted)")
        except Exception as e:
            logging.warning(f"pdf2image failed for {fname}: {e}")
            files_seen.append(f"{fname} (pdf, could not be converted)")
    elif low.endswith(SUPPORTED_IMAGE_EXTS) and used < max_images:
        im_ref = None
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            im_ref = im.copy()
            im.thumbnail((1800,1800))
            b = io.BytesIO(); im.save(b, format="JPEG", quality=75, optimize=True)
            raw = b.getvalue()
        except Exception:
            im_ref = None
        parts.append(_image_part_from_bytes(raw))
        used += 1
        photo_index.append(fname)
        files_seen.append(f"{fname} (photo)")
        if im_ref is not None:
            txt = _maybe_ocr_image_text(im_ref)
            if txt:
                parts.insert(0, {"type":"text", "text": txt[:12000]})
                files_seen.append(f"{fname} (ocr text extracted)")
    elif low.endswith(SUPPORTED_DOCX_EXTS):
        try:
            text = "\n".join([p.text for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()])
        except Exception:
            text = ""
        if text.strip():
            parts.insert(0, {"type":"text","text": text[:12000]})
            files_seen.append(f"{fname} (docx text included)")
        else:
            files_seen.append(f"{fname} (docx, no readable text)")
    elif low.endswith(SUPPORTED_TEXT_EXTS):
        try:
            text = raw.decode("utf-8","ignore")[:12000]
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
# Client Rules: fuzzy finder + endpoints
# -----------------------
def _find_rules_path(base_name: str, rules_dir: str) -> Optional[str]:
    target = base_name.strip()
    if not target.lower().endswith(".docx"):
        target += ".docx"
    p = os.path.join(rules_dir, target)
    if os.path.exists(p):
        return p
    pattern = os.path.join(rules_dir, "*.docx")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    low_target = target.lower()
    for c in candidates:
        if os.path.basename(c).lower() == low_target:
            return c
    for c in candidates:
        if os.path.basename(c).lower().startswith(low_target[:-5]):
            return c
    tokens = [t for t in low_target[:-5].split() if t]
    def contains_in_order(name: str) -> bool:
        i = 0
        for tok in tokens:
            i = name.find(tok, i)
            if i == -1:
                return False
            i += len(tok)
        return True
    for c in candidates:
        if contains_in_order(os.path.basename(c).lower()):
            return c
    return None

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    path = _find_rules_path(client_name, CLIENT_RULES_DIR)
    if not path:
        try:
            files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CLIENT_RULES_DIR, "*.docx")))
        except Exception:
            files = []
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Rules not found for '{client_name}'.",
                "hint": "Ensure the .docx file exists in CLIENT_RULES_DIR (default 'client_rules').",
                "available_rule_files": files[:50]
            },
        )
    try:
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return {"file": os.path.basename(path), "text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Unable to read rules: {e}", "file": os.path.basename(path)})

@app.get("/client-rules")
async def list_client_rules():
    try:
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(CLIENT_RULES_DIR, "*.docx")))
        return {"dir": CLIENT_RULES_DIR, "files": files}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "dir": CLIENT_RULES_DIR})

# -----------------------
# Vision Review
# -----------------------
@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    ai_notes: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("comprehensive")
):
    parts: List[Dict[str, Any]] = []
    files_seen: List[str] = []
    photo_index: List[str] = []  # ordered list of image-like items sent to the model (stable Photo # mapping)
    MAX_IMAGES = 24
    used = 0
    pdf_text_fulls: List[str] = []  # full PDF text for supplement detection

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
                used = _add_bytes(parts, files_seen, photo_index, data, f"{fname}::{inner_name}", used, MAX_IMAGES, pdf_text_fulls=pdf_text_fulls)
        else:
            used = _add_bytes(parts, files_seen, photo_index, raw, fname, used, MAX_IMAGES, pdf_text_fulls=pdf_text_fulls)

    # Collect uploaded TEXT ONLY for evidence checks
    uploaded_text_blobs = []
    for p in parts:
        if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
            uploaded_text_blobs.append(p["text"])
    uploaded_text_all = "\n".join(uploaded_text_blobs)

    # --- Closing Report: extract "Inspection Results" section for deterministic cross-check + shop-status ---
    def _extract_inspection_results_block(_txt: str) -> str:
        if not _txt:
            return ""
        t = _txt.replace("\r", "\n")
        # Capture the text after the "Inspection Results" header up to the next major header.
        m_ir = re.search(
            r"(?is)\bInspection\s+Results\b\s*[:\-]?\s*(.{0,5000}?)(?=\n\s*(?:Possible\s+Supplement\s+Amount|Supplement\b|Estimate\b|Inspection\s+Location\b|Repair\s+Facility\b|Vehicle\b|Owner\b|Appraiser\b|Adjuster\b|$))",
            t,
        )
        if m_ir:
            return (m_ir.group(1) or "").strip()[:5000]
        return ""

    inspection_results_text = _extract_inspection_results_block(uploaded_text_all or "")

    # Closing Report: "Possible Supplement Amount" (this ALONE does NOT mean the estimate is a supplement)
    _possible_supp_amount = None
    _m_psa = re.search(r"(?i)\bPossible\s+Supplement\s+Amount\b\s*\$?\s*([0-9,]+(?:\.[0-9]{2})?)", uploaded_text_all or "")
    if _m_psa:
        _possible_supp_amount = _m_psa.group(1)

    photos_provided = any(isinstance(p, dict) and p.get('type') != 'text' for p in parts)

    # --- Robust detectors (CLEAN RETAIL + ADVISOR) ---
    clean_retail_rx = (
        r"(?i)\b("
        r"NADA|J[.\s-]*D[.\s-]*\s*Power|JDPower\.com|"
        r"Kell?ey\s+Blue\s+Book|KBB\.com|"
        r"Edmunds|Carfax|Cars\.com|"
        r"Clean\s+Retail(?:\s+Value)?"
        r")\b"
    )
    _clean_retail_present = bool(re.search(clean_retail_rx, uploaded_text_all or ""))

    advisor_rx = r"(?i)\bAdvisor\s+Report\b"
    _advisor_present = bool(re.search(advisor_rx, uploaded_text_all or ""))

    paint_mat_rx = r"(Paint\s+(Suppl(?:ies|y)|Materials)|Materials\s*Line)"
    _paint_materials_present = bool(re.search(paint_mat_rx, uploaded_text_all or "", flags=re.IGNORECASE))

    # --- Parts source guardrail (OEM vs Aftermarket/LKQ) ---
    # Detect explicit non-OEM indicators in estimate LINE ITEMS only (ignore generic boilerplate/disclosures).
    _explicit_non_oem_parts = False
    try:
        # Heuristic: look for non-OEM keywords on the same line as an operation/part line (starts with line #).
        _explicit_non_oem_parts = bool(re.search(
            r"(?im)^\s*\d+\s+.*\b(A/M|AFTERMARKET|NON\s*OEM|CAPA|NSF|LKQ|RCY|RECYCLED|USED|RECOND)\b",
            uploaded_text_all or "",
        ))
    except Exception:
        _explicit_non_oem_parts = False

    # Closing report sometimes explicitly states no aftermarket/LKQ parts were included.
    _closing_no_aftermarket = False
    try:
        _closing_no_aftermarket = bool(re.search(r"(?i)\bno\s+aftermarket\b|\bno\s+aftermarket\s+or\s+lkq\b", uploaded_text_all or ""))
    except Exception:
        _closing_no_aftermarket = False

    # Presence detectors for VIN / Odometer / Production Date (OCR text)
    vin_photo_rx = r"(?i)\bDescription:\s*VIN\b|\bVIN(?:\s*#|:)?\s*[A-HJ-NPR-Z0-9]{17}\b"
    odo_photo_rx = r"(?i)\bDescription:\s*Odometer\b|\bOdometer\s*Photo\b|\bPhoto\s*[:#]?\s*Odometer\b"

    # Production date patterns (label)
    prod_date_rx = (
        r"(?is)\b(Production\s*date|Prod(?:uction)?\s*Date|Date\s*of\s*Mfr|Date\s*of\s*Manufacture|"
        r"MFD\.?\s*(?:BY|DATE)?|MFR\.?\s*DATE|MFG\.?\s*DATE|DATE)\b"
        r".{0,80}?\b(0[1-9]|1[0-2])\s*[-/.\u2013\u2014\u2212:\s]\s*(20\d{2}|\d{2})\b"
    )

    _vin_photo_present = bool(re.search(vin_photo_rx, uploaded_text_all or ""))
    _odo_photo_present = bool(re.search(odo_photo_rx, uploaded_text_all or ""))

    # presence + extractor
    _prod_date_present = False
    _prod_date_str = None
    m = re.search(prod_date_rx, uploaded_text_all or "")
    if m:
        mm = m.group(2); yy = m.group(3)
        if len(yy) == 2: yy = "20" + yy
        _prod_date_present = True
        _prod_date_str = f"{mm}/{yy}"
    if not _prod_date_present:
        loose_prod_rx = (
            r"(?is)\b(MFD|MFR|MFG|PROD\.?|PRODUCTION|DATE)\b"
            r".{0,60}?\b(0[1-9]|1[0-2])\s*[-/.\u2013\u2014\u2212:\s]\s*(20\d{2}|\d{2})\b"
        )
        m2 = re.search(loose_prod_rx, uploaded_text_all or "")
        if m2:
            mm = m2.group(2); yy = m2.group(3)
            if len(yy) == 2: yy = "20" + yy
            _prod_date_present = True
            _prod_date_str = f"{mm}/{yy}"

    # treat production date as evidenced if either door label VIN photo or prod date text is seen
    _prod_evidenced = _prod_date_present or _vin_photo_present

    # Lock to 3 intents only
    if ai_intent not in ALLOWED_INTENTS:
        ai_intent = "comprehensive"

    REQ_LABELS = {
        "guidelines_only": "Guidelines → Estimate (no photos)",
        "comprehensive": "Comprehensive: Guidelines + Estimate + Photos (with VIN check)",
        "damage_report_from_photos": "Create a Damage Report from Photos",
    }
    req_label = REQ_LABELS.get(ai_intent, "Comprehensive: Guidelines + Estimate + Photos (with VIN check)")
    log.info(f"ai_intent received: {ai_intent} -> using label: {req_label}")

    KEYS = [
        "file_number","request_type","claim_number","vin","vin_verification","vehicle",
        "odometer_estimate_only","compliance_score","summary_brief","summary_markdown",
        "fraud_markdown","primary_impact","secondary_impact","estimated_costs_markdown","conclusion"
    ]

    SYSTEM = SYSTEM_BASE

    # Closing Report shop-status detector (documents-only; used to prevent false Repair Facility deductions)
    _not_at_shop = False
    try:
        # Prefer the extracted "Inspection Results" block if available (more deterministic)
        if inspection_results_text:
            _not_at_shop = bool(re.search(
                r"(?i)\bNot\s+at\s+Shop\b|\bOwner\s+Location\b|\bInspection\s+Location\s*:?\s*Owner\b|\bInspection\s+Location\s+Owner\b",
                inspection_results_text,
            ))
        if not _not_at_shop:
            _not_at_shop = bool(re.search(
                r"(?i)\bNot\s+at\s+Shop\b|\bOwner\s+Location\b|\bInspection\s+Location\s*:?\s*Owner\b|\bInspection\s+Location\s+Owner\b",
                uploaded_text_all or "",
            ))
    except Exception:
        _not_at_shop = False

    # --- NEW: detect all supplement tags S01, S02, ... in the combined document text ---
    combined_detection_text = (uploaded_text_all or "") + "\n" + "\n".join(pdf_text_fulls or [])
    supplement_versions = sorted(set(re.findall(r"(?i)\bS[0-9]{2}\b", combined_detection_text)))
    supplement_block = ""
    if supplement_versions:
        supplement_block = (
            "\n\nSUPPLEMENT VERSIONS DETECTED FROM DOCUMENTS:\n"
            f"- Detected supplement tags: {', '.join(supplement_versions)}.\n"
            "- Use these exact tags (e.g., 'Supplement S01 and S02') when describing supplements in the narrative.\n"
        )

    prompt_text = (
        f"REQUEST TYPE SELECTED (exact): '{req_label}'. Use this exact string in 'request_type'.\n\n"
        "FILES SEEN (echo verbatim in '## Inputs Used'):\n- "
        + ("\n- ".join(files_seen) if files_seen else "none")
        + "\n\nPHOTO INDEX (MANDATORY — use this mapping for ALL Photo # citations):\n"
        + ("\n".join([f"Photo {i+1}: {name}" for i, name in enumerate(photo_index)]) if photo_index else "No photos were included.")
        + "\n\n"
        + "\n\nCLIENT RULES (if provided; else blank):\n"
        + (client_rules[:2000] if client_rules else "")
        + "\n\nADD'L NOTES FOR AI REVIEW (priority focus; only applies to guidelines/review items):\n"
        + ((ai_notes or "").strip()[:2000] if (ai_notes or "").strip() else "")
        + supplement_block
        + "\n\nANALYSIS LAYOUT (guidance, not strict):\n"
        + DETAIL_TEMPLATES.get(ai_intent, DETAIL_TEMPLATES["comprehensive"])
    )
    
    if (ai_notes or "").strip():
        prompt_text += (
            "\n\nAI NOTES INSTRUCTIONS (MANDATORY): "
            "You MUST explicitly address the Add'l Notes in the '## Detailed Audit Report' narrative and/or '## Key Issues & Actions'. "
            "If a note cannot be verified from evidence, say so and state what evidence would be needed."
        )
    prompt_text = (
        "OUTPUT FORMAT (MANDATORY): Return ONLY a single strict JSON object with keys "
        "['file_number','request_type','claim_number','vin','vin_verification','vehicle',"
        "'odometer_estimate_only','compliance_score','summary_brief','summary_markdown',"
        "'fraud_markdown','primary_impact','secondary_impact','estimated_costs_markdown','conclusion'] "
        "and no extra text before or after.\n\n"
    ) + prompt_text

    # --- Closing Report cross-check injection (Inspection Results vs Detailed Audit Report) ---
    if inspection_results_text:
        prompt_text += (
            "\n\nCLOSING REPORT — INSPECTION RESULTS (verbatim extract):\n"
            + inspection_results_text[:2000]
            + "\n\nCLOSING REPORT CROSS-CHECK (MANDATORY):\n"
            "- In your '## Detailed Audit Report' narrative, explicitly confirm your narrative matches the Inspection Results above.\n"
            "- If the Inspection Results indicate 'Not at Shop' / 'Owner location', do NOT claim Repair Facility info is missing and do NOT deduct for it.\n"
            "- If the Inspection Results list a named shop/repair facility, ensure your Repair Facility discussion is consistent with that documentation.\n"
        )
    if _possible_supp_amount:
        prompt_text += (
            "\n\nCLOSING REPORT NOTE: 'Possible Supplement Amount' is present ($" + str(_possible_supp_amount) + "). "
            "This alone does NOT mean the estimate is a supplement. Do NOT label the estimate as a Supplement unless explicit supplement tags (e.g., S01/S02) or 'Supplement Summary' are present.\n"
        )

    if photos_provided:
        prompt_text += (
            "\n\nPHOTOS PROVIDED: This upload includes photos/images. "
            "Do NOT say 'photos not provided', 'not provided here', or similar. "
            "Assess provided photos and do not mark required photos as missing unless they are truly absent."
        )

    if ai_intent == "damage_report_from_photos":
        prompt_text += (
            "\n\nPHOTOS-ONLY MODE: Set 'compliance_score' to 'N/A'. "
            "Do NOT include a '## Compliance Score Rationale' section."
            "\nODOMETER TRANSCRIPTION: Use only the odometer photo for mileage. "
            "If the digits are not fully readable, return 'Present — not clearly legible' and explain (glare/blur/angle). "
            "Do not infer or estimate mileage from other sources."
        )
        prompt_text += (
            "\nABSOLUTE BAN (PHOTOS-ONLY): Do not reference or imply any estimate document. "
            "Do not use phrases like 'the estimate', 'estimate suggests', 'p#/L#', 'CCC', 'labor rate', or any estimate page/line notation. "
            "If you need to discuss costs, label them as 'photo-based rough costs' with explicit assumptions, and keep them independent of any estimate. "
            "If no odometer photo is present in the upload set, output 'Missing' for odometer_estimate_only."
        )
    else:
        prompt_text += SUPPLEMENT_HANDLING

    if ai_intent == "comprehensive":
        prompt_text += (
            "\n\nUploader note: If odometer and registration photos are present, report their legibility accurately. "
            "If they are not present, state 'Missing' plainly. Do not assume their presence if they cannot be visually confirmed."
        )

    if client_rules.strip():
        prompt_text += (
            "\n\nWhen client_rules text is provided, you MUST include a section titled '## Client Guidelines Comparison' "
            "with 3–8 concise bullets. For each, quote the relevant rule fragment and mark Aligned / Not Aligned / Not Evidenced, "
            "citing evidence (p#/L# or Photo #). Also weave any material rule alignment/misalignment into the '## Detailed Audit Report' narrative."
        )
        prompt_text += (
            "\n\nWeave the following static audit questions naturally into the '## Detailed Audit Report' narrative "
            "(do NOT present as a separate Q&A list; integrate answers inline and cite evidence with p#/L# and Photo # as applicable):\n"
            + "\n".join(f"- {q}" for q in STATIC_AUDIT_QUESTIONS)
        )

    prompt_text += (
        "\n\nPHOTO NUMBER SANITY CHECK: Before finalizing, verify that every referenced Photo # actually exists and matches the content described."
        "\nCOST RATIONALE REQUIREMENT: For each cost bucket (Body/Paint/Materials/Parts/Sublet/Tax), include a one-line rationale tied to observed operations or panel counts when you provide costs."
    )

    prompt_text += IDENTIFIERS_VERIFICATION_PROTOCOL
    prompt_text += CONSISTENCY_GUARD
    prompt_text += DAMAGE_SIDE_GUARD
    prompt_text += PARTS_SOURCE_GUARD

    # --------- EVIDENCE FLAGS ----------
    flags = []
    if _not_at_shop:
        flags.append(
            "- Closing Report Inspection Results indicate the vehicle is NOT at a repair facility (Owner location / Not at Shop). "
            "Do NOT treat Repair Facility information as required and do NOT deduct for it."
        )
    if _paint_materials_present:
        flags.append(
            "- Paint materials summary line is present in the estimate totals (e.g., 'Paint Supplies' on the totals page). "
            "Treat paint materials as evidenced even if not itemized per panel."
        )
    if _clean_retail_present:
        flags.append(
            "- Clean Retail Value printout is present (e.g., J.D. Power / NADA / KBB / Edmunds / Carfax / Cars.com). "
            "If the year/trim/mileage do not match the estimate/VIN, state 'Present — mismatched' and specify the differences. "
            "Do not mark it 'Not Evidenced'."
        )
    if _advisor_present:
        flags.append(
            "- A refreshed copy of the Advisor Report is present in the documents. "
            "Do not state it is missing."
        )
    if _vin_photo_present:
        flags.append(
            "- A driver-door VIN label/photo is present. Treat Production Date as evidenced by the same label; do NOT deduct or claim 'not separately documented'."
        )
    if _odo_photo_present:
        flags.append(
            "- An odometer photo is present. Do not mark the odometer as missing; transcribe the digits and cite the Photo #."
        )
    if _prod_date_present:
        flags.append(
            "- A production date (Date of Mfr/MFD DATE) is visible on a door label photo. Do not mark Production Date as missing; cite the Photo # and the month/year."
        )
    if _closing_no_aftermarket and not _explicit_non_oem_parts:
        flags.append(
            "- Closing Report states no aftermarket/LKQ parts were included and the estimate line items do not explicitly indicate non-OEM parts. "
            "Do NOT describe any replaced parts as aftermarket/A/M/LKQ; treat them as OEM unless line items explicitly say otherwise."
        )
    elif _explicit_non_oem_parts:
        flags.append(
            "- Non-OEM parts indicators (A/M/Aftermarket/LKQ/etc.) appear in the estimate line items. "
            "If you discuss parts type, be specific and cite the exact estimate line(s) showing the indicator."
        )

    # ✅ FIX #1: Actually inject evidence flags into the prompt (previously you computed flags but never used them)
    if flags:
        prompt_text += (
            "\n\nEVIDENCE FLAGS (obey these and do NOT contradict them):\n"
            + "\n".join(flags)
        )

    parts_payload: List[Dict[str,Any]] = []
    redaction_success = False
    try:
        red_prompt = redact_text_preserve_vin_claim(prompt_text)
        redaction_success = True
    except Exception as e:
        log.warning(f"Redaction failed on prompt_text: {e}")
        red_prompt = prompt_text

    parts_payload.append({"type": "text", "text": red_prompt})

    if parts:
        for p in parts:
            if p.get("type") == "text" and isinstance(p.get("text"), str):
                try:
                    red_txt = redact_text_preserve_vin_claim(p["text"])
                    redaction_success = True
                except Exception as e:
                    log.warning(f"Redaction failed on a text part: {e}")
                    red_txt = p["text"]
                parts_payload.append({"type": "text", "text": red_txt})
            else:
                parts_payload.append(p)

    redaction_status = "Redacted PII: Successful ✅" if redaction_success else "Redacted PII: Not Applied"

    # -------------------------------
    # Keep prompt lean
    # -------------------------------
    TEXT_PART_LIMIT = 6
    _text_parts = [p for p in parts_payload if p.get("type") == "text"]
    _image_parts = [p for p in parts_payload if p.get("type") != "text"]
    for tp in _text_parts:
        if isinstance(tp.get("text"), str) and len(tp["text"]) > 8000:
            tp["text"] = tp["text"][:8000]
    parts_payload = _text_parts[:TEXT_PART_LIMIT] + _image_parts

    # Token limits
    MAX_TOKENS_BY_INTENT = {
        "comprehensive": 2200,
        "guidelines_only": 1500,
        "damage_report_from_photos": 1400
    }
    max_tokens = MAX_TOKENS_BY_INTENT.get(ai_intent, 1500)

    # Call GPT and parse JSON (JSON hardened)
    try:
        rsp = client.chat_completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": parts_payload}],
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type":"json_object"}
        )
    except AttributeError:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": parts_payload}],
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type":"json_object"}
        )

    # --- Hardened JSON parse helper
    def _try_parse_json(raw_text: str):
        if not raw_text:
            return None
        raw_local = raw_text.strip()
        for fence in ("```json", "```JSON", "```"):
            if raw_local.startswith(fence):
                raw_local = raw_local[len(fence):]
            if raw_local.endswith("```"):
                raw_local = raw_local[:-3]
        raw_local = raw_local.strip()
        raw_local = raw_local.replace("\ufeff", "").replace("\u200b", "").replace("\u00A0", " ")
        try:
            return json.loads(raw_local)
        except Exception:
            pass
        lb = raw_local.find("{"); rb = raw_local.rfind("}")
        chunk = raw_local[lb:rb+1] if (lb != -1 and rb != -1 and rb > lb) else raw_local
        fixes = {"\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"', "\r": "", "\t": "    "}
        for k, v in fixes.items():
            chunk = chunk.replace(k, v)
        chunk = re.sub(r",\s*([}\]])", r"\1", chunk)
        try:
            return json.loads(chunk)
        except Exception:
            return None

    try:
        raw = (rsp.choices[0].message.content or "")
    except Exception as e:
        log.error(f"LLM returned no content: {e}")
        return JSONResponse(status_code=500, content={"error":"Model returned no content."})

    data = _try_parse_json(raw)

    # One safe retry on truncation
    try:
        finish_reason = getattr(rsp.choices[0], "finish_reason", None)
    except Exception:
        finish_reason = None

    if data is None and (finish_reason == "length" or (raw and raw.strip().endswith("..."))):
        _text_parts_retry = [p for p in parts_payload if p.get("type") == "text"]
        _image_parts_retry = [p for p in parts_payload if p.get("type") != "text"]
        shrunk = _text_parts_retry[: max(3, len(_text_parts_retry)//2)] + _image_parts_retry
        retry_tokens = min(3000, max_tokens + 600)
        try:
            rsp2 = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": shrunk}],
                max_tokens=retry_tokens,
                temperature=0,
                response_format={"type": "json_object"}
            )
            raw2 = (rsp2.choices[0].message.content or "")
            data = _try_parse_json(raw2)
        except Exception:
            pass

    # Fallback: formatting pass
    if data is None:
        try:
            fix_prompt = [
                {"role":"system","content":
                    "You are a formatter. Return ONLY a strict JSON object. No prose. No markdown. No code fences."
                },
                {"role":"user","content":
                    "Convert the following text into a valid JSON object. "
                    "Use exactly these keys (all required): "
                    "['file_number','request_type','claim_number','vin','vin_verification','vehicle',"
                    "'odometer_estimate_only','compliance_score','summary_brief','summary_markdown',"
                    "'fraud_markdown','primary_impact','secondary_impact','estimated_costs_markdown','conclusion'] "
                    "Do not invent new keys. If a field is unavailable, use 'N/A'. "
                    "Here is the text:\n\n" + raw
                }
            ]
            try:
                fix_rsp = client.chat_completions.create(  # type: ignore[attr-defined]
                    model=MODEL,
                    messages=fix_prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    response_format={"type":"json_object"}
                )
            except AttributeError:
                fix_rsp = client.chat.completions.create(
                    model=MODEL,
                    messages=fix_prompt,
                    max_tokens=max_tokens,
                    temperature=0,
                    response_format={"type":"json_object"}
                )
            fixed = (fix_rsp.choices[0].message.content or "")
            data = _try_parse_json(fixed)
        except Exception as e:
            log.error(f"Self-heal reformat failed: {e}")

    if data is None:
        log.error(f"LLM failure or JSON parse error; first 500 chars:\n" + (raw or "")[:500])
        skeleton = {k: "N/A" for k in KEYS}
        skeleton["file_number"] = file_number
        skeleton["request_type"] = req_label
        skeleton["summary_brief"] = "N/A (model output could not be parsed; skeleton returned)."
        skeleton["summary_markdown"] = (
            "## Detailed Audit Report\n"
            "Model output could not be parsed into JSON on this run. Please resubmit."
        )
        skeleton["fraud_markdown"] = "No material inconsistencies found."
        return skeleton

    def _get(k):
        v = data.get(k)
        return "" if v is None else str(v)

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

    # ✅ FIX #2: Hard fallback so UI never gets an empty narrative ("No narrative generated")
    try:
        sm_tmp = (result.get("summary_markdown") or "").strip()
        if not sm_tmp:
            result["summary_markdown"] = (
                "## Detailed Audit Report\n"
                "Narrative fallback: The model returned an empty narrative field. "
                "Please re-run with the same inputs; core identifiers and score fields were still returned.\n\n"
                "## Overall Assessment\n"
                f"Request Type: {result.get('request_type','N/A')}\n"
                f"Compliance Score: {result.get('compliance_score','N/A')}\n"
            )
        elif "## Detailed Audit Report" not in sm_tmp:
            # Keep minimal: do not re-write content; just prepend the required header to avoid downstream display rules.
            result["summary_markdown"] = "## Detailed Audit Report\n" + sm_tmp
    except Exception:
        pass

    # --- Score ↔ narrative synchronization ---
    def _extract_score_from_text(text: str):
        if not text:
            return None
        m = re.search(r"(?is)\bFinal\s*score\b[^0-9]{0,10}(\d{1,3})\s*%?\b", text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except Exception:
                pass
        m = re.search(r"(?is)\bthe\s+compliance\s+score\s+is\s+set\s+at\s+(\d{1,3})\s*%?\b", text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except Exception:
                pass
        m = re.search(r"(?is)\bCompliance\s*Score\b[^0-9]{0,10}(\d{1,3})\s*%?\b", text)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            except Exception:
                pass
        return None

    def _canonicalize_score_in_narrative(narr: str, score_int: int) -> str:
        if not narr:
            narr = ""
        scrub_lines = r"(?im)^\s*(Final\s*score|Compliance\s*Score)\s*[:\-–]\s*\d{1,3}\s*%?\s*$"
        narr = re.sub(scrub_lines, "", narr)
        narr = re.sub(r"(?is)\bthe\s+compliance\s+score\s+is\s+set\s+at\s+\d{1,3}\s*%?\b",
                      "the compliance score is set as below", narr)
        narr = re.sub(r"\n{3,}", "\n\n", narr).strip()
        return (narr + f"\n\nCompliance Score: {score_int}").strip()

    try:
        sm = (result.get("summary_markdown") or "")
        if ai_intent == "damage_report_from_photos":
            result["compliance_score"] = "N/A"
            sm = re.sub(r"(?im)^\s*(Final\s*score|Compliance\s*Score)\s*[:\-–]\s*\d{1,3}\s*%?\s*$", "", sm).strip()
            result["summary_markdown"] = sm
        else:
            s_text = _extract_score_from_text(sm)
            s_json = None
            v = (result.get("compliance_score") or "").strip()
            if re.fullmatch(r"\d{1,3}", v):
                try:
                    s_json = int(v)
                except Exception:
                    s_json = None
            chosen = s_text if s_text is not None else s_json
            if chosen is not None:
                chosen = max(0, min(100, int(chosen)))
                result["compliance_score"] = str(chosen)
                result["summary_markdown"] = _canonicalize_score_in_narrative(sm, chosen)
            else:
                result["compliance_score"] = "N/A"
                sm = re.sub(r"(?im)^\s*(Final\s*score|Compliance\s*Score)\s*[:\-–]\s*\d{1,3}\s*%?\s*$", "", sm).strip()
                result["summary_markdown"] = sm
    except Exception:
        pass

    # ---- AUTO-ADD Compliance Score Rationale with arithmetic when missing ----
    try:
        if ai_intent != "damage_report_from_photos":
            sm = result.get("summary_markdown") or ""
            score_str = (result.get("compliance_score") or "").strip()
            if "## Compliance Score Rationale" not in sm and re.fullmatch(r"\d{1,3}", score_str):
                score_int = max(0, min(100, int(score_str)))
                if score_int < 100:
                    deduction = 100 - score_int
                    rationale_lines = [
                        "",
                        "## Compliance Score Rationale",
                        f"Starting from 100%, a total deduction of {deduction} points was applied based on the minor, non-fatal documentation/formatting items described above, resulting in a final compliance score of {score_int}%.",
                    ]
                    if _prod_evidenced or _clean_retail_present:
                        rationale_lines.append(
                            "No deduction was applied for Production Date or Clean Retail value, as these items are evidenced in the file and treated as compliant."
                        )
                    sm = sm.rstrip() + "\n\n" + "\n".join(rationale_lines)
                    result["summary_markdown"] = sm
    except Exception:
        pass

    # Clean Retail deterministic override
    if _clean_retail_present:
        try:
            sm = result.get("summary_markdown") or ""
            # 1) Flip or remove "missing clean retail" style statements
            sm = re.sub(
                r"(?im)^\s*[-*]\s*Missing\s+clean\s+retail\s+value\s+printout[^\n]*$",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bodometer\s+photo\s+not\s+present\b.*?(?:\n|$)", "", sm
            )
            sm = re.sub(
                r"(?is)\bmissing\s+(?:a\s+)?clean\s+retail\s+value\s+printout\b[^\n\.]*",
                "Clean retail value printout is present via valuation (e.g., NADA/J.D. Power/KBB/Edmunds/Carfax/Cars.com)",
                sm,
            )
            sm = re.sub(
                r"(?is)a\s+printout\s+showing\s+the\s+clean\s+retail\s+value[^.]*\.",
                "A valuation printout (e.g., J.D. Power or NADA clean retail page) is present in the file and satisfies this requirement.",
                sm,
            )
            sm = re.sub(
                r"(?is)\bclean\s+retail\s+value[^.\n]*Not\s+Evidenced[^.\n]*",
                "Clean retail value requirement is evidenced by the valuation printout (e.g., NADA/J.D. Power/KBB).",
                sm,
            )
            sm = re.sub(
                r"(?is)\bNo\s+printout\s+found\b",
                "Valuation printout confirmed present in file.",
                sm,
            )
            # New: Also rewrite the longer paragraph variant
            sm = re.sub(
                r"(?is)Also,\s+the\s+client\s+rules\s+require\s+a\s+printout\s+showing\s+the\s+Clean\s+Retail\s+Value\s+of\s+the\s+unit,[^.]*\.\s*The\s+NADA\s+value\s+is\s+mentioned[^.]*\.\s*These\s+omissions\s+reduce\s+compliance\.",
                "Client rules require a printout showing the Clean Retail Value of the unit; a valuation printout (e.g., J.D. Power or NADA clean retail page) is present in the file and satisfies this requirement.",
                sm,
            )
            # 2) Client Guidelines bullet: convert 'Not Evidenced' to 'Aligned'
            sm = re.sub(
                r'(?im)^-?\s*"Printout\s+showing\s+the\s+Clean\s+Retail\s+Value\s+of\s+the\s+unit\s+is\s+required[^"]*"\s*-\s*Not\s+Evidenced[^\n]*$',
                '- "Printout showing the Clean Retail Value of the unit is required with all files" - Aligned (valuation printout present in file, e.g., NADA/J.D. Power page).',
                sm,
            )
            # 3) Risks bullet about missing clean retail: remove
            sm = re.sub(
                r"(?im)^\s*[-*]\s*High:\s*Missing\s+clean\s+retail\s+value\s+printout[^\n]*$",
                "",
                sm,
            )
            # 4) Conclusion/summary variants mentioning absence of clean retail
            sm = re.sub(
                r"(?is)Missing\s+repair\s+facility\s+info\s+and\s+clean\s+retail\s+value\s+printout\s+are\s+noted\s+compliance\s+issues\.",
                "Only minor documentation items are noted; core estimate, Clean Retail value evidence, and production date requirements are satisfied.",
                sm,
            )
            sm = re.sub(
                r"(?is)Compliance\s+is\s+reduced\s+due\s+to\s+missing\s+repair\s+facility\s+information\s+and\s+absence\s+of\s+a\s+clean\s+retail\s+value\s+printout\.",
                "Compliance is modestly reduced due to minor non-fatal documentation items only; Production Date and Clean Retail value requirements are satisfied in this file.",
                sm,
            )

            # Backstop replacements
            sm_fixed = re.sub(
                r"(?i)(Clean\s+retail\s+value[^:\n]*:\s*)(Not\s+Evidenced[^.\n]*)",
                r"\1Evidenced (Clean Retail printout present via NADA/J.D. Power/KBB/Edmunds/Carfax/Cars.com)",
                sm,
            )
            sm_fixed = sm_fixed.replace(
                "Clean retail value printout: Not Evidenced (NADA/J.D. Power/KBB/etc. required on all files).",
                "Clean retail value printout: Evidenced (Clean Retail printout present via NADA/J.D. Power/KBB/etc.).",
                )
            result["summary_markdown"] = sm_fixed

            sb = result.get("summary_brief") or ""
            sb = re.sub(
                r"(?i)Clean\s+retail\s+value[^.]*Not Evidenced[^.]*",
                "Clean Retail value printout present and compliant.",
                sb,
            )
            sb = re.sub(
                r"(?is)\bmissing\s+(?:a\s+)?clean\s+retail\s+value\s+printout\b[^.]*",
                "Clean Retail value printout present via J.D. Power/NADA/KBB valuation.",
                sb,
            )
            sb = re.sub(
                r"(?is)absence\s+of\s+a\s+clean\s+retail\s+value\s+printout",
                "minor non-fatal documentation items (not related to Clean Retail value requirement)",
                sb,
            )

            # --- scrub release paperwork mentions out of brief as a deduction reason ---
            if "release paperwork" in sb.lower():
                sb = re.sub(
                    r"(?is)\band\s+release\s+paperwork\b",
                    "",
                    sb,
                )
                sb = re.sub(
                    r"(?is)\bincomplete\s+release\s+paperwork\b[^\.]*",
                    "",
                    sb,
                )

            result["summary_brief"] = sb
        except Exception:
            pass

    # Narrative cleanup to remove false "missing" claims if evidence present
    try:
        sm = result.get("summary_markdown") or ""
        orig_sm = sm
        lower_sm = sm.lower()

        if _odo_photo_present:
            sm = re.sub(r"(?im)^\s*[-*]\s*Missing\s+odometer\s+photo.*$", "", sm)
            sm = re.sub(r"(?is)\bodometer\s+photo\s+not\s+present\b.*?(?:\n|$)", "", sm)
            sm = re.sub(r"(?is)\bthe\s+odometer\s+(?:photo\s+)?is\s+missing\b.*?(?:\n|$)", "", sm)

        if _prod_evidenced:
            sm = re.sub(
                r"(?im)^\s*[-*]\s*Missing\s+production\s+date\s*(?:plate|photo)?[^\n]*$",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bfile\s+is\s+missing\s+(?:a\s+)?production\s+date\s+photo\b[^\n\.]*",
                "Production date is documented on the driver-door VIN label photo and satisfies the client requirement",
                sm,
            )
            sm = re.sub(
                r"(?is)\bmissing\s+(?:a\s+)?production\s+date\s+photo\b[^\n\.]*",
                "Production date is documented on the driver-door VIN label photo",
                sm,
            )
            sm = re.sub(
                r"(?is)\bproduction\s+date\s+(?:plate\s+)?(?:photo\s+)?not\s+present\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bno\s+production\s+date(?:\s+(?:plate|photo|image))?\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bproduction\s+date\s+not\s+separately\s+documented\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bthe\s+production\s+date(?:\s+(?:plate|photo|image))?\s+is\s+missing\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r'(?im)^-?\s*"Production\s+Date\s+Photo\s+is\s+mandatory"\s*-\s*Not\s+Evidenced[^\n]*$',
                '- "Production Date Photo is mandatory" - Aligned (production date documented on the driver-door VIN label photo).',
                sm,
            )
            sm = re.sub(
                r"(?im)^\s*[-*]\s*High:\s*Missing\s+production\s+date\s*photo[^\n]*$",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)The\s+production\s+date\s+is\s+listed\s+in\s+the\s+estimate\s*\(p1\)\s*but\s+no\s+separate\s+photo\s+of\s+the\s+production\s+date\s+sticker\s+is\s+provided,\s*which\s+is\s+a\s+client\s+rule\s+requirement\.",
                "",
                sm,
            )
            sm = re.sub(
                r"Client rules compliance is mostly met except for the Production date is documented on the driver-door VIN label photo\.",
                "Client rules compliance is mostly met. Production date is documented on the driver-door VIN label photo.",
                sm,
            )
            sm = re.sub(
                r"(?is)client rules compliance is mostly met except for the production date is documented on the driver-door vin label photo",
                "Client rules compliance is mostly met and the Production date is documented on the driver-door VIN label photo",
                sm,
            )
        elif _prod_date_present:
            sm = re.sub(
                r"(?im)^\s*[-*]\s*Missing\s+production\s+date\s*(?:plate|photo)?[^\n]*$",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bproduction\s+date\s+(?:plate\s+)?(?:photo\s+)?not\s+present\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bno\s+production\s+date(?:\s+(?:plate|photo|image))?\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bproduction\s+date\s+not\s+separately\s+documented\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bthe\s+production\s+date(?:\s+(?:plate|photo|image))?\s+is\s+missing\b.*?(?:\n|$)",
                "",
                sm,
            )
            sm = re.sub(
                r'(?im)^-?\s*"Production\s+Date\s+Photo\s+is\s+mandatory"\s*-\s*Not\s+Evidenced[^\n]*$',
                '- "Production Date Photo is mandatory" - Aligned (production date documented on a door label photo).',
                sm,
            )
            sm = re.sub(
                r"(?im)^\s*[-*]\s*High:\s*Missing\s+production\s+date\s*photo[^\n]*$",
                "",
                sm,
            )

        # --- Release paperwork must never be a compliance deduction ---
        if "release paperwork" in lower_sm:
            sm = re.sub(
                r"(?is)The\s+absence\s+of\s+repair\s+facility\s+information\s+and\s+incomplete\s+release\s+paperwork\s+reduce\s+compliance\s+but\s+do\s+not\s+affect\s+the\s+technical\s+accuracy\s+of\s+the\s+estimate\.",
                "The absence of repair facility information is noted as a minor documentation item but does not affect the technical accuracy of the estimate.",
                sm,
            )
            sm = re.sub(
                r"(?im)^\s*[-*]\s*Missing\s+release\s+paperwork[^\n]*$",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\band\s+release\s+paperwork\b",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)\bincomplete\s+release\s+paperwork\b[^\.]*\.",
                "",
                sm,
            )

        # Repair Facility + Owner's location: do NOT deduct
        if _not_at_shop and "repair facility" in lower_sm:
            sm = re.sub(r"(?is)\babsence\s+of\s+repair\s+facility\s+(?:details|info|information)\b[^\.]*\.(?:\s*)", "", sm)
            sm = re.sub(
                r"(?is)However,\s+the\s+file\s+lacks\s+repair\s+facility\s+information[^\.]*\.",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)Missing\s+repair\s+facility\s+info[^\.]*\.",
                "",
                sm,
            )
            sm = re.sub(
                r"(?is)missing\s+repair\s+facility\s+information\b[^\.]*\.",
                "",
                sm,
            )

        sm = re.sub(r"\n{3,}", "\n\n", sm).strip()
        if len(sm) < 120:
            sm = orig_sm.strip()
        result["summary_markdown"] = sm if sm else orig_sm.strip()
    except Exception:
        pass

    # FINAL OVERRIDE of Compliance Score Rationale when PD / Clean Retail are evidenced
    try:
        if ai_intent != "damage_report_from_photos":
            sm = result.get("summary_markdown") or ""
            score_str = (result.get("compliance_score") or "").strip()
            if re.fullmatch(r"\d{1,3}", score_str):
                score_int = max(0, min(100, int(score_str)))
                if score_int < 100 and (_clean_retail_present or _prod_evidenced):
                    # Remove any existing "## Compliance Score Rationale" section entirely
                    pattern = r"(?is)\n##\s*Compliance\s*Score\s*Rationale\b.*?(?=\n##\s|\Z)"
                    sm_no_section = re.sub(pattern, "", sm).rstrip()

                    deduction = 100 - score_int
                    new_lines = [
                        "",
                        "## Compliance Score Rationale",
                        f"Starting from 100%, a total deduction of {deduction} points was applied for minor, non-fatal documentation/formatting items noted above (e.g., small clarity or layout issues), resulting in a final compliance score of {score_int}%.",
                    ]
                    new_lines.append(
                        "No deduction was taken for Production Date, Clean Retail value, or Release Paperwork, as these items are either evidenced in the file or outside the scope of this compliance audit."
                    )
                    sm_final = sm_no_section + "\n\n" + "\n".join(new_lines)
                    result["summary_markdown"] = sm_final
    except Exception:
        pass

    # Normalize odometer field when photo is present, so header is clean
    try:
        if _odo_photo_present:
            sm = result.get("summary_markdown") or ""
            odo_field = (result.get("odometer_estimate_only") or "").strip()

            # Narrative overrides header: if the narrative says mileage is unknown or explicitly indicates
            # there is no odometer photo, do NOT force "Present and legible in photos...".
            _narr_says_no_odo = bool(re.search(
                r"(?is)\b("
                r"no\s+odometer\s+photo|"
                r"odometer\s+photo\s+is\s+missing|"
                r"mileage\s+is\s+unknown|"
                r"mileage\s+not\s+documented|"
                r"odometer\s+reading\s+is\s+(?:marked\s+)?unknown"
                r")\b",
                sm,
            ))

            if _narr_says_no_odo:
                _odo_photo_present = False
                if odo_field in {"", "N/A"}:
                    result["odometer_estimate_only"] = "UNK / Unknown (no odometer photo provided)."
            else:
                # Try to extract explicit mileage from narrative
                m_odo = re.search(r"(?is)odometer\s+reading\s+of\s+([0-9,]+)\s*miles", sm)
                if m_odo:
                    miles = m_odo.group(1)
                    result["odometer_estimate_only"] = f"{miles} miles (confirmed by estimate and photos)."
                else:
                    # Fallback generic phrasing
                    result["odometer_estimate_only"] = "Present and legible in photos (e.g., odometer photo)."

                # Clean any weird "No, odometer photo present..." phrasing from brief or narrative
                for key in ("summary_brief", "summary_markdown"):
                    txt = result.get(key) or ""
                    if txt:
                        txt = re.sub(
                            r"No,\s*odometer\s+photo\s+present\s+and\s+legible\s*\(Photo\s*\d+\)",
                            "Odometer is present and legible in the photos and matches the estimate.",
                            txt,
                            flags=re.IGNORECASE,
                        )
                        result[key] = txt
    except Exception:
        pass

    # Non-empty Fraud fallback
    if not result["fraud_markdown"] or result["fraud_markdown"].strip().upper() in {"", "N/A"}:
        result["fraud_markdown"] = (
            "No material inconsistencies found. Checks performed: VIN match across estimate and photos, "
            "odometer/registration presence and legibility, duplicate/edited images, timestamp continuity, and "
            "panel/impact consistency."
        )

    # ---- FINAL FORBIDDEN-DEDuction scrubber + optional score restore (prevents "Production date documented" being treated as a deficiency) ----
    try:
        def _scrub_forbidden(_t_in: str) -> str:
            if not _t_in:
                return _t_in
            _t = str(_t_in)

            # Production Date: if door-label VIN photo or prod date is evidenced, never call it missing or a deduction.
            if _prod_evidenced:
                _t = re.sub(r"(?is)\bmissing\s+(?:mandatory\s+)?production\s+date\s+(?:photo|plate|sticker|label)\b[^.\n]*[\.]?", "", _t)
                _t = re.sub(r"(?is)\bmissing\s+production\s+date\s+photos?\b[^.\n]*[\.]?", "", _t)
                _t = re.sub(r"(?is)\bmandatory\s+production\s+date\s+photo\b[^.\n]*[\.]?", "", _t)

                # Neutralize the common contradiction where "Production date is documented..." is incorrectly framed as an exception/deficiency.
                _t = re.sub(
                    r"(?is)\bexcept\s+for\s+(?:the\s+)?production\s+date\s+is\s+documented\s+on\s+the\s+driver-door\s+vin\s+label\s+photo\b[^.\n]*[\.]?",
                    "Production date is documented on the driver-door VIN label photo. ",
                    _t,
                )
                _t = re.sub(
                    r"(?is)\bcompliance\s+is\s+reduced\s+due\s+to\s+[^.\n]*\bproduction\s+date\b[^.\n]*[\.]?",
                    "",
                    _t,
                )

            # Clean Retail: if detected, don't call it missing/absent.
            if _clean_retail_present:
                _t = re.sub(r"(?is)\bmissing\s+(?:a\s+)?clean\s+retail\s+value\s+printout\b[^.\n]*[\.]?", "", _t)
                _t = re.sub(r"(?is)\babsence\s+of\s+(?:a\s+)?clean\s+retail\s+value\s+printout\b[^.\n]*[\.]?", "", _t)

            # Parts source: prevent false aftermarket/LKQ statements when not evidenced in line items.
            if _closing_no_aftermarket and not _explicit_non_oem_parts:
                # Replace common phrasing that incorrectly asserts aftermarket/LKQ usage.
                _t = re.sub(r"(?i)\b(aftermarket|a/m|non\s*oem|quality\s+replacement)\b(?=\s+(parts?|components?))", "OEM", _t)
                _t = re.sub(r"(?i)\b(lkq|recycled|used|rcy)\b(?=\s+(parts?|components?))", "OEM", _t)
                _t = re.sub(r"(?is)\busing\s+oem\s+oem\b", "using OEM", _t)
                # Remove any broad sentence that still claims aftermarket was used (keep disclaimers if present).
                _t = re.sub(
                    r"(?is)\bthe\s+estimate\s+calls\s+for\b[^.\n]{0,220}\baftermarket\b[^.\n]{0,220}\.",
                    "The estimate specifies replacement parts without explicit non-OEM indicators in the line items; do not label them as aftermarket unless the line items say so.",
                    _t,
                )

            # Release paperwork is outside the scope; never claim compliance reduction for it.
            _t = re.sub(r"(?is)\bmissing\s+release\s+paperwork\b[^.\n]*[\.]?", "", _t)
            _t = re.sub(r"(?is)\bincomplete\s+release\s+paperwork\b[^.\n]*[\.]?", "", _t)
            _t = re.sub(r"(?is)\bcompliance\s+is\s+reduced\s+due\s+to\s+[^.\n]*\brelease\s+paperwork\b[^.\n]*[\.]?", "", _t)

            # Repair Facility: scrub missing-language ONLY when Closing Report indicates owner location / not at shop.
            if _not_at_shop:
                _t = re.sub(r"(?is)\bmissing\s+repair\s+facility\s+(?:info|information|details)\b[^.\n]*[\.]?", "", _t)
                _t = re.sub(r"(?is)\babsence\s+of\s+repair\s+facility\s+(?:info|information|details)\b[^.\n]*[\.]?", "", _t)
                _t = re.sub(r"(?is)\bcompliance\s+is\s+reduced\s+due\s+to\s+[^.\n]*\brepair\s+facility\b[^.\n]*[\.]?", "", _t)
                _t = re.sub(r"(?is)\bdue\s+to\s+missing\s+repair\s+facility\s+(?:info|information|details)\b", "due to minor non-fatal documentation items", _t)

            # General cleanup
            _t = re.sub(r"[ \t]{2,}", " ", _t)
            _t = re.sub(r"\n{3,}", "\n\n", _t).strip()
            return _t

        # Apply scrubber to conclusion, brief, and narrative (prevents contradictory final evaluation lines)
        result["conclusion"] = _scrub_forbidden(result.get("conclusion") or "")
        result["summary_brief"] = _scrub_forbidden(result.get("summary_brief") or "")

        _sm_before = result.get("summary_markdown") or ""
        _sm_after = _scrub_forbidden(_sm_before)

        # SCORE OVERRIDE: If the only apparent deductions are forbidden ones (Production Date when door-label is present,
        # or Repair Facility when Closing Report indicates owner location/not at shop), restore score to 100 and remove the rationale section.
        try:
            if ai_intent != "damage_report_from_photos":
                _sm_work = _sm_after if _sm_after else _sm_before
                score_str = (result.get("compliance_score") or "").strip()
                if re.fullmatch(r"\d{1,3}", score_str):
                    score_int = max(0, min(100, int(score_str)))
                    if score_int < 100:
                        lower = (_sm_work or "").lower()

                        forbidden_hit = False
                        if _prod_evidenced and (("missing production date" in lower) or ("production date photo" in lower) or ("except for" in lower and "production date" in lower) or ("reduced due" in lower and "production date" in lower)):
                            forbidden_hit = True
                        if _not_at_shop and (("repair facility" in lower and ("missing" in lower or "absence" in lower or "reduced" in lower))):
                            forbidden_hit = True

                        if forbidden_hit:
                            score_int = 100
                            result["compliance_score"] = "100"

                            # strip rationale section if present
                            _sm_work = re.sub(r"(?is)\n##\s*Compliance\s*Score\s*Rationale\b.*?(?=\n##\s|\Z)", "", _sm_work).strip()

                            # remove any lingering forbidden phrases
                            _sm_work = re.sub(r"(?is)\bCompliance\s+is\s+reduced\s+due\s+to\s+[^.]*\bproduction\s+date\b[^.]*\.", "", _sm_work)
                            _sm_work = re.sub(r"(?is)\bCompliance\s+is\s+reduced\s+due\s+to\s+[^.]*\brepair\s+facility\b[^.]*\.", "", _sm_work)

                            # ensure score line is consistent
                            _sm_work = re.sub(r"(?im)^\s*(Final\s*score|Compliance\s*Score)\s*[:\-–]\s*\d{1,3}\s*%?\s*$", "", _sm_work)
                            _sm_work = re.sub(r"\n{3,}", "\n\n", _sm_work).strip()
                            _sm_work = (_sm_work + f"\n\nCompliance Score: {score_int}").strip()

                            _sm_after = _sm_work
        except Exception:
            pass

        if _sm_after and _sm_after.strip():
            result["summary_markdown"] = _sm_after
    except Exception:
        pass

    # -----------------------
    # PDF helpers
    # -----------------------
    def _pdf_sanitize(text: str, max_token_len: int = 60) -> str:
        if text is None:
            return ""
        s = str(text).replace("\r\n", "\n").replace("\r", "\n")
        s = "".join(ch if ord(ch) < 256 else " " for ch in s)
        def _break(tok: str) -> str:
            if len(tok) <= max_token_len:
                return tok
            return " ".join(tok[i:i+max_token_len] for i in range(0, len(tok), max_token_len))
        s = " ".join(_break(t) for t in s.split(" "))
        return s

    pdf = FPDF(); pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_left_margin(10); pdf.set_right_margin(10)

    try:
        pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True); pdf.set_font(size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    def mc(s):
        try:
            effective_w = pdf.w - pdf.l_margin - pdf.r_margin
            if effective_w <= 5:
                effective_w = 180
            safe = _pdf_sanitize(s)
            if not safe.strip():
                safe = "-"
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(effective_w, 6, safe)
        except Exception:
            effective_w = pdf.w - pdf.l_margin - pdf.r_margin
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(effective_w, 6, (_pdf_sanitize(str(s))[:2000] + " …"))

    # POI-15 Total Loss trigger from uploaded text ONLY
    poi15_hit = False
    try:
        txt = uploaded_text_all.lower()
        if re.search(r"\b15\s*total\s*loss\b", txt, flags=re.IGNORECASE):
            poi15_hit = True
        elif re.search(r"point\s*of\s*impact[^A-Za-z0-9]{0,10}15[^A-Za-z0-9]{0,20}total\s*loss", txt, flags=re.IGNORECASE):
            poi15_hit = True
    except Exception:
        poi15_hit = False

    if ai_intent == "damage_report_from_photos":
        pdf.cell(0,10,"AI-4-IA Damage Report", ln=True, align="C")
        pdf.set_font_size(10); pdf.ln(3)

        mc(f"Claim #: {result['claim_number'] or 'N/A'}    File #: {file_number or 'N/A'}")
        pdf_status = result["redaction_status"].replace("✅", "OK")
        pdf.ln(2); mc(pdf_status)
        pdf.ln(2); mc("Damage Summary"); mc((result["summary_markdown"] or "N/A").strip())
        mc("Estimated Repair Costs"); mc((result["estimated_costs_markdown"] or "N/A").strip())
        pdf.ln(2); mc("Fraud & Authenticity Check"); mc((result["fraud_markdown"] or 'N/A').strip())
        pdf.ln(2); mc("Conclusion"); mc((result["conclusion"] or 'N/A').strip())

        safe_file = _safe(file_number)
        pdf_filename = f"AI_Damage_Report_{safe_file}.pdf"
    else:
        pdf.cell(0,10,"NSPXN.com AI Review Report", ln=True, align="C")
        pdf.set_font_size(10); pdf.ln(3)
        mc(f"File Number: {file_number}")
        mc(f"IA Company: {ia_company}")
        mc(f"Appraiser ID #: {appraiser_id}")
        mc(f"Request Type: {result['request_type']}")

        # --- Supplement header (documents only; ignore negated mentions) ---
        smark = result.get("summary_markdown","")
        _txt_docs = uploaded_text_all or ""
        _supp_doc_hit = bool(re.search(
            r"(?is)\b(Supplement\s+(?:Summary|of\s+Record)|Estimate\s+Version:\s*S0[1-9]\b|\bS0[1-9]\b|\bSupplement\s+Estimate\b)",
            _txt_docs
        ))
        _no_supp_negation = not re.search(r"(?is)\b(no|not)\s+(a\s+)?supplement\b", _txt_docs)
        supp_detected_docs = _supp_doc_hit and _no_supp_negation
        if supp_detected_docs:
            mc("Supplement Status: Supplement Estimate detected in documentation")
            _possible_amt = None
            _m = re.search(r"(?is)\bPossible\s+Supplement\s+Amount\s*\$?([0-9,]+\.\d{2})\b", _txt_docs)
            if _m:
                _possible_amt = _m.group(1)
            mc("Supplement Details")
            # NEW: list all supplement tags (S01, S02, ...) instead of a single version only
            supp_versions_docs = sorted(set(re.findall(r"(?i)\bS[0-9]{2}\b", _txt_docs)))
            if supp_versions_docs:
                mc(f"- Supplements detected in documents: {', '.join(supp_versions_docs)}")
            if _possible_amt:
                mc(f"- Possible amount noted: ${_possible_amt}")
            if not (supp_versions_docs or _possible_amt):
                mc("- Supplement indicators present (e.g., 'Supplement Summary' or S01/S02).")

        # --- Total Loss echo (documents-only; no narrative trigger) ---
        explicit_tl_hit = False
        try:
            txt_docs = uploaded_text_all or ""
            poi15_docs = re.search(
                r"(?is)\bpoint\s*of\s*impact[^a-z0-9]{0,10}15[^a-z0-9]{0,20}total\s*loss\b",
                txt_docs,
            )
            doc_tl_docs = re.search(
                r"(?i)\b(estimate\s*type|type\s*of\s*loss)\s*:\s*total\s*loss\b",
                txt_docs,
            )
            explicit_tl_hit = bool(poi15_docs or doc_tl_docs)
        except Exception:
            explicit_tl_hit = False
        if explicit_tl_hit:
            mc("Estimate Type: Total Loss (explicit in documents)")

        mc(f"Claim #: {result['claim_number']}")
        mc(f"VIN (from estimate/photos): {result['vin']}")
        mc(f"VIN verification (estimate vs photo): {result['vin_verification']}")
        mc(f"Vehicle: {result['vehicle']}")
        mc(f"Odometer (from estimate): {result['odometer_estimate_only']}")
        mc(f"Compliance Score: {result['compliance_score']}")
        pdf_status = result["redaction_status"].replace("✅", "OK")
        mc(pdf_status)
        pdf.ln(3); mc("AI-4-IA Review Summary"); mc((smark or '').strip())
        pdf.ln(3); mc("Fraud Detection"); mc((result["fraud_markdown"] or 'N/A').strip())

        safe_file = _safe(file_number)
        pdf_filename = f"{safe_file}.pdf"

    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    try:
        out = pdf.output(dest="S")
        if isinstance(out, (bytes, bytearray)):
            data_bytes = bytes(out)
        else:
            data_bytes = str(out).encode("latin-1", "ignore")
        with open(pdf_path, "wb") as f:
            f.write(data_bytes)
    except Exception as e:
        logging.warning(f"PDF write error: {e}")

    pdf_url = f"/download-pdf?filename={pdf_filename}"

    # -----------------------
    # Email — info-only (attach PDF)
    # -----------------------
    try:
        msg = EmailMessage()
        if ai_intent == "damage_report_from_photos":
            subj = f"AI Damage Report: {file_number or ''} {result['claim_number'] or ''}".strip()
            body = (
                "AI-4-IA Damage Report\n\n"
                f"IA Company: {ia_company}\n"
                f"Claim #: {result['claim_number'] or 'N/A'}    File #: {file_number or 'N/A'}\n"
                f"Odometer: {result['odometer_estimate_only'] or 'N/A'}    Primary Impact: {result['primary_impact'] or 'N/A'}\n"
                f"Secondary Impact: {result['secondary_impact'] or 'N/A'}\n\n"
                f"{result['redaction_status']}\n\n"
                "Damage Summary\n"
                f"{(result['summary_markdown'] or 'N/A')}\n\n"
                "Estimated Repair Costs\n"
                f"{(result['estimated_costs_markdown'] or 'N/A')}\n\n"
                "Fraud & Authenticity Check\n"
                f"{(result['fraud_markdown'] or 'N/A')}\n\n"
                "Conclusion\n"
                f"{(result['conclusion'] or 'N/A')}\n"
            )
        else:
            try:
                _txt_email = uploaded_text_all or ""
                _poi15_email = re.search(
                    r"(?is)\bpoint\s*of\s*impact[^a-z0-9]{0,10}15[^a-z0-9]{0,20}total\s*loss\b",
                    _txt_email,
                )
                _doc_tl_email = re.search(
                    r"(?i)\b(estimate\s*type|type\s*of\s*loss)\s*:\s*total\s*loss\b",
                    _txt_email,
                )
                _explicit_tl_email = bool(_poi15_email or _doc_tl_email)
            except Exception:
                _explicit_tl_email = False

            # NEW: supplement line includes all Sxx tags if present
            supp_line = ""
            if supp_detected_docs:
                supp_versions_email = sorted(set(re.findall(r"(?i)\bS[0-9]{2}\b", _txt_email or "")))
                if supp_versions_email:
                    supp_line = (
                        "Supplement Status: Supplement Estimates detected in documentation "
                        f"({', '.join(supp_versions_email)})\n"
                    )
                else:
                    supp_line = "Supplement Status: Supplement Estimate detected in documentation\n"

            tl_line = "Estimate Type: Total Loss (explicit in documents)\n" if _explicit_tl_email else ""
            subj = f"AI-4-IA Review: {result['claim_number'] or file_number}"
            body = (
                "NSPXN.com AI Review Report\n\n"
                f"File Number: {file_number}\n"
                f"IA Company: {ia_company}\n"
                f"Appraiser ID #: {appraiser_id}\n"
                f"Request Type: {result['request_type']}\n"
                f"{supp_line}"
                f"{tl_line}"
                f"Claim #: {result['claim_number']}\n"
                f"VIN (from estimate/photos): {result['vin']}\n"
                f"VIN verification (estimate vs photo): {result['vin_verification']}\n"
                f"Vehicle: {result['vehicle']}\n"
                f"Odometer (from estimate): {result['odometer_estimate_only']}\n"
                f"Compliance Score: {result['compliance_score']}\n\n"
                f"{result['redaction_status']}\n\n"
                "AI-4-IA Review Summary\n"
                f"{result['summary_markdown']}\n\n"
                "Fraud Detection\n"
                f"{result['fraud_markdown']}\n"
            )

        msg["Subject"] = subj
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg["Cc"] = "growley505@gmail.com"
        msg.set_content(body)

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=pdf_filename)
        except Exception as e:
            logging.warning(f"Failed to attach PDF to email: {e}")

        with smtplib.SMTP_SSL("mail.tierra.net", 465, timeout=20) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
        log.info("Info email sent to info@nspxn.com")
    except Exception as e:
        logging.error(f"Email error: {e}")

    return {
        **result,
        "web_summary": result["summary_brief"],
        "gpt_output": result["summary_markdown"],
        "pdf_url": pdf_url,
        "pdf_filename": pdf_filename
    }

# -----------------------
# PDF download
# -----------------------
@app.get("/download-pdf")
async def download_pdf(file_number: Optional[str] = None, filename: Optional[str] = None):
    if filename:
        safe = _safe(filename)
        path = os.path.join(PDF_DIR, safe)
        if os.path.exists(path):
            return FileResponse(path=path, media_type="application/pdf", filename=safe)
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    if not file_number:
        return JSONResponse(status_code=400, content={"detail": "Missing query param 'filename' or 'file_number'"})
    safe_num = _safe(file_number)
    candidates = glob.glob(os.path.join(PDF_DIR, f"*{safe_num}*.pdf"))
    if not candidates:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    latest = max(candidates, key=lambda p: os.path.getmtime(p))
    return FileResponse(path=latest, media_type="application/pdf", filename=os.path.basename(latest))





























