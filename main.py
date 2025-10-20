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
    return re.sub(r"[^\w.\-]+", "-", (s or "").strip()).strip("-_.")

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

    # Comprehensive — with Detailed Appraisal Report + simplified cross-check
    "comprehensive": (
        "## Inputs Used\n"
        "- List the estimate pages/lines and photo numbers you used, plus any rules text (if provided).\n\n"

        "## Executive Summary\n"
        "- 3–6 bullets capturing the big picture: estimate integrity, rule alignment (only if rules text was supplied), "
        "and photo consistency.\n\n"

        "## Detailed Appraisal Report\n"
        "- Write this section as a **formal, paragraph-style appraisal report** summarizing the entire claim. "
        "Include: scope of impact, damage by zone/panel, repair vs. replace rationale, parts type (OEM/LKQ/Aftermarket), "
        "labor operations, refinish/overlap considerations, rate validation, paint materials handling, sublet usage, "
        "tax/markup accuracy, and overall estimate integrity. Cite photos and estimate lines (e.g., 'Photo 3', 'p2/L14'). "
        "Close with compliance to any provided client rules and a clear final recommendation (Repairable vs. Total Loss). "
        "Minimum 10–14 sentences (one continuous narrative, not bullets).\n\n"

        "## Photo-by-Photo Damage Ledger\n"
        "| Photo # | View/Angle | Panels/Parts Visible | Condition (dent/crease/scrape/misalignment) | Identifiers (VIN/odo/plate/reg) | Legibility |\n"
        "|---:|---|---|---|---|---|\n"
        "- One row per photo used in the analysis (≥6 rows if ≥6 photos exist). If an identifier is present but unreadable, mark **Present — not clearly legible**.\n\n"

        "## Brief Damage Descriptions\n"
        "- 6–12 bullets. Each bullet: **part/panel** + **condition** + **suggested op** (repair/replace/refinish/blend) + **Photo #**.\n\n"

        "## Estimate Line Extract (top relevant lines)\n"
        "| Est. p#/L# | Part/Op | Labor Hrs | Rate | Part Type | Price | Notes |\n"
        "|---|---|---:|---:|---|---:|---|\n"
        "- Include 8+ lines if available, focusing on items tied to observed damages. Use 'Notes' for overlap/blend or rationale.\n\n"

        "## Estimate Compliance Cross-Check (brief)\n"
        "Status: **Compliant / Non-compliant / Not Evidenced**. Cite only the most important evidence (p#/L# or Photo #). Keep it short.\n"
        "| Topic | Evidence (p#/L# or value; Photo # if relevant) | Status | Required Fix |\n"
        "|---|---|:--:|---|\n"
        "| Labor Rates |  |  |  |\n"
        "| Refinish/Overlap |  |  |  |\n"
        "| Paint Materials |  |  |  |\n"
        "| OEM Procedures |  |  |  |\n"
        "| Sublet |  |  |  |\n"
        "| Tax/Markup |  |  |  |\n\n"

        "## Client Guidelines Comparison (if rules text was supplied)\n"
        "- 3–8 bullets. Quote the relevant rule fragment and note **Aligned / Not Aligned / Not Evidenced**, with evidence refs (p#/L#, Photo #). "
        "If no client_rules were provided, omit this section.\n\n"

        "## Risks / Missing Evidence\n"
        "- Short bullets with severity (High/Med/Low) and a one-line remediation.\n\n"

        "## Compliance Score Rationale\n"
        "- REQUIRED if score < 100: start at **100** and list each deficiency with evidence refs (p#/L# and/or Photo #), "
        "severity (Minor/Moderate/Major), and numeric deduction. Show the arithmetic to the final score.\n\n"

        "## Final Evaluation\n"
        "- Compliance Score: NN% with a single-sentence justification. "
        "If no fraud indicators are identified, state **'No material inconsistencies found.'** Do not use 'N/A'."
    ),

    # Photos-only — unchanged template
    "damage_report_from_photos": (
        "# AI-4-IA Damage Report\n"
        "Create a concise, professional damage report **based only on the provided photos (and any optional text)**.\n\n"

        "## Inputs Used\n"
        "- List exact Photo #s and any text used.\n\n"

        "## Quick Stats\n"
        "- Claim # (if visible): <value or N/A>\n"
        "- File # (echo from request): <value or N/A>\n"
        "- Odometer (if visible): <value or N/A>\n"
        "- Primary Impact: <area(s)>\n"
        "- Secondary Impact: <area(s) or 'None observed'>\n\n"

        "## Photo-by-Photo Damage Ledger\n"
        "| Photo # | View/Angle | Panels/Parts Visible | Condition (dent/crease/scrape/misalignment) | Identifiers (VIN/odo/plate/reg) | Legibility |\n"
        "|---:|---|---|---|---|---|\n"
        "- One row per photo used in the analysis (≥6 rows if ≥6 photos exist). If an identifier is present but unreadable, mark **Present — not clearly legible**.\n\n"

        "## Damage Summary\n"
        "- 6–12 bullets with **panel/part + condition + suggested op**, citing **Photo #**.\n\n"

        "## Detailed Appraisal Report\n"
        "- Provide a **detailed appraisal narrative** based on the photos: impact zones, repair/replace reasoning, "
        "likely parts source (OEM/LKQ/Aftermarket) when inferable, refinish/overlap notes, and cost implications. "
        "Reference specific Photo #s. Minimum 8–12 sentences (one continuous narrative, not bullets).\n\n"

        "## Estimated Repair Costs\n"
        "- Provide a reasonable high-level breakdown (Body Labor, Paint Labor, Paint Materials, Parts, Sublet, Tax) and brief rationale.\n\n"

        "## Fraud & Authenticity Check\n"
        "- Any inconsistencies between photos/metadata/identifiers; if none, say so.\n\n"

        "## Compliance Score Rationale\n"
        "- REQUIRED if score < 100: explain deductions based on evidence completeness, clarity/legibility, and "
        "internal consistency among photos. Show the arithmetic from 100 to the final score.\n\n"

        "## Conclusion\n"
        "- 1–2 sentences summarizing repairability and scope. "
        "If no fraud indicators are identified, state **'No material inconsistencies found.'** Do not use 'N/A'.\n"
    ),
}

ALLOWED_INTENTS = {"guidelines_only","comprehensive","damage_report_from_photos"}

SYSTEM_BASE = (
    "You are an auto-claims appraisal assistant. Return ONLY valid JSON (no code fences). "
    "Populate exactly these keys (always include all, use 'N/A' when not applicable): "
    "['file_number','request_type','claim_number','vin','vin_verification','vehicle',"
    "'odometer_estimate_only','compliance_score','summary_brief','summary_markdown',"
    "'fraud_markdown','primary_impact','secondary_impact','estimated_costs_markdown','conclusion']. "
    "Use evidence only from the provided inputs. Cite estimate page/line as 'p#/L#' and photos as 'Photo #'. "
    "Avoid guessing; if uncertain, say 'N/A' and why. summary_brief must be <= 280 chars (plain text)."
)

# (Retained) No hallucinated rules + numeric score + fraud never N/A (instruction-level)
SYSTEM_BASE += (
    " Do not state or imply any client rule unless it appears verbatim in the provided client_rules text. "
    "If client_rules is blank, write the entire report without referencing client rules. "
    "If a value cannot be confirmed from the visible evidence, set it to 'N/A' and briefly state why. "
    " Compliance Score must be a numeric percentage 0–100 (never 'N/A'). "
    "If no client rules are supplied, base the score on estimate↔photo internal consistency, "
    "evidence completeness, and clarity/legibility. Provide a one-sentence rationale. "
    "If compliance_score < 100, include a dedicated section titled '## Compliance Score Rationale' "
    "which itemizes every deficiency with exact evidence references (estimate p#/L# and/or Photo #), "
    "assigns an explicit deduction per item, and shows the arithmetic to the final score. "
    "Use a consistent scheme (e.g., Minor −5, Moderate −10, Major −20) and never go below 0. "
    "The 'fraud_markdown' section must never be 'N/A'. If nothing material is found, write "
    "'No material inconsistencies found.' and briefly note what was checked (VIN match, date/metadata, "
    "obvious photo tampering, duplicated images)."
)

# (Simplified) Encourage narrative; tables optional; rationale only when <100
SYSTEM_BASE += (
    " Focus on a cohesive, professional appraisal. Prefer narrative over rigid tables. "
    "Include a section named '## Detailed Appraisal Report'. "
    "Include '## Compliance Score Rationale' only when compliance_score < 100, and show deductions from 100 with brief evidence refs (p#/L# or Photo #). "
    "If you include tables, keep them concise and only when they help clarity. "
    "Avoid placeholder rows/columns; do not invent data. "
    "When client_rules text is provided, also include a section titled '## Client Guidelines Comparison' with 3–8 concise bullets quoting the relevant rule fragment and citing evidence (p#/L#, Photo #); weave any material rule alignment/misalignment into the Detailed Appraisal Report narrative."
)

# >>> PATCH A addition (unchanged): require a long narrative section
SYSTEM_BASE += (
    " Your 'summary_markdown' MUST include a top-level section named '## Detailed Appraisal Report' "
    "containing a cohesive narrative of at least 10–14 sentences (not bullets). "
    "It must synthesize: impact zones, per-panel damages, repair vs. replace rationale, parts type (OEM/LKQ/Aftermarket), "
    "labor ops, refinish/overlap, rate/materials/sublet/tax handling, and estimate integrity. "
    "It must cite concrete evidence inline (e.g., p2/L14, Photo 3)."
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
            pages = convert_from_bytes(raw, dpi=260)  # slightly sharper thumbnails
            files_seen.append(f"{fname} (pdf, {len(pages)} page(s))")
            for im in pages[:max_images - used]:
                b = io.BytesIO()
                im.save(b, format="JPEG", quality=85, optimize=True)
                parts.append(_image_part_from_bytes(b.getvalue()))
                used += 1
        except Exception as e:
            logging.warning(f"pdf2image failed for {fname}: {e}")
            files_seen.append(f"{fname} (pdf, could not be converted)")
    elif low.endswith(SUPPORTED_IMAGE_EXTS) and used < max_images:
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            im.thumbnail((1800,1800))
            b = io.BytesIO(); im.save(b, format="JPEG", quality=85, optimize=True)
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
    """
    Try to find a rules file in this priority order:
      1) Exact .docx match
      2) Case-insensitive exact .docx
      3) Case-insensitive 'startswith' .docx
      4) Any .docx containing the tokens (in order)
    Returns absolute path or None.
    """
    target = base_name.strip()
    if not target.lower().endswith(".docx"):
        target += ".docx"

    # 1) exact
    p = os.path.join(rules_dir, target)
    if os.path.exists(p):
        return p

    pattern = os.path.join(rules_dir, "*.docx")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    low_target = target.lower()

    # 2) case-insensitive exact
    for c in candidates:
        if os.path.basename(c).lower() == low_target:
            return c

    # 3) case-insensitive startswith
    for c in candidates:
        if os.path.basename(c).lower().startswith(low_target[:-5]):  # strip ".docx"
            return c

    # 4) contains tokens in order
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
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("comprehensive")
):
    parts: List[Dict[str, Any]] = []
    files_seen: List[str] = []
    MAX_IMAGES = 24
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

    KEYS = [
        "file_number","request_type","claim_number","vin","vin_verification","vehicle",
        "odometer_estimate_only","compliance_score","summary_brief","summary_markdown",
        "fraud_markdown","primary_impact","secondary_impact","estimated_costs_markdown","conclusion"
    ]

    SYSTEM = SYSTEM_BASE

    prompt_text = (
        f"REQUEST TYPE SELECTED (exact): '{req_label}'. Use this exact string in 'request_type'.\n\n"
        "FILES SEEN (echo verbatim in '## Inputs Used'):\n- "
        + ("\n- ".join(files_seen) if files_seen else "none") + "\n\n"
        "CLIENT RULES (if provided; else blank):\n" + (client_rules[:2000] if client_rules else "") + "\n\n"
        "ANALYSIS LAYOUT (guidance, not strict):\n" + DETAIL_TEMPLATES.get(ai_intent, DETAIL_TEMPLATES["comprehensive"])
    )

    # Odometer/registration legibility nudge for comprehensive
    if ai_intent == "comprehensive":
        prompt_text += (
            "\n\nUploader note: Odometer and Registration photos were provided. "
            "If you cannot clearly read them, report 'Present — not clearly legible' rather than 'Missing'."
        )
    # If client_rules provided, require a dedicated Guidelines comparison section and narrative tie-in
    if client_rules.strip():
        prompt_text += (
            "\n\nWhen client_rules text is provided, you MUST include a section titled '## Client Guidelines Comparison' "
            "with 3–8 concise bullets. For each, quote the relevant rule fragment and mark Aligned / Not Aligned / Not Evidenced, "
            "citing evidence (p#/L#, Photo #). Also weave any material rule alignment/misalignment into the '## Detailed Appraisal Report' narrative."
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

    redaction_status = "Redacted PII: Successful ✅" if redaction_success else "Redacted PII: Not Applied"

    MAX_TOKENS_BY_INTENT = {
        "comprehensive": 1500,
        "guidelines_only": 1000,
        "damage_report_from_photos": 900
    }
    max_tokens = MAX_TOKENS_BY_INTENT.get(ai_intent, 1000)

    # Call GPT and parse JSON
    try:
        rsp = client.chat_completions.create(  # type: ignore[attr-defined]
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": safe_user_parts}],
            max_tokens=max_tokens,
            temperature=0
        )
    except AttributeError:
        # compatible with newer SDKs
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role":"system","content": SYSTEM},
                      {"role":"user","content": safe_user_parts}],
            max_tokens=max_tokens,
            temperature=0
        )
    try:
        raw = (rsp.choices[0].message.content or "").strip()
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
    except Exception as e:
        log.error(f"LLM failure or JSON parse error: {e}")
        return JSONResponse(status_code=500, content={"error":"Model output could not be parsed as JSON."})

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

    # Non-empty Fraud fallback (prevents 'N/A' in output/PDF)
    if not result["fraud_markdown"] or result["fraud_markdown"].strip().upper() in {"", "N/A"}:
        result["fraud_markdown"] = (
            "No material inconsistencies found. Checks performed: VIN match across estimate and photos, "
            "odometer/registration presence and legibility, duplicate/edited images, timestamp continuity, and "
            "panel/impact consistency."
        )

    # -----------------------
    # PDF helpers (sanitizer for FPDF)
    # -----------------------
    def _pdf_sanitize(text: str, max_token_len: int = 60) -> str:
        """Make text safe for FPDF multi_cell: strip non-latin-1 and break long tokens."""
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

    # -----------------------
    # PDF — setup (margins/autobreak) + SAFE mc()
    # -----------------------
    pdf = FPDF(); pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.set_left_margin(10); pdf.set_right_margin(10)

    try:
        pdf.add_font("DejaVu","", "DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    # Bullet-proof mc(): fixed width + left-margin reset + sanitize + fail-soft
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

    # -----------------------
    # Compose PDF content
    # -----------------------
    if ai_intent == "damage_report_from_photos":
        pdf.cell(0,10,"AI-4-IA Damage Report", ln=True, align="C")
        pdf.set_font_size(10); pdf.ln(3)

        mc(f"Claim #: {result['claim_number'] or 'N/A'}    File #: {file_number or 'N/A'}")
        pdf_status = result["redaction_status"].replace("✅", "OK")
        pdf.ln(2); mc(pdf_status)
        pdf.ln(2); mc("Damage Summary"); mc((result["summary_markdown"] or "N/A").strip())
        pdf.ln(2); mc("Estimated Repair Costs"); mc((result["estimated_costs_markdown"] or "N/A").strip())
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
        mc(f"Claim #: {result['claim_number']}")
        mc(f"VIN (from estimate/photos): {result['vin']}")
        mc(f"VIN verification (estimate vs photo): {result['vin_verification']}")
        mc(f"Vehicle: {result['vehicle']}")
        mc(f"Odometer (from estimate): {result['odometer_estimate_only']}")
        mc(f"Compliance Score: {result['compliance_score']}")
        pdf_status = result["redaction_status"].replace("✅", "OK")
        mc(pdf_status)
        pdf.ln(3); mc("AI-4-IA Review Summary"); mc((result["summary_markdown"] or '').strip())
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

    # Always return an exact filename link (prevents mismatched PDFs)
    pdf_url = f"/download-pdf?filename={pdf_filename}"

    # -----------------------
    # Email — info-only (matches older working behavior)
    # -----------------------
    try:
        msg = EmailMessage()

        if ai_intent == "damage_report_from_photos":
            subj = f"AI Damage Report: {file_number or ''} {result['claim_number'] or ''}".strip()
            # >>> ONLY CHANGE: IA Company line added above Claim/File.
            body = f"""AI-4-IA Damage Report

IA Company: {ia_company}
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
"""
        else:
            subj = f"AI-4-IA Review: {result['claim_number'] or file_number}"
            body = f"""NSPXN.com AI Review Report

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
"""

        msg["Subject"] = subj
        msg["From"] = "info@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(body)

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
    # 1) Explicit filename wins (recommended path used by API response)
    if filename:
        safe = _safe(filename)
        path = os.path.join(PDF_DIR, safe)
        if os.path.exists(path):
            return FileResponse(path=path, media_type="application/pdf", filename=safe)
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # 2) Back-compat: pick the most recent PDF containing this file_number
    if not file_number:
        return JSONResponse(status_code=400, content={"detail": "Missing query param 'filename' or 'file_number'"})

    safe_num = _safe(file_number)
    candidates = glob.glob(os.path.join(PDF_DIR, f"*{safe_num}*.pdf"))
    if not candidates:
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    latest = max(candidates, key=lambda p: os.path.getmtime(p))
    return FileResponse(path=latest, media_type="application/pdf", filename=os.path.basename(latest))





