import os
import re
import io
import base64
import logging
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse

from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from pdf2image import convert_from_bytes
import pytesseract

from fpdf import FPDF

try:
    from docx import Document as DocxDocument
    DOCX_OK = True
except Exception:
    DOCX_OK = False

# Optional OpenAI integration if key present
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    try:
        from openai import OpenAI
        oai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        USE_OPENAI = False

# -----------------------------
# App bootstrap
# -----------------------------

app = FastAPI(title="NSPXN AI Audit - Refined")
logger = logging.getLogger("nspxn")
logging.basicConfig(level=logging.INFO)

ALLOWED_CORS = os.environ.get("ALLOW_ORIGINS", "https://nspxn.com,https://www.nspxn.com").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_CORS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers: OCR and parsing
# -----------------------------

VIN_REGEX = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
CLAIM_REGEX = re.compile(r"(?:Claim\s*(?:#|No\.?|Number)[:\s]*)([A-Za-z0-9\-\./]+)", re.IGNORECASE)
YEAR_REGEX = re.compile(r"\b(19[6-9]\d|20[0-4]\d|2050)\b")
MAKE_MODEL_REGEX = re.compile(r"Vehicle[:\s]*(\d{4})?\s*([A-Za-z][A-Za-z\-\s]+)\s+([A-Za-z0-9][A-Za-z0-9\-\s]+)", re.IGNORECASE)
LABOR_RATE_REGEX = re.compile(r"(Body|Paint|Refinish|Mechanical|Frame|Structural)\s*(?:Labor)?\s*[:\-]?\s*\$?\s*(\d{1,3}(?:\.\d{2})?)", re.IGNORECASE)
TAX_REGEX = re.compile(r"(?:Sales\s*Tax|Tax\s*Rate)[:\s]*([\d]{1,2}(?:\.\d{1,2})?)\s*%|(?:Sales\s*Tax|Tax)\s*\$?\s*(\d{1,5}(?:\.\d{2})?)", re.IGNORECASE)

DAMAGE_KEYWORDS = [
    "bumper", "fender", "door", "hood", "grille", "headlamp", "taillamp", "quarter panel",
    "rocker", "windshield", "mirror", "roof", "trunk", "decklid", "liftgate", "applique"
]

REQUIRED_PHOTO_HINTS = {
    "vin": ["vin", "vehicle identification", "17-digit"],
    "odometer": ["odo", "miles", "odometer"],
    "plate": ["plate", "license"],
    "corner": ["corner", "front left", "front right", "rear left", "rear right"]
}

def ocr_image(img: Image.Image) -> str:
    # Preprocess to improve OCR
    gray = ImageOps.grayscale(img)
    sharp = ImageEnhance.Contrast(gray).enhance(1.5)
    sharp = sharp.filter(ImageFilter.MedianFilter(3))
    text = pytesseract.image_to_string(sharp)
    return text

def ocr_pdf_first_page(pdf_bytes: bytes, dpi: int = 200) -> Tuple[str, Image.Image]:
    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=dpi)
    if not images:
        return "", None
    img = images[0]
    text = ocr_image(img)
    return text, img

def find_first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    if m:
        # Return first non-None group if groups exist
        if m.lastindex:
            for i in range(1, m.lastindex + 1):
                if m.group(i):
                    return m.group(i).strip()
        return m.group(0).strip()
    return None

def extract_estimate_core(text: str) -> Dict[str, Any]:
    data = {
        "claim_number": None,
        "vin": None,
        "vehicle_year": None,
        "vehicle_make": None,
        "vehicle_model": None,
        "labor_rates": {},
        "tax_rate": None,
        "has_tax_line": False,
        "damage_lines_found": [],
    }

    # Claim #
    claim = find_first(CLAIM_REGEX, text)
    if not claim:
        # fallbacks
        m = re.search(r"\bClaim\b.*?[:#]\s*([A-Za-z0-9\-\./]+)", text, re.IGNORECASE|re.DOTALL)
        if m:
            claim = m.group(1).strip()
    data["claim_number"] = claim

    # VIN
    vin = find_first(VIN_REGEX, text)
    data["vin"] = vin

    # Year, Make, Model heuristics
    # Try common "Vehicle: 2018 Toyota Camry" like lines
    m = re.search(r"(?:Vehicle|Unit|Yr/Make/Model)[:\s-]*([0-9]{4})\s+([A-Za-z][A-Za-z\-\s]+?)\s+([A-Za-z0-9][A-Za-z0-9\-\s]+)", text, re.IGNORECASE)
    if m:
        data["vehicle_year"] = m.group(1).strip()
        data["vehicle_make"] = re.sub(r"\s+", " ", m.group(2)).strip()
        data["vehicle_model"] = re.sub(r"\s+", " ", m.group(3)).strip()
    else:
        # Try Yr/Make/Model in columns
        ym = YEAR_REGEX.search(text)
        if ym:
            data["vehicle_year"] = ym.group(0)
        # Rough make/model by proximity to year
        if data["vehicle_year"]:
            after = text[text.find(data["vehicle_year"]) : text.find(data["vehicle_year"]) + 120]
            parts = re.findall(r"[A-Za-z]{3,}", after)
            if parts:
                data["vehicle_make"] = parts[0]
                if len(parts) > 1:
                    data["vehicle_model"] = " ".join(parts[1:3])

    # Labor rates
    for m in LABOR_RATE_REGEX.finditer(text):
        k = m.group(1).lower()
        v = m.group(2)
        try:
            data["labor_rates"][k] = float(v)
        except Exception:
            pass

    # Tax
    for m in TAX_REGEX.finditer(text):
        pct, amt = m.groups()
        if pct:
            try:
                data["tax_rate"] = float(pct)
            except Exception:
                pass
        data["has_tax_line"] = True

    # Damage lines (simple keyword scan)
    damage_lines = []
    for kw in DAMAGE_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
            damage_lines.append(kw)
    data["damage_lines_found"] = sorted(set(damage_lines))

    return data

def find_vins_in_images(images: List[Image.Image]) -> List[str]:
    vins = []
    for img in images:
        try:
            t = ocr_image(img)
            for v in VIN_REGEX.findall(t):
                vv = v.strip().upper()
                if vv not in vins:
                    vins.append(vv)
        except Exception:
            continue
    return vins

def detect_required_photos(images: List[Image.Image]) -> Dict[str, bool]:
    found = {k: False for k in REQUIRED_PHOTO_HINTS.keys()}
    for img in images:
        try:
            t = ocr_image(img).lower()
        except Exception:
            t = ""
        for key, hints in REQUIRED_PHOTO_HINTS.items():
            if any(h in t for h in hints):
                found[key] = True
    # Heuristic: if >=4 images, assume corner set present even if no text
    if not found["corner"] and len(images) >= 4:
        found["corner"] = True
    return found

def read_guidelines(files: List[UploadFile]) -> str:
    texts = []
    for f in files:
        name = (f.filename or "").lower()
        b = f.file.read()
        if name.endswith(".docx") and DOCX_OK:
            try:
                doc = DocxDocument(io.BytesIO(b))
                chunks = []
                for p in doc.paragraphs:
                    chunks.append(p.text)
                texts.append("\n".join(chunks))
                continue
            except Exception:
                pass
        # PDFs and others -> OCR
        try:
            pages = convert_from_bytes(b, dpi=150)
            buf = []
            for p in pages[:4]:  # cap for speed
                buf.append(ocr_image(p))
            texts.append("\n".join(buf))
        except Exception:
            # Fallback plain decode
            try:
                texts.append(b.decode("utf-8", errors="ignore"))
            except Exception:
                pass
    return "\n\n".join(texts)

def check_against_guidelines(estimate_text: str, guidelines_text: str) -> Dict[str, Any]:
    checks = []

    # labor rate instruction
    if re.search(r"labor rates|approved labor rates|use.*labor rates", guidelines_text, re.IGNORECASE):
        has_rates = bool(LABOR_RATE_REGEX.search(estimate_text))
        checks.append({"rule": "Labor rates present on estimate per client guidance", "status": "PASS" if has_rates else "FAIL"})

    # tax guidance
    if re.search(r"tax|sales tax|tax rate", guidelines_text, re.IGNORECASE):
        has_tax = bool(TAX_REGEX.search(estimate_text))
        checks.append({"rule": "Tax / Sales tax listed on estimate", "status": "PASS" if has_tax else "FAIL"})

    # required photos
    if re.search(r"photo|photos|images|4 corners|odometer|vin|license", guidelines_text, re.IGNORECASE):
        checks.append({"rule": "Required photos to be provided (VIN, Odometer, 4-Corners, Plate)", "status": "CHECKED"})

    # parts usage (OEM/LKQ/AM)
    if re.search(r"\b(OEM|LKQ|aftermarket|AM|recycled|Recon)\b", guidelines_text, re.IGNORECASE):
        # Simple presence check; deeper validation would parse estimate line items
        present_any = bool(re.search(r"\b(OEM|LKQ|aftermarket|AM|recycled|Recon)\b", estimate_text, re.IGNORECASE))
        checks.append({"rule": "Parts usage labeled (OEM/AM/LKQ) as required", "status": "PASS" if present_any else "WARN"})

    # NADA / valuation mention
    if re.search(r"\bNADA\b|\bvaluation\b|\bclean retail\b", guidelines_text, re.IGNORECASE):
        checks.append({"rule": "Valuation (NADA/Clean Retail) requirement noted", "status": "CHECKED"})

    return {"guideline_checks": checks}

def summarize_damage_comparison(estimate_text: str, photos_text: str) -> str:
    # If OpenAI is available, ask it for a compact summary using both texts
    if USE_OPENAI:
        try:
            prompt = f"""You are an auto-damage audit assistant. Cross-check estimate damage lines with photo evidence.
Keep this under 120 words. Be concrete and neutral.
Estimate excerpt:
{estimate_text[:4000]}

OCR from photos:
{photos_text[:4000]}
"""
            resp = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content": prompt}],
                temperature=0.2,
                max_tokens=220
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            pass

    # Fallback heuristic summary
    found_in_est = sorted(set([kw for kw in DAMAGE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", estimate_text, re.IGNORECASE)]))
    found_in_ph = sorted(set([kw for kw in DAMAGE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", photos_text, re.IGNORECASE)]))
    overlap = [k for k in found_in_est if k in found_in_ph]
    missing = [k for k in found_in_est if k not in found_in_ph]
    extras = [k for k in found_in_ph if k not in found_in_est]
    parts = []
    if overlap:
        parts.append(f"Photo evidence supports estimate items for: {', '.join(overlap)}.")
    if missing:
        parts.append(f"Estimate mentions not clearly evidenced in photos: {', '.join(missing)}.")
    if extras:
        parts.append(f"Photos show potential unlisted damages: {', '.join(extras)}.")
    if not parts:
        parts.append("Could not confidently match damages between estimate and photos with heuristic OCR.")
    return " ".join(parts)

def build_pdf_report(payload: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI-4-IA Audit Summary", ln=1)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Claim #: {payload.get('claim_number') or 'Unknown'}", ln=1)
    pdf.cell(0, 8, f"VIN: {payload.get('vin') or 'Unknown'} (VIN photo match: {payload.get('vin_photo_match')})", ln=1)
    pdf.cell(0, 8, f"Vehicle: {payload.get('vehicle_year') or '?'} {payload.get('vehicle_make') or ''} {payload.get('vehicle_model') or ''}".strip(), ln=1)
    pdf.cell(0, 8, f"Compliance Score: {payload.get('compliance_score', 0)}%", ln=1)

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Findings", ln=1)

    def add_bullet(title, lines: List[str]):
        pdf.set_font("Arial", "B", 11)
        pdf.multi_cell(0, 6, f"- {title}")
        pdf.set_font("Arial", "", 11)
        for line in lines:
            pdf.multi_cell(0, 6, f"  • {line}")

    # Required photos
    rp = payload.get("required_photos", {})
    rp_lines = [f"{k.capitalize()}: {'Present' if v else 'Missing'}" for k, v in rp.items()]
    add_bullet("Required Photos", rp_lines or ["No photo analysis"])

    # Labor rates
    lr = payload.get("labor_rates", {})
    if lr:
        lr_lines = [f"{k.title()}: ${v:.2f}" for k, v in lr.items()]
    else:
        lr_lines = ["Not found on estimate"]
    add_bullet("Labor Rates", lr_lines)

    # Taxes
    tax = payload.get("tax_rate")
    tax_lines = [f"Tax rate present: {tax}%" if tax is not None else "Tax rate not found"]
    add_bullet("Taxes", tax_lines)

    # Damage comparison
    dmg = payload.get("damage_summary", "")
    add_bullet("Damage Comparison Summary", [dmg] if dmg else ["No summary available"])

    # Guideline checks
    gl = payload.get("guideline_checks", [])
    gl_lines = [f"{c['rule']}: {c['status']}" for c in gl]
    add_bullet("Client Guideline Review", gl_lines or ["No client guidelines uploaded"])

    out = pdf.output(dest="S").encode("latin-1")
    return out

def score_compliance(fields: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 100
    notes = []

    if not fields.get("claim_number"):
        score -= 10; notes.append("Missing Claim # (-10)")
    if not fields.get("vin"):
        score -= 10; notes.append("Missing VIN (-10)")
    if not fields.get("vehicle_year"):
        score -= 5; notes.append("Missing Vehicle Year (-5)")
    if not fields.get("vehicle_make"):
        score -= 5; notes.append("Missing Make (-5)")
    if not fields.get("vehicle_model"):
        score -= 5; notes.append("Missing Model (-5)")

    # Labor rates
    if not fields.get("labor_rates"):
        score -= 15; notes.append("Labor rates not listed (-15)")

    # Tax
    if fields.get("tax_rate") is None:
        score -= 10; notes.append("Tax rate not present (-10)")

    # VIN photo
    vin_photo_present = fields.get("vin_photo_present", False)
    if not vin_photo_present:
        score -= 15; notes.append("VIN photo not found (-15)")

    # VIN match
    if vin_photo_present and fields.get("vin_photo_match") == "MISMATCH":
        score -= 40; notes.append("VIN mismatch (-40)")

    # Required photos
    req = fields.get("required_photos", {})
    for k in ["corner", "vin", "odometer", "plate"]:
        if not req.get(k, False):
            score -= 5; notes.append(f"Missing required photo: {k} (-5)")

    score = max(0, min(100, score))
    return score, notes

# -----------------------------
# Routes
# -----------------------------

@app.get("/", response_class=PlainTextResponse)
def root():
    return "NSPXN AI Audit API is running."

@app.post("/analyze")
async def analyze(
    estimate: UploadFile = File(..., description="Estimate PDF (first page contains Claim/VIN/Vehicle)"),
    photos: List[UploadFile] = File([], description="Damage/VIN/odometer/plate photos"),
    guidelines: List[UploadFile] = File([], description="Client guidelines DOCX/PDF"),
):
    try:
        est_bytes = await estimate.read()
        est_text, est_img = ocr_pdf_first_page(est_bytes, dpi=200)
        core = extract_estimate_core(est_text)

        # Photos
        img_objs = []
        photos_text_parts = []
        for p in photos:
            b = await p.read()
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
                img_objs.append(img)
                photos_text_parts.append(ocr_image(img))
            except Exception:
                # If user uploaded a PDF as 'photo', OCR first page
                try:
                    pgs = convert_from_bytes(b, first_page=1, last_page=1, dpi=200)
                    if pgs:
                        img = pgs[0]
                        img_objs.append(img)
                        photos_text_parts.append(ocr_image(img))
                except Exception:
                    continue
        photos_text = "\n".join(photos_text_parts)

        # VIN from photos
        vin_list = find_vins_in_images(img_objs)
        vin_photo_present = len(vin_list) > 0
        core["vin_photo_present"] = vin_photo_present
        vin_match_status = "UNKNOWN"
        if vin_photo_present and core.get("vin"):
            if core["vin"].upper() in [v.upper() for v in vin_list]:
                vin_match_status = "MATCH"
            else:
                vin_match_status = "MISMATCH"
        core["vin_photo_match"] = vin_match_status

        # Required photos heuristic
        core["required_photos"] = detect_required_photos(img_objs)

        # Guidelines
        gl_text = ""
        if guidelines:
            gl_text = read_guidelines(guidelines)
        gl_checks = check_against_guidelines(est_text, gl_text).get("guideline_checks", [])
        core["guideline_checks"] = gl_checks

        # Damage comparison summary
        core["damage_summary"] = summarize_damage_comparison(est_text, photos_text)

        # Score
        compliance_score, deductions = score_compliance(core)
        core["compliance_score"] = compliance_score
        core["deductions"] = deductions

        # Normalize output field names to your existing UI
        out = {
            "claim_number": core.get("claim_number"),
            "vin": core.get("vin"),
            "vehicle_year": core.get("vehicle_year"),
            "vehicle_make": core.get("vehicle_make"),
            "vehicle_model": core.get("vehicle_model"),
            "labor_rates": core.get("labor_rates"),
            "tax_rate": core.get("tax_rate"),
            "required_photos": core.get("required_photos"),
            "vin_photo_match": core.get("vin_photo_match"),
            "compliance_score": core.get("compliance_score"),
            "damage_summary": core.get("damage_summary"),
            "guideline_checks": core.get("guideline_checks"),
            "deductions": core.get("deductions"),
        }

        # PDF report
        pdf_bytes = build_pdf_report({
            **out,
        })
        pdf_path = "/tmp/ai_audit_report.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Return JSON and base64 pdf for convenience
        return JSONResponse({
            "ok": True,
            "result": out,
            "report_pdf_b64": base64.b64encode(pdf_bytes).decode("ascii"),
            "message": "Analysis complete."
        })

    except Exception as e:
        logger.exception("Analyze failed")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/generate-report-pdf")
async def generate_report_pdf(
    payload: dict
):
    try:
        pdf_bytes = build_pdf_report(payload)
        path = "/tmp/ai_audit_report.pdf"
        with open(path, "wb") as f:
            f.write(pdf_bytes)
        return FileResponse(path, filename="AI_Audit_Report.pdf", media_type="application/pdf")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# Health for Render
@app.get("/healthz")
def healthz():
    return {"status": "ok"}






















