from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
import os
import re
import base64
import io
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")

app = FastAPI()

# Global request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        logger.debug(f"Received {request.method} {request.url.path} with headers={dict(request.headers)}")
        body = await request.body()
        logger.debug(f"Raw request body (hex): {body.hex()}")
        form = await request.form() if request.method in ["POST", "PUT"] else None
        if form:
            logger.debug(f"Raw form data: {dict(form)}")
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Middleware error: {str(e)} with body={body.decode('utf-8', errors='ignore') if 'body' in locals() else 'No body'}")
        return JSONResponse(status_code=400, content={"error": f"Invalid request: {str(e)}"})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com",
        "https://www.nspxn.com",
        "http://nspxn.com",
        "http://www.nspxn.com",
        "https://nspxn.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------- Client Rules Loader (auto-load + pasted override) -----------------------
import glob
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
CLIENT_RULES_DIRS = [
    _THIS_DIR / "client_rules",
    Path.cwd() / "client_rules",
]
_rules_cache: Dict[str, str] = {}

def _read_docx_path(fp: Path) -> str:
    doc = Document(str(fp))
    return "\n".join(p.text for p in doc.paragraphs)

def _read_txt_path(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def _discover_rules_files() -> List[Path]:
    files: List[Path] = []
    for d in CLIENT_RULES_DIRS:
        if d.is_dir():
            files.extend(map(Path, glob.glob(str(d / "*.docx"))))
            files.extend(map(Path, glob.glob(str(d / "*.txt"))))
    return files

def load_rules_dir(force_refresh: bool = False) -> Dict[str, str]:
    """Load and cache .docx/.txt rules; key = filename sans extension."""
    global _rules_cache
    if _rules_cache and not force_refresh:
        return _rules_cache
    result: Dict[str, str] = {}
    files = _discover_rules_files()
    if not files:
        logger.warning("No client rules found in: %s", ", ".join(str(p) for p in CLIENT_RULES_DIRS))
    for fp in files:
        name = fp.stem
        try:
            text = _read_docx_path(fp) if fp.suffix.lower() == ".docx" else _read_txt_path(fp)
            text = text.strip()
            if text:
                result[name] = text
                logger.debug("Loaded client rules file: %s", fp)
        except Exception as e:
            logger.exception("Failed reading rules %s: %s", fp, e)
    _rules_cache = result
    return _rules_cache

def resolve_rules(selected_client: Optional[str], pasted_text: Optional[str]) -> str:
    """
    Priority:
      1) pasted_text (non-empty)
      2) match selected_client against file names (case-insensitive, partial ok)
      3) if only one rules file exists, use it
      4) else ''
    """
    pasted = (pasted_text or "").strip()
    if pasted:
        return pasted

    rules_map = load_rules_dir()
    if selected_client:
        keys = list(rules_map.keys())
        exact = next((k for k in keys if k.lower() == selected_client.lower()), None)
        if exact:
            return rules_map[exact]
        lowsel = selected_client.lower()
        partials = [k for k in keys if lowsel in k.lower()]
        if len(partials) == 1:
            return rules_map[partials[0]]

    if len(rules_map) == 1:
        return next(iter(rules_map.values()))

    return ""
# ---------------------------------------------------------------------------------------------------

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = ImageOps.autocontrast(img)
    img = ImageOps.invert(img)
    return img

def extract_text_from_pdf(file) -> str:
    try:
        file.seek(0)
        images = convert_from_bytes(file.read(), dpi=200)
        text_output = ""
        for i, img in enumerate(images, 1):
            processed = preprocess_image(img)
            try:
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 3')
            except Exception as e:
                logger.warning(f"PSM 3 failed for page {i}: {str(e)}, retrying with PSM 6")
                ocr_text = pytesseract.image_to_string(processed, lang="eng", config='--psm 6')
            if len(ocr_text.strip()) < 50 or re.search(r"[\:/\d\s]{50,}", ocr_text):
                logger.warning(f"Page {i} OCR output skipped (garbled): {ocr_text[:100]}...")
                continue
            text_output += f"\n[Page {i}]\n{ocr_text}"
        if not text_output.strip():
            logger.error("No valid text extracted from PDF")
        return text_output
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return f"\n❌ OCR error: {str(e)}"

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    text = '\n'.join(p.text for p in doc.paragraphs if p.text.strip())
    logger.debug(f"Extracted DOCX text: {text[:500]}...")
    return text

def extract_text_from_images(image_files: List[UploadFile]) -> str:
    text_output = ""
    for i, img in enumerate(image_files, 1):
        try:
            img.file.seek(0)
            image = Image.open(io.BytesIO(img.file.read()))
            processed = preprocess_image(image)
            ocr_text = pytesseract.image_to_string(processed, lang="eng")
            text_output += f"\n[Image {i}]\n{ocr_text}"
            logger.debug(f"Extracted text from image {i}: {ocr_text[:500]}...")
        except Exception as e:
            logger.error(f"Image {i} OCR error: {str(e)}")
    return text_output

@app.post("/vision-review")
async def vision_review(
    request: Request,  # ✅ fix: we log headers below, so accept request
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...),
    # ✅ estimate (PDF/DOCX) + images
    estimate: UploadFile = File(...),
    image_files: List[UploadFile] = File(...),
    # ✅ NEW: client rules support (backward compatible with multiple field names)
    client_name: Optional[str] = Form(default=None),
    client_rules: Optional[str] = Form(default=None),
    rules_text: Optional[str] = Form(default=None),
    rules: Optional[str] = Form(default=None),
    guidelines: Optional[str] = Form(default=None),
):
    logger.debug(f"Starting vision_review with headers={dict(request.headers)}")
    # Validate form fields with detailed logging
    logger.debug(f"Received form data: file_number='{file_number}', ia_company='{ia_company}', appraiser_id='{appraiser_id}'")
    if not all([field.strip() for field in [file_number, ia_company, appraiser_id]]):
        missing = [name for name, val in [('file_number', file_number), ('ia_company', ia_company), ('appraiser_id', appraiser_id)] if not val.strip()]
        logger.error(f"Validation failed: Empty fields - {', '.join(missing)}")
        return JSONResponse(status_code=422, content={"error": f"Missing or empty required fields: {', '.join(missing)}"})
    
    # Validate estimate file type
    logger.debug(f"Estimate file: filename='{estimate.filename}', content_type='{estimate.content_type}'")
    if not estimate.filename.lower().endswith(('.pdf', '.docx')):
        logger.error(f"Validation failed: Invalid estimate file type - {estimate.filename}")
        return JSONResponse(status_code=422, content={"error": f"Estimate must be a PDF or DOCX file, got {estimate.filename}"})
    
    # Process estimate
    estimate_text = extract_text_from_pdf(estimate) if estimate.filename.lower().endswith('.pdf') else extract_text_from_docx(estimate)
    if "OCR error" in estimate_text:
        logger.error(f"OCR error on estimate {estimate.filename}: {estimate_text}")
        return JSONResponse(status_code=422, content={"error": "Failed to extract text from estimate due to OCR error"})
    
    # Process images
    image_text = extract_text_from_images(image_files)
    if not image_text.strip():
        logger.warning("No valid text extracted from images")

    # ✅ Resolve client guidelines:
    #  - pasted text from any known field names wins
    #  - else load by client_name from /client_rules
    pasted_candidates = [client_rules, rules_text, rules, guidelines]
    pasted_merged = next((x for x in pasted_candidates if (x or "").strip()), "")
    effective_rules = resolve_rules(client_name, pasted_merged)
    if not effective_rules:
        # Keep behavior graceful but explicit
        logger.warning("Client rules NOT resolved. client_name=%r; pasted_present=%r",
                       client_name, any((x or "").strip() for x in pasted_candidates))
        effective_rules = "No client guidelines available"

    logger.debug(f"Client rules (resolved) preview: {effective_rules[:500]}...")

    # Simple comparison (unchanged, but now uses effective_rules)
    comparison = "Comparison Report:\n"
    comparison += f"Estimate Text: {estimate_text[:500]}...\n"
    comparison += f"Image Text: {image_text[:500]}...\n"
    comparison += f"Client Guidelines: {effective_rules[:500]}...\n"
    matches = set(re.findall(r'\w+', estimate_text.lower())) & set(re.findall(r'\w+', image_text.lower())) & set(re.findall(r'\w+', effective_rules.lower()))
    if matches:
        comparison += f"Matching terms across estimate, images, and guidelines: {', '.join(sorted(matches))}\n"
    else:
        comparison += "No matching terms found.\n"

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=11)
    try:
        pdf.cell(200, 10, txt="NSPXN.com Comparison Report", ln=True, align='C')
        pdf.ln(5)
        pdf.multi_cell(0, 10, f"File Number: {file_number}")
        pdf.multi_cell(0, 10, f"IA Company: {ia_company}")
        pdf.multi_cell(0, 10, f"Appraiser ID #: {appraiser_id}")
        pdf.ln(5)
        pdf.multi_cell(0, 10, "Comparison Summary:", align='L')
        pdf.set_font("DejaVu", size=9)
        pdf.multi_cell(0, 10, comparison)

        pdf_path = f"{file_number}.pdf"
        pdf.output(pdf_path)
    except Exception as pdf_e:
        logger.error(f"PDF generation error: {str(pdf_e)}")
        return JSONResponse(status_code=500, content={"error": f"PDF generation failed: {str(pdf_e)}", "comparison": comparison})

    # Send email
    msg = EmailMessage()
    msg["Subject"] = f"Comparison Report: {file_number}"
    msg["From"] = "noreply@nspxn.com"
    msg["To"] = "info@nspxn.com"
    email_body = f"""NSPXN.com Comparison Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Comparison Summary:
{comparison}
"""
    msg.set_content(email_body.encode("utf-8", errors="ignore").decode("utf-8"))
    with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
        try:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
        except Exception as email_e:
            logger.error(f"Email sending error: {str(email_e)}")

    return {
        "comparison": comparison,
        "file_number": file_number,
        "ia_company": ia_company,
        "appraiser_id": appraiser_id
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.debug(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)














