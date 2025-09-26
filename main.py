from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, base64, json, logging

import smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, Image

from openai import OpenAI

# =========================
# Config & setup
# =========================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable is NOT set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL_DEFAULT = os.getenv("OAI_MODEL", "gpt-4o-mini")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com", "https://www.nspxn.com",
        "http://nspxn.com",  "http://www.nspxn.com",
        "https://nspxn.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OCR helpers (FAST + reliable)
# =========================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.9)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr_pdf_text(pdf_bytes: bytes, limit_pages: Optional[int] = None, dpi: int = 200, psms=("--psm 6","--psm 3")) -> str:
    """OCR a few pages only (fast), try a couple PSMs for robustness."""
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)
        if limit_pages:
            pages = pages[:limit_pages]
        blocks = []
        for i, p in enumerate(pages, 1):
            proc = preprocess_image(p)
            txt = ""
            for psm in psms:
                try:
                    txt = pytesseract.image_to_string(proc, lang="eng", config=psm)
                except Exception:
                    continue
                if len(txt.strip()) >= 15:
                    break
            if txt.strip():
                blocks.append(f"\n[Page {i}]\n{txt}")
        return "".join(blocks)
    except Exception as e:
        logger.warning(f"OCR PDF error: {e}")
        return ""

def ocr_docx_text(file_like: io.BytesIO) -> str:
    try:
        doc = Document(file_like)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.warning(f"DOCX read error: {e}")
        return ""

# =========================
# VIN utils (normalize + checksum) — ALWAYS RUN
# =========================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
_translit = {**{str(i): i for i in range(10)},
             **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def normalize_vin(s: str) -> Optional[str]:
    s = (s or "").strip().upper().replace(" ", "").replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s

def vin_checksum_ok(v: str) -> bool:
    if len(v) != 17: return False
    try:
        total = sum(_translit[ch] * _weights[i] for i, ch in enumerate(v))
        check = total % 11
        return v[8] == ("X" if check == 10 else str(check))
    except Exception:
        return False

def best_vin_candidate(raw_text: str) -> Optional[str]:
    # Look near a VIN label first
    label_hits = re.findall(r"(?:VIN[:\s\-]*)([A-HJ-NPR-Z0-9]{10,20})", raw_text, re.IGNORECASE)
    cands = []
    for c in label_hits + re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", raw_text):
        v = normalize_vin(c)
        if v: cands.append(v)
    # Prefer checksum-valid
    for v in cands:
        if vin_checksum_ok(v): return v
    return cands[0] if cands else None

# =========================
# Vehicle description (robust)
# =========================
MAKE_MAP = {
    "NISS":"Nissan","NISSAN":"Nissan","CHEV":"Chevrolet","CHEVY":"Chevrolet",
    "TOY":"Toyota","TOYOTA":"Toyota","FORD":"Ford","HONDA":"Honda","HYUN":"Hyundai",
    "HYUNDAI":"Hyundai","KIA":"Kia","BMW":"BMW","MERCEDES":"Mercedes-Benz","MB":"Mercedes-Benz",
    "VW":"Volkswagen","VOLKS":"Volkswagen","SUBARU":"Subaru","MAZDA":"Mazda","DODGE":"Dodge"
}
STOP_TOKENS = {"GASOLINE","DIESEL","HYBRID","ELECTRIC","BLACK","WHITE","BLUE","RED","SILVER","GRAY","GREY",
               "4D","2D","SED","SDN","SUV","COUPE","HATCH","TRUCK","WAGON","AWD","FWD","RWD","2.5L","3.5L","L",
               "GDI","DIRECT","INJECTION","TURBO","PAINT","CLEAR","COAT","COLOR"}

def extract_vehicle(text: str) -> Optional[str]:
    lines = [re.sub(r"\s{2,}", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    # Prefer lines starting with a year and then make/model tokens
    cand = None
    for ln in lines:
        if re.search(r"^\s*(19|20)\d{2}\b", ln) and not re.search(r"\b(AM|PM)\b", ln):
            tokens = ln.split()
            year = tokens[0]
            if not year.isdigit(): continue
            tail = tokens[1:]
            kept = []
            for t in tail:
                raw = re.sub(r"[^\w\-]", "", t).upper()
                if raw in STOP_TOKENS: break
                if raw in ("A/M","OEM"): break
                kept.append(t)
                if len(kept) >= 4: break  # Year + up to 4 tokens: Make Model Trim
            if kept:
                # Normalize make token 1 if possible
                mk = MAKE_MAP.get(kept[0].upper(), kept[0].capitalize())
                desc = " ".join([year, mk] + kept[1:])
                cand = desc
                break
    if not cand:
        # Fallback: search for VEHICLE section next line
        for i, ln in enumerate(lines):
            if "VEHICLE" in ln.upper() and i+1 < len(lines):
                nxt = lines[i+1]
                if re.search(r"(19|20)\d{2}", nxt):
                    cand = nxt.strip()
                    break
    return cand

# =========================
# Basic fields (cheap)
# =========================
def extract_claim(text: str) -> Optional[str]:
    for pat in [r"Claim\s*[:#]\s*([A-Za-z0-9\-_\/]+)", r"Claim\s*(?:No\.?|Number|#)\s*[: ]\s*([A-Za-z0-9\-_\/]+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    return None

# =========================
# Minimal, safe GPT call (with fallback)
# =========================
def safe_chat_completion(messages, max_tokens=900, model=MODEL_DEFAULT):
    try:
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0
        )
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "rate" in msg:
            logger.warning("429 RateLimit → falling back to gpt-3.5-turbo")
            try:
                return client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=messages, max_tokens=max_tokens, temperature=0
                )
            except Exception as e2:
                logger.error(f"Fallback failed: {e2}")
                return None
        logger.error(f"OpenAI error: {e}")
        return None

# =========================
# Intent routing — ONLY what’s asked
# =========================
def parse_intent(ai_request: str) -> str:
    t = (ai_request or "").lower()
    if "comprehensive" in t or ("guideline" in t and "photo" in t):
        return "comprehensive"
    if "guideline" in t or "client" in t:
        return "guidelines_only"
    if "photo" in t:
        return "photos_only"
    if "vin" in t:
        return "vin_only"
    if "invoice" in t:
        return "invoices_only"
    return "freeform"

# =========================
# Post-filters to prevent hallucinations
# =========================
def strip_photo_sections(text: str) -> str:
    # Remove any "Client Photo Rules"/"Estimate↔Photos Comparison" sections if they slipped in
    text = re.sub(r"(?s)##\s*Client\s*Photo\s*Rules.*?(?=\n##|\Z)", "", text)
    text = re.sub(r"(?s)##\s*Estimate.?↔.?Photos\s*Comparison.*?(?=\n##|\Z)", "", text)
    # Kill single-line stray claims
    text = re.sub(r"(?im)^-.*photo.*$", "", text)
    return text.strip()

# =========================
# Routes
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...),
    ai_request: str = Form("")
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error": "Appraiser ID is required."})

    # --- Partition uploads
    pdfs: List[Tuple[str, bytes]] = []
    images: List[Tuple[str, bytes]] = []
    docs: List[str] = []

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            pdfs.append((name, raw))
        elif name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append((name, raw))
        elif name.endswith(".docx"):
            docs.append(ocr_docx_text(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            docs.append(raw.decode("utf-8", errors="ignore"))

    intent = parse_intent(ai_request)
    logger.info(f"Intent: {intent} | Request: {ai_request}")

    # --- OCR estimate text (fast)
    limit = 12 if intent == "comprehensive" else 5
    estimate_text = ""
    if pdfs:
        estimate_text = ocr_pdf_text(pdfs[0][1], limit_pages=limit)
        # If we failed to pull a VIN, do a small high-DPI retry on first 2 pages
        if not best_vin_candidate(estimate_text):
            estimate_text_retry = ocr_pdf_text(pdfs[0][1], limit_pages=2, dpi=240, psms=("--psm 6","--psm 11","--psm 3"))
            # Merge retry where helpful
            if len(estimate_text_retry) > len(estimate_text):
                estimate_text = estimate_text_retry + "\n" + estimate_text
    elif docs:
        estimate_text = "\n\n".join(docs)

    # --- Always extract VIN + Vehicle, and Claim
    vin_from_est = best_vin_candidate(estimate_text) or "N/A"
    vehicle_desc = extract_vehicle(estimate_text) or "N/A"
    claim_number = extract_claim(estimate_text) or "N/A"

    # --- Evidence flags to stop hallucinations
    photos_present = len(images) > 0
    has_clean_value = bool(re.search(r"(clean\s*retail|NADA|KBB|Black\s*Book|J\.?D\.?\s*Power|valuation)", estimate_text, re.IGNORECASE))
    has_advisor = bool(re.search(r"(advisor\s*report|ccc\s*one\s*advisor)", estimate_text, re.IGNORECASE))

    # --- Build GPT (ONLY if needed for the chosen intent)
    def vision_images():
        out = []
        for _, blob in images:
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        return out

    gpt_output = ""
    if intent == "guidelines_only":
        system = (
            "You are an auto-damage compliance auditor.\n"
            "TASK: Compare CLIENT GUIDELINES to the ESTIMATE only.\n"
            "CONSTRAINTS:\n"
            "- PhotosPresent={photos_present}. If false, do NOT mention photos at all.\n"
            "- CleanRetailProvided={crv}. AdvisorReportProvided={advisor}.\n"
            "  Only claim 'included' when the flag is true; otherwise mark 'missing' or 'not provided'.\n"
            "- Keep sections: Client Quick Summary, Fatal Errors (if any), Rule Compliance details, Summary.\n"
            "- Be concise and factual. No speculation."
        ).format(photos_present=str(photos_present), crv=str(has_clean_value), advisor=str(has_advisor))
        user_parts = [
            {"type":"text","text":"CLIENT GUIDELINES:\n" + (client_rules or "")[:12000]},
            {"type":"text","text":"\n\nESTIMATE (OCR):\n" + (estimate_text or "")[:12000]},
            {"type":"text","text":f"\n\nEVIDENCE FLAGS:\nPhotosPresent={photos_present}\nCleanRetailProvided={has_clean_value}\nAdvisorReportProvided={has_advisor}"},
            {"type":"text","text":f"\n\nAPPRAISER REQUEST: {ai_request}"},
        ]
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=900)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()
        if not photos_present:
            gpt_output = strip_photo_sections(gpt_output)

    elif intent == "comprehensive":
        system = (
            "You are an auto-damage auditor.\n"
            "TASK: Guidelines + Estimate + Photos (if any).\n"
            "CONSTRAINTS:\n"
            "- PhotosPresent={photos_present}. If false, explicitly say 'No photos were provided' and do NOT invent photo content.\n"
            "- CleanRetailProvided={crv}. AdvisorReportProvided={advisor}; only claim 'included' if true.\n"
            "Sections: Client Quick Summary Compliance → Fatal Errors → Client Photo Rules (omit if no photos) → "
            "Parts/Tax/Labor → Estimate↔Photos Comparison (omit if no photos) → Summary.\n"
            "Be concise and factual."
        ).format(photos_present=str(photos_present), crv=str(has_clean_value), advisor=str(has_advisor))
        user_parts = [
            {"type":"text","text":"CLIENT GUIDELINES:\n" + (client_rules or "")[:8000]},
            {"type":"text","text":"\n\nESTIMATE (OCR):\n" + (estimate_text or "")[:10000]},
            {"type":"text","text":f"\n\nEVIDENCE FLAGS:\nPhotosPresent={photos_present}\nCleanRetailProvided={has_clean_value}\nAdvisorReportProvided={has_advisor}"},
        ]
        if photos_present:
            user_parts.extend(vision_images())
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=1000)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()
        if not photos_present:
            gpt_output = strip_photo_sections(gpt_output)

    elif intent == "photos_only":
        if not photos_present:
            gpt_output = "No photos were provided with this request."
        else:
            system = (
                "Compare the ESTIMATE to the attached PHOTOS.\n"
                "Sections: Photo Coverage, Visible Damage vs Estimate, Discrepancies, Summary.\n"
                "No client guideline analysis; no VIN talk."
            )
            user_parts = [{"type":"text","text":"ESTIMATE (OCR):\n" + (estimate_text or "")[:8000]}]
            user_parts.extend(vision_images())
            rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=800)
            gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()

    elif intent == "vin_only":
        # No GPT call
        gpt_output = (
            "VIN Extraction (Estimate Only)\n"
            f"- VIN from estimate: {vin_from_est}\n"
            "- Note: VIN photo verification not requested."
        )

    elif intent == "invoices_only":
        # Minimal invoice OCR + compare
        invoices_text = ""
        for name, raw in pdfs:
            if "invoice" in name or "receipt" in name or "supplement" in name:
                invoices_text += ocr_pdf_text(raw, limit_pages=6)
        if not invoices_text and pdfs:
            invoices_text = ocr_pdf_text(pdfs[0][1], limit_pages=6)
        system = (
            "Audit whether supplement/estimate lines are substantiated by attached invoices.\n"
            "Write bullets: key invoice items + totals → state if each supports the supplement.\n"
            "Call out any missing docs."
        )
        user_parts = [
            {"type":"text","text":"ESTIMATE (OCR):\n" + (estimate_text or "")[:6000]},
            {"type":"text","text":"\n\nINVOICES (OCR):\n" + (invoices_text or "")[:6000]},
            {"type":"text","text":f"\n\nAPPRAISER REQUEST: {ai_request}"},
        ]
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=800)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()

    else:
        # Freeform — but still respect photos_present flags
        system = (
            "Fulfill the user's request exactly and concisely. "
            f"PhotosPresent={photos_present}. If false, do not mention photos."
        )
        user_parts = [{"type":"text","text":"ESTIMATE (OCR):\n" + (estimate_text or "")[:8000]}]
        if photos_present:
            user_parts.extend(vision_images())
        if client_rules:
            user_parts.append({"type":"text","text":"\n\nCLIENT GUIDELINES:\n" + client_rules[:8000]})
        user_parts.append({"type":"text","text":f"\n\nAPPRAISER REQUEST: {ai_request}"} )
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=900)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()
        if not photos_present:
            gpt_output = strip_photo_sections(gpt_output)

    # =========================
    # Light score (only for guideline-type requests)
    # =========================
    def light_guideline_score(txt: str, rules: str) -> int:
        score = 100
        if "labor" in rules.lower() and not re.search(r"(labor|rate).{0,80}\$", txt, re.IGNORECASE | re.DOTALL):
            score -= 10
        if "tax" in rules.lower() and "tax" not in txt.lower():
            score -= 10
        return max(0, min(100, score))
    comp_score = light_guideline_score(estimate_text, client_rules) if intent in ("guidelines_only","comprehensive") else 100

    # =========================
    # PDF (layout unchanged)
    # =========================
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200, 10, txt="NSPXN.com AI Review Report", ln=True, align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0, 6, f"File Number: {file_number}")
    pdf.multi_cell(0, 6, f"IA Company: {ia_company}")
    pdf.multi_cell(0, 6, f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0, 6, f"Claim #: {claim_number}")
    pdf.multi_cell(0, 6, f"VIN (from estimate): {vin_from_est}")
    pdf.multi_cell(0, 6, f"VIN verification (estimate vs photo): Not requested")
    pdf.multi_cell(0, 6, f"Vehicle: {vehicle_desc}")
    pdf.multi_cell(0, 6, f"Compliance Score: {comp_score}%")

    pdf.ln(4)
    pdf.set_font_size(12); pdf.cell(0, 8, txt="AI-4-IA Review Summary", ln=True)
    pdf.set_font_size(10); pdf.multi_cell(0, 6, gpt_output or "No narrative generated.")

    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0, 8, txt="Estimate ↔ Photos Consistency Review", ln=True)
    pdf.set_font_size(10)
    if photos_present and intent in ("photos_only","comprehensive"):
        pdf.multi_cell(0, 6, "Included in narrative above (single-pass review).")
    else:
        pdf.multi_cell(0, 6, "Not requested or no photos provided.")

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        with open(pdf_path, "wb") as f: f.write(pdf_bytes)
        logger.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        logger.error(f"PDF write error: {e}")

    # =========================
    # Email (unchanged structure)
    # =========================
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Appraiser Request: {ai_request or 'N/A'}

Claim #: {claim_number}
VIN (from estimate): {vin_from_est}
VIN verification (estimate vs photo): Not requested
Vehicle: {vehicle_desc}

Compliance Score: {comp_score}%

Summary:
{gpt_output}
"""
        msg.set_content(body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        logger.error(f"Email error (continuing): {e}")

    return {
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim_number,
        "vehicle": vehicle_desc,
        "vin_estimate": vin_from_est,
        "vin_verification": "Not requested",
        "score": f"{comp_score}%"
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"
    fp = os.path.join(rules_dir, f"{client_name}.docx")
    if os.path.exists(fp):
        try:
            doc = Document(fp)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return {"text": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=404, content={"error": "Rules not found for this client."})











