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
from pytesseract import Output
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from openai import OpenAI

# =========================
# Config
# =========================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ai4ia")

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY is not set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL_DEFAULT = os.getenv("OAI_MODEL", "gpt-4o-mini")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nspxn.com","https://www.nspxn.com",
        "http://nspxn.com","http://www.nspxn.com",
        "https://nspxn.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OCR helpers
# =========================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.9)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def ocr_pdf_pages(pdf_bytes: bytes, limit_pages: Optional[int] = None, dpi: int = 200) -> List[Image.Image]:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    return pages[:limit_pages] if limit_pages else pages

def ocr_pdf_text(pdf_bytes: bytes, limit_pages: Optional[int] = None, dpi: int = 200, psms=("--psm 6","--psm 3")) -> str:
    try:
        pages = ocr_pdf_pages(pdf_bytes, limit_pages, dpi)
        out = []
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
                out.append(f"\n[Page {i}]\n{txt}")
        return "".join(out)
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
# VIN + Vehicle + Mileage (robust) — ALWAYS RUN
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

def pick_best_vin(cands: List[str]) -> Optional[str]:
    uniq = []
    seen = set()
    for c in cands:
        v = normalize_vin(c)
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    # prefer checksum-valid
    for v in uniq:
        if vin_checksum_ok(v): return v
    return uniq[0] if uniq else None

def find_vin_in_text(text: str) -> Optional[str]:
    label_hits = re.findall(r"(?:VIN[:\s\-]*)([A-HJ-NPR-Z0-9]{10,20})", text, re.IGNORECASE)
    line_hits = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text)
    return pick_best_vin(label_hits + line_hits)

def find_vin_via_data(img: Image.Image) -> Optional[str]:
    """Use Tesseract word boxes: find 'VIN' then nearest 17-char token on the same line."""
    try:
        proc = preprocess_image(img)
        data = pytesseract.image_to_data(proc, lang="eng", output_type=Output.DICT)
        n = len(data["text"])
        for i in range(n):
            word = (data["text"][i] or "").strip().upper()
            if word == "VIN":
                line = data["line_num"][i]
                # collect tokens from same line
                cands = []
                for j in range(n):
                    if data["line_num"][j] == line:
                        t = (data["text"][j] or "").strip().upper()
                        if re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", t):
                            cands.append(t)
                if cands:
                    return pick_best_vin(cands)
    except Exception:
        pass
    return None

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
    for ln in lines:
        if re.search(r"^\s*(19|20)\d{2}\b", ln) and not re.search(r"\b(AM|PM)\b", ln):
            tokens = ln.split()
            year = tokens[0]; tail = tokens[1:]
            kept = []
            for t in tail:
                raw = re.sub(r"[^\w\-]", "", t).upper()
                if raw in STOP_TOKENS: break
                if raw in ("A/M","OEM"): break
                kept.append(t)
                if len(kept) >= 4: break
            if kept:
                mk = MAKE_MAP.get(kept[0].upper(), kept[0].capitalize())
                return " ".join([year, mk] + kept[1:])
    # fallback: VEHICLE header → next line
    for i, ln in enumerate(lines):
        if "VEHICLE" in ln.upper() and i+1 < len(lines):
            nxt = lines[i+1]
            if re.search(r"(19|20)\d{2}", nxt):
                return nxt.strip()
    return None

def extract_mileage(text: str) -> Optional[str]:
    pats = [
        r"(?:Odometer|Odo|Mileage|Miles)\s*[:\-]?\s*([\d,]{2,7})\b",
        r"\b([\d,]{2,7})\s*(?:mi|miles)\b",
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = m.group(1)
            return val
    return None

# =========================
# Totals/rates parsing (anchor GPT to facts)
# =========================
def find_money(label: str, text: str) -> Optional[str]:
    pat = rf"{label}[^$\n]{{0,40}}(?:\$)?\s*([\d,]+\.\d{{2}})"
    m = re.search(pat, text, re.IGNORECASE)
    return m.group(1) if m else None

def find_rate_block(text: str) -> List[str]:
    hits = re.findall(r"\$\s?\d{2,3}\.\d{2}\s*(?:/hr|per\s*hour|hr)", text, re.IGNORECASE)
    return list(dict.fromkeys(hits))[:6]

def find_tax_rate(text: str) -> Optional[str]:
    m = re.search(r"(\d{1,2}\.\d{3,4}|\d{1,2}\.\d{1,2}|\d{1,2})\s*%\s*(?:tax|sales)", text, re.IGNORECASE)
    return m.group(1) + "%" if m else None

def parse_estimate_facts(text: str) -> Dict[str, Any]:
    return {
        "totals": {
            "total": find_money(r"(Total|Grand\s*Total|Total Cost of Repairs)", text),
            "parts": find_money("Parts", text),
            "body_labor": find_money(r"(Body\s*Labor|BL)", text),
            "paint_labor": find_money(r"(Paint\s*Labor|PL)", text),
            "paint_supplies": find_money(r"(Paint\s*Suppl(?:y|ies)|Materials|P&M)", text),
            "misc": find_money("Misc", text),
            "other": find_money("Other Charges", text),
        },
        "rates": find_rate_block(text),
        "tax_rate": find_tax_rate(text),
    }

# =========================
# Minimal GPT (with 429 fallback)
# =========================
def safe_chat_completion(messages, max_tokens=900, model=MODEL_DEFAULT):
    try:
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0
        )
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
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
# Post-filter to prevent photo/doc hallucinations
# =========================
def strip_photo_claims(text: str) -> str:
    text = re.sub(r"(?is)##\s*Client\s*Photo\s*Rules.*?(?=\n##|\Z)", "", text)
    text = re.sub(r"(?is)##\s*Estimate.?↔.?Photos\s*Comparison.*?(?=\n##|\Z)", "", text)
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
        elif name.endswith((".jpg",".jpeg",".png",".webp")):
            images.append((name, raw))
        elif name.endswith(".docx"):
            docs.append(ocr_docx_text(io.BytesIO(raw)))
        elif name.endswith(".txt"):
            docs.append(raw.decode("utf-8", errors="ignore"))

    intent = parse_intent(ai_request)
    logger.info(f"Intent: {intent} | Request: {ai_request}")

    # --- OCR (fast): always read a few pages for fields
    limit = 12 if intent == "comprehensive" else 5
    estimate_text = ""
    pages_200 = []
    if pdfs:
        estimate_text = ocr_pdf_text(pdfs[0][1], limit_pages=limit, dpi=200)
        pages_200 = ocr_pdf_pages(pdfs[0][1], limit_pages=min(3, limit), dpi=200)
        # If VIN still missing, try two-page 240dpi + word-box VIN scan
        if not find_vin_in_text(estimate_text):
            pages_240 = ocr_pdf_pages(pdfs[0][1], limit_pages=2, dpi=240)
            # merge text
            for p in pages_240:
                estimate_text = f"{estimate_text}\n" + pytesseract.image_to_string(preprocess_image(p), lang="eng", config="--psm 6")
            # try word-box scan
            if not find_vin_in_text(estimate_text):
                for p in pages_240:
                    vin_box = find_vin_via_data(p)
                    if vin_box:
                        estimate_text += f"\nVIN_FOUND:{vin_box}"
                        break
    elif docs:
        estimate_text = "\n\n".join(docs)

    # --- ALWAYS extract VIN, Vehicle, Mileage from estimate
    vin_from_est = find_vin_in_text(estimate_text) or "N/A"
    vehicle_desc = extract_vehicle(estimate_text) or "N/A"
    est_mileage = extract_mileage(estimate_text)  # may be None if truly absent
    claim_number = (re.search(r"Claim\s*(?:No\.?|Number|#)[:\s]*([A-Za-z0-9\-_/]+)", estimate_text, re.IGNORECASE) or re.search(r"Claim\s*[:#]\s*([A-Za-z0-9\-_/]+)", estimate_text, re.IGNORECASE))
    claim_number = claim_number.group(1) if claim_number else "N/A"

    # Evidence flags to prevent hallucinations
    photos_present = len(images) > 0
    has_clean_value = bool(re.search(r"(clean\s*retail|NADA|KBB|Black\s*Book|J\.?D\.?\s*Power|valuation)", estimate_text, re.IGNORECASE))
    has_advisor = bool(re.search(r"(advisor\s*report|ccc\s*one\s*advisor)", estimate_text, re.IGNORECASE))

    # Parse numeric facts so GPT can anchor to real values
    facts = parse_estimate_facts(estimate_text)

    # --- Build narrative strictly per intent
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
            "HARD RULES:\n"
            f"- PhotosPresent={photos_present}. If false, DO NOT mention photos.\n"
            f"- CleanRetailProvided={has_clean_value}. AdvisorReportProvided={has_advisor}. "
            "Only say 'included' if the flag is True; otherwise mark 'missing/not provided'.\n"
            "- Use these extracted facts when present (do not invent numbers):\n"
            f"{json.dumps(facts, indent=2)}\n"
            "- Sections: Client Quick Summary, Fatal Errors (if any), Parts/Tax/Labor, Documentation Requirements, Summary.\n"
            "- Be concise, factual, and concrete."
        )
        user_parts = [
            {"type":"text","text":"CLIENT GUIDELINES:\n" + (client_rules or "")[:12000]},
            {"type":"text","text":"\n\nESTIMATE (OCR):\n" + (estimate_text or "")[:12000]},
            {"type":"text","text":f"\n\nAPPRAISER REQUEST: {ai_request}"},
        ]
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=1000)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()
        if not photos_present:
            gpt_output = strip_photo_claims(gpt_output)

    elif intent == "comprehensive":
        system = (
            "You are an auto-damage auditor.\n"
            "TASK: Guidelines + Estimate (+ Photos only if provided).\n"
            "HARD RULES:\n"
            f"- PhotosPresent={photos_present}. If false, explicitly state no photos and OMIT photo sections.\n"
            f"- CleanRetailProvided={has_clean_value}. AdvisorReportProvided={has_advisor}. "
            "Only say 'included' if the flag is True; else 'missing'.\n"
            "- Use these extracted facts when present:\n"
            f"{json.dumps(facts, indent=2)}\n"
            "Sections: Client Quick Summary Compliance → Fatal Errors → (Photo Rules, omit if no photos) → "
            "Parts/Tax/Labor → (Estimate↔Photos Comparison, omit if no photos) → Summary.\n"
            "Be concise and specific."
        )
        user_parts = [
            {"type":"text","text":"CLIENT GUIDELINES:\n" + (client_rules or "")[:8000]},
            {"type":"text","text":"\n\nESTIMATE (OCR):\n" + (estimate_text or "")[:10000]},
        ]
        if photos_present:
            user_parts.extend(vision_images())
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=1100)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()
        if not photos_present:
            gpt_output = strip_photo_claims(gpt_output)

    elif intent == "photos_only":
        if not photos_present:
            gpt_output = "No photos were provided with this request."
        else:
            system = (
                "Compare the ESTIMATE to the attached PHOTOS.\n"
                "Sections: Photo Coverage, Visible Damage vs Estimate, Discrepancies, Summary.\n"
                "Do not analyze client guidelines."
            )
            user_parts = [{"type":"text","text":"ESTIMATE (OCR):\n" + (estimate_text or "")[:8000]}]
            user_parts.extend(vision_images())
            rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=900)
            gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()

    elif intent == "vin_only":
        gpt_output = (
            "VIN Extraction (Estimate Only)\n"
            f"- VIN from estimate: {vin_from_est}\n"
            f"- Mileage (from estimate): {est_mileage or 'Not documented'}\n"
            "- Notes: VIN photo verification not requested."
        )

    elif intent == "invoices_only":
        invoices_text = ""
        for name, raw in pdfs:
            if "invoice" in name or "receipt" in name or "supplement" in name:
                invoices_text += ocr_pdf_text(raw, limit_pages=6)
        if not invoices_text and pdfs:
            invoices_text = ocr_pdf_text(pdfs[0][1], limit_pages=6)
        system = (
            "Audit whether supplement/estimate lines are substantiated by attached invoices.\n"
            "Write bullets: key invoice items + totals → state if each supports the supplement. "
            "Call out any missing docs."
        )
        user_parts = [
            {"type":"text","text":"ESTIMATE (OCR):\n" + (estimate_text or "")[:6000]},
            {"type":"text","text":"\n\nINVOICES (OCR):\n" + (invoices_text or "")[:6000]},
            {"type":"text","text":f"\n\nAPPRAISER REQUEST: {ai_request}"},
        ]
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=900)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()

    else:
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
        rsp = safe_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user_parts}], max_tokens=1000)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable (OpenAI error).").strip()
        if not photos_present:
            gpt_output = strip_photo_claims(gpt_output)

    # =========================
    # Light score only for guideline-type requests
    # =========================
    def light_guideline_score(txt: str, rules: str) -> int:
        score = 100
        if rules:
            if "labor" in rules.lower() and not re.search(r"(labor|rate).{0,80}\$", txt, re.IGNORECASE | re.DOTALL):
                score -= 10
            if "tax" in rules.lower() and "tax" not in txt.lower():
                score -= 10
        return max(0, min(100, score))
    comp_score = light_guideline_score(estimate_text, client_rules) if intent in ("guidelines_only","comprehensive") else 100

    # =========================
    # PDF (unchanged layout)
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
    if est_mileage:
        pdf.multi_cell(0, 6, f"Odometer (from estimate): {est_mileage}")
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
{"Odometer (from estimate): " + est_mileage if est_mileage else ""}

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
        "odometer_estimate": est_mileage or "Not documented",
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












