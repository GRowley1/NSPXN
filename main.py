
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, base64, json, logging, time, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageStat, Image

from openai import OpenAI

# =========================
# Config & Logging
# =========================
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("ai4ia")

OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
MODEL = os.getenv("OAI_MODEL", "gpt-4o")

if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    raise RuntimeError("❌ OPENAI_API_KEY is not set")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# =========================
# FastAPI
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep liberal for now; tighten in deploy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# OCR helpers
# =========================
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img

def pdf_text_ocr(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 20) -> str:
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=dpi)[:max_pages]
    except Exception as e:
        log.warning(f"pdf to image failed: {e}")
        return ""
    out = []
    for i, pg in enumerate(pages, 1):
        try:
            txt = pytesseract.image_to_string(_pp(pg), lang="eng", config="--psm 6")
            if txt.strip():
                out.append(f"[Page {i}]\n{txt}")
        except Exception as e:
            log.warning(f"OCR page {i} failed: {e}")
    return "\n\n".join(out)

def docx_text(blob: bytes) -> str:
    try:
        d = Document(io.BytesIO(blob))
        return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        log.warning(f"DOCX read error: {e}")
        return ""

# =========================
# Extractors (VIN/claim/vehicle/odo)
# =========================
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")

def normalize_vin(s: str) -> Optional[str]:
    s = (s or "").upper().replace(" ", "")
    s = s.replace("O", "0").replace("I", "1").replace("Q", "0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s):
        return None
    return s

_trans = {**{str(i): i for i in range(10)},
          **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_w = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def vin_checksum_ok(v: str) -> bool:
    try:
        tot = sum(_trans[ch]*_w[i] for i,ch in enumerate(v))
        chk = tot % 11
        return v[8] == ("X" if chk==10 else str(chk))
    except Exception:
        return False

def best_vin(cands: List[str]) -> Optional[str]:
    # prefer checksum-valid first, else first normalized
    for c in cands:
        v = normalize_vin(c)
        if v and vin_checksum_ok(v):
            return v
    for c in cands:
        v = normalize_vin(c)
        if v:
            return v
    return None

def extract_vin_from_text(text: str) -> Optional[str]:
    if not text: return None
    # prefer labeled
    lab = re.findall(r"\bVIN\s*[:#\-]?\s*([A-HJ-NPR-Z0-9]{10,20})", text, re.IGNORECASE)
    if lab:
        v = best_vin(lab)
        if v: return v
    # fallback: any 17-char vin-like
    cands = re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", text, re.IGNORECASE)
    return best_vin(cands)

def extract_claim_from_text(text: str) -> Optional[str]:
    if not text: return None
    pats = [
        r"Claim\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
        r"Assignment\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
        r"Reference\s*(?:No\.?|Number|#)?\s*[:\-]?\s*([A-Za-z0-9\-_\/]+)",
    ]
    for p in pats:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            tok = m.group(1).strip().rstrip(".,;")
            # require at least one digit to avoid random words
            if re.search(r"\d", tok):
                return tok
    return None

MAKE_MAP = {
    "CHEV":"Chevrolet","CHEVY":"Chevrolet","MB":"Mercedes-Benz","MERCEDES":"Mercedes-Benz",
    "VW":"Volkswagen","VOLKS":"Volkswagen","GMC":"GMC","TOYOTA":"Toyota","HONDA":"Honda",
    "NISSAN":"Nissan","HYUNDAI":"Hyundai","KIA":"Kia","FORD":"Ford","DODGE":"Dodge","RAM":"Ram",
    "SUBARU":"Subaru","MAZDA":"Mazda","BMW":"BMW","AUDI":"Audi","JEEP":"Jeep"
}
VEH_STOP = set("""VIN WB ODOMETER MILEAGE ENGINE L TURBO SUPERCHARGED DIESEL GASOLINE
EXT INTR EXTERIOR INTERIOR COLOR COLORS TRIM TRANS TRANSMISSION 2D 4D P/U WB" WB’ WB’ WB”""".split())

def extract_vehicle_from_text(text: str) -> Optional[str]:
    if not text: return None
    # try to find a long line that begins with yyyy and make-like token
    cand = None
    for line in text.splitlines():
        ln = line.strip()
        m = re.match(r"^(19|20)\d{2}\s+([A-Za-z]{3,})\s+(.+)$", ln)
        if m:
            year = m.group(0).split()[0]
            # keep the longest plausible line as best candidate
            if cand is None or len(ln) > len(cand):
                cand = ln
    if not cand:
        return None
    parts = cand.split()
    year = parts[0]
    make_raw = parts[1].upper()
    make = MAKE_MAP.get(make_raw, parts[1].capitalize())

    model_tokens = []
    for tok in parts[2:]:
        t = re.sub(r"[^\w\-\"’”/]", "", tok).upper()
        if t in VEH_STOP or re.match(r"^\d+(\.\d+)?L$", t):
            break
        # stop if we hit VIN: or License: etc.
        if t in ("VIN:", "VIN"):
            break
        model_tokens.append(tok)
        if len(" ".join(model_tokens)) > 80:  # don't run away
            break

    model = " ".join(model_tokens).strip() or "Vehicle"
    return f"{year} {make} {model}"

def extract_odometer_from_photos(imgs: List[Tuple[str, bytes]]) -> Optional[str]:
    for name, blob in imgs:
        try:
            im = Image.open(io.BytesIO(blob))
            txt = pytesseract.image_to_string(_pp(im), lang="eng")
            m = re.search(r"\b(\d{1,3}(?:,\d{3})+|\d{2,6})\b\s*(?:mi|miles|km)\b", txt, re.IGNORECASE)
            if m:
                return m.group(1)
        except Exception as e:
            log.warning(f"Odometer OCR error {name}: {e}")
    return None

# =========================
# Photo harvesting & checks
# =========================
def harvest_photos_from_pdf(pdf_bytes: bytes, max_pages: int = 20) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)[:max_pages]
        for i, page in enumerate(pages, 1):
            g = _pp(page)
            var = ImageStat.Stat(g).var[0] if g.mode == "L" else sum(ImageStat.Stat(g).var)/3
            looks_photo = var > 120
            if looks_photo:
                buf = io.BytesIO()
                page.save(buf, format="JPEG", quality=85)
                out.append((f"pdf-photo-p{i}.jpg", buf.getvalue()))
    except Exception as e:
        log.warning(f"harvest photos error: {e}")
    return out

def required_photos_status(image_blobs: List[Tuple[str, bytes]], estimate_text: str) -> Tuple[List[str], Dict[str, bool]]:
    required = ["four corners", "odometer", "vin", "license plate"]
    present = {k: False for k in required}

    ext_like = 0
    for name, blob in image_blobs:
        try:
            im = Image.open(io.BytesIO(blob))
            txt = pytesseract.image_to_string(_pp(im), lang="eng")
            if re.search(r"\b[A-HJ-NPR-Z0-9]{17}\b", txt): present["vin"] = True
            if re.search(r"(license|registration)\s*plate|\b[A-Z0-9]{5,8}\b", txt, re.IGNORECASE): present["license plate"] = True
            if re.search(r"\d{1,3}(?:,\d{3})*\s*(mi|miles|km)\b", txt, re.IGNORECASE): present["odometer"] = True
            # exterior heuristic
            var = ImageStat.Stat(_pp(im)).var[0] if im.mode == "L" else sum(ImageStat.Stat(_pp(im)).var)/3
            if var > 150 and len(txt.strip()) < 10:
                ext_like += 1
        except Exception as e:
            log.debug(f"image parse warn {name}: {e}")
    if ext_like >= 2:
        present["four corners"] = True

    missing = [k for k,v in present.items() if not v]
    return missing, present

def vin_from_photos(image_blobs: List[Tuple[str, bytes]]) -> Optional[str]:
    cands = []
    for name, blob in image_blobs:
        try:
            im = Image.open(io.BytesIO(blob))
            for r in (0,90,180,270):
                img = im.rotate(r, expand=True)
                txt = pytesseract.image_to_string(_pp(img), lang="eng", config="--psm 7")
                cands.extend(re.findall(r"\b([A-HJ-NPR-Z0-9]{17})\b", txt))
        except Exception as e:
            log.debug(f"vin photo err {name}: {e}")
    return best_vin(cands)

# =========================
# Client rules integration (deductions)
# =========================
def client_rule_deductions(estimate_text: str, client_rules: str) -> Tuple[int, List[str]]:
    """Return (deduction_total, reasons[])"""
    total = 0
    notes = []
    if re.search(r"NADA.*(required|must)", client_rules, re.IGNORECASE):
        if "NADA" not in estimate_text.upper():
            total -= 25; notes.append("-25: NADA printout missing per client rules")
    if re.search(r"Advisor\s*Report", client_rules, re.IGNORECASE):
        if re.search(r"Advisor", estimate_text, re.IGNORECASE) is None:
            total -= 25; notes.append("-25: Advisor Report not mentioned")
    return total, notes

def labor_tax_deductions(text: str, client_rules: str) -> Tuple[int, List[str]]:
    adj = 0; reasons = []
    def has_rate(label: str) -> bool:
        pat = rf"{label}[^\n]{{0,120}}?\$\s*\d{{2,3}}(?:\.\d+)?\s*(?:/hr|/hour|per\s*hour|hr)"
        return re.search(pat, text, re.IGNORECASE) is not None
    labels = ["Body Labor", "Paint Labor", "Mechanical Labor", "Structural Labor"]
    if not any(has_rate(lbl) for lbl in labels):
        adj -= 50; reasons.append("-50: No labor rates found (any section)")

    if re.search(r"tax\s*(required|must|utilize|apply)", client_rules, re.IGNORECASE):
        if not re.search(r"(sales\s*tax|tax)[^\n]{0,80}?(\d{1,3}\.\d+%|\d{1,3}%|\$\s*\d+(\.\d{2})?)", text, re.IGNORECASE):
            adj -= 25; reasons.append("-25: Sales tax not found but required by rules")
    return adj, reasons

# =========================
# PDF helpers
# =========================
def pdf_title(pdf: FPDF, title: str):
    pdf.set_font_size(12)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font_size(10)

# =========================
# Routes
# =========================
@app.get("/")
async def ok():
    return {"ok": True}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(""),
    appraiser_id: str = Form(""),
    ai_intent: str = Form("comprehensive"),
    client_name: str = Form("")
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error":"Appraiser ID is required."})

    # ---- read uploads (no caching anywhere; only current request) ----
    texts: List[str] = []
    images: List[Tuple[str, bytes]] = []
    vision_imgs: List[Dict[str, Any]] = []

    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"):
            t = pdf_text_ocr(raw, dpi=200, max_pages=12)
            texts.append(t)
            # harvest photo-like pages from within the PDF
            for nm, jb in harvest_photos_from_pdf(raw):
                images.append((nm, jb))
                vision_imgs.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(jb).decode("utf-8")}})
        elif name.endswith(".docx"):
            texts.append(docx_text(raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp")):
            images.append((name, raw))
            vision_imgs.append({"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(raw).decode("utf-8")}})
        elif name.endswith(".txt"):
            texts.append(raw.decode("utf-8","ignore"))
        else:
            # ignore unsupported
            pass

    est_text = "\n\n".join(texts)[:25000]

    # --- Client rules: honor pasted text first; if empty and client_name provided, load from disk
    if not client_rules.strip() and client_name.strip():
        client_rules = load_client_rules_from_disk(client_name.strip())
    # Also, if a DOCX was uploaded and looks like rules (contains "Client Quick Summary" or "Photo Rules"), parse it
    if not client_rules.strip():
        for f in files:
            try:
                if (f.filename or "").lower().endswith(".docx"):
                    raw = await f.read()
                    dtx = docx_text(raw)
                    if re.search(r"Client\s+Quick\s+Summary|Photo\s+Rules|Fatal\s+Error", dtx, re.IGNORECASE):
                        client_rules = dtx
                        break
            except Exception:
                pass
    client_rules = client_rules or ""


    # ---- extract fields from THIS request only ----
    vin_est = extract_vin_from_text(est_text)
    vin_photo = vin_from_photos(images)
    vin_final = vin_est or vin_photo

    claim_number = extract_claim_from_text(est_text) or "N/A"
    vehicle_desc = extract_vehicle_from_text(est_text) or "N/A"
    odo_photo = extract_odometer_from_photos(images)

    missing_photos, photo_presence = required_photos_status(images, est_text)

    # ---- VIN verification string ----
    if images:
        if vin_est and vin_photo:
            vin_verify = "MATCH" if normalize_vin(vin_est)==normalize_vin(vin_photo) else "MISMATCH"
        elif vin_est and not vin_photo:
            vin_verify = "NOT VERIFIED"
        else:
            vin_verify = "NOT VERIFIED"
    else:
        vin_verify = "PHOTOS NOT PROVIDED"

    # ---- Build GPT narrative (short, deterministic) ----
    sections = [
        f"VIN Verification: {vin_verify}",
        "1) Client Quick Summary",
        "2) Fatal Errors",
        "3) Client Photo Rules",
        "4) Documentation Requirements",
        "5) Rates and Sales Tax Rules",
        "6) Estimate ↔ Photos Comparison",
        "7) Summary & Next Steps",
    ]
    if (ai_intent or "").strip().lower() == "comprehensive":
        sys = (
            "Comprehensive review: compare client guidelines to the estimate AND compare the estimate to the photos. "
            "If photos are not provided, OMIT all photo-related sections. "
            "MANDATORY OUTPUT SHAPE:
"
            "VIN Verification: <MATCH | MISMATCH | NOT VERIFIED | PHOTOS NOT PROVIDED>
"
            "1) Client Quick Summary (2–3 bullets)
"
            "2) Fatal Errors (bullet list; only truly fatal items)
"
            "3) Client Photo Rules (only if photos_present=true) — each item begins with [Compliant] | [Non-compliant] | [Not found]
"
            "4) Estimate/Supplement Release Rules — bracketed tags per item
"
            "5) Parts Application Rules — bracketed tags per item
"
            "6) Total Loss Rules — bracketed tags per item (or 'Not applicable')
"
            "7) Tow Charge Rules — bracketed tags per item
"
            "8) Supplement Handling Rules — bracketed tags per item
"
            "9) Betterment/Depreciation Rules — bracketed tags per item
"
            "10) Documentation Requirements — bracketed tags per item (explicitly call out Clean Retail Value printout and Advisor Report)
"
            "11) Rates and Sales Tax Rules — bracketed tags per item
"
            "12) Miscellaneous Rules — bracketed tags per item
"
            "13) Estimate ↔ Photos Comparison (only if photos_present=true): damage match, discrepancies, missing views/measurements
"
            "14) Summary & Next Steps (2 bullets)
"
            "Be concise, specific, and only use the provided materials. "
            "Always end with a single line: Final Evaluation: NN%."
        )
    else:
        sys = (
            "You are an auto-claims appraisal assistant. Be concise and specific. "
            "Where you cannot verify, say 'Not found in provided documents'. "
            "Always end with a single line: Final Evaluation: NN%."
        )
    user_parts: List[Dict[str, Any]] = []
    user_parts.append({"type":"text","text": f"CLIENT GUIDELINES:\n{client_rules[:8000]}"})
    user_parts.append({"type":"text","text": f"ESTIMATE TEXT (OCR):\n{est_text[:12000]}"})
    if images:
        user_parts.extend(vision_imgs)

    try:
        rsp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role":"system","content": sys},
                {"role":"user","content": user_parts}
            ],
            max_tokens=1000,
            temperature=0
        )
        gpt_out = (rsp.choices[0].message.content or "").strip()
    except Exception as e:
        log.warning(f"GPT error: {e}")
        gpt_out = "AI narrative unavailable."

    # ---- Score calculation (authoritative single number) ----
    # 1) try to read from AI
    ai_score = None
    m = re.search(r"(Final|Total|Compliance)\s*(Score|Evaluation)?\s*[:\-]?\s*(\d{1,3})\s*%?", gpt_out, re.IGNORECASE)
    if m:
        try:
            ai_score = int(m.group(3))
        except:
            ai_score = None

    # 2) deterministic deductions
    labor_adj, labor_notes = labor_tax_deductions(est_text, client_rules)
    rule_adj, rule_notes = client_rule_deductions(est_text, client_rules)
    photo_adj = -25 * len([k for k in ["four corners","odometer","vin","license plate"] if k in missing_photos])

    deduce_notes = []
    deduce_notes.extend(labor_notes)
    deduce_notes.extend(rule_notes)
    if photo_adj:
        for miss in [k for k in ["four corners","odometer","vin","license plate"] if k in missing_photos]:
            deduce_notes.append(f"-25: Missing required photo — {miss}")

    computed = max(0, 100 + labor_adj + rule_adj + photo_adj)
    score = max(0, min(100, ai_score if ai_score is not None else computed))

    # Clean AI output of duplicate score lines
    gpt_out_clean = re.sub(r'(?im)^(?:Final|Total|Compliance)\s*(?:Score|Evaluation)?\s*[:\-]?\s*\d{1,3}\s*%.*$','',gpt_out).strip()

    # ---- Build PDF (Unicode-safe) ----
    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
        pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(0, 10, "NSPXN.com AI Review Report", ln=True, align="C")
    pdf.set_font_size(10); pdf.ln(3)
    pdf.multi_cell(0,6,f"File Number: {file_number}")
    pdf.multi_cell(0,6,f"IA Company: {ia_company}")
    pdf.multi_cell(0,6,f"Appraiser ID #: {appraiser_id}")
    pdf.ln(2)
    pdf.multi_cell(0,6,f"Claim #: {claim_number}")
    pdf.multi_cell(0,6,f"VIN (from estimate/photos): {vin_final or 'N/A'}")
    pdf.multi_cell(0,6,f"VIN verification (estimate vs photo): {vin_verify}")
    pdf.multi_cell(0,6,f"Vehicle: {vehicle_desc}")
    if odo_photo:
        pdf.multi_cell(0,6,f"Odometer (from photos): {odo_photo}")
    pdf.multi_cell(0,6,f"Compliance Score: {score}%")

    pdf.ln(3)
    pdf_title(pdf, "AI-4-IA Review Summary")
    pdf.multi_cell(0,6, gpt_out_clean or "No narrative.")

    # Compliance Audit Overview
    pdf.ln(3)
    pdf_title(pdf, "Compliance Audit Overview (Deductions)")
    if deduce_notes:
        for n in deduce_notes:
            pdf.multi_cell(0,6, n)
    else:
        pdf.multi_cell(0,6, "No deductions applied based on deterministic checks.")

    # Save PDF
    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        log.warning(f"PDF write error: {e}")

    # Optional email (kept simple; wrap in try)
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim_number}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        msg.set_content(f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Claim #: {claim_number}
VIN: {vin_final or 'N/A'}
Vehicle: {vehicle_desc}

Compliance Score: {score}%

AI Summary:
{gpt_out_clean}
""")
        # Disabled by default in this sandbox to avoid real outbound
        # with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
        #     smtp.login("info@nspxn.com", "grr2025GRR")
        #     smtp.send_message(msg)
    except Exception as e:
        log.info(f"email skipped/failed: {e}")

    return {
        "file_number": file_number,
        "claim_number": claim_number,
        "vin": vin_final or "N/A",
        "vin_verification": vin_verify,
        "vehicle": vehicle_desc,
        "compliance_score": score,
        "missing_photos": missing_photos,
        "deductions": deduce_notes,
        "gpt_output": gpt_out_clean,
        "pdf_url": f"/download-pdf?file_number={file_number}",
        "pdf_filename": f"{file_number}.pdf"
    }

from docx import Document as _RulesDoc

def load_client_rules_from_disk(client_name: str) -> str:
    """
    Load client rules text from ./client_rules/{client_name}.docx if present.
    Returns empty string if not found or on error.
    """
    try:
        rules_dir = os.getenv("CLIENT_RULES_DIR", "client_rules")
        path = os.path.join(rules_dir, f"{client_name}.docx")
        if os.path.exists(path):
            d = _RulesDoc(path)
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        log.warning(f"Rules load error for {client_name}: {e}")
    return ""

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    """
    Front-end helper: fetch rules text by client name from server disk.
    """
    txt = load_client_rules_from_disk(client_name)
    if not txt:
        return JSONResponse(status_code=404, content={"error":"Rules not found for this client."})
    return {"text": txt}

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    safe = re.sub(r"[^\w.\-]+","-", file_number).strip("-_.")
    path = os.path.join(PDF_DIR, f"{safe}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{safe}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})
