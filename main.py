
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Tuple, Optional, Dict, Any
import os, re, io, json, logging, base64, smtplib
from email.message import EmailMessage

from fpdf import FPDF
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
from pytesseract import Output
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

from openai import OpenAI

# ----------------- Config -----------------
PDF_DIR = os.getenv("PDF_DIR", "/tmp")
os.makedirs(PDF_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ai4ia")

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("❌ OPENAI_API_KEY not set.")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.getenv("OAI_MODEL", "gpt-4o-mini")

# Whitelisted request types (trimmed per user spec)
INTENTS = {
    "guidelines_only": "Compare client guidelines to the estimate only",
    "comprehensive": "Guidelines + Estimate + Photos (VIN/Vehicle & Rates/Tax included)",
    "photos_only": "Estimate ↔ Photos comparison only",
    "invoices_with_photos": "Supplement ↔ Invoices (+ Photos)",
    "docs_checklist": "Documentation checklist (Clean Retail Value, Advisor Report, etc.)"
}

# ----------------- App -----------------
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

# ----------------- Compatibility stub (prevents old-call crashes) -----------------
def compare_estimate_with_photos(*args, **kwargs):
    """
    Safe no-JSON stub to remain compatible with older frontends that still invoke this.
    Returns a concise narrative string; never raises JSONDecodeError/NameError.
    """
    images = kwargs.get("images") or (args[1] if len(args) > 1 else [])
    if not images:
        return "Photos were not provided; photo comparison omitted."
    return "Photo comparison included in narrative above."

# ----------------- OCR/Text helpers -----------------
def _pp(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.9)
    img = img.filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img)
    return img

def pdf_pages(pdf_bytes: bytes, limit_pages: Optional[int] = None, dpi: int = 200) -> List[Image.Image]:
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    return pages[:limit_pages] if limit_pages else pages

def fast_pdf_text(pdf_bytes: bytes, limit_pages: Optional[int] = None) -> str:
    """
    Fast text-layer extraction (PyPDF2). Returns empty string if module not present or no text layer.
    """
    text = ""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = reader.pages
        if limit_pages:
            pages = pages[:limit_pages]
        for i, page in enumerate(pages, 1):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                text += f"\n[Page {i}]\n{t}"
    except Exception as e:
        log.info(f"PyPDF2 fast extract skipped: {e}")
    return text or ""

def ocr_pdf_text(pdf_bytes: bytes, limit_pages: Optional[int] = None, dpi: int = 200, psms=("--psm 6","--psm 3")) -> str:
    try:
        pages = pdf_pages(pdf_bytes, limit_pages, dpi)
        out = []
        for i, p in enumerate(pages, 1):
            proc = _pp(p)
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
        log.warning(f"OCR PDF error: {e}")
        return ""

def ocr_docx_text(file_like: io.BytesIO) -> str:
    try:
        doc = Document(file_like)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        log.warning(f"DOCX read error: {e}")
        return ""

# ----------------- VIN & Vehicle (ALWAYS) -----------------
VIN_ALLOWED = set("0123456789ABCDEFGHJKLMNPRSTUVWXYZ")
_translit = {**{str(i): i for i in range(10)},
             **dict(A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,J=1,K=2,L=3,M=4,N=5,P=7,R=9,S=2,T=3,U=4,V=5,W=6,X=7,Y=8,Z=9)}
_weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

def _normalize_vin(s: str) -> Optional[str]:
    s = (s or "").upper()
    s = re.sub(r"[^A-HJ-NPR-Z0-9]", "", s)
    s = s.replace("O","0").replace("I","1").replace("Q","0")
    if len(s) != 17 or any(ch not in VIN_ALLOWED for ch in s): return None
    return s

def _vin_ok(v: str) -> bool:
    try:
        total = sum(_translit[ch] * _weights[i] for i, ch in enumerate(v))
        check = total % 11
        return v[8] == ("X" if check == 10 else str(check))
    except Exception:
        return False

VIN_RELAXED = re.compile(r"(?:V\.?I\.?N\.?|VIN|Vehicle\\s+Identification\\s+Number)\\b[^A-Z0-9]{0,12}((?:[A-HJ-NPR-Z0-9][\\s\\-]*){17})", re.IGNORECASE)
VIN_TIGHT   = re.compile(r"\\b([A-HJ-NPR-Z0-9]{17})\\b")

def vin_from_text(text: str) -> Optional[str]:
    cands = []
    for m in VIN_RELAXED.finditer(text or ""):
        cands.append(m.group(1))
    cands += VIN_TIGHT.findall(text or "")
    uniq = []
    seen = set()
    for c in cands:
        v = _normalize_vin(c)
        if v and v not in seen:
            uniq.append(v); seen.add(v)
    for v in uniq:
        if _vin_ok(v): return v
    return uniq[0] if uniq else None

def vin_from_pdf_boxes(pages: List[Image.Image]) -> Optional[str]:
    try:
        for p in pages:
            proc = _pp(p)
            data = pytesseract.image_to_data(proc, lang="eng", output_type=Output.DICT)
            n = len(data["text"])
            for i in range(n):
                token = (data["text"][i] or "").strip().upper()
                if token in ("VIN","V.I.N.","VEHICLE","IDENTIFICATION"):
                    line = data["line_num"][i]
                    left_anchor = data["left"][i]
                    tokens = []
                    for j in range(n):
                        if data["line_num"][j] == line and data["left"][j] >= left_anchor - 5:
                            tokens.append((data["text"][j] or "").strip().upper())
                    joined = " ".join(tokens)
                    v = vin_from_text(joined)
                    if v: return v
                    top = data["top"][i] - 10
                    band = proc.crop((0, max(0, top), proc.width, min(proc.height, top + data["height"][i] + 20)))
                    txt = pytesseract.image_to_string(band, lang="eng", config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
                    v2 = vin_from_text(txt)
                    if v2: return v2
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

def vehicle_from_text(text: str) -> Optional[str]:
    lines = [re.sub(r"\\s{2,}", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    for ln in lines:
        if re.search(r"^\\s*(19|20)\\d{2}\\b", ln) and not re.search(r"\\b(AM|PM)\\b", ln):
            toks = ln.split()
            year = toks[0]; tail = toks[1:]
            keep = []
            for t in tail:
                raw = re.sub(r"[^\\w\\-]", "", t).upper()
                if raw in STOP_TOKENS or raw in ("A/M","OEM"): break
                keep.append(t)
                if len(keep) >= 4: break
            if keep:
                mk = MAKE_MAP.get(keep[0].upper(), keep[0].capitalize())
                return " ".join([year, mk] + keep[1:])
    for i, ln in enumerate(lines):
        if "VEHICLE" in ln.upper() and i+1 < len(lines):
            nxt = lines[i+1]
            if re.search(r"(19|20)\\d{2}", nxt):
                return nxt.strip()
    return None

def mileage_from_text(text: str) -> Optional[str]:
    for p in [r"(?:Odometer|Odo|Mileage|Miles)\\s*[:\\-]?\\s*([\\d,]{2,7})\\b",
              r"\\b([\\d,]{2,7})\\s*(?:mi|miles)\\b"]:
        m = re.search(p, text, re.IGNORECASE)
        if m: return m.group(1)
    return None

def claim_from_text(text: str) -> Optional[str]:
    pats = [
        r"Claim\\s*(?:No\\.?|Number|#)\\s*[: ]\\s*([A-Za-z0-9\\-_\\/]+)",
        r"CLM\\s*#\\s*[: ]\\s*([A-Za-z0-9\\-_\\/]+)",
        r"Claim\\s*[:#]\\s*([A-Za-z0-9\\-_\\/]+)"
    ]
    for pat in pats:
        m = re.search(pat, text, re.IGNORECASE)
        if m: return m.group(1).strip()
    return None

def extract_reported_days(text: str) -> Optional[int]:
    m = re.search(r"Days?\\s*to\\s*Repair\\s*[:\\-]?\\s*([0-9]+)", text or "", re.IGNORECASE)
    try:
        return int(m.group(1)) if m else None
    except:
        return None

# ----------------- Facts helpers (for grounded GPT) -----------------
def money(label: str, text: str) -> Optional[str]:
    m = re.search(rf"{label}[^$\\n]{{0,60}}(?:\\$)?\\s*([\\d,]+\\.\\d{{2}})", text, re.IGNORECASE)
    return m.group(1) if m else None

def rate_hits(text: str) -> List[str]:
    hits = re.findall(r"\\$\\s?\\d{2,3}\\.\\d{2}\\s*(?:/hr|per\\s*hour|hr)", text, re.IGNORECASE)
    return list(dict.fromkeys(hits))[:6]

def tax_rate(text: str) -> Optional[str]:
    m = re.search(r"(\\d{1,2}(?:\\.\\d{1,4})?)\\s*%\\s*(?:tax|sales)", text, re.IGNORECASE)
    return m.group(1) + "%" if m else None

def hours(label: str, text: str) -> Optional[float]:
    m = re.search(rf"{label}[^0-9\\n]{{0,40}}(\\d+(?:\\.\\d+)?)\\s*h", text, re.IGNORECASE)
    try: return float(m.group(1)) if m else None
    except: return None

def estimate_facts(text: str) -> Dict[str, Any]:
    body_h = hours("Body\\s*Labor|BL", text) or 0.0
    paint_h = hours("Paint\\s*Labor|PL", text) or 0.0
    mech_h  = hours("Mech|Mechanical\\s*Labor", text) or 0.0
    frame_h = hours("Frame|Structural\\s*Labor", text) or 0.0
    total_h = round(body_h + paint_h + mech_h, 2) + (frame_h or 0.0)
    days_formula = round(total_h/5.0, 1) if total_h else None
    return {
        "totals": {
            "total": money(r"(Total|Grand\\s*Total|Total Cost of Repairs)", text),
            "parts": money("Parts", text),
            "body_labor": money(r"(Body\\s*Labor|BL)", text),
            "paint_labor": money(r"(Paint\\s*Labor|PL)", text),
            "paint_supplies": money(r"(Paint\\s*Suppl(?:y|ies)|Materials|P&M)", text),
            "misc": money("Misc", text),
            "other": money("Other Charges", text),
            "subtotal": money("Subtotal", text),
            "sales_tax": money("(Sales\\s*Tax|Tax)", text),
        },
        "rates": rate_hits(text),
        "tax_rate": tax_rate(text),
        "hours": {
            "body": body_h or None, "paint": paint_h or None,
            "mech": mech_h or None, "frame": frame_h or None,
            "total_hours": total_h or None,
            "days_formula_hrs_div_5": days_formula,
            "days_reported": extract_reported_days(text)
        }
    }

# ----------------- GPT wrapper -----------------
def chat(messages, max_tokens=900, model=MODEL):
    try:
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0
        )
    except Exception as e:
        if "429" in str(e) or "rate" in str(e).lower():
            log.warning("429 RateLimit → fallback gpt-3.5-turbo")
            try:
                return client.chat.completions.create(
                    model="gpt-3.5-turbo", messages=messages, max_tokens=max_tokens, temperature=0
                )
            except Exception as e2:
                log.error(f"Fallback failed: {e2}")
                return None
        log.error(f"OpenAI error: {e}")
        return None

def strip_photo_claims(text: str) -> str:
    text = re.sub(r"(?is)##\\s*Client\\s*Photo\\s*Rules.*?(?=\\n##|\\Z)", "", text)
    text = re.sub(r"(?is)##\\s*Estimate.?↔.?Photos\\s*Comparison.*?(?=\\n##|\\Z)", "", text)
    text = re.sub(r"(?im)^-.*photo.*$", "", text)
    return text.strip()

def correct_false_negatives(text: str, mileage_present: bool, reported_days: Optional[int]) -> str:
    if not text: return text
    lines = text.splitlines()
    out = []
    for ln in lines:
        bad = False
        if mileage_present and re.search(r"(?i)mileage\\s+(not|missing)|mileage\\s+noted\\s+on\\s+the\\s+estimate\\s*:\\s*no", ln):
            bad = True
        if (reported_days is not None) and re.search(r"(?i)(repair\\s+days|approx(imate)?\\s+repair\\s+days).*(not|missing)", ln):
            bad = True
        if not bad:
            out.append(ln)
    return "\\n".join(out).strip()

# ----------------- API -----------------
@app.get("/")
async def root():
    return {"status":"ok"}

@app.post("/vision-review")
async def vision_review(
    files: List[UploadFile] = File(...),
    client_rules: str = Form(""),
    file_number: str = Form(...),
    ia_company: str = Form(...),
    appraiser_id: str = Form(...),
    ai_intent: str = Form("guidelines_only")
):
    if not appraiser_id.strip():
        return JSONResponse(status_code=400, content={"error":"Appraiser ID is required."})

    intent = ai_intent if ai_intent in INTENTS else "guidelines_only"
    request_type_label = INTENTS.get(intent, intent)

    # Partition uploads
    pdfs: List[Tuple[str, bytes]] = []
    images: List[Tuple[str, bytes]] = []
    docs: List[str] = []
    for f in files:
        raw = await f.read()
        name = (f.filename or "upload").lower()
        if name.endswith(".pdf"): pdfs.append((name, raw))
        elif name.endswith((".jpg",".jpeg",".png",".webp")): images.append((name, raw))
        elif name.endswith(".docx"): docs.append(ocr_docx_text(io.BytesIO(raw)))
        elif name.endswith(".txt"): docs.append(raw.decode("utf-8", errors="ignore"))

    photos_present = len(images) > 0

    # Estimate text (fast): text-layer first, then OCR fallback for sparse files
    limit = 12 if intent == "comprehensive" else 5
    est_text = ""
    if pdfs:
        est_text = fast_pdf_text(pdfs[0][1], limit_pages=limit)
        if len(est_text.strip()) < 120:  # fallback only if text-layer is too thin
            est_text = ocr_pdf_text(pdfs[0][1], limit_pages=limit, dpi=200)
    elif docs:
        est_text = "\\n\\n".join(docs)

    # ALWAYS extract VIN / vehicle / claim / mileage
    vin = None
    if pdfs:
        vin = vin_from_text(est_text)
        if not vin:
            vin = vin_from_pdf_boxes(pdf_pages(pdfs[0][1], limit_pages=3, dpi=240))
        if not vin:
            est_text_hi = ocr_pdf_text(pdfs[0][1], limit_pages=3, dpi=240, psms=("--psm 6","--psm 11"))
            vin = vin_from_text(est_text_hi)
    else:
        vin = vin_from_text(est_text)
    vin = vin or "N/A"

    vehicle = vehicle_from_text(est_text) or "N/A"
    mileage = mileage_from_text(est_text)
    claim = claim_from_text(est_text) or "N/A"

    # Evidence flags & facts
    has_clean_value = bool(re.search(r"(clean\\s*retail|NADA|KBB|Black\\s*Book|J\\.?D\\.?\\s*Power|valuation)", est_text, re.IGNORECASE))
    has_advisor = bool(re.search(r"(advisor\\s*report|ccc\\s*one\\s*advisor)", est_text, re.IGNORECASE))
    facts = estimate_facts(est_text)
    reported_days = facts.get("hours", {}).get("days_reported")
    mileage_present = bool(mileage)

    def vision_images():
        out=[]
        for _, blob in images:
            b64 = base64.b64encode(blob).decode("utf-8")
            out.append({"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}})
        return out

    gpt_output = ""

    if intent == "guidelines_only":
        system = (
            "Auto-damage compliance auditor.\\n"
            f"PhotosPresent={photos_present}. Do NOT mention photos if false.\\n"
            f"CleanRetailProvided={has_clean_value}. AdvisorReportProvided={has_advisor}.\\n"
            f"- MileagePresent={mileage_present}; DaysReported={reported_days}; "
            f"DaysByFormula={facts.get('hours',{}).get('days_formula_hrs_div_5')}.\\n"
            "- If MileagePresent=True, explicitly cite the mileage from the estimate and DO NOT say mileage is missing.\\n"
            "- If DaysReported is a number, DO NOT say repair days are missing; cite 'Days to Repair: X'. "
            "If DaysReported is None but hours exist, compute approximate days (hours/5) and label it as calculated.\\n"
            "Sections (bullet points):\\n"
            "1) Client Quick Summary (1-3 lines).\\n"
            "2) Fatal Errors (only if any, be explicit).\\n"
            "3) Parts/Tax/Labor Compliance (rates, P&M, tax).\\n"
            "4) Documentation Requirements (Clean Retail Value, Advisor, open items).\\n"
            "5) Summary & Recommendations (1-2 lines).\\n"
            + json.dumps(facts, indent=2)
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\\n"+(client_rules or "")[:12000]},
            {"type":"text","text":"\\n\\nESTIMATE (OCR/TEXT):\\n"+(est_text or "")[:12000]},
        ]
        rsp = chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=900)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()
        if not photos_present: gpt_output = strip_photo_claims(gpt_output)
        gpt_output = correct_false_negatives(gpt_output, mileage_present, reported_days)

    elif intent == "comprehensive":
        system = (
            "Auto-damage auditor.\\n"
            f"PhotosPresent={photos_present}. If false, omit photo sections and explicitly state no photos were provided.\\n"
            f"CleanRetailProvided={has_clean_value}. AdvisorReportProvided={has_advisor}. Only say 'included' if True; else 'missing'.\\n"
            f"- MileagePresent={mileage_present}; DaysReported={reported_days}; "
            f"DaysByFormula={facts.get('hours',{}).get('days_formula_hrs_div_5')}.\\n"
            "- If MileagePresent=True, explicitly cite the mileage from the estimate and DO NOT say mileage is missing.\\n"
            "- If DaysReported is a number, DO NOT say repair days are missing; cite 'Days to Repair: X'. "
            "If DaysReported is None but hours exist, compute approximate days (hours/5) and label it as calculated.\\n"
            "Sections: Client Quick Summary Compliance → Fatal Errors → "
            + ("Client Photo Rules → Estimate↔Photos Comparison → " if photos_present else "")
            + "Parts/Tax/Labor → Summary.\\n"
            "Be concise and concrete. Use provided facts; do not invent values.\\n"
            + json.dumps(facts, indent=2)
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\\n"+(client_rules or "")[:8000]},
            {"type":"text","text":"\\n\\nESTIMATE (OCR/TEXT):\\n"+(est_text or "")[:10000]},
        ]
        if photos_present: user.extend(vision_images())
        rsp = chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=950)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()
        if not photos_present: gpt_output = strip_photo_claims(gpt_output)
        gpt_output = correct_false_negatives(gpt_output, mileage_present, reported_days)

    elif intent == "photos_only":
        if not photos_present:
            gpt_output = "No photos were provided with this request."
        else:
            system = "Compare ESTIMATE to PHOTOS only. Sections: Photo Coverage, Visible Damage vs Estimate, Discrepancies, Summary."
            user = [{"type":"text","text":"ESTIMATE (OCR/TEXT):\\n"+(est_text or "")[:8000]}]
            user.extend(vision_images())
            rsp = chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=700)
            gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()

    elif intent == "invoices_with_photos":
        invoices_text = ""
        for name, raw in pdfs:
            if any(k in name for k in ("invoice","receipt","supplement")):
                invoices_text += ocr_pdf_text(raw, limit_pages=5)
        if not invoices_text and pdfs:
            invoices_text = ocr_pdf_text(pdfs[0][1], limit_pages=5)

        photo_note = "Photos were NOT provided; do not invent photo content." if not photos_present else "Photos were provided."
        system = (
            "Audit whether the supplement/estimate is substantiated by attached invoices and (if present) by the photos.\\n"
            f"{photo_note}\\n"
            "Write sections: Invoices Summary, Support vs Estimate Lines, (Photo Corroboration, omit if no photos), Missing Documentation, Summary."
        )
        user = [
            {"type":"text","text":"ESTIMATE (OCR/TEXT):\\n"+(est_text or "")[:6000]},
            {"type":"text","text":"\\n\\nINVOICES (OCR):\\n"+(invoices_text or "")[:6000]},
        ]
        if photos_present: user.extend(vision_images())
        rsp = chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=800)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()
        if not photos_present: gpt_output = strip_photo_claims(gpt_output)
        gpt_output = correct_false_negatives(gpt_output, mileage_present, reported_days)

    elif intent == "docs_checklist":
        system = (
            "Documentation checklist only. State whether the estimate includes: Clean Retail Value printout, Advisor Report, "
            "and any other required docs mentioned in guidelines. If not found, mark 'missing'. Be terse and factual."
        )
        user = [
            {"type":"text","text":"CLIENT GUIDELINES:\\n"+(client_rules or "")[:6000]},
            {"type":"text","text":"\\n\\nESTIMATE (OCR/TEXT):\\n"+(est_text or "")[:8000]},
            {"type":"text","text":f"\\n\\nDetected:\\nCleanRetailProvided={has_clean_value}\\nAdvisorReportProvided={has_advisor}"},
        ]
        rsp = chat([{"role":"system","content":system},{"role":"user","content":user}], max_tokens=400)
        gpt_output = (rsp.choices[0].message.content if rsp else "Automated narrative unavailable.").strip()

    # light score only for guideline-type requests
    def light_score(txt: str, rules: str) -> int:
        score = 100
        if rules and intent in ("guidelines_only","comprehensive","docs_checklist"):
            if "labor" in rules.lower() and not re.search(r"(labor|rate).{0,80}\\$", txt, re.IGNORECASE | re.DOTALL): score -= 10
            if "tax" in rules.lower() and "tax" not in txt.lower(): score -= 10
        return max(0, min(100, score))
    comp_score = light_score(est_text, client_rules)

    # ----------------- PDF -----------------
    pdf = FPDF(); pdf.add_page()
    try:
        pdf.add_font("DejaVu","","DejaVuSans.ttf", uni=True); pdf.set_font("DejaVu", size=11)
    except Exception:
        pdf.set_font("Arial", size=11)

    pdf.cell(200,10,"NSPXN.com AI Review Report",ln=True,align="C")
    pdf.ln(5); pdf.set_font_size(10)
    pdf.multi_cell(0,6,f"File Number: {file_number}")
    pdf.multi_cell(0,6,f"IA Company: {ia_company}")
    pdf.multi_cell(0,6,f"Request Type: {request_type_label}")
    pdf.multi_cell(0,6,f"Appraiser ID #: {appraiser_id}")
    pdf.ln(4)
    pdf.multi_cell(0,6,f"Claim #: {claim}")
    pdf.multi_cell(0,6,f"VIN (from estimate): {vin}")
    vin_line = "Not requested"
    if intent in ("comprehensive",):
        vin_line = "Included in narrative" if photos_present else "Photos not provided"
    pdf.multi_cell(0,6,f"VIN verification (estimate vs photo): {vin_line}")
    pdf.multi_cell(0,6,f"Vehicle: {vehicle}")
    if mileage: pdf.multi_cell(0,6,f"Odometer (from estimate): {mileage}")
    if reported_days is not None:
        pdf.multi_cell(0,6,f"Days to Repair (reported): {reported_days}")
    elif facts.get("hours",{}).get("days_formula_hrs_div_5"):
        pdf.multi_cell(0,6,f"Approx. Days to Repair (calc.): {facts['hours']['days_formula_hrs_div_5']}")
    pdf.multi_cell(0,6,f"Compliance Score: {comp_score}%")

    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0,8,"AI-4-IA Review Summary",ln=True)
    pdf.set_font_size(10); pdf.multi_cell(0,6,gpt_output or "No narrative generated.")

    pdf.ln(4); pdf.set_font_size(12); pdf.cell(0,8,"Estimate ↔ Photos Consistency Review",ln=True)
    pdf.set_font_size(10)
    if intent in ("comprehensive","photos_only","invoices_with_photos") and photos_present:
        pdf.multi_cell(0,6,"Included in narrative above (single-pass review).")
    else:
        pdf.multi_cell(0,6,"Not requested or no photos provided.")

    pdf_path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    try:
        with open(pdf_path,"wb") as f: f.write(pdf.output(dest="S").encode("latin-1"))
        log.info(f"PDF saved → {pdf_path}")
    except Exception as e:
        log.error(f"PDF write error: {e}")

    # ----------------- Email -----------------
    try:
        msg = EmailMessage()
        msg["Subject"] = f"AI-4-IA Review: {claim}"
        msg["From"] = "noreply@nspxn.com"
        msg["To"] = "info@nspxn.com"
        body = f"""NSPXN.com AI4IA Review Report

File Number: {file_number}
IA Company: {ia_company}
Appraiser ID #: {appraiser_id}

Request Type: {request_type_label}

Claim #: {claim}
VIN (from estimate): {vin}
VIN verification (estimate vs photo): {vin_line}
Vehicle: {vehicle}
{('Odometer (from estimate): ' + mileage) if mileage else ''}
{('Days to Repair (reported): ' + str(reported_days)) if reported_days is not None else (('Approx. Days to Repair (calc.): ' + str(facts.get('hours',{}).get('days_formula_hrs_div_5'))) if facts.get('hours',{}).get('days_formula_hrs_div_5') else '')}

Compliance Score: {comp_score}%

Summary:
{gpt_output}
"""
        msg.set_content(body)
        with smtplib.SMTP_SSL("mail.tierra.net", 465) as smtp:
            smtp.login("info@nspxn.com", "grr2025GRR")
            smtp.send_message(msg)
    except Exception as e:
        log.error(f"Email error (continuing): {e}")

    return {
        "request_type": request_type_label,
        "gpt_output": gpt_output,
        "file_number": file_number,
        "claim_number": claim,
        "vehicle": vehicle,
        "vin_estimate": vin,
        "vin_verification": vin_line,
        "odometer_estimate": mileage or "Not documented",
        "days_to_repair": reported_days if reported_days is not None else (facts.get("hours",{}).get("days_formula_hrs_div_5") or "Not documented"),
        "score": f"{comp_score}%"
    }

@app.get("/download-pdf")
async def download_pdf(file_number: str):
    path = os.path.join(PDF_DIR, f"{file_number}.pdf")
    if os.path.exists(path):
        return FileResponse(path=path, media_type="application/pdf", filename=f"{file_number}.pdf")
    return JSONResponse(status_code=404, content={"detail":"Not Found"})

@app.get("/client-rules/{client_name}")
async def get_client_rules(client_name: str):
    rules_dir = "client_rules"
    fp = os.path.join(rules_dir, f"{client_name}.docx")
    if os.path.exists(fp):
        try:
            doc = Document(fp)
            text = "\\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return {"text": text}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    return JSONResponse(status_code=404, content={"error":"Rules not found for this client."})
